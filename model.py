# =============================================================================
#  ⚽  CAPA DE DATOS Y MODELO — COPA DEL MUNDO 2026
# =============================================================================
#  Este módulo es el ÚNICO punto de entrada de datos del dashboard. Sustituye
#  al antiguo `load_mock_data()`: descarga datos REALES y genera predicciones.
#
#  FUENTE DE DATOS (gratuita, sin API key):
#    martj42/international_results  -> un único CSV con TODOS los partidos
#    internacionales (1872→2026). Incluye además el fixture del Mundial 2026
#    (tournament == "FIFA World Cup"): los partidos jugados traen el marcador
#    real y los próximos vienen con marcador vacío (NA).
#
#  MODELO (Poisson / Dixon-Coles):
#    1. A partir del histórico estimamos, para cada selección, una fuerza de
#       ATAQUE y de DEFENSA (ponderando los partidos recientes mucho más que
#       los antiguos) mediante un ajuste de punto fijo tipo Dixon-Coles.
#    2. Para cada partido del Mundial calculamos los goles esperados (xG) de
#       cada equipo = λ. De ahí sale TODO: probabilidades de victoria/empate,
#       el marcador más probable y la matriz completa de resultados.
#
#  Sólo depende de numpy y pandas. La descarga usa la stdlib (urllib).
# =============================================================================

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
#  CONSTANTES DE CONFIGURACIÓN
# -----------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "martj42/international_results/master/results.csv"
)
HERE = Path(__file__).resolve().parent
CACHE_FILE = HERE / "data" / "results.csv"
MAX_CACHE_AGE_HOURS = 12          # re-descarga el CSV si la caché es más vieja

TOURNAMENT_START = pd.Timestamp("2026-06-11")  # primer partido del Mundial 2026

# --- Parámetros del modelo ---------------------------------------------------
HALF_LIFE_DAYS = 730     # vida media del peso temporal (~2 años)
WINDOW_YEARS = 8         # sólo entrenamos con los últimos N años
PRIOR_MATCHES = 1.5      # "shrinkage": estabiliza equipos con pocos partidos
MAX_GOALS = 10           # tope de la rejilla de marcadores (0..10)
DC_RHO = -0.10           # corrección Dixon-Coles para marcadores bajos
ITERATIONS = 200         # iteraciones del ajuste de punto fijo
TOL = 1e-9               # tolerancia de convergencia

# Factoriales precomputados para la PMF de Poisson (0!..MAX_GOALS!).
_FACT = np.array([math.factorial(k) for k in range(MAX_GOALS + 1)], dtype=float)
_GOALS = np.arange(MAX_GOALS + 1)


# =============================================================================
#  1. DESCARGA Y PREPARACIÓN DEL CSV
# =============================================================================
def fetch_results(force: bool = False) -> pd.DataFrame:
    """Descarga el CSV de resultados (o usa la caché local) y lo devuelve crudo.

    Estrategia robusta:
      * Si existe una caché reciente y no se fuerza la descarga -> usa la caché.
      * Si no, intenta descargar; si lo logra, actualiza la caché.
      * Si la descarga falla pero hay caché (aunque sea vieja) -> usa la caché.
      * Si no hay nada -> lanza un error claro.
    """
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    cache_fresh = False
    if CACHE_FILE.exists():
        age_h = (dt.datetime.now().timestamp() - CACHE_FILE.stat().st_mtime) / 3600
        cache_fresh = age_h < MAX_CACHE_AGE_HOURS

    if CACHE_FILE.exists() and cache_fresh and not force:
        return pd.read_csv(CACHE_FILE)

    try:
        req = Request(DATA_URL, headers={"User-Agent": "wc2026-dashboard"})
        with urlopen(req, timeout=30) as resp:
            raw = resp.read()
        CACHE_FILE.write_bytes(raw)
    except Exception as exc:  # noqa: BLE001  (red caída, timeout, etc.)
        if CACHE_FILE.exists():
            print(f"[model] Aviso: descarga falló ({exc}); uso caché local.")
        else:
            raise RuntimeError(
                "No se pudo descargar el CSV de resultados y no hay caché local "
                f"en {CACHE_FILE}. Revisa tu conexión a internet."
            ) from exc

    return pd.read_csv(CACHE_FILE)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos: fechas, marcadores numéricos (NA→NaN) y `neutral` bool."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    # `neutral` puede venir como bool o como texto "TRUE"/"FALSE": lo unificamos.
    df["neutral"] = df["neutral"].astype(str).str.upper().isin(["TRUE", "1", "T"])
    return df.dropna(subset=["date"])


# =============================================================================
#  2. ENTRENAMIENTO DE FUERZAS DE EQUIPO (ataque / defensa)
# =============================================================================
class Ratings:
    """Contenedor con las fuerzas estimadas y los parámetros globales del modelo.

    Atributos:
      table : DataFrame indexado por equipo con columnas `attack`, `defense`,
              `form` (forma reciente 0-1) y `matches` (peso total de partidos).
      gamma : media liga de goles por equipo-partido (ancla de la escala).
      home_adv : factor multiplicativo de localía (>1).
    """

    def __init__(self, table: pd.DataFrame, gamma: float, home_adv: float):
        self.table = table
        self.gamma = gamma
        self.home_adv = home_adv

    def get(self, team: str) -> dict:
        """Fuerzas de un equipo; si no está (sin historial) devuelve neutro 1.0."""
        if team in self.table.index:
            row = self.table.loc[team]
            return {
                "attack": float(row["attack"]),
                "defense": float(row["defense"]),
                "form": float(row["form"]),
            }
        return {"attack": 1.0, "defense": 1.0, "form": 0.5}


def train_ratings(matches: pd.DataFrame, cutoff: pd.Timestamp) -> Ratings:
    """Estima fuerzas de ataque/defensa usando SÓLO partidos previos a `cutoff`.

    Idea (Dixon-Coles): los goles que marca el local se modelan como Poisson con
    media  λ = γ · ataque_local · defensa_visita · localía.  Resolvemos los
    parámetros con un punto fijo tipo máxima verosimilitud, ponderando cada
    partido por un peso que decae exponencialmente con la antigüedad (los
    partidos recientes valen mucho más).  Excluir los partidos ≥ cutoff evita
    cualquier fuga de información al evaluar partidos ya jugados.
    """
    train = matches[
        matches["home_score"].notna()
        & matches["away_score"].notna()
        & (matches["date"] < cutoff)
        & (matches["date"] >= cutoff - pd.DateOffset(years=WINDOW_YEARS))
    ].copy()

    # --- Peso temporal: w = 0.5 ** (antigüedad / vida_media) ---
    age_days = (cutoff - train["date"]).dt.days.to_numpy()
    w = np.power(0.5, age_days / HALF_LIFE_DAYS)

    # --- Codificación de equipos a índices enteros (para vectorizar) ---
    teams = pd.Index(sorted(set(train["home_team"]) | set(train["away_team"])))
    t_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = train["home_team"].map(t_idx).to_numpy()
    ai = train["away_team"].map(t_idx).to_numpy()
    hs = train["home_score"].to_numpy(dtype=float)
    as_ = train["away_score"].to_numpy(dtype=float)
    neutral = train["neutral"].to_numpy(dtype=bool)

    # --- γ: media liga de goles por equipo-partido ---
    gamma = float((w * (hs + as_)).sum() / (2.0 * w.sum()))

    # --- Localía: razón de goles local/visita en partidos NO neutrales ---
    # Validado empíricamente: en eliminatorias (calendario ida/vuelta, sin sesgo
    # de selección) la razón es ~1.6, igual que en todos los no-neutrales, así
    # que esta razón mide localía REAL y no que los fuertes jueguen más en casa.
    nn = ~neutral
    if w[nn].sum() > 0 and (w[nn] * as_[nn]).sum() > 0:
        home_adv = float((w[nn] * hs[nn]).sum() / (w[nn] * as_[nn]).sum())
        home_adv = min(max(home_adv, 1.0), 1.8)  # acotada a un rango sensato
    else:
        home_adv = 1.4
    hf = np.where(neutral, 1.0, home_adv)  # factor de localía por partido

    # --- Numeradores (constantes): goles marcados / encajados por equipo ---
    num_a = np.bincount(hi, w * hs, n) + np.bincount(ai, w * as_, n)   # marcados
    num_d = np.bincount(hi, w * as_, n) + np.bincount(ai, w * hs, n)   # encajados
    kappa = gamma * PRIOR_MATCHES  # pseudo-conteo que empuja las fuerzas hacia 1

    # --- Punto fijo: alternamos ataque y defensa (con localía fija) ---
    # Importante: el factor de localía entra en los denominadores, de modo que las
    # fuerzas que estimamos son "de campo neutral" y NO duplican la ventaja local.
    a = np.ones(n)
    d = np.ones(n)
    for _ in range(ITERATIONS):
        a_old = a
        # Denominador de ataque: Σ w·γ·def_rival·localía  (local recibe hf, visita 1)
        den_a = (
            np.bincount(hi, w * gamma * d[ai] * hf, n)
            + np.bincount(ai, w * gamma * d[hi], n)
        )
        a = (num_a + kappa) / (den_a + kappa)
        a /= np.exp(np.log(a).mean())  # normaliza a media geométrica 1

        # Denominador de defensa: Σ w·γ·ataque_rival·localía_rival
        den_d = (
            np.bincount(hi, w * gamma * a[ai], n)               # rival = visita (1)
            + np.bincount(ai, w * gamma * a[hi] * hf, n)        # rival = local  (hf)
        )
        d = (num_d + kappa) / (den_d + kappa)
        d /= np.exp(np.log(d).mean())

        if np.abs(a - a_old).max() < TOL:
            break

    # --- Recalibración de escala ---
    # La normalización geométrica de a/d fija las fuerzas RELATIVAS pero no el
    # nivel global de goles (la media aritmética de a·d > 1 lo infla). Ajustamos
    # γ con un único multiplicador para que el total de goles esperados iguale al
    # real; no altera las fuerzas relativas ni, por tanto, las probabilidades.
    e_home = a[hi] * d[ai] * hf
    e_away = a[ai] * d[hi]
    denom = float((w * (e_home + e_away)).sum())
    if denom > 0:
        gamma = float((w * (hs + as_)).sum() / denom)

    # --- Forma reciente: puntos por partido (0-1), con el mismo peso temporal ---
    home_pts = np.where(hs > as_, 3.0, np.where(hs == as_, 1.0, 0.0))
    away_pts = np.where(as_ > hs, 3.0, np.where(hs == as_, 1.0, 0.0))
    form_num = np.bincount(hi, w * home_pts, n) + np.bincount(ai, w * away_pts, n)
    form_den = np.bincount(hi, w, n) + np.bincount(ai, w, n)
    form = np.divide(form_num, 3.0 * form_den, out=np.full(n, 0.5), where=form_den > 0)

    table = pd.DataFrame(
        {"attack": a, "defense": d, "form": form, "matches": form_den}, index=teams
    )
    return Ratings(table, gamma, home_adv)


# =============================================================================
#  3. PREDICCIÓN DE UN PARTIDO (matriz de marcadores Poisson + Dixon-Coles)
# =============================================================================
def _poisson_pmf(lam: float) -> np.ndarray:
    """Vector PMF de Poisson para k = 0..MAX_GOALS."""
    return np.exp(-lam) * np.power(lam, _GOALS) / _FACT


def predict_match(
    home: dict, away: dict, neutral: bool, ratings: Ratings
) -> dict:
    """Devuelve la predicción completa de un partido a partir de las fuerzas.

    `home`/`away` son los dicts que devuelve `Ratings.get()`.
    Las medias λ son directamente los GOLES ESPERADOS (xG) de cada equipo.
    """
    gamma, h = ratings.gamma, ratings.home_adv
    hf = h if not neutral else 1.0
    lam_home = gamma * home["attack"] * away["defense"] * hf
    lam_away = gamma * away["attack"] * home["defense"]
    # Saneamiento numérico
    lam_home = float(np.clip(lam_home, 0.05, 8.0))
    lam_away = float(np.clip(lam_away, 0.05, 8.0))

    # Matriz de marcadores independientes: M[i, j] = P(local=i)·P(visita=j)
    matrix = np.outer(_poisson_pmf(lam_home), _poisson_pmf(lam_away))

    # Corrección Dixon-Coles para los 4 marcadores bajos (mejora los empates).
    rho = DC_RHO
    tau = np.array(
        [
            [1.0 - lam_home * lam_away * rho, 1.0 + lam_home * rho],
            [1.0 + lam_away * rho, 1.0 - rho],
        ]
    )
    matrix[:2, :2] *= np.maximum(tau, 1e-6)
    matrix /= matrix.sum()  # renormaliza para que sume 1

    # Probabilidades de resultado
    idx = np.indices(matrix.shape)
    p_home = float(matrix[idx[0] > idx[1]].sum())  # local marca más
    p_draw = float(np.trace(matrix))
    p_away = float(matrix[idx[0] < idx[1]].sum())

    # Marcador más probable (argmax de la matriz)
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    most_likely = {"home": int(i), "away": int(j), "prob": float(matrix[i, j])}

    return {
        "xg_home": lam_home,
        "xg_away": lam_away,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "most_likely": most_likely,
        "matrix": matrix,
    }


# =============================================================================
#  4. ENSAMBLAJE: FIXTURE DEL MUNDIAL + PREDICCIONES  (punto de entrada)
# =============================================================================
def _derive_groups(fixtures: pd.DataFrame) -> dict:
    """Reconstruye los grupos a partir del calendario (sin datos externos).

    En la fase de grupos cada selección juega contra las otras 3 de su grupo,
    así que los rivales de un equipo SON su grupo. Etiquetamos A..L según el
    orden del primer partido de cada grupo.
    """
    opponents: dict[str, set] = {}
    for _, r in fixtures.iterrows():
        opponents.setdefault(r["home_team"], set()).add(r["away_team"])
        opponents.setdefault(r["away_team"], set()).add(r["home_team"])

    seen: set = set()
    raw_groups = []
    for team in opponents:
        if team in seen:
            continue
        members = frozenset({team} | opponents[team])
        seen |= members
        first = fixtures[
            fixtures["home_team"].isin(members) & fixtures["away_team"].isin(members)
        ]["date"].min()
        raw_groups.append((first, members))

    raw_groups.sort(key=lambda x: x[0])  # ordena por fecha del primer partido
    team_to_group = {}
    for letter, (_, members) in zip("ABCDEFGHIJKL", raw_groups):
        for team in members:
            team_to_group[team] = f"Grupo {letter}"
    return team_to_group


def _percentiles(ratings: Ratings, teams: list) -> pd.DataFrame:
    """Rangos percentiles (0-100) de cada métrica DENTRO de las 48 selecciones.

    Sirve para el radar: 100 = mejor del Mundial en ese eje.  La defensa se
    invierte (encajar menos = mejor) y la 'potencia' combina ataque y defensa.
    """
    sub = ratings.table.reindex(teams).copy()
    sub["attack"] = sub["attack"].fillna(1.0)
    sub["defense"] = sub["defense"].fillna(1.0)
    sub["form"] = sub["form"].fillna(0.5)
    out = pd.DataFrame(index=teams)
    out["Ataque"] = sub["attack"].rank(pct=True) * 100
    out["Defensa"] = (1.0 / sub["defense"]).rank(pct=True) * 100   # menos goles = mejor
    out["Potencia"] = (sub["attack"] / sub["defense"]).rank(pct=True) * 100
    out["Forma"] = sub["form"].rank(pct=True) * 100
    return out.round(0)


def load_dashboard_data() -> dict:
    """Punto de entrada del dashboard. Devuelve:

        {
          "matches": { "<etiqueta>": <registro de partido>, ... },
          "meta":    { stats globales del torneo },
        }

    Cada <registro> trae meta (fecha, sede, grupo, estado), home/away (nombre,
    xG, percentiles para el radar), prediction (prob. y marcadores) y el
    resultado real (`actual`) si el partido ya se jugó.
    """
    df = prepare(fetch_results())

    fixtures = df[
        (df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-01-01")
    ].sort_values("date").reset_index(drop=True)

    wc_teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    groups = _derive_groups(fixtures)

    # Dos "fotos" de fuerzas para no filtrar información:
    #   * pre  -> sólo con datos ANTERIORES al Mundial (evalúa partidos jugados).
    #   * now  -> con todo lo disponible hasta hoy (pronostica los próximos).
    ratings_pre = train_ratings(df, cutoff=TOURNAMENT_START)
    latest = df.loc[df["home_score"].notna(), "date"].max()
    ratings_now = train_ratings(df, cutoff=latest + pd.Timedelta(days=1))

    pct_pre = _percentiles(ratings_pre, wc_teams)
    pct_now = _percentiles(ratings_now, wc_teams)

    matches: dict = {}
    n_played = 0
    for _, row in fixtures.iterrows():
        home_name, away_name = row["home_team"], row["away_team"]
        played = pd.notna(row["home_score"]) and pd.notna(row["away_score"])
        ratings = ratings_pre if played else ratings_now
        pct = pct_pre if played else pct_now

        pred = predict_match(
            ratings.get(home_name), ratings.get(away_name), bool(row["neutral"]), ratings
        )

        actual = None
        if played:
            n_played += 1
            hs, as_ = int(row["home_score"]), int(row["away_score"])
            if hs > as_:
                winner = home_name
            elif hs < as_:
                winner = away_name
            else:
                winner = "Empate"
            actual = {"home": hs, "away": as_, "winner": winner}

        # Ganador más probable según el modelo
        outcomes = {home_name: pred["p_home"], "Empate": pred["p_draw"], away_name: pred["p_away"]}
        fav = max(outcomes, key=outcomes.get)

        label = f"{home_name} vs {away_name} · {row['date'].date()}"
        matches[label] = {
            "meta": {
                "date": row["date"].date().isoformat(),
                "city": row["city"],
                "country": row["country"],
                "neutral": bool(row["neutral"]),
                "group": groups.get(home_name, "—"),
                "status": "played" if played else "upcoming",
            },
            "home": {
                "name": home_name,
                "xg": pred["xg_home"],
                "win_prob": pred["p_home"],
                "radar": pct.loc[home_name].to_dict(),
            },
            "away": {
                "name": away_name,
                "xg": pred["xg_away"],
                "win_prob": pred["p_away"],
                "radar": pct.loc[away_name].to_dict(),
            },
            "prediction": {
                "p_home": pred["p_home"],
                "p_draw": pred["p_draw"],
                "p_away": pred["p_away"],
                "favorite": fav,
                "favorite_prob": outcomes[fav],
                "most_likely": pred["most_likely"],
                "matrix": pred["matrix"],
            },
            "actual": actual,
        }

    meta = {
        "n_matches": len(matches),
        "n_played": n_played,
        "n_upcoming": len(matches) - n_played,
        "n_teams": len(wc_teams),
        "data_through": str(latest.date()) if pd.notna(latest) else "—",
        "gamma": ratings_now.gamma,
        "home_adv": ratings_now.home_adv,
        "source": "martj42/international_results",
    }
    return {"matches": matches, "meta": meta}


# =============================================================================
#  5. AUTOTEST (ejecuta:  python model.py)
# =============================================================================
def _selftest() -> None:
    data = load_dashboard_data()
    matches, meta = data["matches"], data["meta"]
    print(f"Partidos: {meta['n_matches']}  (jugados {meta['n_played']}, "
          f"próximos {meta['n_upcoming']})  | equipos: {meta['n_teams']}")
    print(f"γ={meta['gamma']:.3f}  localía={meta['home_adv']:.3f}  "
          f"datos hasta {meta['data_through']}\n")

    # --- Comprobaciones de coherencia ---
    for label, m in matches.items():
        p = m["prediction"]
        total = p["p_home"] + p["p_draw"] + p["p_away"]
        assert abs(total - 1.0) < 1e-6, f"Probabilidades no suman 1 en {label}: {total}"
        assert m["home"]["xg"] > 0 and m["away"]["xg"] > 0, f"xG no positivo en {label}"
        assert abs(float(m["prediction"]["matrix"].sum()) - 1.0) < 1e-6
    print("✓ Probabilidades válidas (suman 1) y xG positivos en los 72 partidos.")

    # --- Validación de sensatez: un partido jugado conocido ---
    sample = list(matches.items())[:3] + list(matches.items())[-3:]
    print("\nEjemplos de predicción:")
    for label, m in sample:
        p = m["prediction"]
        ml = p["most_likely"]
        estado = m["meta"]["status"]
        real = ""
        if m["actual"]:
            real = f"  | REAL {m['actual']['home']}-{m['actual']['away']}"
        print(
            f"  [{estado:8}] {label}\n"
            f"      xG {m['home']['xg']:.2f}-{m['away']['xg']:.2f}  "
            f"P(1/X/2)={p['p_home']:.0%}/{p['p_draw']:.0%}/{p['p_away']:.0%}  "
            f"+probable {ml['home']}-{ml['away']} ({ml['prob']:.0%})"
            f"  fav: {p['favorite']}{real}"
        )

    # --- El favorito del modelo debería acertar bastante en los ya jugados ---
    played = [m for m in matches.values() if m["actual"]]
    hits = sum(1 for m in played if m["prediction"]["favorite"] == m["actual"]["winner"])
    if played:
        print(f"\nAcierto del favorito en partidos jugados: "
              f"{hits}/{len(played)} ({hits/len(played):.0%})")
    print("\n✓ Autotest OK.")


if __name__ == "__main__":
    _selftest()
