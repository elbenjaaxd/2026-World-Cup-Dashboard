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

# --- Tiros a puerta (ESTIMACIÓN) ---------------------------------------------
# No existe una fuente gratuita y accesible de tiros para estos partidos (FotMob
# bloqueó su API), así que estimamos los tiros a puerta a partir del xG:
#     tiros_a_puerta ≈ xG / conversión
# usando la tasa media de conversión tiro-a-puerta→gol del fútbol internacional
# (~0.30 goles por tiro a puerta). Es una aproximación transparente, NO un dato
# real, y así se etiqueta en la interfaz.
SOT_CONVERSION = 0.30

# --- Recomendaciones de apuestas ---------------------------------------------
# 'Recomendado' = mayor CONFIANZA del modelo. No se comparan con cuotas de casas
# de apuestas (no hay fuente gratuita), así que esto NO mide valor frente al
# mercado, sino la probabilidad que el modelo asigna a cada selección.
OVER_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)  # líneas de Más/Menos goles
CONF_MIN = 0.58       # probabilidad mínima del modelo para "recomendar" un pick
TIER_ALTA = 0.70      # umbral de confianza alta
TIER_MEDIA = 0.60     # umbral de confianza media (por debajo: especulativa)


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


def _poisson_sf(lam: float, k: int) -> float:
    """P(X >= k) para X ~ Poisson(lam). Se usa en las líneas de tiros a puerta."""
    if k <= 0:
        return 1.0
    k = min(k, MAX_GOALS + 1)
    cdf = float(np.sum(np.exp(-lam) * np.power(lam, _GOALS[:k]) / _FACT[:k]))
    return float(min(1.0, max(0.0, 1.0 - cdf)))


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

    # Mercados de goles derivados de la MISMA matriz (cálculo exacto):
    #   * Más/Menos N goles  -> suma de las celdas con (local+visita) > N
    #   * Ambos marcan (BTTS) -> suma de las celdas con local>=1 y visita>=1
    totals = idx[0] + idx[1]
    p_over = {ln: float(matrix[totals > ln].sum()) for ln in OVER_LINES}
    p_btts = float(matrix[(idx[0] >= 1) & (idx[1] >= 1)].sum())

    # Tiros a puerta estimados a partir del xG (ver SOT_CONVERSION). Son una
    # estimación, no un dato real; sirven de media para una Poisson en los picks.
    sot_home = lam_home / SOT_CONVERSION
    sot_away = lam_away / SOT_CONVERSION

    # Marcador más probable (argmax de la matriz)
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    most_likely = {"home": int(i), "away": int(j), "prob": float(matrix[i, j])}

    return {
        "xg_home": lam_home,
        "xg_away": lam_away,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over": p_over,
        "p_btts": p_btts,
        "sot_home": sot_home,
        "sot_away": sot_away,
        "most_likely": most_likely,
        "matrix": matrix,
    }


# =============================================================================
#  3b. MERCADOS Y RECOMENDACIONES DE APUESTAS
# =============================================================================
#  A partir de la predicción de un partido elegimos la MEJOR selección del
#  modelo en cada mercado (1X2, doble oportunidad, Más/Menos 2.5, ambos marcan
#  y tiros a puerta) junto con su probabilidad. Las que superan `CONF_MIN` se
#  consideran "recomendadas". Para los partidos ya jugados podemos GRADUAR esos
#  picks contra el resultado real y enseñar un histórico de aciertos.
# -----------------------------------------------------------------------------
def _tier(p: float) -> str:
    """Etiqueta de confianza de un pick según su probabilidad."""
    if p >= TIER_ALTA:
        return "Alta"
    if p >= TIER_MEDIA:
        return "Media"
    return "Especulativa"


def recommend_picks(rec: dict) -> list:
    """Mejor selección del modelo en cada mercado para UN partido.

    Devuelve una lista de 'picks' (un dict por mercado) con la probabilidad que
    el modelo asigna. No consulta cuotas: 'recomendado' = mayor confianza del
    modelo, no valor frente a una casa de apuestas.
    """
    pred = rec["prediction"]
    home, away = rec["home"]["name"], rec["away"]["name"]
    picks = []

    # 1) Resultado (1X2): el más probable de los tres.
    outcomes = [(home, pred["p_home"]), ("Empate", pred["p_draw"]), (away, pred["p_away"])]
    team, prob = max(outcomes, key=lambda x: x[1])
    sel = "Empate" if team == "Empate" else f"Gana {team}"
    picks.append({"market": "Resultado (1X2)", "selection": sel, "prob": prob,
                  "kind": "1x2", "team": team})

    # 2) Doble oportunidad: la combinación 1X / X2 / 12 más probable.
    dcs = [
        (f"{home} o empate", pred["p_home"] + pred["p_draw"], [home, "Empate"]),
        (f"{away} o empate", pred["p_away"] + pred["p_draw"], [away, "Empate"]),
        (f"{home} o {away} (sin empate)", pred["p_home"] + pred["p_away"], [home, away]),
    ]
    sel, prob, covers = max(dcs, key=lambda x: x[1])
    picks.append({"market": "Doble oportunidad", "selection": sel, "prob": prob,
                  "kind": "dc", "covers": covers})

    # 3) Goles Más/Menos 2.5 (el lado más probable).
    over25 = pred["p_over"][2.5]
    if over25 >= 0.5:
        picks.append({"market": "Goles 2.5", "selection": "Más de 2.5", "prob": over25,
                      "kind": "ou", "line": 2.5, "side": "over"})
    else:
        picks.append({"market": "Goles 2.5", "selection": "Menos de 2.5", "prob": 1 - over25,
                      "kind": "ou", "line": 2.5, "side": "under"})

    # 4) Ambos equipos marcan (BTTS).
    btts = pred["p_btts"]
    if btts >= 0.5:
        picks.append({"market": "Ambos marcan", "selection": "Sí", "prob": btts,
                      "kind": "btts", "side": "yes"})
    else:
        picks.append({"market": "Ambos marcan", "selection": "No", "prob": 1 - btts,
                      "kind": "btts", "side": "no"})

    # 5) Tiros a puerta del equipo dominante (ESTIMADO; no verificable). Tomamos
    #    la línea más alta cuya probabilidad (Poisson de media = tiros estimados)
    #    siga siendo razonable; si ninguna llega, caemos a la línea más baja.
    if rec["home"]["sot"] >= rec["away"]["sot"]:
        equipo, mean = home, rec["home"]["sot"]
    else:
        equipo, mean = away, rec["away"]["sot"]
    line, p = 2.5, _poisson_sf(mean, 3)
    for cand in (4.5, 3.5):
        pc = _poisson_sf(mean, math.ceil(cand))
        if pc >= TIER_MEDIA:
            line, p = cand, pc
            break
    picks.append({"market": "Tiros a puerta", "selection": f"{equipo} +{line}", "prob": p,
                  "kind": "sot", "team": equipo, "line": line})

    for pk in picks:
        pk["prob"] = float(pk["prob"])
        pk["tier"] = _tier(pk["prob"])
    return picks


def _grade_pick(pick: dict, actual: dict):
    """True/False si el pick acertó; None si no es verificable (tiros a puerta)."""
    kind = pick["kind"]
    h, a = actual["home"], actual["away"]
    if kind == "1x2":
        return actual["winner"] == pick["team"]
    if kind == "dc":
        return actual["winner"] in pick["covers"]
    if kind == "ou":
        over = (h + a) > pick["line"]
        return over if pick["side"] == "over" else (not over)
    if kind == "btts":
        yes = h >= 1 and a >= 1
        return yes if pick["side"] == "yes" else (not yes)
    return None  # tiros a puerta: no hay dato real con que graduar


def _build_recommendations(matches: dict) -> list:
    """Picks recomendados (prob >= CONF_MIN) de los PRÓXIMOS partidos, ordenados."""
    recs = []
    for label, m in matches.items():
        if m["meta"]["status"] != "upcoming":
            continue
        for pk in m["picks"]:
            if pk["prob"] < CONF_MIN:
                continue
            recs.append({
                "label": label, "date": m["meta"]["date"], "group": m["meta"]["group"],
                "home": m["home"]["name"], "away": m["away"]["name"],
                "market": pk["market"], "selection": pk["selection"],
                "prob": pk["prob"], "tier": pk["tier"], "kind": pk["kind"],
            })
    recs.sort(key=lambda r: r["prob"], reverse=True)
    return recs


def _backtest(matches: dict) -> dict:
    """Gradúa los picks recomendados en los partidos JUGADOS (histórico real).

    Usa los picks calculados con las fuerzas PREVIAS al torneo (los partidos
    jugados se predicen con `ratings_pre`), así que no hay fuga de información.
    Los tiros a puerta no se gradúan (no hay dato real).
    """
    graded = []
    for m in matches.values():
        if m["meta"]["status"] != "played":
            continue
        for pk in m["picks"]:
            if pk["prob"] < CONF_MIN:
                continue
            res = _grade_pick(pk, m["actual"])
            if res is None:
                continue
            graded.append((pk, res))

    n = len(graded)
    hits = sum(1 for _, r in graded if r)
    by_market, by_tier = {}, {}
    for pk, r in graded:
        for bucket, key in ((by_market, pk["market"]), (by_tier, pk["tier"])):
            d = bucket.setdefault(key, {"n": 0, "hits": 0})
            d["n"] += 1
            d["hits"] += int(bool(r))
    for bucket in (by_market, by_tier):
        for d in bucket.values():
            d["rate"] = d["hits"] / d["n"] if d["n"] else 0.0
    return {"n": n, "hits": hits, "rate": hits / n if n else 0.0,
            "by_market": by_market, "by_tier": by_tier, "conf_min": CONF_MIN}


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
                "sot": pred["sot_home"],
                "radar": pct.loc[home_name].to_dict(),
            },
            "away": {
                "name": away_name,
                "xg": pred["xg_away"],
                "win_prob": pred["p_away"],
                "sot": pred["sot_away"],
                "radar": pct.loc[away_name].to_dict(),
            },
            "prediction": {
                "p_home": pred["p_home"],
                "p_draw": pred["p_draw"],
                "p_away": pred["p_away"],
                "p_over": pred["p_over"],
                "p_btts": pred["p_btts"],
                "favorite": fav,
                "favorite_prob": outcomes[fav],
                "most_likely": pred["most_likely"],
                "matrix": pred["matrix"],
            },
            "actual": actual,
        }
        # Mejor pick del modelo por mercado (se calcula sobre el registro ya armado).
        matches[label]["picks"] = recommend_picks(matches[label])

    # Recomendaciones (próximos) y su histórico de aciertos (jugados).
    recommendations = _build_recommendations(matches)
    backtest = _backtest(matches)

    meta = {
        "n_matches": len(matches),
        "n_played": n_played,
        "n_upcoming": len(matches) - n_played,
        "n_teams": len(wc_teams),
        "data_through": str(latest.date()) if pd.notna(latest) else "—",
        "gamma": ratings_now.gamma,
        "home_adv": ratings_now.home_adv,
        "conf_min": CONF_MIN,
        "source": "martj42/international_results",
    }
    return {
        "matches": matches,
        "meta": meta,
        "recommendations": recommendations,
        "backtest": backtest,
    }


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
        # Mercados nuevos: rangos válidos y monotonía de las líneas Más/Menos.
        assert 0.0 <= p["p_btts"] <= 1.0, f"BTTS fuera de rango en {label}"
        assert all(0.0 <= v <= 1.0 for v in p["p_over"].values())
        assert p["p_over"][1.5] >= p["p_over"][2.5] >= p["p_over"][3.5], f"O/U no monótono en {label}"
        assert m["home"]["sot"] > 0 and m["away"]["sot"] > 0, f"tiros<=0 en {label}"
        assert m["picks"], f"sin picks en {label}"
    print("✓ Probabilidades, xG, mercados O/U·BTTS, tiros a puerta y picks OK en los 72 partidos.")

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

    # --- Recomendaciones de apuestas y su histórico ---
    recs, bt = data["recommendations"], data["backtest"]
    print(f"\nRecomendaciones (próximos, prob≥{CONF_MIN:.0%}): {len(recs)}")
    for r in recs[:8]:
        print(f"  {r['prob']:.0%} [{r['tier']:12}] {r['market']:18} "
              f"{r['selection']:30} · {r['home']} vs {r['away']}")
    if bt["n"]:
        print(f"\nBacktest en jugados (prob≥{bt['conf_min']:.0%}): "
              f"{bt['hits']}/{bt['n']} ({bt['rate']:.0%})")
        for mkt, d in sorted(bt["by_market"].items()):
            print(f"    {mkt:18} {d['hits']:>3}/{d['n']:<3} ({d['rate']:.0%})")

    print("\n✓ Autotest OK.")


if __name__ == "__main__":
    _selftest()
