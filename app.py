# =============================================================================
#  ⚽  DASHBOARD DE ANÁLISIS Y PREDICCIÓN — MUNDIAL 2026
# =============================================================================
#  App web interactiva que descarga datos REALES de selecciones y predice cada
#  partido del Mundial 2026: goles esperados (xG), probabilidad de victoria /
#  empate, marcador más probable y la matriz completa de resultados.
#
#  Ejecutar con:   streamlit run app.py
#
#  ARQUITECTURA (modular a propósito):
#    1. model.load_dashboard_data()  -> ÚNICO punto de datos. Descarga el CSV
#       histórico real, entrena el modelo Poisson/Dixon-Coles y devuelve las
#       predicciones de los 72 partidos de fase de grupos.  (Ver model.py)
#    2. Funciones de gráficos (Plotly): construyen cada figura desde el registro
#       de un partido.
#    3. Funciones de render (Streamlit): pintan cada sección de la pantalla.
#    4. main(): orquesta todo.
# =============================================================================

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import model

# -----------------------------------------------------------------------------
#  PALETA Y CONSTANTES DE ESTILO
# -----------------------------------------------------------------------------
COLOR_LOCAL = "#38BDF8"      # Azul cian  -> Equipo Local
COLOR_VISITANTE = "#FB923C"  # Ámbar      -> Equipo Visitante
COLOR_EMPATE = "#94A3B8"     # Gris       -> Empate
COLOR_FONDO = "#0E1117"      # Fondo oscuro (coincide con el dark theme nativo)
COLOR_CARD = "#161B22"       # Fondo de las tarjetas de KPI
COLOR_TEXTO = "#E6EDF3"      # Texto principal

# Ejes del radar de fuerzas. Coinciden con las claves que devuelve el modelo
# (percentiles 0-100 de cada selección dentro de las 48 del Mundial).
EJES_RADAR = ["Ataque", "Defensa", "Forma", "Potencia"]

# Escala de color del mapa de calor (oscuro -> cian), legible sobre fondo dark.
ESCALA_HEATMAP = [
    [0.0, "#0E1117"], [0.25, "#0F3A52"], [0.5, "#11638C"],
    [0.75, "#1E90C2"], [1.0, COLOR_LOCAL],
]


# =============================================================================
#  1. CAPA DE DATOS  (cacheada)
# =============================================================================
@st.cache_data(ttl=21600, show_spinner="⏳ Descargando datos y entrenando el modelo…")
def get_dashboard_data() -> dict:
    """Envuelve `model.load_dashboard_data()` con la caché de Streamlit.

    El trabajo pesado (descarga + entrenamiento) se ejecuta una sola vez y se
    reutiliza entre interacciones. El botón '🔄 Actualizar datos' la invalida.
    """
    return model.load_dashboard_data()


# =============================================================================
#  2. CAPA DE GRÁFICOS (Plotly)
# =============================================================================
def _aplicar_tema_oscuro(fig: go.Figure) -> go.Figure:
    """Aplica un look oscuro y transparente para integrar la figura con el fondo."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_TEXTO, size=13),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def build_outcome_bar_fig(rec: dict) -> go.Figure:
    """Barras con la probabilidad de cada resultado: Local / Empate / Visitante."""
    pred = rec["prediction"]
    etiquetas = [rec["home"]["name"], "Empate", rec["away"]["name"]]
    valores = [pred["p_home"] * 100, pred["p_draw"] * 100, pred["p_away"] * 100]
    colores = [COLOR_LOCAL, COLOR_EMPATE, COLOR_VISITANTE]

    fig = go.Figure(
        go.Bar(
            x=etiquetas, y=valores, marker_color=colores,
            text=[f"{v:.0f}%" for v in valores], textposition="outside",
        )
    )
    fig.update_layout(
        title="Probabilidad de resultado (1 · X · 2)",
        yaxis_title="Probabilidad (%)",
        yaxis_range=[0, max(valores) * 1.18],
        showlegend=False,
    )
    return _aplicar_tema_oscuro(fig)


def build_scoreline_heatmap_fig(rec: dict, max_goals: int = 6) -> go.Figure:
    """Mapa de calor de la probabilidad de cada marcador exacto.

    Eje Y = goles del local, eje X = goles del visitante. Se resalta con un
    recuadro el marcador más probable.
    """
    matrix = rec["prediction"]["matrix"]
    z = matrix[: max_goals + 1, : max_goals + 1] * 100  # a porcentaje
    ejes = list(range(max_goals + 1))
    # Solo etiquetamos las celdas con probabilidad relevante (>=1%) para no saturar.
    texto = [["" if v < 1 else f"{v:.0f}" for v in fila] for fila in z]

    fig = go.Figure(
        go.Heatmap(
            z=z, x=ejes, y=ejes, colorscale=ESCALA_HEATMAP,
            text=texto, texttemplate="%{text}", textfont=dict(size=11),
            colorbar=dict(title="%"), hovertemplate=(
                f"{rec['home']['name']} %{{y}} – %{{x}} {rec['away']['name']}"
                "<br>Probabilidad: %{z:.1f}%<extra></extra>"
            ),
        )
    )

    # Recuadro en el marcador más probable
    ml = rec["prediction"]["most_likely"]
    if ml["home"] <= max_goals and ml["away"] <= max_goals:
        fig.add_shape(
            type="rect",
            x0=ml["away"] - 0.5, x1=ml["away"] + 0.5,
            y0=ml["home"] - 0.5, y1=ml["home"] + 0.5,
            line=dict(color=COLOR_VISITANTE, width=3),
        )

    fig.update_layout(
        title="Probabilidad por marcador exacto (%)",
        xaxis_title=f"Goles · {rec['away']['name']}",
        yaxis_title=f"Goles · {rec['home']['name']}",
        yaxis=dict(autorange="reversed", dtick=1),
        xaxis=dict(dtick=1),
    )
    return _aplicar_tema_oscuro(fig)


def build_strength_radar_fig(rec: dict) -> go.Figure:
    """Radar de fuerzas (percentil 0-100 entre las 48 selecciones del Mundial)."""
    ejes_cerrados = EJES_RADAR + [EJES_RADAR[0]]

    def _valores(team_key: str) -> list:
        radar = rec[team_key]["radar"]
        vals = [radar.get(e, 0) for e in EJES_RADAR]
        return vals + [vals[0]]  # cerrar el polígono

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=_valores("home"), theta=ejes_cerrados, fill="toself",
        name=rec["home"]["name"], line_color=COLOR_LOCAL,
    ))
    fig.add_trace(go.Scatterpolar(
        r=_valores("away"), theta=ejes_cerrados, fill="toself",
        name=rec["away"]["name"], line_color=COLOR_VISITANTE,
    ))
    fig.update_layout(
        title="Perfil de fuerzas (percentil entre las 48 selecciones)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
    )
    return _aplicar_tema_oscuro(fig)


def build_goals_markets_fig(rec: dict) -> go.Figure:
    """Barras horizontales con la probabilidad de los mercados de goles.

    Líneas de Más de 1.5 / 2.5 / 3.5 goles + 'Ambos marcan'. Todas salen de la
    misma matriz de marcadores Poisson, así que son cálculos exactos del modelo.
    """
    pred = rec["prediction"]
    lineas = [1.5, 2.5, 3.5]
    categorias = [f"Más de {ln}" for ln in lineas] + ["Ambos marcan"]
    valores = [pred["p_over"][ln] * 100 for ln in lineas] + [pred["p_btts"] * 100]
    colores = [COLOR_LOCAL] * len(lineas) + [COLOR_VISITANTE]

    fig = go.Figure(
        go.Bar(
            x=valores, y=categorias, orientation="h", marker_color=colores,
            text=[f"{v:.0f}%" for v in valores], textposition="outside",
        )
    )
    fig.update_layout(
        title="Mercados de goles (probabilidad del modelo)",
        xaxis_title="Probabilidad (%)", xaxis_range=[0, 100],
        yaxis=dict(autorange="reversed"), showlegend=False,
    )
    return _aplicar_tema_oscuro(fig)


# =============================================================================
#  3. CAPA DE PRESENTACIÓN (Streamlit)
# =============================================================================
def inject_css() -> None:
    """Refuerza el modo oscuro y da a las métricas un aspecto de 'tarjeta'."""
    st.markdown(
        f"""
        <style>
            .stApp {{ background-color: {COLOR_FONDO}; color: {COLOR_TEXTO}; }}
            [data-testid="stMetric"],
            div[data-testid="metric-container"] {{
                background-color: {COLOR_CARD};
                border: 1px solid #2A2F3A;
                border-radius: 12px;
                padding: 16px 18px;
            }}
            [data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(label: str, rec: dict) -> None:
    """Encabezado: equipos, estado (jugado/próximo), grupo, fecha y sede."""
    meta, home, away = rec["meta"], rec["home"], rec["away"]
    st.title("⚽ Dashboard Mundial 2026 — Análisis y Predicciones")

    if meta["status"] == "played":
        act = rec["actual"]
        st.subheader(f"{home['name']}  {act['home']} – {act['away']}  {away['name']}")
        estado = "✅ Partido jugado"
    else:
        st.subheader(f"{home['name']}  vs  {away['name']}")
        estado = "🔮 Próximo partido"

    sede = f"{meta['city']}, {meta['country']}"
    if meta["neutral"]:
        sede += " · cancha neutral"
    st.caption(f"{estado}  ·  {meta['group']}  ·  {meta['date']}  ·  {sede}")
    st.divider()


def render_kpis(rec: dict) -> None:
    """Fila superior de tarjetas: xG de cada equipo, favorito y marcador probable."""
    st.markdown("#### Métricas clave del partido")
    pred = rec["prediction"]
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(f"xG · {rec['home']['name']}", f"{rec['home']['xg']:.2f}")
    col2.metric(f"xG · {rec['away']['name']}", f"{rec['away']['xg']:.2f}")
    col3.metric(
        "Ganador más probable", pred["favorite"],
        f"{pred['favorite_prob']:.0%} prob.", delta_color="off",
    )
    ml = pred["most_likely"]
    col4.metric(
        "Resultado más probable", f"{ml['home']}–{ml['away']}",
        f"{ml['prob']:.0%} prob.", delta_color="off",
    )
    st.divider()


def render_markets(rec: dict) -> None:
    """Mercados de goles (Más/Menos, BTTS) y tiros a puerta estimados."""
    pred, home, away = rec["prediction"], rec["home"], rec["away"]
    st.markdown("#### Mercados de goles y tiros a puerta")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Tiros a puerta · {home['name']}", f"{home['sot']:.1f}")
    col2.metric(f"Tiros a puerta · {away['name']}", f"{away['sot']:.1f}")
    col3.metric("Ambos marcan (Sí)", f"{pred['p_btts']:.0%}")
    col4.metric("Más de 2.5 goles", f"{pred['p_over'][2.5]:.0%}")

    st.plotly_chart(build_goals_markets_fig(rec), width="stretch")
    st.caption(
        "Los tiros a puerta son una **estimación** a partir del xG "
        "(tiros ≈ xG ÷ 0.30); no hay datos reales de tiros para estos partidos. "
        "Los mercados de goles y 'ambos marcan' sí son cálculos exactos de la "
        "matriz de marcadores."
    )
    st.divider()


def render_charts(rec: dict) -> None:
    """Gráficos: barras de probabilidad, mapa de calor de marcadores y radar."""
    st.markdown("#### Visualización de las predicciones")
    col_izq, col_der = st.columns(2)
    with col_izq:
        st.plotly_chart(build_outcome_bar_fig(rec), width="stretch")
    with col_der:
        st.plotly_chart(build_scoreline_heatmap_fig(rec), width="stretch")

    st.plotly_chart(build_strength_radar_fig(rec), width="stretch")
    st.divider()


def render_evaluation(rec: dict) -> None:
    """Sólo para partidos jugados: compara la predicción con la realidad."""
    if rec["meta"]["status"] != "played":
        return
    pred, act = rec["prediction"], rec["actual"]
    ml = pred["most_likely"]
    acerto_ganador = pred["favorite"] == act["winner"]
    acerto_marcador = ml["home"] == act["home"] and ml["away"] == act["away"]

    msg = (
        f"**Modelo vs. realidad** — El favorito del modelo era "
        f"**{pred['favorite']}** ({pred['favorite_prob']:.0%}); ganó "
        f"**{act['winner']}**. Marcador más probable previsto "
        f"**{ml['home']}–{ml['away']}**; real **{act['home']}–{act['away']}**."
    )
    if acerto_marcador:
        st.success(msg + "  🎯 ¡Marcador exacto acertado!")
    elif acerto_ganador:
        st.success(msg + "  ✅ Ganador acertado.")
    else:
        st.warning(msg + "  ❌ El resultado sorprendió al modelo.")
    st.divider()


def to_tournament_df(matches: dict) -> pd.DataFrame:
    """Aplana las predicciones de todos los partidos a un DataFrame para la tabla."""
    filas = []
    for label, m in matches.items():
        p = m["prediction"]
        ml = p["most_likely"]
        filas.append({
            "Fecha": m["meta"]["date"],
            "Grupo": m["meta"]["group"],
            "Estado": "Jugado" if m["meta"]["status"] == "played" else "Próximo",
            "Local": m["home"]["name"],
            "Visitante": m["away"]["name"],
            "xG L": round(m["home"]["xg"], 2),
            "xG V": round(m["away"]["xg"], 2),
            "P(1)": f"{p['p_home']:.0%}",
            "P(X)": f"{p['p_draw']:.0%}",
            "P(2)": f"{p['p_away']:.0%}",
            "+2.5": f"{p['p_over'][2.5]:.0%}",
            "BTTS": f"{p['p_btts']:.0%}",
            "+ probable": f"{ml['home']}–{ml['away']}",
            "Real": (f"{m['actual']['home']}–{m['actual']['away']}"
                     if m["actual"] else "—"),
        })
    return pd.DataFrame(filas)


def render_raw_data(matches: dict, label: str) -> None:
    """Sección colapsable con la tabla de predicciones de todo el torneo."""
    with st.expander("🔎 Ver tabla de predicciones del torneo"):
        df = to_tournament_df(matches)
        st.markdown(f"**Partido seleccionado — {label}**")
        sel = df[(df["Local"] == matches[label]["home"]["name"])
                 & (df["Visitante"] == matches[label]["away"]["name"])
                 & (df["Fecha"] == matches[label]["meta"]["date"])]
        st.dataframe(sel.reset_index(drop=True), width="stretch")

        st.markdown("**Torneo completo (72 partidos)**")
        st.dataframe(df.reset_index(drop=True), width="stretch", height=420)


# --- Constantes de la pestaña de apuestas ---
_TIER_ICON = {"Alta": "🟢 Alta", "Media": "🟡 Media", "Especulativa": "🔵 Especulativa"}


def render_recommendations(data: dict) -> None:
    """Pestaña de picks recomendados + histórico de aciertos en lo ya jugado."""
    recs, bt, meta = data["recommendations"], data["backtest"], data["meta"]

    st.markdown("### 🎯 Recomendaciones de apuestas")
    st.caption(
        f"Selecciones de mayor **confianza del modelo** (probabilidad ≥ "
        f"{meta['conf_min']:.0%}) para los **próximos** partidos. No se comparan "
        "con cuotas de casas de apuestas: indican la confianza del modelo, no un "
        "valor garantizado frente al mercado."
    )

    if not recs:
        st.info("No hay próximos partidos con picks por encima del umbral de confianza.")
    else:
        df = pd.DataFrame(recs)
        df["partido"] = df["home"] + "  vs  " + df["away"]
        mercados = sorted(df["market"].unique())
        partidos = list(dict.fromkeys(df["partido"]))  # mantiene el orden por prob.

        sel_p = st.multiselect(
            "Partidos:", partidos, default=partidos,
            help="Filtra para ver solo el/los partido(s) que te interesan.",
        )
        col_f1, col_f2 = st.columns([3, 2])
        sel_m = col_f1.multiselect("Mercados:", mercados, default=mercados)
        min_p = col_f2.slider(
            "Confianza mínima:", 50, 95, int(meta["conf_min"] * 100), step=1,
        ) / 100.0

        vista = df[
            df["partido"].isin(sel_p)
            & df["market"].isin(sel_m)
            & (df["prob"] >= min_p)
        ].sort_values("prob", ascending=False)
        if vista.empty:
            st.info("Ningún pick cumple los filtros seleccionados.")
        else:
            tabla = pd.DataFrame({
                "Partido": vista["partido"],
                "Fecha": vista["date"],
                "Grupo": vista["group"],
                "Mercado": vista["market"],
                "Selección": vista["selection"],
                "Prob.": (vista["prob"] * 100).round(0).astype(int).astype(str) + "%",
                "Confianza": vista["tier"].map(_TIER_ICON),
            })
            st.dataframe(tabla.reset_index(drop=True), width="stretch", height=460)
            st.caption(f"{len(vista)} picks · ordenados por probabilidad del modelo.")

    # --- Histórico: cómo habrían rendido estos picks en lo ya jugado ---
    st.divider()
    st.markdown("#### 📈 Rendimiento en partidos ya jugados")
    if bt["n"] == 0:
        st.info("Aún no hay partidos jugados para evaluar el histórico de picks.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Picks evaluados", bt["n"])
        c2.metric("Aciertos", bt["hits"])
        c3.metric("Tasa de acierto", f"{bt['rate']:.0%}")

        filas = [
            {"Mercado": m, "Picks": d["n"], "Aciertos": d["hits"],
             "Acierto": f"{d['rate']:.0%}"}
            for m, d in sorted(bt["by_market"].items(), key=lambda x: -x[1]["rate"])
        ]
        st.dataframe(pd.DataFrame(filas), width="stretch")
        st.caption(
            "Picks generados con las fuerzas **previas** al torneo (sin fuga de "
            "información). Los tiros a puerta no se evalúan (no hay datos reales). "
            "Nótese que 'Doble oportunidad' y '1X2' rinden muy por encima de "
            "'Goles 2.5' y 'Ambos marcan', que son mercados cercanos al azar."
        )

    st.divider()
    st.warning(
        "⚠️ **Juega con responsabilidad.** Estas recomendaciones son la salida de "
        "un modelo estadístico con fines educativos; no garantizan resultados y no "
        "consideran las cuotas del mercado. Apostar conlleva riesgo de pérdida."
    )


# =============================================================================
#  4. ORQUESTACIÓN PRINCIPAL
# =============================================================================
def main() -> None:
    st.set_page_config(
        page_title="Dashboard Mundial 2026",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # --- Panel lateral: control y actualización de datos ---
    st.sidebar.header("Panel de control")
    if st.sidebar.button("🔄 Actualizar datos", width="stretch"):
        model.fetch_results(force=True)   # fuerza nueva descarga del CSV
        get_dashboard_data.clear()        # invalida la caché de Streamlit
        st.rerun()

    # --- Carga de datos (descarga + modelo, cacheado) ---
    data = get_dashboard_data()
    matches, meta = data["matches"], data["meta"]

    # --- Filtro y selección de partido ---
    filtro = st.sidebar.radio(
        "Mostrar partidos:", ["Todos", "Próximos", "Jugados"], horizontal=True,
    )
    def _incluir(m: dict) -> bool:
        if filtro == "Próximos":
            return m["meta"]["status"] == "upcoming"
        if filtro == "Jugados":
            return m["meta"]["status"] == "played"
        return True

    opciones = [
        ("🔮 " if m["meta"]["status"] == "upcoming" else "✅ ") + label
        for label, m in matches.items() if _incluir(m)
    ]
    if not opciones:
        st.sidebar.warning("No hay partidos para este filtro.")
        st.stop()

    elegido = st.sidebar.selectbox("Selecciona el partido:", options=opciones)
    label = elegido[2:]  # quita el emoji + espacio para recuperar la clave real
    rec = matches[label]

    # --- KPIs del torneo en el sidebar ---
    st.sidebar.divider()
    st.sidebar.markdown(
        f"**Torneo:** {meta['n_matches']} partidos\n\n"
        f"✅ Jugados: {meta['n_played']}  ·  🔮 Próximos: {meta['n_upcoming']}\n\n"
        f"Selecciones: {meta['n_teams']}  ·  datos hasta {meta['data_through']}"
    )
    st.sidebar.caption(
        f"Fuente: `{meta['source']}` (datos internacionales reales). "
        "Modelo Poisson/Dixon-Coles con decaimiento temporal. "
        "xG = goles esperados del modelo."
    )

    # --- Render: dos pestañas (análisis del partido / recomendaciones) ---
    tab_partido, tab_apuestas = st.tabs(
        ["📊 Análisis del partido", "🎯 Recomendaciones de apuestas"]
    )
    with tab_partido:
        render_header(label, rec)
        render_kpis(rec)
        render_markets(rec)
        render_evaluation(rec)
        render_charts(rec)
        render_raw_data(matches, label)
    with tab_apuestas:
        render_recommendations(data)


if __name__ == "__main__":
    main()
