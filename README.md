# ⚽ Dashboard Mundial 2026 — Análisis y Predicciones

App web interactiva que **descarga datos reales** de selecciones nacionales y **predice cada
partido del Mundial 2026**: goles esperados (xG), probabilidad de victoria/empate, marcador más
probable, la matriz completa de resultados, mercados de goles (Más/Menos y "ambos marcan"),
tiros a puerta estimados y una pestaña de **recomendaciones de apuestas** con su histórico de
aciertos.

Construida con **Streamlit** + **Pandas** + **Plotly**. El modelo de predicción
(Poisson/Dixon-Coles) está implementado a mano sobre **NumPy**, así que no requiere scipy ni
ninguna API de pago.

## ✨ Funcionalidades

- **xG (goles esperados)** de cada equipo en cada partido.
- **Probabilidades 1 · X · 2** (victoria local / empate / victoria visitante).
- **Marcador más probable** y un **mapa de calor** con la probabilidad de cada resultado exacto.
- **Mercados de goles**: Más/Menos 1.5 · 2.5 · 3.5 y **ambos marcan (BTTS)**, calculados de forma
  exacta a partir de la matriz de marcadores.
- **Tiros a puerta estimados** por equipo (a partir del xG; ver nota más abajo).
- **Radar de fuerzas**: ataque, defensa, forma y potencia de cada selección (percentil frente a
  las 48 del torneo).
- **Pestaña de recomendaciones de apuestas**: la mejor selección del modelo por mercado (resultado,
  doble oportunidad, Más/Menos 2.5, ambos marcan, tiros a puerta), ordenada por confianza y con
  filtros. Incluye un **histórico de aciertos** que evalúa, sin fuga de información, cómo habrían
  rendido esos picks en los partidos ya jugados.
- Cubre los **72 partidos de fase de grupos**. Los ya jugados muestran el resultado real junto a
  la predicción previa del modelo ("modelo vs. realidad"); los próximos muestran el pronóstico.
- Filtro por estado (todos / próximos / jugados) y botón para **actualizar los datos**.

## 🚀 Instalación y uso

```bash
# 1. Crear y activar un entorno virtual (Python 3.10+)
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la app
streamlit run app.py
```

La primera ejecución descarga el CSV de datos (~3.7 MB) y lo cachea en `data/`. Si no hay
conexión, la app reutiliza la última copia descargada.

## 🧠 Cómo funciona

| Archivo            | Responsabilidad |
|--------------------|-----------------|
| `model.py`         | Único punto de datos. Descarga el histórico real, entrena el modelo y genera las predicciones de los 72 partidos. |
| `app.py`           | Interfaz Streamlit (gráficos, selector de partidos, KPIs). |
| `requirements.txt` | Dependencias (streamlit, pandas, plotly, numpy). |

**Fuente de datos:** [`martj42/international_results`](https://github.com/martj42/international_results)
— un único CSV con todos los partidos internacionales (1872→2026) que **ya incluye el fixture del
Mundial 2026** (los partidos sin jugar vienen con marcador vacío).

**Modelo:** Poisson/Dixon-Coles. A partir del histórico se estima una fuerza de **ataque** y
**defensa** por selección, ponderando los partidos recientes mucho más que los antiguos (vida
media ~2 años) y teniendo en cuenta la ventaja de localía de los anfitriones (EE. UU., México,
Canadá). Para cada partido se calculan los goles esperados (λ = xG) y, con ellos, la matriz de
marcadores con una corrección Dixon-Coles para los resultados bajos.

Verificación rápida del modelo (sin abrir la UI):

```bash
python model.py        # autotest: imprime predicciones de ejemplo y comprobaciones
```

## ⚠️ Notas

- El **xG** aquí son los goles esperados del modelo (λ de Poisson), no un xG basado en tiros
  (ninguna fuente gratuita publica datos de tiros para estos partidos).
- Los **tiros a puerta son una estimación** derivada del xG (tiros ≈ xG ÷ 0.30, la tasa media de
  conversión tiro-a-puerta→gol del fútbol internacional), **no un dato real**. Se intentó FotMob,
  pero bloqueó su API tras un token, así que no hay tiros reales accesibles para estos partidos.
- Las **recomendaciones de apuestas** reflejan la **confianza del modelo**, no valor frente a las
  cuotas de una casa (no se usan cuotas). El histórico muestra que doble oportunidad y 1X2 rinden
  bien, mientras que Más/Menos 2.5 y BTTS están cerca del azar. **Juega con responsabilidad**: es
  una herramienta educativa y apostar conlleva riesgo de pérdida.
- Solo se cubre la **fase de grupos**: los cruces eliminatorios aún no tienen equipos definidos.

## 📄 Licencia

Uso personal/educativo. Los datos pertenecen a sus respectivas fuentes.
