# ⚽ Dashboard Mundial 2026 — Análisis y Predicciones

App web interactiva que **descarga datos reales** de selecciones nacionales y **predice cada
partido del Mundial 2026**: goles esperados (xG), probabilidad de victoria/empate, marcador más
probable y la matriz completa de resultados.

Construida con **Streamlit** + **Pandas** + **Plotly**. El modelo de predicción
(Poisson/Dixon-Coles) está implementado a mano sobre **NumPy**, así que no requiere scipy ni
ninguna API de pago.

## ✨ Funcionalidades

- **xG (goles esperados)** de cada equipo en cada partido.
- **Probabilidades 1 · X · 2** (victoria local / empate / victoria visitante).
- **Marcador más probable** y un **mapa de calor** con la probabilidad de cada resultado exacto.
- **Radar de fuerzas**: ataque, defensa, forma y potencia de cada selección (percentil frente a
  las 48 del torneo).
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
- Solo se cubre la **fase de grupos**: los cruces eliminatorios aún no tienen equipos definidos.

## 📄 Licencia

Uso personal/educativo. Los datos pertenecen a sus respectivas fuentes.
