# Leveraged Liquidation ABM

Proyecto de simulacion basada en agentes para estudiar cascadas de liquidacion en
un mercado financiero con traders apalancados. El modelo combina ruido exogeno,
impacto de mercado no lineal, margin calls y liquidaciones forzadas.

## Estructura del proyecto

- `model/`: implementacion principal del modelo.
  - `market.py`: mercado e impacto de ordenes sobre el precio.
  - `noise_traders.py`: agentes que generan ordenes aleatorias.
  - `leveraged_traders.py`: traders apalancados sujetos a margin calls.
  - `model.py`: coordinacion del mercado, agentes y metricas.
- `web/`: demo visual en HTML, CSS y JavaScript.
- `verify_model.py`: comprobaciones basicas del comportamiento del modelo.
- `experiments.py`: ejecucion del analisis de sensibilidad.
- `analyze_sensitivity.py`: resumen y ranking de sensibilidad.
- `plot_sensitivity.py`: generacion de graficas de sensibilidad.
- `calibrate_optimizers.py`: calibracion de parametros.
- `validate_calibrated_model.py`: validacion Monte Carlo del modelo calibrado.
- `validate_stylized.py`: analisis de stylized facts.

## Requisitos

El proyecto usa Python y las siguientes librerias principales:

- `mesa`
- `numpy`
- `pandas`
- `matplotlib`

Instalacion orientativa:

```bash
pip install mesa numpy pandas matplotlib
```

## Uso

Para ejecutar una comprobacion rapida del modelo:

```bash
python verify_model.py
```

Para lanzar la visualizacion interactiva:

```text
web/index.html
```

## Experimentos

Analisis de sensibilidad:

```bash
python experiments.py
python analyze_sensitivity.py
python plot_sensitivity.py
```

Calibracion y validacion:

```bash
python calibrate_optimizers.py
python validate_calibrated_model.py
python validate_stylized.py
```

Los scripts generan archivos `.csv` y figuras con los resultados de las
simulaciones. Estos resultados pueden regenerarse ejecutando los comandos
anteriores.
