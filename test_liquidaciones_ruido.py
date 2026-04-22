import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from model.model import MarketModel

print("\n" + "=" * 80)
print("EXPERIMENTO FINAL: El impacto del Apalancamiento en el Riesgo Sistémico")
print("=" * 80)

# Dejamos el ruido fijo: 50 noise traders con sigma moderado
F_NOISE = 50
F_SIGMA = 0.5

# Grid de validacion: Variamos la cantidad de Traders Apalancados
valores_N_traders = [0, 10, 50, 100] 
num_ejecuciones = 50
num_pasos = 50

resultados = []
retornos_hist = {}

for n_traders in valores_N_traders:
    print(f"Simulando: N_traders={n_traders:3d} apalancados ({num_ejecuciones} runs)...")

    volatilidades_run = []
    kurtosis_run = []
    skewness_run = []
    margin_calls_run = []
    liquidaciones_run = []

    for run in range(num_ejecuciones):
        # AQUÍ ESTÁ LA MAGIA: Ahora introducimos N_traders al ecosistema
        model = MarketModel(
            N_noise=F_NOISE, 
            N_traders=n_traders, 
            sigma_noise=F_SIGMA,
            margen_mantenimiento=0.25 # Puedes jugar con esto después
        )

        precios = [model.market.price]
        margin_calls_totales = 0
        liquidaciones_totales = 0

        for _ in range(num_pasos):
            model.step()
            precios.append(model.market.price)
            # Recogemos métricas sistémicas
            margin_calls_totales += model.margin_calls_count
            liquidaciones_totales += model.market.last_liquidations

        precios_array = np.array(precios)
        retornos = np.diff(precios_array) / precios_array[:-1]

        volatilidades_run.append(np.std(retornos))
        kurtosis_run.append(kurtosis(retornos))
        skewness_run.append(skew(retornos))
        margin_calls_run.append(margin_calls_totales)
        liquidaciones_run.append(liquidaciones_totales)

        # Guardamos historiales para los histogramas
        if run == 0:
            retornos_hist[n_traders] = retornos.copy()

    resultados.append(
        {
            "Traders Apalancados": n_traders,
            "Volatilidad Media": np.mean(volatilidades_run),
            "Kurtosis (Colas Gruesas)": np.mean(kurtosis_run),
            "Asimetría (Crash Skew)": np.mean(skewness_run),
            "Total Margin Calls": np.mean(margin_calls_run),
            "Volumen Liquidado": np.mean(liquidaciones_run)
        }
    )

df_resultados = pd.DataFrame(resultados)

print("\n" + "=" * 80)
print("RESULTADOS DEL IMPACTO SISTÉMICO")
print("=" * 80)
print(df_resultados.round(4).to_string(index=False))

# --- GRÁFICA ESTRELLA PARA TU TFG ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Emergencia de Riesgo Sistémico por Apalancamiento", fontsize=14)

# Histograma Comparativo
axes[0].set_title("Distribución de Retornos")
axes[0].hist(retornos_hist[0], bins=20, alpha=0.5, label="0 Apalancados (Mercado Sano)", color="blue")
axes[0].hist(retornos_hist[100], bins=20, alpha=0.5, label="100 Apalancados (Riesgo Sistémico)", color="red")
axes[0].set_xlabel("Retorno")
axes[0].set_ylabel("Frecuencia")
axes[0].legend()

# Evolución de las Colas Gruesas
axes[1].set_title("Evolución de Kurtosis vs Apalancados")
axes[1].plot(df_resultados["Traders Apalancados"], df_resultados["Kurtosis (Colas Gruesas)"], marker='o', color='purple', linewidth=2)
axes[1].set_xlabel("Número de Traders Apalancados")
axes[1].set_ylabel("Exceso de Kurtosis (Colas Gruesas)")
axes[1].grid(alpha=0.3)
axes[1].axhline(0, color='black', linestyle='--', alpha=0.5, label="Umbral Normal")
axes[1].legend()

plt.tight_layout()
plt.show()