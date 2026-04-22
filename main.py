import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

from model.model import MarketModel

print("\n" + "=" * 80)
print("EXPERIMENTO 2D: Sensibilidad a N_noise y sigma_noise")
print("Objetivo: validar la responsabilidad de los noise traders")
print("=" * 80)

# Grid de validacion: poblacion y agresividad del ruido
valores_N_noise = [10, 50, 100]
valores_sigma = [0.2, 0.5, 1.0]
num_ejecuciones = 50
num_pasos = 50

resultados = []
trayectorias_ejemplo = []
serie_ejemplo = None
retornos_hist = {}

for n_traders in valores_N_noise:
    for sigma in valores_sigma:
        print(f"Simulando: N_noise={n_traders:3d} | sigma={sigma:.1f} ({num_ejecuciones} runs)...")

        volatilidades_run = []
        rango_precios_run = []
        kurtosis_run = []
        skewness_run = []
        drift_run = []

        media_flujo_run = []
        desv_flujo_run = []
        flujo_abs_medio_run = []
        max_flujo_abs_run = []
        share_flujo_positivo_run = []
        share_retorno_positivo_run = []

        for run in range(num_ejecuciones):
            model = MarketModel(N_noise=n_traders, N_traders=0, sigma_noise=sigma)

            precios = [model.market.price]

            for _ in range(num_pasos):
                model.step()
                precios.append(model.market.price)

            precios_array = np.array(precios)
            retornos = np.diff(precios_array) / precios_array[:-1]
            flujos = np.array(model.market.flow_history)

            volatilidades_run.append(np.std(retornos))
            rango_precios_run.append(np.max(precios_array) - np.min(precios_array))
            kurtosis_run.append(kurtosis(retornos))
            skewness_run.append(skew(retornos))
            drift_run.append((precios_array[-1] / precios_array[0]) - 1)

            media_flujo_run.append(np.mean(flujos))
            desv_flujo_run.append(np.std(flujos))
            flujo_abs_medio_run.append(np.mean(np.abs(flujos)))
            max_flujo_abs_run.append(np.max(np.abs(flujos)))
            share_flujo_positivo_run.append(np.mean(flujos > 0))
            share_retorno_positivo_run.append(np.mean(retornos > 0))

            if run == 0:
                trayectorias_ejemplo.append(
                    {
                        "N_noise": n_traders,
                        "Sigma": sigma,
                        "Precio inicial": precios_array[0],
                        "Precio final": precios_array[-1],
                        "Min precio": np.min(precios_array),
                        "Max precio": np.max(precios_array),
                        "Flujo medio": np.mean(flujos),
                        "Flujo abs medio": np.mean(np.abs(flujos)),
                    }
                )

            if run == 0 and n_traders == 50 and sigma == 0.5:
                serie_ejemplo = {
                    "N_noise": n_traders,
                    "Sigma": sigma,
                    "precios": precios_array.copy(),
                    "flujos": flujos.copy(),
                }

            if run == 0 and (n_traders, sigma) in [(10, 0.2), (100, 1.0)]:
                retornos_hist[(n_traders, sigma)] = retornos.copy()

        resultados.append(
            {
                "N_noise": n_traders,
                "Sigma": sigma,
                "Volatilidad Media": np.mean(volatilidades_run),
                "Volatilidad (Desv. Est.)": np.std(volatilidades_run),
                "Rango Precio Medio": np.mean(rango_precios_run),
                "Kurtosis Media": np.mean(kurtosis_run),
                "Asimetria Media": np.mean(skewness_run),
                "Drift Medio": np.mean(drift_run),
                "Flujo Medio": np.mean(media_flujo_run),
                "Flujo (Desv. Est.)": np.mean(desv_flujo_run),
                "Flujo Abs Medio": np.mean(flujo_abs_medio_run),
                "Max Flujo Abs Medio": np.mean(max_flujo_abs_run),
                "Share Flujo Positivo": np.mean(share_flujo_positivo_run),
                "Share Retorno Positivo": np.mean(share_retorno_positivo_run),
            }
        )

df_resultados = pd.DataFrame(resultados)
df_trayectorias = pd.DataFrame(trayectorias_ejemplo)

print("\n" + "=" * 80)
print("RESULTADOS FINALES AGREGADOS")
print("=" * 80)

tabla_matriz_vol = df_resultados.pivot(index="N_noise", columns="Sigma", values="Volatilidad Media")
print("\nMATRIZ DE VOLATILIDAD MEDIA:")
print("-" * 50)
print(tabla_matriz_vol.round(4))

tabla_matriz_flujo = df_resultados.pivot(index="N_noise", columns="Sigma", values="Flujo Abs Medio")
print("\nMATRIZ DE FLUJO ABSOLUTO MEDIO:")
print("-" * 50)
print(tabla_matriz_flujo.round(4))

tabla_matriz_kurt = df_resultados.pivot(index="N_noise", columns="Sigma", values="Kurtosis Media")
print("\nMATRIZ DE KURTOSIS MEDIA (referencia de puro ruido):")
print("-" * 50)
print(tabla_matriz_kurt.round(4))

tabla_matriz_sesgo = df_resultados.pivot(index="N_noise", columns="Sigma", values="Flujo Medio")
print("\nMATRIZ DE FLUJO MEDIO (deberia estar cerca de 0):")
print("-" * 50)
print(tabla_matriz_sesgo.round(4))

print("\nTRAYECTORIAS EJEMPLO (run 0 de cada combinacion):")
print("-" * 80)
print(df_trayectorias.round(4).to_string(index=False))

print("\nTABLA PLANA COMPLETA:")
print("-" * 80)
print(df_resultados.round(4).to_string(index=False))

# =========================
# GRAFICAS DE VALIDACION
# =========================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Validacion visual de noise traders", fontsize=14)

# 1. Heatmap de volatilidad media
im1 = axes[0, 0].imshow(tabla_matriz_vol.values, cmap="YlOrRd", aspect="auto")
axes[0, 0].set_title("Heatmap: Volatilidad media")
axes[0, 0].set_xlabel("sigma_noise")
axes[0, 0].set_ylabel("N_noise")
axes[0, 0].set_xticks(range(len(tabla_matriz_vol.columns)))
axes[0, 0].set_xticklabels(tabla_matriz_vol.columns)
axes[0, 0].set_yticks(range(len(tabla_matriz_vol.index)))
axes[0, 0].set_yticklabels(tabla_matriz_vol.index)
for i in range(tabla_matriz_vol.shape[0]):
    for j in range(tabla_matriz_vol.shape[1]):
        axes[0, 0].text(
            j,
            i,
            f"{tabla_matriz_vol.values[i, j]:.4f}",
            ha="center",
            va="center",
            color="black",
            fontsize=9,
        )
fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

# 2. Heatmap de flujo absoluto medio
im2 = axes[0, 1].imshow(tabla_matriz_flujo.values, cmap="Blues", aspect="auto")
axes[0, 1].set_title("Heatmap: Flujo absoluto medio")
axes[0, 1].set_xlabel("sigma_noise")
axes[0, 1].set_ylabel("N_noise")
axes[0, 1].set_xticks(range(len(tabla_matriz_flujo.columns)))
axes[0, 1].set_xticklabels(tabla_matriz_flujo.columns)
axes[0, 1].set_yticks(range(len(tabla_matriz_flujo.index)))
axes[0, 1].set_yticklabels(tabla_matriz_flujo.index)
for i in range(tabla_matriz_flujo.shape[0]):
    for j in range(tabla_matriz_flujo.shape[1]):
        axes[0, 1].text(
            j,
            i,
            f"{tabla_matriz_flujo.values[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=9,
        )
fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

# 3. Histograma de retornos
axes[1, 0].set_title("Histogramas de retornos")
axes[1, 0].set_xlabel("Retorno")
axes[1, 0].set_ylabel("Frecuencia")
if (10, 0.2) in retornos_hist:
    axes[1, 0].hist(
        retornos_hist[(10, 0.2)],
        bins=20,
        alpha=0.6,
        label="N_noise=10, sigma=0.2",
        color="steelblue",
    )
if (100, 1.0) in retornos_hist:
    axes[1, 0].hist(
        retornos_hist[(100, 1.0)],
        bins=20,
        alpha=0.6,
        label="N_noise=100, sigma=1.0",
        color="darkorange",
    )
axes[1, 0].legend()

# 4. Espacio reservado en la figura 2x2
axes[1, 1].axis("off")
axes[1, 1].text(
    0.05,
    0.8,
    "La serie temporal ejemplo se muestra\n"
    "en una figura separada para mantener\n"
    "precio y flujo en paneles apilados.",
    fontsize=11,
)

plt.tight_layout()
plt.show()

# Serie temporal ejemplo: precio arriba, flujo abajo
if serie_ejemplo is not None:
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig2.suptitle(
        f"Serie temporal ejemplo: N_noise={serie_ejemplo['N_noise']}, "
        f"sigma={serie_ejemplo['Sigma']}",
        fontsize=14,
    )

    tiempo_precios = np.arange(len(serie_ejemplo["precios"]))
    tiempo_flujos = np.arange(1, len(serie_ejemplo["flujos"]) + 1)

    ax1.plot(tiempo_precios, serie_ejemplo["precios"], color="firebrick", linewidth=2)
    ax1.set_ylabel("Precio")
    ax1.set_title("Precio")
    ax1.grid(alpha=0.3)

    ax2.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax2.bar(tiempo_flujos, serie_ejemplo["flujos"], color="slateblue", alpha=0.8)
    ax2.set_xlabel("Paso")
    ax2.set_ylabel("Order flow")
    ax2.set_title("Flujo agregado")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
