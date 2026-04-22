import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model.model import MarketModel


print("\n" + "=" * 80)
print("CALIBRACION DE HETEROGENEIDAD INICIAL")
print("Objetivo: encontrar una zona fragil pero no rota")
print("=" * 80)

# Storyline:
# 1) mantenemos el capital fijo en un rango razonable
# 2) barrer solo la escala de la posicion inicial
# 3) medir leverage inicial y respuesta a un shock pequeno

num_runs = 500
N_traders = 50
N_noise = 0

capital_min = 50
capital_max = 150
base_position_min = 5
base_position_max = 15

theta = 0.5
margin_mantenimiento = 0.25
price_0 = 100.0
shock = -0.01  # -1%: shock pequeno para ver fragilidad sin romper todo

position_scales = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
near_threshold_tol = 0.10

rows = []
sample_distributions = []


def build_model(position_scale):
    model = MarketModel(
        N_noise=N_noise,
        N_traders=N_traders,
        capital_min=capital_min,
        capital_max=capital_max,
        position_min=base_position_min * position_scale,
        position_max=base_position_max * position_scale,
        theta=theta,
        margen_mantenimiento=margin_mantenimiento,
    )
    return model


for position_scale in position_scales:
    print(f"Simulando position_scale={position_scale:.2f} ({num_runs} runs)...")

    threshold = 1 / margin_mantenimiento

    capital_means = []
    position_means = []
    leverage_means = []
    leverage_stds = []
    share_above_threshold = []
    share_near_threshold = []

    trigger_rates = []
    liquidated_shares = []
    full_liquidation_rates = []
    post_shock_price_drops = []

    sample_capitals = None
    sample_positions = None
    sample_leverages = None

    for run in range(num_runs):
        model = build_model(position_scale)

        capitals = np.array([t.capital for t in model.leveraged_traders])
        positions = np.array([t.position for t in model.leveraged_traders])
        leverages = np.array([t.leverage for t in model.leveraged_traders])

        capital_means.append(np.mean(capitals))
        position_means.append(np.mean(positions))
        leverage_means.append(np.mean(leverages))
        leverage_stds.append(np.std(leverages))

        share_above_threshold.append(np.mean(leverages > threshold))
        share_near_threshold.append(
            np.mean((leverages >= (1 - near_threshold_tol) * threshold) & (leverages < threshold))
        )

        if run == 0:
            sample_capitals = capitals.copy()
            sample_positions = positions.copy()
            sample_leverages = leverages.copy()

        # Shock pequeno para medir sensibilidad sin colapso artificial
        model.market.price = price_0 * (1 + shock)
        price_before_step = model.market.price
        model.step()
        price_after_step = model.market.price

        trigger_rates.append(model.margin_calls_count / N_traders if N_traders else 0.0)
        liquidated_shares.append(
            model.market.last_liquidations / np.sum(positions) if np.sum(positions) > 0 else 0.0
        )
        full_liquidation_rates.append(
            np.mean([t.position <= 1e-6 for t in model.leveraged_traders])
        )
        post_shock_price_drops.append(
            (price_after_step / price_before_step) - 1
        )

    rows.append(
        {
            "position_scale": position_scale,
            "Capital Medio": np.mean(capital_means),
            "Position Media": np.mean(position_means),
            "Leverage Medio": np.mean(leverage_means),
            "Leverage (Desv. Est.)": np.mean(leverage_stds),
            "Share Above Threshold": np.mean(share_above_threshold),
            "Share Near Threshold": np.mean(share_near_threshold),
            "Margin Call Rate (shock -1%)": np.mean(trigger_rates),
            "Liquidated Share (shock -1%)": np.mean(liquidated_shares),
            "Full Liquidation Rate (shock -1%)": np.mean(full_liquidation_rates),
            "Price Drop After Step": np.mean(post_shock_price_drops),
        }
    )

    if sample_capitals is not None:
        sample_distributions.append(
            {
                "position_scale": position_scale,
                "threshold": threshold,
                "capitals": sample_capitals,
                "positions": sample_positions,
                "leverages": sample_leverages,
            }
        )


df = pd.DataFrame(rows)

print("\n" + "=" * 80)
print("RESUMEN DE CALIBRACION")
print("=" * 80)
print(df.round(4).to_string(index=False))

print("\nLECTURA RAPIDA:")
print("- Queremos una zona con Share Above Threshold moderado, no cercano a 1.")
print("- Queremos un shock pequeno que dispare algunas margin calls, no un vaciado total.")
print("- Si la tasa de liquidacion total es muy alta, el sistema sigue demasiado fragil.")

# Grficas resumen
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Calibracion de fragilidad inicial", fontsize=14)

axes[0, 0].plot(df["position_scale"], df["Leverage Medio"], marker="o", color="seagreen")
axes[0, 0].axhline(1 / margin_mantenimiento, color="red", linestyle="--", label="Threshold")
axes[0, 0].set_title("Leverage medio vs escala de posicion")
axes[0, 0].set_xlabel("position_scale")
axes[0, 0].set_ylabel("Leverage medio")
axes[0, 0].legend()

axes[0, 1].plot(df["position_scale"], df["Share Above Threshold"], marker="o", color="steelblue")
axes[0, 1].plot(df["position_scale"], df["Share Near Threshold"], marker="o", color="darkorange")
axes[0, 1].set_title("Fragilidad inicial")
axes[0, 1].set_xlabel("position_scale")
axes[0, 1].set_ylabel("Share")
axes[0, 1].legend(["Above threshold", "Near threshold"])

axes[1, 0].plot(df["position_scale"], df["Margin Call Rate (shock -1%)"], marker="o", color="crimson")
axes[1, 0].plot(df["position_scale"], df["Full Liquidation Rate (shock -1%)"], marker="o", color="purple")
axes[1, 0].set_title("Respuesta a shock pequeno")
axes[1, 0].set_xlabel("position_scale")
axes[1, 0].set_ylabel("Rate")
axes[1, 0].legend(["Margin calls", "Full liquidation"])

axes[1, 1].plot(df["position_scale"], df["Liquidated Share (shock -1%)"], marker="o", color="slateblue")
axes[1, 1].set_title("Volumen liquidado tras shock pequeno")
axes[1, 1].set_xlabel("position_scale")
axes[1, 1].set_ylabel("Liquidated share")

plt.tight_layout()
plt.show()

# Histogramas de ejemplo: el primer y ultimo caso de la calibracion
if sample_distributions:
    fig2, axes2 = plt.subplots(len(sample_distributions), 3, figsize=(14, 4 * len(sample_distributions)))
    if len(sample_distributions) == 1:
        axes2 = np.array([axes2])

    for i, sample in enumerate(sample_distributions):
        axes2[i, 0].hist(sample["capitals"], bins=15, color="steelblue", alpha=0.8)
        axes2[i, 0].set_title(f"Capital\nscale={sample['position_scale']:.2f}")
        axes2[i, 0].set_xlabel("Capital")

        axes2[i, 1].hist(sample["positions"], bins=15, color="darkorange", alpha=0.8)
        axes2[i, 1].set_title("Position")
        axes2[i, 1].set_xlabel("Position")

        axes2[i, 2].hist(sample["leverages"], bins=15, color="seagreen", alpha=0.8)
        axes2[i, 2].axvline(sample["threshold"], color="red", linestyle="--", linewidth=2)
        axes2[i, 2].set_title("Leverage")
        axes2[i, 2].set_xlabel("Leverage")

    plt.tight_layout()
    plt.show()
