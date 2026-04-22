import numpy as np
import pandas as pd

from model.model import MarketModel


print("\n" + "=" * 80)
print("COMPARACION DE CALIBRACIONES DEL MARKET MODEL")
print("Objetivo: elegir una zona fragil pero no rota")
print("=" * 80)

# Comparacion controlada de dos calibraciones candidatas
position_scales = [0.25, 0.30]
capital_min = 50
capital_max = 150
base_position_min = 5
base_position_max = 15

N_noise_values = [10, 50, 100]
sigma_values = [0.2, 0.5, 1.0]
num_runs = 100
num_steps = 50

rows = []

for position_scale in position_scales:
    position_min = base_position_min * position_scale
    position_max = base_position_max * position_scale
    threshold = 1 / 0.25

    print(
        f"\n--- position_scale={position_scale:.2f} "
        f"(position_min={position_min:.2f}, position_max={position_max:.2f}) ---"
    )

    for N_noise in N_noise_values:
        for sigma in sigma_values:
            print(f"Simulando N_noise={N_noise:3d} | sigma={sigma:.1f} ({num_runs} runs)...")

            final_prices = []
            max_drawdowns = []
            margin_calls = []
            liquidations = []
            active_traders_end = []
            price_drops_after_first_step = []

            for run in range(num_runs):
                model = MarketModel(
                    N_noise=N_noise,
                    N_traders=50,
                    sigma_noise=sigma,
                    capital_min=capital_min,
                    capital_max=capital_max,
                    position_min=position_min,
                    position_max=position_max,
                    theta=0.5,
                    margen_mantenimiento=0.25,
                )

                price_path = [model.market.price]

                for _ in range(num_steps):
                    model.step()
                    price_path.append(model.market.price)

                price_array = np.array(price_path)
                peak = np.maximum.accumulate(price_array)
                drawdown = np.max((peak - price_array) / peak)

                final_prices.append(price_array[-1])
                max_drawdowns.append(drawdown)
                margin_calls.append(model.datacollector.get_model_vars_dataframe()["MarginCalls"].sum())
                liquidations.append(model.datacollector.get_model_vars_dataframe()["Liquidations"].sum())
                active_traders_end.append(sum(t.active for t in model.leveraged_traders))
                price_drops_after_first_step.append((price_array[1] / price_array[0]) - 1)

            rows.append(
                {
                    "position_scale": position_scale,
                    "Threshold": threshold,
                    "N_noise": N_noise,
                    "Sigma": sigma,
                    "Final Price Mean": np.mean(final_prices),
                    "Final Price Std": np.std(final_prices),
                    "Max Drawdown Mean": np.mean(max_drawdowns),
                    "Margin Calls Mean": np.mean(margin_calls),
                    "Liquidations Mean": np.mean(liquidations),
                    "Active Traders End Mean": np.mean(active_traders_end),
                    "Price Drop After First Step": np.mean(price_drops_after_first_step),
                }
            )

df = pd.DataFrame(rows)

print("\n" + "=" * 80)
print("RESULTADOS DE LA COMPARACION")
print("=" * 80)
print(df.round(4).to_string(index=False))

print("\nLECTURA RAPIDA:")
print("- Si position_scale=0.25 tiene drawdown moderado y 0.30 ya se vuelve mucho mas agresivo, la historia queda bien.")
print("- Si ambas son demasiado duras, hay que bajar mas la escala.")
