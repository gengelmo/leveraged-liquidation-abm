from model.model import MarketModel
import numpy as np
import pandas as pd


print("\n" + "=" * 80)
print("MONTE CARLO: Calibracion de liquidate()")
print("Objetivo: medir agresividad de las ventas forzadas")
print("=" * 80)

rng = np.random.default_rng(42)

# Grid de parametros a calibrar
theta_values = [0.1, 0.25, 0.5, 0.75, 1.0]
margen_values = [0.10, 0.15, 0.25, 0.35, 0.50]

# Casos aleatorios base. Mantener estos rangos hace que la calibracion
# sea comparable con la inicializacion actual del modelo.
num_trials = 1000
price_0 = 100.0
capital_range = (50.0, 150.0)
position_range = (5.0, 15.0)

rows = []
examples = []


def build_case(theta, margen_mantenimiento):
    model = MarketModel(N_noise=0, N_traders=1)
    trader = model.leveraged_traders[0]

    trader.capital = rng.uniform(*capital_range)
    trader.position = rng.uniform(*position_range)
    trader.prev_price = price_0
    model.market.price = price_0

    trader.theta = theta
    trader.margen_mantenimiento = margen_mantenimiento

    return model, trader


for theta in theta_values:
    for margen_mantenimiento in margen_values:
        print(
            f"Simulando theta={theta:.2f} | margen_mantenimiento={margen_mantenimiento:.2f} "
            f"({num_trials} trials)..."
        )

        trigger_flags = []
        sold_fractions = []
        sold_fractions_triggered = []
        full_liquidation_flags = []
        orders = []
        excess_leverages = []
        leverages = []

        for trial in range(num_trials):
            model, trader = build_case(theta, margen_mantenimiento)

            capital_before = trader.capital
            position_before = trader.position
            value_before = trader.value
            leverage_before = trader.leverage
            threshold = margen_mantenimiento * value_before
            margin_call = capital_before < threshold

            if margin_call:
                order = trader.liquidate()
            else:
                order = 0.0

            sold_amount = abs(order)
            sold_fraction = sold_amount / position_before if position_before > 0 else 0.0
            full_liquidation = trader.position <= 1e-6
            excess_leverage = max(0.0, leverage_before - (1 / margen_mantenimiento))

            trigger_flags.append(margin_call)
            sold_fractions.append(sold_fraction)
            if margin_call:
                sold_fractions_triggered.append(sold_fraction)
            full_liquidation_flags.append(full_liquidation and margin_call)
            orders.append(order)
            excess_leverages.append(excess_leverage)
            leverages.append(leverage_before)

            if trial == 0:
                examples.append(
                    {
                        "theta": theta,
                        "margen_mantenimiento": margen_mantenimiento,
                        "capital": capital_before,
                        "position": position_before,
                        "leverage": leverage_before,
                        "threshold": threshold,
                        "margin_call": margin_call,
                        "order": order,
                        "sold_fraction": sold_fraction,
                        "remaining_position": trader.position,
                        "active": trader.active,
                    }
                )

        rows.append(
            {
                "theta": theta,
                "margen_mantenimiento": margen_mantenimiento,
                "Trigger Rate": np.mean(trigger_flags),
                "Avg Sold Fraction": np.mean(sold_fractions),
                "Avg Sold Fraction | Triggered": np.mean(sold_fractions_triggered)
                if sold_fractions_triggered
                else 0.0,
                "Full Liquidation Rate": np.mean(full_liquidation_flags),
                "Avg Order": np.mean(orders),
                "Avg Excess Leverage": np.mean(excess_leverages),
                "Avg Leverage": np.mean(leverages),
            }
        )


df = pd.DataFrame(rows)
df_examples = pd.DataFrame(examples)

print("\n" + "=" * 80)
print("RESUMEN AGREGADO")
print("=" * 80)

tabla_trigger = df.pivot(index="theta", columns="margen_mantenimiento", values="Trigger Rate")
print("\nPROBABILIDAD DE ENTRAR EN LIQUIDACION:")
print("-" * 60)
print(tabla_trigger.round(4))

tabla_sold = df.pivot(index="theta", columns="margen_mantenimiento", values="Avg Sold Fraction | Triggered")
print("\nFRACCION MEDIA VENDIDA SOLO EN CASOS TRIGGERED:")
print("-" * 60)
print(tabla_sold.round(4))

tabla_full = df.pivot(index="theta", columns="margen_mantenimiento", values="Full Liquidation Rate")
print("\nTASA DE LIQUIDACION TOTAL:")
print("-" * 60)
print(tabla_full.round(4))

tabla_order = df.pivot(index="theta", columns="margen_mantenimiento", values="Avg Order")
print("\nORDEN MEDIA DEVUELTA POR liquidate():")
print("-" * 60)
print(tabla_order.round(4))

print("\nEJEMPLOS DEL PRIMER TRIAL DE CADA COMBINACION:")
print("-" * 80)
print(df_examples.round(4).to_string(index=False))

print("\nTABLA PLANA COMPLETA:")
print("-" * 80)
print(df.round(4).to_string(index=False))
