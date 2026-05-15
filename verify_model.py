import numpy as np

from model.model import MarketModel


def check_initial_fragility(model):
    """
    Comprueba la fragilidad inicial antes de ejecutar pasos del modelo.
    Sirve para ver si los traders ya nacen cerca o dentro de margin call.
    """
    initial_margin_calls = 0
    leverages = []
    margins = []
    capitals = []
    positions = []

    for agent in model.leveraged_traders:
        leverages.append(agent.leverage)
        margins.append(agent.margen_mantenimiento)
        capitals.append(agent.capital)
        positions.append(agent.position)

        if agent.capital < agent.margen_mantenimiento * agent.value:
            initial_margin_calls += 1

    return {
        "initial_margin_calls": initial_margin_calls,
        "initial_avg_leverage": float(np.mean(leverages)) if leverages else 0.0,
        "initial_max_leverage": float(np.max(leverages)) if leverages else 0.0,
        "initial_avg_margin": float(np.mean(margins)) if margins else 0.0,
        "initial_min_capital": float(np.min(capitals)) if capitals else 0.0,
        "initial_max_position": float(np.max(positions)) if positions else 0.0,
    }


def run_model(params, steps):
    """
    Crea el modelo, comprueba su fragilidad inicial y ejecuta la simulación.
    """
    model = MarketModel(**params)

    initial_checks = check_initial_fragility(model)

    for _ in range(steps):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()

    return df, initial_checks


def print_basic_checks(df, initial_checks, label):
    print()
    print("=" * 80)
    print(f"VERIFICACIÓN: {label}")
    print("=" * 80)

    print()
    print("Columnas generadas:")
    print(list(df.columns))

    print()
    print("Fragilidad inicial:")
    print(
        f"Traders inicialmente en margin call: "
        f"{initial_checks['initial_margin_calls']}"
    )
    print(
        f"Apalancamiento medio inicial: "
        f"{initial_checks['initial_avg_leverage']:.4f}x"
    )
    print(
        f"Apalancamiento máximo inicial: "
        f"{initial_checks['initial_max_leverage']:.4f}x"
    )
    print(
        f"Margen medio inicial: "
        f"{initial_checks['initial_avg_margin']:.4f}"
    )
    print(
        f"Capital mínimo inicial: "
        f"{initial_checks['initial_min_capital']:.4f}"
    )
    print(
        f"Posición máxima inicial: "
        f"{initial_checks['initial_max_position']:.4f}"
    )

    print()
    print("Últimas filas:")
    columns_to_show = [
        "Price",
        "Returns",
        "MarginCalls",
        "Liquidations",
        "ActiveTraders",
        "AvgLeverage",
        "AvgCapital",
        "TotalPosition",
        "UniqueMarginCalled",
        "InitialTotalPosition",
    ]

    print(df[columns_to_show].tail(15).to_string())

    initial_price_observed = df["Price"].iloc[0]
    final_price = df["Price"].iloc[-1]
    initial_total_position = df["InitialTotalPosition"].iloc[0]
    final_total_position = df["TotalPosition"].iloc[-1]
    total_liquidations = df["Liquidations"].sum()

    liquidation_ratio = (
        total_liquidations / initial_total_position
        if initial_total_position > 0
        else 0.0
    )

    print()
    print("Resumen:")
    print(f"Precio primera observación: {initial_price_observed:.4f}")
    print(f"Precio final: {final_price:.4f}")
    print(
        f"Retorno acumulado observado: "
        f"{(final_price / initial_price_observed - 1):.4%}"
    )
    print(f"Margin calls acumulados: {df['MarginCalls'].sum():.0f}")
    print(
        f"Traders únicos con margin call: "
        f"{df['UniqueMarginCalled'].iloc[-1]:.0f}"
    )
    print(f"Liquidaciones totales: {total_liquidations:.4f}")
    print(f"Traders activos finales: {df['ActiveTraders'].iloc[-1]:.0f}")
    print(f"Posición inicial total: {initial_total_position:.4f}")
    print(f"Posición final total: {final_total_position:.4f}")
    print(f"Ratio liquidado sobre posición inicial: {liquidation_ratio:.4%}")

    print()
    print("Checks lógicos:")

    checks = {
        "InitialTotalPosition constante": df["InitialTotalPosition"].nunique() == 1,
        "UniqueMarginCalled no decrece": (
            df["UniqueMarginCalled"].diff().fillna(0) >= 0
        ).all(),
        "ActiveTraders no aumenta": (
            df["ActiveTraders"].diff().fillna(0) <= 0
        ).all(),
        "TotalPosition no aumenta": (
            df["TotalPosition"].diff().fillna(0) <= 1e-8
        ).all(),
        "Liquidations no negativas": (df["Liquidations"] >= 0).all(),
        "Price siempre positivo": (df["Price"] > 0).all(),
        "Liquidations solo si hay posición": (
            df["Liquidations"] <= df["InitialTotalPosition"]
        ).all(),
    }

    for name, passed in checks.items():
        status = "OK" if passed else "REVISAR"
        print(f"- {name}: {status}")


def main():
    conservative_params = {
        "N_noise": 50,
        "N_traders": 50,
        "sigma_noise": 0.5,
        "lambda_": 0.01,
        "alpha": 0.5,
        "margen_mantenimiento": 0.25,
        "margen_mantenimiento_spread": 0.02,
        "capital_scale": 1.2,
        "position_scale": 0.15,
    }

    baseline_params = {
        "N_noise": 50,
        "N_traders": 50,
        "sigma_noise": 0.5,
        "lambda_": 0.01,
        "alpha": 0.5,
        "margen_mantenimiento": 0.25,
        "margen_mantenimiento_spread": 0.02,
        "capital_scale": 1.0,
        "position_scale": 0.25,
    }

    stress_params = {
        "N_noise": 50,
        "N_traders": 100,
        "sigma_noise": 1.0,
        "lambda_": 0.02,
        "alpha": 0.8,
        "margen_mantenimiento": 0.35,
        "margen_mantenimiento_spread": 0.02,
        "capital_scale": 0.7,
        "position_scale": 0.5,
    }

    scenarios = [
        ("Escenario conservador", conservative_params),
        ("Escenario base", baseline_params),
        ("Escenario de estrés", stress_params),
    ]

    for label, params in scenarios:
        df, initial_checks = run_model(params, steps=100)
        print_basic_checks(df, initial_checks, label)


if __name__ == "__main__":
    main()