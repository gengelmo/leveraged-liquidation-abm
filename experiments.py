import pandas as pd
import numpy as np

from model.model import MarketModel


BASE_PARAMS = {
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


def compute_initial_fragility(model):
    """
    Calcula métricas de fragilidad antes de empezar la simulación.

    Sirve para saber si una configuración nace ya en una situación frágil,
    por ejemplo con traders inicialmente en margin call.
    """
    initial_margin_calls = 0
    leverages = []

    for agent in model.leveraged_traders:
        leverages.append(agent.leverage)

        if agent.capital < agent.margen_mantenimiento * agent.value:
            initial_margin_calls += 1

    return {
        "initial_margin_calls": initial_margin_calls,
        "initial_avg_leverage": np.mean(leverages) if leverages else 0.0,
        "initial_max_leverage": np.max(leverages) if leverages else 0.0,
    }


def run_simulation(params, steps=500):
    """
    Ejecuta una simulación y devuelve:
    - el DataFrame generado por Mesa
    - las métricas de fragilidad inicial
    """
    model = MarketModel(**params)

    initial_metrics = compute_initial_fragility(model)

    # Guardamos el estado inicial antes del primer paso.
    # Así el precio inicial aparece como 100 en el DataFrame.
    model.datacollector.collect(model)

    for _ in range(steps):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()

    return df, initial_metrics


def compute_metrics(df):
    """
    Calcula métricas agregadas de una simulación.
    """
    prices = df["Price"]

    # Quitamos el primer retorno porque corresponde al estado inicial guardado.
    returns = df["Returns"].iloc[1:]
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    volatility = returns.std()

    # Drawdown clásico: caída desde máximo previo.
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    max_drawdown = drawdown.min()

    # Caída máxima respecto al precio inicial del modelo.
    initial_price = prices.iloc[0]
    drop_from_initial = (prices - initial_price) / initial_price
    max_drop_from_initial = drop_from_initial.min()

    # Presión acumulada de margin calls.
    # Esto mide trader-pasos en margin call, no traders únicos.
    margin_call_pressure = df["MarginCalls"].sum()

    # Número inicial de traders apalancados.
    # Como guardamos el estado inicial antes de simular, iloc[0] es el valor inicial.
    initial_traders = df["ActiveTraders"].iloc[0]

    # Traders distintos que alguna vez han tenido margin call.
    unique_margin_called = df["UniqueMarginCalled"].iloc[-1]

    unique_margin_called_ratio = (
        unique_margin_called / initial_traders
        if initial_traders > 0
        else 0.0
    )

    # Volumen total liquidado.
    total_liquidations = df["Liquidations"].sum()

    # Ratio liquidado respecto a la exposición inicial agregada.
    initial_total_position = df["InitialTotalPosition"].iloc[0]

    liquidation_ratio = (
        total_liquidations / initial_total_position
        if initial_total_position > 0
        else 0.0
    )

    # Estado final del sistema.
    final_price = prices.iloc[-1]
    final_active_traders = df["ActiveTraders"].iloc[-1]
    final_total_position = df["TotalPosition"].iloc[-1]

    final_active_ratio = (
        final_active_traders / initial_traders
        if initial_traders > 0
        else 0.0
    )

    inactive_ratio = 1.0 - final_active_ratio

    # Distintos umbrales de crash usando drawdown clásico.
    crash_20 = int(max_drawdown < -0.20)
    crash_50 = int(max_drawdown < -0.50)
    crash_80 = int(max_drawdown < -0.80)

    # Umbrales de caída respecto al precio inicial.
    crash_initial_20 = int(max_drop_from_initial < -0.20)
    crash_initial_50 = int(max_drop_from_initial < -0.50)
    crash_initial_80 = int(max_drop_from_initial < -0.80)

    return {
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "max_drop_from_initial": max_drop_from_initial,
        "margin_call_pressure": margin_call_pressure,
        "unique_margin_called": unique_margin_called,
        "unique_margin_called_ratio": unique_margin_called_ratio,
        "total_liquidations": total_liquidations,
        "liquidation_ratio": liquidation_ratio,
        "crash_20": crash_20,
        "crash_50": crash_50,
        "crash_80": crash_80,
        "crash_initial_20": crash_initial_20,
        "crash_initial_50": crash_initial_50,
        "crash_initial_80": crash_initial_80,
        "final_price": final_price,
        "final_active_traders": final_active_traders,
        "final_active_ratio": final_active_ratio,
        "inactive_ratio": inactive_ratio,
        "final_total_position": final_total_position,
    }


def run_sensitivity(parameter_name, values, n_iter=50, steps=500):
    """
    Ejecuta un análisis de sensibilidad univariante.

    Varía un parámetro manteniendo todos los demás fijos.
    """
    results = []

    for value in values:
        for iteration in range(n_iter):
            params = BASE_PARAMS.copy()
            params[parameter_name] = value

            df, initial_metrics = run_simulation(params, steps=steps)
            metrics = compute_metrics(df)

            results.append({
                "parameter": parameter_name,
                "value": value,
                "iteration": iteration,
                **initial_metrics,
                **metrics,
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    parameter_ranges = {
        "N_noise": [25, 50, 100, 200],
        "N_traders": [25, 50, 100, 200],
        "sigma_noise": [0.25, 0.5, 1.0, 1.5],
        "lambda_": [0.005, 0.01, 0.02, 0.04],
        "alpha": [0.4, 0.6, 0.8, 1.0],
        "margen_mantenimiento": [0.15, 0.25, 0.35, 0.45],
        "margen_mantenimiento_spread": [0.0, 0.02, 0.05, 0.10],
        "capital_scale": [0.8, 1.0, 1.2, 1.5],
        "position_scale": [0.10, 0.15, 0.25, 0.35],
    }

    all_results = []

    for parameter_name, values in parameter_ranges.items():
        print(f"Running sensitivity for {parameter_name}...")

        df_param = run_sensitivity(
            parameter_name=parameter_name,
            values=values,
            n_iter=50,
            steps=500,
        )

        all_results.append(df_param)

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv("sensitivity_results.csv", index=False)

    print("Done. Results saved to sensitivity_results.csv")