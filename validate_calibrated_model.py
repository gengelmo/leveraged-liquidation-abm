from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model.model import MarketModel


# ============================================================
# Configuración general
# ============================================================

N_VALIDATION_RUNS = 200
N_STEPS = 500
SEED_BASE = 200_000

CALIBRATION_FILE = Path("calibration_results") / "best_calibrated_params.csv"

OUTPUT_DIR = Path("validation_results")
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================
# Parámetros fijos del modelo
# ============================================================

FIXED_PARAMS = {
    "N_noise": 50,
    "N_traders": 50,
    "margen_mantenimiento": 0.25,
    "margen_mantenimiento_spread": 0.02,
}


# ============================================================
# Funciones auxiliares
# ============================================================

def load_calibrated_params():
    """
    Lee los parámetros calibrados desde best_calibrated_params.csv.

    El archivo viene de la calibración por random search.
    """
    df = pd.read_csv(CALIBRATION_FILE)

    if df.empty:
        raise ValueError("El archivo de calibración está vacío.")

    best = df.iloc[0]

    calibrated_params = {
        "lambda_": float(best["lambda_"]),
        "alpha": float(best["alpha"]),
        "sigma_noise": float(best["sigma_noise"]),
        "capital_scale": float(best["capital_scale"]),
        "position_scale": float(best["position_scale"]),
    }

    params = {
        **FIXED_PARAMS,
        **calibrated_params,
    }

    return params


def compute_initial_fragility(model):
    """
    Calcula métricas iniciales antes de empezar la simulación.
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


def compute_max_drawdown(prices):
    """
    Calcula el drawdown máximo clásico:
    caída máxima desde un máximo previo.
    """
    prices = np.asarray(prices)
    running_max = np.maximum.accumulate(prices)
    drawdowns = (prices - running_max) / running_max

    return drawdowns.min()


def run_one_simulation(params, steps, seed):
    """
    Ejecuta una simulación completa y devuelve:
    - métricas agregadas
    - serie temporal generada por el DataCollector
    """
    np.random.seed(seed)

    model = MarketModel(**params)

    initial_metrics = compute_initial_fragility(model)

    # Guardamos el estado inicial para que la serie empiece en precio 100.
    model.datacollector.collect(model)

    for _ in range(steps):
        model.step()

    df = model.datacollector.get_model_vars_dataframe().reset_index(drop=True)

    prices = df["Price"]
    returns = df["Returns"].iloc[1:]
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    volatility = returns.std()

    max_drawdown = compute_max_drawdown(prices)

    initial_price = prices.iloc[0]
    max_drop_from_initial = ((prices - initial_price) / initial_price).min()

    margin_call_pressure = df["MarginCalls"].sum()

    initial_traders = df["ActiveTraders"].iloc[0]
    final_active_traders = df["ActiveTraders"].iloc[-1]

    unique_margin_called = df["UniqueMarginCalled"].iloc[-1]

    unique_margin_called_ratio = (
        unique_margin_called / initial_traders
        if initial_traders > 0
        else 0.0
    )

    total_liquidations = df["Liquidations"].sum()
    initial_total_position = df["InitialTotalPosition"].iloc[0]
    final_total_position = df["TotalPosition"].iloc[-1]

    liquidation_ratio = (
        total_liquidations / initial_total_position
        if initial_total_position > 0
        else 0.0
    )

    final_active_ratio = (
        final_active_traders / initial_traders
        if initial_traders > 0
        else 0.0
    )

    inactive_ratio = 1.0 - final_active_ratio

    final_price = prices.iloc[-1]

    crash_20 = int(max_drawdown < -0.20)
    crash_50 = int(max_drawdown < -0.50)
    crash_80 = int(max_drawdown < -0.80)

    crash_initial_20 = int(max_drop_from_initial < -0.20)
    crash_initial_50 = int(max_drop_from_initial < -0.50)
    crash_initial_80 = int(max_drop_from_initial < -0.80)

    metrics = {
        **initial_metrics,
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

    return metrics, df


# ============================================================
# Gráficas
# ============================================================

def plot_price_trajectories(time_series_df):
    """
    Grafica varias trayectorias de precio calibradas.
    """
    plt.figure(figsize=(8, 5))

    for run_id, subset in time_series_df.groupby("run_id"):
        plt.plot(subset["step"], subset["Price"], alpha=0.35, linewidth=1)

    plt.xlabel("Paso")
    plt.ylabel("Precio")
    plt.title("Trayectorias de precio bajo parámetros calibrados")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "calibrated_price_trajectories.png", dpi=200)
    plt.close()


def plot_distribution(metrics_df, column, xlabel, filename, bins=30):
    """
    Histograma de una métrica de validación.
    """
    plt.figure(figsize=(7, 4))

    plt.hist(metrics_df[column], bins=bins, alpha=0.75)

    plt.xlabel(xlabel)
    plt.ylabel("Frecuencia")
    plt.title(f"Distribución de {xlabel}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close()


def plot_scatter(metrics_df, x_col, y_col, xlabel, ylabel, filename):
    """
    Gráfica de dispersión entre dos métricas.
    """
    plt.figure(figsize=(7, 5))

    plt.scatter(metrics_df[x_col], metrics_df[y_col], alpha=0.7)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} frente a {xlabel}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close()


def plot_cumulative_liquidations(time_series_df):
    """
    Grafica liquidaciones acumuladas para varias trayectorias.
    """
    plt.figure(figsize=(8, 5))

    for run_id, subset in time_series_df.groupby("run_id"):
        cumulative_liq = subset["Liquidations"].cumsum()
        plt.plot(subset["step"], cumulative_liq, alpha=0.35, linewidth=1)

    plt.xlabel("Paso")
    plt.ylabel("Liquidaciones acumuladas")
    plt.title("Liquidaciones acumuladas bajo parámetros calibrados")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "calibrated_cumulative_liquidations.png", dpi=200)
    plt.close()


def plot_unique_margin_called(time_series_df):
    """
    Grafica traders únicos afectados por margin call.
    """
    plt.figure(figsize=(8, 5))

    for run_id, subset in time_series_df.groupby("run_id"):
        plt.plot(
            subset["step"],
            subset["UniqueMarginCalled"],
            alpha=0.35,
            linewidth=1,
        )

    plt.xlabel("Paso")
    plt.ylabel("Traders únicos con margin call")
    plt.title("Contagio de margin calls bajo parámetros calibrados")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "calibrated_unique_margin_called.png", dpi=200)
    plt.close()


def generate_figures(metrics_df, time_series_sample):
    """
    Genera todas las gráficas principales de validación.
    """
    plot_price_trajectories(time_series_sample)

    plot_cumulative_liquidations(time_series_sample)

    plot_unique_margin_called(time_series_sample)

    plot_distribution(
        metrics_df,
        column="max_drop_from_initial",
        xlabel="Máxima caída desde precio inicial",
        filename="distribution_max_drop_from_initial.png",
    )

    plot_distribution(
        metrics_df,
        column="max_drawdown",
        xlabel="Máximo drawdown",
        filename="distribution_max_drawdown.png",
    )

    plot_distribution(
        metrics_df,
        column="liquidation_ratio",
        xlabel="Ratio de liquidación",
        filename="distribution_liquidation_ratio.png",
    )

    plot_distribution(
        metrics_df,
        column="unique_margin_called_ratio",
        xlabel="Ratio de traders afectados",
        filename="distribution_unique_margin_called_ratio.png",
    )

    plot_scatter(
        metrics_df,
        x_col="liquidation_ratio",
        y_col="max_drop_from_initial",
        xlabel="Ratio de liquidación",
        ylabel="Máxima caída desde precio inicial",
        filename="scatter_liquidation_vs_drop.png",
    )

    plot_scatter(
        metrics_df,
        x_col="unique_margin_called_ratio",
        y_col="liquidation_ratio",
        xlabel="Ratio de traders afectados",
        ylabel="Ratio de liquidación",
        filename="scatter_contagion_vs_liquidation.png",
    )


# ============================================================
# Resumen
# ============================================================

def build_summary(metrics_df):
    """
    Construye una tabla resumen con medias, desviaciones y percentiles.
    """
    selected_metrics = [
        "volatility",
        "max_drawdown",
        "max_drop_from_initial",
        "margin_call_pressure",
        "unique_margin_called_ratio",
        "liquidation_ratio",
        "final_price",
        "final_active_ratio",
        "inactive_ratio",
        "final_total_position",
    ]

    rows = []

    for metric in selected_metrics:
        values = metrics_df[metric].dropna()

        rows.append({
            "metric": metric,
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "p25": values.quantile(0.25),
            "median": values.median(),
            "p75": values.quantile(0.75),
            "max": values.max(),
        })

    summary = pd.DataFrame(rows)

    crash_summary = pd.DataFrame([
        {
            "metric": "crash_20_probability",
            "mean": metrics_df["crash_20"].mean(),
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "max": np.nan,
        },
        {
            "metric": "crash_50_probability",
            "mean": metrics_df["crash_50"].mean(),
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "max": np.nan,
        },
        {
            "metric": "crash_80_probability",
            "mean": metrics_df["crash_80"].mean(),
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "max": np.nan,
        },
        {
            "metric": "crash_initial_20_probability",
            "mean": metrics_df["crash_initial_20"].mean(),
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "max": np.nan,
        },
        {
            "metric": "crash_initial_50_probability",
            "mean": metrics_df["crash_initial_50"].mean(),
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "max": np.nan,
        },
        {
            "metric": "crash_initial_80_probability",
            "mean": metrics_df["crash_initial_80"].mean(),
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "max": np.nan,
        },
    ])

    summary = pd.concat([summary, crash_summary], ignore_index=True)

    return summary


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("VALIDACIÓN DEL MODELO CALIBRADO")
    print("=" * 80)

    params = load_calibrated_params()

    print()
    print("Parámetros calibrados usados:")
    for key, value in params.items():
        print(f"- {key}: {value}")

    metrics_rows = []
    time_series_rows = []

    for run_id in range(N_VALIDATION_RUNS):
        print(f"Validación simulación {run_id + 1}/{N_VALIDATION_RUNS}")

        seed = SEED_BASE + run_id

        metrics, df_ts = run_one_simulation(
            params=params,
            steps=N_STEPS,
            seed=seed,
        )

        metrics["run_id"] = run_id
        metrics_rows.append(metrics)

        # Guardamos todas las series temporales.
        df_ts = df_ts.copy()
        df_ts["run_id"] = run_id
        df_ts["step"] = df_ts.index

        time_series_rows.append(df_ts)

    metrics_df = pd.DataFrame(metrics_rows)
    time_series_df = pd.concat(time_series_rows, ignore_index=True)

    metrics_file = OUTPUT_DIR / "calibrated_validation_metrics.csv"
    time_series_file = OUTPUT_DIR / "calibrated_time_series_all.csv"

    metrics_df.to_csv(metrics_file, index=False)
    time_series_df.to_csv(time_series_file, index=False)

    # Para gráficas de trayectorias, usamos una muestra para que no quede ilegible.
    sample_run_ids = metrics_df.sample(
        n=min(30, len(metrics_df)),
        random_state=123,
    )["run_id"]

    time_series_sample = time_series_df[
        time_series_df["run_id"].isin(sample_run_ids)
    ].copy()

    time_series_sample.to_csv(
        OUTPUT_DIR / "calibrated_time_series_sample.csv",
        index=False,
    )

    summary = build_summary(metrics_df)

    summary_file = OUTPUT_DIR / "calibrated_validation_summary.csv"
    summary.to_csv(summary_file, index=False)

    generate_figures(metrics_df, time_series_sample)

    print()
    print("=" * 80)
    print("VALIDACIÓN COMPLETADA")
    print("=" * 80)
    print()
    print(f"Métricas por simulación guardadas en: {metrics_file}")
    print(f"Series temporales completas guardadas en: {time_series_file}")
    print(f"Resumen guardado en: {summary_file}")
    print(f"Figuras guardadas en: {FIGURES_DIR}")
    print()
    print("Resumen principal:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()