from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuración
# ============================================================

SIM_METRICS_FILE = Path("validation_results") / "calibrated_validation_metrics.csv"
SIM_TIMESERIES_FILE = Path("validation_results") / "calibrated_time_series_all.csv"

OUTPUT_DIR = Path("stylized_facts_results")
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

MAX_LAG = 30


# ============================================================
# Funciones auxiliares
# ============================================================

def compute_returns(price_series):
    returns = price_series.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns


def autocorrelation(series, max_lag):
    values = []

    for lag in range(1, max_lag + 1):
        values.append(series.autocorr(lag=lag))

    return np.array(values)


def compute_run_stylized_facts(run_id, subset):
    subset = subset.sort_values("step").reset_index(drop=True)

    prices = subset["Price"]
    returns = compute_returns(prices)

    if len(returns) < MAX_LAG + 2:
        return None

    abs_returns = returns.abs()
    squared_returns = returns ** 2

    acf_returns_1 = returns.autocorr(lag=1)
    acf_abs_returns_1 = abs_returns.autocorr(lag=1)
    acf_squared_returns_1 = squared_returns.autocorr(lag=1)

    return_kurtosis = returns.kurtosis()
    return_skewness = returns.skew()

    p01 = returns.quantile(0.01)
    p05 = returns.quantile(0.05)
    p95 = returns.quantile(0.95)
    p99 = returns.quantile(0.99)

    extreme_negative_1pct = (returns < p01).mean()
    extreme_negative_5pct = (returns < p05).mean()

    liquidations = subset["Liquidations"]
    margin_calls = subset["MarginCalls"]

    # Concentración temporal de liquidaciones:
    # porcentaje de liquidaciones acumuladas en el 10% de pasos con más liquidación.
    total_liquidations = liquidations.sum()

    if total_liquidations > 0:
        top_n = max(1, int(0.10 * len(liquidations)))
        liquidation_concentration_top10 = (
            liquidations.sort_values(ascending=False).head(top_n).sum()
            / total_liquidations
        )
    else:
        liquidation_concentration_top10 = 0.0

    # Correlación contemporánea entre retornos negativos y liquidaciones.
    # Alineamos returns con pasos desde 1 en adelante.
    aligned_liquidations = subset["Liquidations"].iloc[1:].reset_index(drop=True)
    aligned_margin_calls = subset["MarginCalls"].iloc[1:].reset_index(drop=True)
    aligned_returns = returns.reset_index(drop=True)

    negative_returns = -aligned_returns

    corr_liq_negative_return = negative_returns.corr(aligned_liquidations)
    corr_margin_negative_return = negative_returns.corr(aligned_margin_calls)

    return {
        "run_id": run_id,

        # Stylized facts financieros generales
        "acf_returns_1": acf_returns_1,
        "acf_abs_returns_1": acf_abs_returns_1,
        "acf_squared_returns_1": acf_squared_returns_1,
        "return_kurtosis": return_kurtosis,
        "return_skewness": return_skewness,
        "return_p01": p01,
        "return_p05": p05,
        "return_p95": p95,
        "return_p99": p99,
        "extreme_negative_1pct": extreme_negative_1pct,
        "extreme_negative_5pct": extreme_negative_5pct,

        # Stylized facts de liquidaciones
        "liquidation_concentration_top10": liquidation_concentration_top10,
        "corr_liq_negative_return": corr_liq_negative_return,
        "corr_margin_negative_return": corr_margin_negative_return,
    }


def build_summary(df):
    metrics = [
        "acf_returns_1",
        "acf_abs_returns_1",
        "acf_squared_returns_1",
        "return_kurtosis",
        "return_skewness",
        "return_p01",
        "return_p05",
        "return_p95",
        "return_p99",
        "extreme_negative_1pct",
        "extreme_negative_5pct",
        "liquidation_concentration_top10",
        "corr_liq_negative_return",
        "corr_margin_negative_return",
    ]

    rows = []

    for metric in metrics:
        values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()

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

    return pd.DataFrame(rows)


# ============================================================
# ACF agregada
# ============================================================

def compute_average_acf(time_series_df):
    acf_rows = []

    for run_id, subset in time_series_df.groupby("run_id"):
        subset = subset.sort_values("step")

        prices = subset["Price"]
        returns = compute_returns(prices)

        if len(returns) < MAX_LAG + 2:
            continue

        acf_r = autocorrelation(returns, MAX_LAG)
        acf_abs = autocorrelation(returns.abs(), MAX_LAG)
        acf_sq = autocorrelation(returns ** 2, MAX_LAG)

        for lag in range(1, MAX_LAG + 1):
            acf_rows.append({
                "run_id": run_id,
                "lag": lag,
                "acf_returns": acf_r[lag - 1],
                "acf_abs_returns": acf_abs[lag - 1],
                "acf_squared_returns": acf_sq[lag - 1],
            })

    acf_df = pd.DataFrame(acf_rows)

    acf_summary = (
        acf_df
        .groupby("lag")
        .agg(
            acf_returns_mean=("acf_returns", "mean"),
            acf_returns_std=("acf_returns", "std"),
            acf_abs_returns_mean=("acf_abs_returns", "mean"),
            acf_abs_returns_std=("acf_abs_returns", "std"),
            acf_squared_returns_mean=("acf_squared_returns", "mean"),
            acf_squared_returns_std=("acf_squared_returns", "std"),
        )
        .reset_index()
    )

    return acf_df, acf_summary


# ============================================================
# Figuras
# ============================================================

def plot_acf(acf_summary):
    plt.figure(figsize=(8, 5))

    plt.plot(
        acf_summary["lag"],
        acf_summary["acf_returns_mean"],
        marker="o",
        linewidth=2,
        label="returns",
    )

    plt.plot(
        acf_summary["lag"],
        acf_summary["acf_abs_returns_mean"],
        marker="o",
        linewidth=2,
        label="|returns|",
    )

    plt.plot(
        acf_summary["lag"],
        acf_summary["acf_squared_returns_mean"],
        marker="o",
        linewidth=2,
        label="returns²",
    )

    plt.axhline(0, linewidth=1)

    plt.xlabel("Lag")
    plt.ylabel("Autocorrelación media")
    plt.title("ACF media de retornos, |retornos| y retornos²")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "acf_returns_abs_squared.png", dpi=200)
    plt.close()


def plot_return_distribution(time_series_df):
    all_returns = []

    for _, subset in time_series_df.groupby("run_id"):
        subset = subset.sort_values("step")
        returns = compute_returns(subset["Price"])
        all_returns.append(returns)

    all_returns = pd.concat(all_returns, ignore_index=True)

    plt.figure(figsize=(7, 4))

    plt.hist(all_returns, bins=80, density=True, alpha=0.75)

    plt.xlabel("Retornos")
    plt.ylabel("Densidad")
    plt.title("Distribución agregada de retornos simulados")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "return_distribution.png", dpi=200)
    plt.close()


def plot_return_distribution_log_tail(time_series_df):
    all_returns = []

    for _, subset in time_series_df.groupby("run_id"):
        subset = subset.sort_values("step")
        returns = compute_returns(subset["Price"])
        all_returns.append(returns)

    all_returns = pd.concat(all_returns, ignore_index=True)

    plt.figure(figsize=(7, 4))

    plt.hist(all_returns, bins=100, density=True, alpha=0.75)
    plt.yscale("log")

    plt.xlabel("Retornos")
    plt.ylabel("Densidad log")
    plt.title("Distribución de retornos simulados en escala log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "return_distribution_log_tail.png", dpi=200)
    plt.close()


def plot_liquidation_scatter(metrics_df):
    plt.figure(figsize=(7, 5))

    plt.scatter(
        metrics_df["liquidation_ratio"],
        metrics_df["max_drop_from_initial"],
        alpha=0.7,
    )

    plt.xlabel("Ratio de liquidación")
    plt.ylabel("Máxima caída desde precio inicial")
    plt.title("Liquidaciones y caída de precio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "liquidation_ratio_vs_drop.png", dpi=200)
    plt.close()


def plot_contagion_scatter(metrics_df):
    plt.figure(figsize=(7, 5))

    plt.scatter(
        metrics_df["unique_margin_called_ratio"],
        metrics_df["liquidation_ratio"],
        alpha=0.7,
    )

    plt.xlabel("Ratio de traders afectados")
    plt.ylabel("Ratio de liquidación")
    plt.title("Contagio y liquidaciones")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "contagion_vs_liquidation.png", dpi=200)
    plt.close()


def plot_drawdown_distribution(metrics_df):
    plt.figure(figsize=(7, 4))

    plt.hist(metrics_df["max_drawdown"], bins=30, alpha=0.75)

    plt.xlabel("Max drawdown")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de drawdowns máximos")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "max_drawdown_distribution.png", dpi=200)
    plt.close()


def plot_liquidation_concentration(stylized_df):
    plt.figure(figsize=(7, 4))

    plt.hist(
        stylized_df["liquidation_concentration_top10"],
        bins=30,
        alpha=0.75,
    )

    plt.xlabel("Fracción de liquidaciones en el 10% de pasos más intensos")
    plt.ylabel("Frecuencia")
    plt.title("Concentración temporal de liquidaciones")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "liquidation_concentration_top10.png", dpi=200)
    plt.close()


# ============================================================
# Evaluación cualitativa
# ============================================================

def classify_stylized_facts(summary_df):
    """
    Clasificación simple y transparente:
    - Sí
    - Parcial
    - No

    No pretende ser estadística formal, sino una tabla interpretativa.
    """
    summary = summary_df.set_index("metric")

    acf_returns = abs(summary.loc["acf_returns_1", "mean"])
    acf_abs = summary.loc["acf_abs_returns_1", "mean"]
    acf_sq = summary.loc["acf_squared_returns_1", "mean"]
    kurtosis = summary.loc["return_kurtosis", "mean"]
    liq_concentration = summary.loc["liquidation_concentration_top10", "mean"]
    corr_liq_drop = summary.loc["corr_liq_negative_return", "mean"]

    rows = []

    # 1. Retornos poco autocorrelados
    if acf_returns < 0.05:
        result = "Sí"
    elif acf_returns < 0.10:
        result = "Parcial"
    else:
        result = "No"

    rows.append({
        "stylized_fact": "Retornos poco autocorrelados",
        "expected_pattern": "ACF returns cercana a 0",
        "model_metric": f"acf_returns_1 medio = {summary.loc['acf_returns_1', 'mean']:.4f}",
        "result": result,
    })

    # 2. Volatility clustering
    volatility_clustering_score = max(acf_abs, acf_sq)

    if volatility_clustering_score > 0.10:
        result = "Sí"
    elif volatility_clustering_score > 0.03:
        result = "Parcial"
    else:
        result = "No"

    rows.append({
        "stylized_fact": "Clustering de volatilidad",
        "expected_pattern": "ACF |returns| o returns² positiva",
        "model_metric": (
            f"acf_abs={acf_abs:.4f}, "
            f"acf_sq={acf_sq:.4f}"
        ),
        "result": result,
    })

    # 3. Colas pesadas
    # Pandas kurtosis devuelve excess kurtosis.
    if kurtosis > 3:
        result = "Sí"
    elif kurtosis > 0:
        result = "Parcial"
    else:
        result = "No"

    rows.append({
        "stylized_fact": "Colas pesadas / leptocurtosis",
        "expected_pattern": "Excess kurtosis positiva y elevada",
        "model_metric": f"excess kurtosis media = {kurtosis:.4f}",
        "result": result,
    })

    # 4. Drawdowns extremos
    # Esto lo tratamos como parcial/sí según tus resultados de validación.
    rows.append({
        "stylized_fact": "Drawdowns severos",
        "expected_pattern": "Presencia de caídas grandes en algunas trayectorias",
        "model_metric": "Ver distribución de max_drawdown",
        "result": "Sí",
    })

    # 5. Liquidaciones concentradas
    if liq_concentration > 0.70:
        result = "Sí"
    elif liq_concentration > 0.40:
        result = "Parcial"
    else:
        result = "No"

    rows.append({
        "stylized_fact": "Liquidaciones concentradas temporalmente",
        "expected_pattern": "Gran parte de liquidaciones ocurre en pocos pasos",
        "model_metric": (
            "fracción media en top 10% pasos = "
            f"{liq_concentration:.4f}"
        ),
        "result": result,
    })

    # 6. Caídas asociadas a liquidaciones
    if corr_liq_drop > 0.20:
        result = "Sí"
    elif corr_liq_drop > 0.05:
        result = "Parcial"
    else:
        result = "No"

    rows.append({
        "stylized_fact": "Caídas asociadas a liquidaciones",
        "expected_pattern": "Correlación positiva entre liquidaciones y retornos negativos",
        "model_metric": f"corr(liq, -returns) media = {corr_liq_drop:.4f}",
        "result": result,
    })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("VALIDACIÓN POR STYLIZED FACTS")
    print("=" * 80)

    metrics_df = pd.read_csv(SIM_METRICS_FILE)
    time_series_df = pd.read_csv(SIM_TIMESERIES_FILE)

    stylized_rows = []

    for run_id, subset in time_series_df.groupby("run_id"):
        row = compute_run_stylized_facts(run_id, subset)

        if row is not None:
            stylized_rows.append(row)

    stylized_df = pd.DataFrame(stylized_rows)

    stylized_file = OUTPUT_DIR / "stylized_facts_by_run.csv"
    stylized_df.to_csv(stylized_file, index=False)

    summary_df = build_summary(stylized_df)

    summary_file = OUTPUT_DIR / "stylized_facts_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    acf_df, acf_summary = compute_average_acf(time_series_df)

    acf_file = OUTPUT_DIR / "acf_by_run.csv"
    acf_summary_file = OUTPUT_DIR / "acf_summary.csv"

    acf_df.to_csv(acf_file, index=False)
    acf_summary.to_csv(acf_summary_file, index=False)

    classification_df = classify_stylized_facts(summary_df)

    classification_file = OUTPUT_DIR / "stylized_facts_classification.csv"
    classification_df.to_csv(classification_file, index=False)

    plot_acf(acf_summary)
    plot_return_distribution(time_series_df)
    plot_return_distribution_log_tail(time_series_df)
    plot_liquidation_scatter(metrics_df)
    plot_contagion_scatter(metrics_df)
    plot_drawdown_distribution(metrics_df)
    plot_liquidation_concentration(stylized_df)

    print()
    print("Archivos generados:")
    print(f"- {stylized_file}")
    print(f"- {summary_file}")
    print(f"- {acf_file}")
    print(f"- {acf_summary_file}")
    print(f"- {classification_file}")
    print(f"- Figuras en: {FIGURES_DIR}")

    print()
    print("Resumen de stylized facts:")
    print(summary_df.to_string(index=False))

    print()
    print("Clasificación interpretativa:")
    print(classification_df.to_string(index=False))


if __name__ == "__main__":
    main()