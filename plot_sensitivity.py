import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


SUMMARY_FILE = "sensitivity_summary.csv"
RANKING_FILE = "sensitivity_ranking.csv"
OUTPUT_DIR = Path("figures_sensitivity")


STD_COLUMNS = {
    "volatility_mean": "volatility_std",
    "max_drawdown_mean": "max_drawdown_std",
    "drawdown_severity_mean": "drawdown_severity_std",
    "max_drop_from_initial_mean": "max_drop_from_initial_std",
    "drop_from_initial_severity_mean": "drop_from_initial_severity_std",
    "margin_call_pressure_mean": "margin_call_pressure_std",
    "unique_margin_called_ratio_mean": "unique_margin_called_ratio_std",
    "total_liquidations_mean": "total_liquidations_std",
    "liquidation_ratio_mean": "liquidation_ratio_std",
    "final_price_mean": "final_price_std",
    "final_active_ratio_mean": "final_active_ratio_std",
    "inactive_ratio_mean": "inactive_ratio_std",
}


SUMMARY_METRICS = [
    "volatility_mean",
    "drawdown_severity_mean",
    "drop_from_initial_severity_mean",
    "liquidation_ratio_mean",
    "unique_margin_called_ratio_mean",
    "crash_50_probability",
]


METRIC_LABELS = {
    "volatility_mean": "Volatilidad",
    "drawdown_severity_mean": "Drawdown",
    "drop_from_initial_severity_mean": "Caída desde precio inicial",
    "liquidation_ratio_mean": "Ratio de liquidación",
    "unique_margin_called_ratio_mean": "Traders afectados",
    "crash_50_probability": "Crash > 50%",
}


def plot_metric(summary, parameter, metric, ylabel):
    """
    Genera una gráfica individual de sensibilidad.

    Eje X: valores del parámetro.
    Eje Y: media de la métrica.
    Si existe desviación típica, la muestra como barras de error.
    """
    subset = summary[summary["parameter"] == parameter].sort_values("value")

    x = subset["value"]
    y = subset[metric]

    std_col = STD_COLUMNS.get(metric)

    plt.figure(figsize=(7, 4))

    if std_col is not None and std_col in subset.columns:
        yerr = subset[std_col]
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            linewidth=2,
        )
    else:
        plt.plot(x, y, marker="o", linewidth=2)

    plt.xlabel(parameter)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} según {parameter}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = OUTPUT_DIR / f"{parameter}_{metric}.png"
    plt.savefig(filename, dpi=200)
    plt.close()


def compute_normalized_sensitivity(summary):
    """
    Calcula dos tablas de sensibilidad:

    1. sensitivity_raw:
       Para cada parámetro y métrica calcula:
       sensibilidad = máximo - mínimo.

    2. sensitivity_normalized:
       Normaliza cada columna dividiendo entre el máximo de esa métrica.
       Así todas las métricas quedan entre 0 y 1 y son comparables.

    Finalmente calcula un sensitivity_score como la media de las sensibilidades
    normalizadas.
    """
    rows = []

    for parameter in summary["parameter"].unique():
        subset = summary[summary["parameter"] == parameter]

        row = {"parameter": parameter}

        for metric in SUMMARY_METRICS:
            row[metric] = subset[metric].max() - subset[metric].min()

        rows.append(row)

    sensitivity_raw = pd.DataFrame(rows).set_index("parameter")

    sensitivity_normalized = sensitivity_raw.copy()

    for metric in SUMMARY_METRICS:
        max_value = sensitivity_normalized[metric].max()

        if max_value > 0:
            sensitivity_normalized[metric] = (
                sensitivity_normalized[metric] / max_value
            )
        else:
            sensitivity_normalized[metric] = 0.0

    sensitivity_normalized["sensitivity_score"] = sensitivity_normalized[
        SUMMARY_METRICS
    ].mean(axis=1)

    return sensitivity_raw, sensitivity_normalized


def plot_summary_ranking(sensitivity_normalized):
    """
    Genera un ranking agregado de sensibilidad.
    """
    ranking = sensitivity_normalized.sort_values("sensitivity_score", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(ranking.index, ranking["sensitivity_score"])
    plt.xlabel("Índice de sensibilidad normalizado")
    plt.title("Ranking agregado de sensibilidad")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    filename = OUTPUT_DIR / "summary_ranking_sensitivity.png"
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_summary_heatmap(sensitivity_normalized):
    """
    Genera un heatmap parámetro x métrica.

    Sirve para ver qué parámetro afecta más a cada dimensión del riesgo.
    """
    heatmap_data = sensitivity_normalized[SUMMARY_METRICS].copy()

    heatmap_data = heatmap_data.loc[
        sensitivity_normalized.sort_values(
            "sensitivity_score",
            ascending=False,
        ).index
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(heatmap_data.values, aspect="auto")

    ax.set_xticks(range(len(SUMMARY_METRICS)))
    ax.set_xticklabels(
        [METRIC_LABELS[m] for m in SUMMARY_METRICS],
        rotation=30,
        ha="right",
    )

    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)

    ax.set_title("Mapa de sensibilidad por parámetro y métrica")

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            value = heatmap_data.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, label="Sensibilidad normalizada")
    plt.tight_layout()

    filename = OUTPUT_DIR / "summary_heatmap_sensitivity.png"
    plt.savefig(filename, dpi=200)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    summary = pd.read_csv(SUMMARY_FILE)

    # Convertimos el drawdown a severidad positiva.
    summary["drawdown_severity_mean"] = -summary["max_drawdown_mean"]
    summary["drawdown_severity_std"] = summary["max_drawdown_std"]

    # Convertimos la caída desde precio inicial a severidad positiva.
    summary["drop_from_initial_severity_mean"] = -summary[
        "max_drop_from_initial_mean"
    ]
    summary["drop_from_initial_severity_std"] = summary[
        "max_drop_from_initial_std"
    ]

    important_plots = [
        ("lambda_", "drop_from_initial_severity_mean", "Caída media desde precio inicial"),
        ("position_scale", "drop_from_initial_severity_mean", "Caída media desde precio inicial"),
        ("position_scale", "liquidation_ratio_mean", "Ratio medio de liquidación"),
    ]

    for parameter, metric, ylabel in important_plots:
        plot_metric(summary, parameter, metric, ylabel)

    sensitivity_raw, sensitivity_normalized = compute_normalized_sensitivity(summary)

    sensitivity_raw.to_csv(OUTPUT_DIR / "summary_sensitivity_raw.csv")
    sensitivity_normalized.to_csv(
        OUTPUT_DIR / "summary_sensitivity_normalized.csv"
    )

    plot_summary_ranking(sensitivity_normalized)
    plot_summary_heatmap(sensitivity_normalized)

    print(f"Gráficas guardadas en {OUTPUT_DIR}")
    print("- Gráficas individuales de sensibilidad")
    print("- summary_ranking_sensitivity.png")
    print("- summary_heatmap_sensitivity.png")
    print("- summary_sensitivity_raw.csv")
    print("- summary_sensitivity_normalized.csv")


if __name__ == "__main__":
    main()