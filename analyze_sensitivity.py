import pandas as pd


INPUT_FILE = "sensitivity_results.csv"
SUMMARY_FILE = "sensitivity_summary.csv"
RANKING_FILE = "sensitivity_ranking.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    # Resumen por parámetro y valor.
    # Cada fila representa el comportamiento medio de una configuración
    # en las distintas iteraciones Monte Carlo.
    summary = (
        df.groupby(["parameter", "value"])
        .agg(
            initial_margin_calls_mean=("initial_margin_calls", "mean"),
            initial_avg_leverage_mean=("initial_avg_leverage", "mean"),
            initial_max_leverage_mean=("initial_max_leverage", "mean"),

            volatility_mean=("volatility", "mean"),
            volatility_std=("volatility", "std"),

            max_drawdown_mean=("max_drawdown", "mean"),
            max_drawdown_std=("max_drawdown", "std"),

            max_drop_from_initial_mean=("max_drop_from_initial", "mean"),
            max_drop_from_initial_std=("max_drop_from_initial", "std"),

            margin_call_pressure_mean=("margin_call_pressure", "mean"),
            margin_call_pressure_std=("margin_call_pressure", "std"),

            unique_margin_called_mean=("unique_margin_called", "mean"),
            unique_margin_called_std=("unique_margin_called", "std"),

            unique_margin_called_ratio_mean=("unique_margin_called_ratio", "mean"),
            unique_margin_called_ratio_std=("unique_margin_called_ratio", "std"),

            total_liquidations_mean=("total_liquidations", "mean"),
            total_liquidations_std=("total_liquidations", "std"),

            liquidation_ratio_mean=("liquidation_ratio", "mean"),
            liquidation_ratio_std=("liquidation_ratio", "std"),

            crash_20_probability=("crash_20", "mean"),
            crash_50_probability=("crash_50", "mean"),
            crash_80_probability=("crash_80", "mean"),

            crash_initial_20_probability=("crash_initial_20", "mean"),
            crash_initial_50_probability=("crash_initial_50", "mean"),
            crash_initial_80_probability=("crash_initial_80", "mean"),

            final_price_mean=("final_price", "mean"),
            final_price_std=("final_price", "std"),

            final_active_traders_mean=("final_active_traders", "mean"),
            final_active_traders_std=("final_active_traders", "std"),

            final_active_ratio_mean=("final_active_ratio", "mean"),
            final_active_ratio_std=("final_active_ratio", "std"),

            inactive_ratio_mean=("inactive_ratio", "mean"),
            inactive_ratio_std=("inactive_ratio", "std"),

            final_total_position_mean=("final_total_position", "mean"),
            final_total_position_std=("final_total_position", "std"),
        )
        .reset_index()
    )

    summary.to_csv(SUMMARY_FILE, index=False)

    # Ranking de sensibilidad.
    # Para cada parámetro medimos cuánto cambia cada métrica al variar sus valores.
    ranking_rows = []

    metrics_for_ranking = [
        "volatility_mean",
        "max_drawdown_mean",
        "max_drop_from_initial_mean",
        "liquidation_ratio_mean",
        "unique_margin_called_ratio_mean",
        "crash_50_probability",
    ]

    for parameter in summary["parameter"].unique():
        subset = summary[summary["parameter"] == parameter]

        row = {"parameter": parameter}

        for metric in metrics_for_ranking:
            metric_range = subset[metric].max() - subset[metric].min()
            row[f"{metric}_range"] = metric_range

        ranking_rows.append(row)

    ranking = pd.DataFrame(ranking_rows)

    # Ranking simple: media de rangos normalizados.
    # Así evitamos que métricas con escalas grandes dominen el ranking.
    range_columns = [col for col in ranking.columns if col.endswith("_range")]

    normalized = ranking[range_columns].copy()

    for col in range_columns:
        max_value = normalized[col].max()
        if max_value > 0:
            normalized[col] = normalized[col] / max_value
        else:
            normalized[col] = 0.0

    # Guardamos también los rangos normalizados para interpretar el score.
    for col in range_columns:
        ranking[f"{col}_normalized"] = normalized[col]

    # Índice agregado de sensibilidad.
    ranking["sensitivity_score"] = normalized.mean(axis=1)

    ranking = ranking.sort_values("sensitivity_score", ascending=False)

    ranking.to_csv(RANKING_FILE, index=False)

    print("Archivos generados:")
    print(f"- {SUMMARY_FILE}")
    print(f"- {RANKING_FILE}")
    print()
    print("Ranking agregado de sensibilidad:")
    print(
        ranking[
            [
                "parameter",
                "sensitivity_score",
                "max_drop_from_initial_mean_range",
                "liquidation_ratio_mean_range",
                "unique_margin_called_ratio_mean_range",
                "crash_50_probability_range",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()