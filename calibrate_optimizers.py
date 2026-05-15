from pathlib import Path

import numpy as np
import pandas as pd

from model.model import MarketModel


OUTPUT_DIR = Path("calibration_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Configuración general
# ============================================================

N_STEPS = 500

# Simulaciones por candidato durante la calibración.
N_ITER_CALIBRATION = 10

# Simulaciones para reevaluar las mejores soluciones.
N_ITER_REEVALUATION = 100

# Número de candidatos aleatorios.
N_RANDOM_CANDIDATES = 100

# Número de mejores soluciones a reevaluar.
TOP_N_REEVALUATE = 5


# ============================================================
# Parámetros fijos
# ============================================================

BASE_PARAMS = {
    "N_noise": 50,
    "N_traders": 50,
    "margen_mantenimiento": 0.25,
    "margen_mantenimiento_spread": 0.02,
}


# ============================================================
# Parámetros a calibrar
# ============================================================

PARAMETER_NAMES = [
    "lambda_",
    "alpha",
    "sigma_noise",
    "capital_scale",
    "position_scale",
]


BOUNDS = {
    "lambda_": (0.003, 0.04),
    "alpha": (0.3, 1.0),
    "sigma_noise": (0.25, 1.5),
    "capital_scale": (0.8, 1.6),
    "position_scale": (0.10, 0.40),
}


# ============================================================
# Targets de calibración
# ============================================================

TARGETS = {
    "volatility": 0.02,
    "max_drop_from_initial": -0.50,
    "crash_50_probability": 0.50,
    "liquidation_ratio": 0.30,
    "unique_margin_called_ratio": 0.40,
}


SCALES = {
    "volatility": 0.02,
    "max_drop_from_initial": 0.50,
    "crash_50_probability": 1.00,
    "liquidation_ratio": 0.30,
    "unique_margin_called_ratio": 0.40,
}


WEIGHTS = {
    "volatility": 0.5,
    "max_drop_from_initial": 1.5,
    "crash_50_probability": 1.5,
    "liquidation_ratio": 1.0,
    "unique_margin_called_ratio": 1.0,
}


# ============================================================
# Métricas
# ============================================================

def compute_initial_fragility(model):
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
    prices = np.asarray(prices)
    running_max = np.maximum.accumulate(prices)
    drawdowns = (prices - running_max) / running_max
    return drawdowns.min()


def run_one_simulation(params, steps):
    model = MarketModel(**params)

    initial_metrics = compute_initial_fragility(model)

    # Guardamos estado inicial para que el precio inicial sea 100.
    model.datacollector.collect(model)

    for _ in range(steps):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()

    prices = df["Price"]
    returns = df["Returns"].iloc[1:]
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    volatility = returns.std()
    max_drawdown = compute_max_drawdown(prices)

    initial_price = prices.iloc[0]
    max_drop_from_initial = ((prices - initial_price) / initial_price).min()

    margin_call_pressure = df["MarginCalls"].sum()

    initial_traders = df["ActiveTraders"].iloc[0]
    unique_margin_called = df["UniqueMarginCalled"].iloc[-1]

    unique_margin_called_ratio = (
        unique_margin_called / initial_traders
        if initial_traders > 0
        else 0.0
    )

    total_liquidations = df["Liquidations"].sum()
    initial_total_position = df["InitialTotalPosition"].iloc[0]

    liquidation_ratio = (
        total_liquidations / initial_total_position
        if initial_total_position > 0
        else 0.0
    )

    final_price = prices.iloc[-1]
    final_active_traders = df["ActiveTraders"].iloc[-1]

    final_active_ratio = (
        final_active_traders / initial_traders
        if initial_traders > 0
        else 0.0
    )

    inactive_ratio = 1.0 - final_active_ratio
    final_total_position = df["TotalPosition"].iloc[-1]

    crash_20 = int(max_drawdown < -0.20)
    crash_50 = int(max_drawdown < -0.50)
    crash_80 = int(max_drawdown < -0.80)

    crash_initial_20 = int(max_drop_from_initial < -0.20)
    crash_initial_50 = int(max_drop_from_initial < -0.50)
    crash_initial_80 = int(max_drop_from_initial < -0.80)

    return {
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


def aggregate_simulations(params, n_iterations, steps, seed_base):
    """
    Ejecuta varias simulaciones para una configuración.

    Usamos semillas fijas por iteración para que la comparación entre
    candidatos sea más estable durante la calibración.
    """
    simulation_results = []

    for iteration in range(n_iterations):
        np.random.seed(seed_base + iteration)

        metrics = run_one_simulation(params, steps)
        simulation_results.append(metrics)

    df = pd.DataFrame(simulation_results)

    result = {}

    continuous_metrics = [
        "initial_margin_calls",
        "initial_avg_leverage",
        "initial_max_leverage",
        "volatility",
        "max_drawdown",
        "max_drop_from_initial",
        "margin_call_pressure",
        "unique_margin_called",
        "unique_margin_called_ratio",
        "total_liquidations",
        "liquidation_ratio",
        "final_price",
        "final_active_traders",
        "final_active_ratio",
        "inactive_ratio",
        "final_total_position",
    ]

    for metric in continuous_metrics:
        result[metric] = df[metric].mean()
        result[f"{metric}_std"] = df[metric].std()

    result["crash_20_probability"] = df["crash_20"].mean()
    result["crash_50_probability"] = df["crash_50"].mean()
    result["crash_80_probability"] = df["crash_80"].mean()

    result["crash_initial_20_probability"] = df["crash_initial_20"].mean()
    result["crash_initial_50_probability"] = df["crash_initial_50"].mean()
    result["crash_initial_80_probability"] = df["crash_initial_80"].mean()

    return result


# ============================================================
# Loss
# ============================================================

def compute_loss(metrics):
    loss = 0.0

    for metric, target in TARGETS.items():
        value = metrics[metric]
        scale = SCALES[metric]
        weight = WEIGHTS[metric]

        loss += weight * ((value - target) / scale) ** 2

    return loss


def vector_to_params(x):
    variable_params = {
        name: float(value)
        for name, value in zip(PARAMETER_NAMES, x)
    }

    return {
        **BASE_PARAMS,
        **variable_params,
    }


def evaluate_candidate(x, n_iterations, steps, seed_base):
    params = vector_to_params(x)

    metrics = aggregate_simulations(
        params=params,
        n_iterations=n_iterations,
        steps=steps,
        seed_base=seed_base,
    )

    loss = compute_loss(metrics)

    row = {
        **{name: params[name] for name in PARAMETER_NAMES},
        **metrics,
        "loss": loss,
    }

    return row


# ============================================================
# Random search
# ============================================================

def random_search():
    print()
    print("=" * 80)
    print("CALIBRACIÓN: RANDOM SEARCH")
    print("=" * 80)

    rng = np.random.default_rng(123)
    rows = []

    for candidate_id in range(N_RANDOM_CANDIDATES):
        x = []

        for name in PARAMETER_NAMES:
            low, high = BOUNDS[name]
            x.append(rng.uniform(low, high))

        print(f"Random candidate {candidate_id + 1}/{N_RANDOM_CANDIDATES}")

        row = evaluate_candidate(
            x=x,
            n_iterations=N_ITER_CALIBRATION,
            steps=N_STEPS,
            seed_base=10_000,
        )

        row["method"] = "random_search"
        row["candidate_id"] = candidate_id

        rows.append(row)

    results = pd.DataFrame(rows)
    results = results.sort_values("loss", ascending=True)

    output_file = OUTPUT_DIR / "calibration_random_search.csv"
    results.to_csv(output_file, index=False)

    print()
    print(f"Resultados guardados en: {output_file}")
    print()
    print("Mejores configuraciones random search:")
    print(results.head(10).to_string(index=False))

    return results


# ============================================================
# Reevaluación de mejores soluciones
# ============================================================

def reevaluate_best(random_results):
    print()
    print("=" * 80)
    print("REEVALUACIÓN MONTE CARLO DE MEJORES CONFIGURACIONES")
    print("=" * 80)

    random_top = random_results.head(TOP_N_REEVALUATE).copy()

    reevaluation_rows = []

    for rank, (_, row) in enumerate(random_top.iterrows(), start=1):
        print(f"Reevaluando random_search rank {rank}")

        variable_params = {
            name: row[name]
            for name in PARAMETER_NAMES
        }

        params = {
            **BASE_PARAMS,
            **variable_params,
        }

        metrics = aggregate_simulations(
            params=params,
            n_iterations=N_ITER_REEVALUATION,
            steps=N_STEPS,
            seed_base=100_000 + rank * 10_000,
        )

        loss = compute_loss(metrics)

        reevaluation_row = {
            "source_method": "random_search",
            "source_rank": rank,
            **variable_params,
            **metrics,
            "loss": loss,
        }

        reevaluation_rows.append(reevaluation_row)

    reevaluation = pd.DataFrame(reevaluation_rows)
    reevaluation = reevaluation.sort_values("loss", ascending=True)

    output_file = OUTPUT_DIR / "calibration_reevaluation.csv"
    reevaluation.to_csv(output_file, index=False)

    best_file = OUTPUT_DIR / "best_calibrated_params.csv"
    reevaluation.head(1).to_csv(best_file, index=False)

    print()
    print(f"Reevaluación guardada en: {output_file}")
    print(f"Mejor configuración guardada en: {best_file}")
    print()
    print("Mejores configuraciones reevaluadas:")
    print(reevaluation.head(TOP_N_REEVALUATE).to_string(index=False))

    return reevaluation


# ============================================================
# Main
# ============================================================

def main():
    print("Inicio de calibración")
    print(f"Pasos por simulación: {N_STEPS}")
    print(f"Iteraciones por candidato durante calibración: {N_ITER_CALIBRATION}")
    print(f"Iteraciones por candidato durante reevaluación: {N_ITER_REEVALUATION}")
    print(f"Candidatos aleatorios: {N_RANDOM_CANDIDATES}")
    print()

    print("Targets:")
    for metric, target in TARGETS.items():
        print(f"- {metric}: {target}")

    print()
    print("Parámetros calibrados:")
    for name in PARAMETER_NAMES:
        print(f"- {name}: bounds={BOUNDS[name]}")

    random_results = random_search()
    reevaluation = reevaluate_best(random_results)

    print()
    print("=" * 80)
    print("CALIBRACIÓN COMPLETADA")
    print("=" * 80)
    print()
    print("Mejor configuración final:")
    best = reevaluation.iloc[0]

    for name in PARAMETER_NAMES:
        print(f"{name}: {best[name]:.6f}")

    print()
    print("Métricas principales:")
    for metric in TARGETS.keys():
        print(
            f"{metric}: {best[metric]:.6f} "
            f"(target={TARGETS[metric]})"
        )

    print(f"loss: {best['loss']:.6f}")


if __name__ == "__main__":
    main()