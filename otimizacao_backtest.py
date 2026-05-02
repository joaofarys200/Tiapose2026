"""
Fase 2 da Otimização – Backtest por Splits.

Para cada split do backtest de previsão (12 splits, janela rolling),
constrói grupos com as melhores previsões (best method por Store×Horizon),
executa os métodos de otimização e agrega lucros entre splits (mediana/média).

Referência do guia (Phase 2):
  "…you can run an execution of the same optimization method for each of the
  last weeks you executed forecasts… obtaining for example 20 profits, one for
  each of the weeks analyzed. And then you can aggregate the various profit
  values using a median or average value."

Uso:
    python otimizacao_backtest.py [--iterations 300] [--n-runs 1] [--output-prefix ...]
"""

import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Importa toda a lógica de otimização do módulo principal.
from otimizacao_metaheuristica import (
    DEFAULT_OMEGA,
    DEFAULT_GA_POP_SIZE,
    DEFAULT_GA_TOTAL_EVALS,
    DEFAULT_GA_CROSSOVER_RATE,
    DEFAULT_GA_MUTATION_RATE,
    DEFAULT_SA_COOLING_RATE,
    DEFAULT_SA_T_INITIAL,
    SA_T_FINAL,
    STORES_ORDERED,
    UNITS_CAP,
    GeneticAlgorithmOptimizer,
    Group,
    HillClimbingOptimizer,
    MonteCarloOptimizer,
    NSGA2Optimizer,
    SimulatedAnnealingOptimizer,
    evaluate_solution,
    solution_to_plan_df,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

STORE_CSV = {
    "Baltimore":    "csv/stores/baltimore.csv",
    "Lancaster":    "csv/stores/lancaster.csv",
    "Philadelphia": "csv/stores/philadelphia.csv",
    "Richmond":     "csv/stores/richmond.csv",
}

BACKTEST_CSV   = "csv/forecast/multivariate/multivariate_backtest_all_splits.csv"
BEST_METHODS_CSV = "csv/forecast/multivariate/multivariate_best_methods.csv"

# Parâmetros das janelas (devem coincidir com forecast_multivariate.py)
MAX_H = 7
N_BACKTEST_SPLITS = 12
MIN_TRAIN_SIZE = 180


# ---------------------------------------------------------------------------
# Construção dos grupos de cada split
# ---------------------------------------------------------------------------

def load_store_dates() -> dict[str, list[pd.Timestamp]]:
    """Carrega datas de cada loja (714 linhas)."""
    dates: dict[str, list[pd.Timestamp]] = {}
    for store, csv_path in STORE_CSV.items():
        df = pd.read_csv(csv_path, usecols=["Date"])
        df["Date"] = pd.to_datetime(df["Date"])
        dates[store] = df["Date"].tolist()
    return dates


def build_best_pred_lookup(
    backtest_df: pd.DataFrame, best_methods_df: pd.DataFrame
) -> dict[tuple[str, int, int], int]:
    """
    Constrói lookup: (store, split, horizon) → y_pred (clamp >=0).

    Usa o melhor método por Store×Horizon conforme best_methods_df.
    """
    # Melhor método por (store, horizon).
    best_lookup: dict[tuple[str, int], tuple[str, str]] = {}
    for _, row in best_methods_df.iterrows():
        key = (row["Store"], int(row["Horizon"]))
        if key not in best_lookup:
            best_lookup[key] = (row["Method"], row["LagSet"])

    pred_lookup: dict[tuple[str, int, int], int] = {}
    for (store, horizon), (method, lagset) in best_lookup.items():
        mask = (
            (backtest_df["Store"] == store)
            & (backtest_df["Horizon"] == horizon)
            & (backtest_df["Method"] == method)
            & (backtest_df["LagSet"] == lagset)
        )
        subset = backtest_df[mask]
        for _, row in subset.iterrows():
            customers = max(0, int(round(float(row["y_pred"]))))
            pred_lookup[(store, int(row["Split"]), horizon)] = customers

    return pred_lookup


def build_groups_for_split(
    split_id: int,
    pred_lookup: dict[tuple[str, int, int], int],
    store_dates: dict[str, list[pd.Timestamp]],
    n_total_rows: int = 714,
) -> list[Group]:
    """
    Constrói 28 grupos (4 lojas × 7 dias) para um split específico.

    origin para split i (1-based) =
        (n_total_rows - MAX_H) - N_BACKTEST_SPLITS + 1 + (split_id - 1)
    data do horizonte h = store_dates[origin + (h - 1)]
    """
    latest_origin = n_total_rows - MAX_H
    first_origin  = latest_origin - N_BACKTEST_SPLITS + 1
    origin        = first_origin + (split_id - 1)   # 0-indexed row

    groups: list[Group] = []
    idx = 0
    for store in STORES_ORDERED:
        for h in range(1, MAX_H + 1):
            date = store_dates[store][origin + (h - 1)]
            customers = pred_lookup.get((store, split_id, h), 0)
            groups.append(Group(idx=idx, store=store, date=date, horizon=h, customers=customers))
            idx += 1

    return groups


# ---------------------------------------------------------------------------
# Execução do backtest
# ---------------------------------------------------------------------------

def run_backtest(args: argparse.Namespace) -> None:
    output_path = Path(args.output_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_path)

    # Carregar dados
    backtest_df    = pd.read_csv(BACKTEST_CSV)
    best_methods_df = pd.read_csv(BEST_METHODS_CSV)
    store_dates    = load_store_dates()

    pred_lookup = build_best_pred_lookup(backtest_df, best_methods_df)

    splits = sorted(backtest_df["Split"].unique())
    print(f"Splits encontrados: {splits}")

    # Métodos e objetivos a executar
    objectives = ["o1", "o2"]
    if args.o3_mode in {"weighted", "both"}:
        objectives.append("o3_weighted")
    if args.o3_mode in {"pareto", "both"}:
        objectives.append("o3_pareto")

    default_constraint_modes = (
        [args.constraint_mode] if args.constraint_mode != "both" else ["repair", "penalty"]
    )
    base_methods = ["monte_carlo", "hill_climbing", "simulated_annealing", "genetic_algorithm"]

    # Resultados: lista de linhas {Objective, ConstraintMode, Method, Split, Run, Profit, Units, HR, Feasible}
    all_rows: list[dict] = []

    print("=" * 70)
    print("BACKTEST DE OTIMIZACAO (Fase 2)")
    print("=" * 70)

    for objective in objectives:
        obj_modes   = default_constraint_modes if objective in ("o2", "o3_weighted") else ["none"]
        obj_methods = ["nsga_ii"] if objective == "o3_pareto" else base_methods

        for constraint_mode in obj_modes:
            for method in obj_methods:
                label = f"{objective.upper()} × {method.upper()} × {constraint_mode.upper()}"
                print(f"\n[{label}]")

                scenario_key = f"{objective}|{method}|{constraint_mode}"
                seed_offset  = sum(ord(ch) for ch in scenario_key) * 1000

                for split_id in splits:
                    groups = build_groups_for_split(int(split_id), pred_lookup, store_dates)

                    for run_idx in range(args.n_runs):
                        random.seed(args.seed_base + seed_offset + int(split_id) * 100 + run_idx)

                        if method == "monte_carlo":
                            optimizer = MonteCarloOptimizer(
                                groups, objective, args.omega, constraint_mode, args.iterations
                            )
                        elif method == "hill_climbing":
                            optimizer = HillClimbingOptimizer(
                                groups, objective, args.omega, constraint_mode, args.iterations
                            )
                        elif method == "simulated_annealing":
                            optimizer = SimulatedAnnealingOptimizer(
                                groups, objective, args.omega, constraint_mode,
                                args.iterations, args.sa_temp_initial,
                                args.sa_temp_final, args.sa_cooling_rate,
                            )
                        elif method == "genetic_algorithm":
                            optimizer = GeneticAlgorithmOptimizer(
                                groups, objective, args.omega, constraint_mode,
                                total_evals=args.iterations,
                                pop_size=args.ga_pop_size,
                            )
                        else:
                            optimizer = NSGA2Optimizer(
                                groups, generations=args.iterations,
                                population_size=args.nsga_pop_size,
                            )

                        solution = optimizer.optimize()

                        # Calcular profit diretamente do solution_to_plan_df
                        plan_df = solution_to_plan_df(
                            objective.upper(), method.upper(), groups, solution
                        )
                        total_profit = int(plan_df["Daily_Profit"].sum())
                        feasible = (
                            (solution.total_units <= UNITS_CAP)
                            if objective in ("o2", "o3_weighted", "o3_pareto")
                            else True
                        )

                        all_rows.append(
                            {
                                "Objective": objective.upper(),
                                "ConstraintMode": constraint_mode.upper(),
                                "Method": method.upper(),
                                "Split": int(split_id),
                                "Run": run_idx + 1,
                                "Profit": total_profit,
                                "Units": int(solution.total_units),
                                "HR": int(solution.total_hr),
                                "Feasible": int(feasible),
                                "Fitness": float(solution.fitness_o1),
                            }
                        )

                    print(f"  Split {int(split_id):2d} -> done", end="\r")
                print()  # newline after split loop

    # Guardar resultados detalhados por split
    detail_df = pd.DataFrame(all_rows)
    detail_file = f"{output_prefix}_splits.csv"
    detail_df.to_csv(detail_file, index=False)
    print(f"\nDetailed results saved: {detail_file}")

    # Agregar por Objective × ConstraintMode × Method (mediana/média sobre splits×runs)
    summary_rows: list[dict] = []
    for (obj, mode, meth), grp in detail_df.groupby(
        ["Objective", "ConstraintMode", "Method"]
    ):
        feasible_only = grp[grp["Feasible"] == 1]
        profit_series = feasible_only["Profit"] if len(feasible_only) > 0 else grp["Profit"]
        summary_rows.append(
            {
                "Objective":      obj,
                "ConstraintMode": mode,
                "Method":         meth,
                "Median_Profit":  round(float(profit_series.median()), 1),
                "Mean_Profit":    round(float(profit_series.mean()), 1),
                "Std_Profit":     round(float(profit_series.std()), 1),
                "Min_Profit":     int(profit_series.min()),
                "Max_Profit":     int(profit_series.max()),
                "Median_Units":   round(float(grp["Units"].median()), 1),
                "Median_HR":      round(float(grp["HR"].median()), 1),
                "Feasible_Rate":  round(float(grp["Feasible"].mean()), 3),
                "N_Splits":       int(grp["Split"].nunique()),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["Objective", "ConstraintMode", "Method"]
    )

    print("\n" + "=" * 70)
    print("RESUMO AGREGADO (mediana/media de lucro sobre splits)")
    print("=" * 70)
    print("\n" + summary_df.to_string(index=False))

    summary_file = f"{output_prefix}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved: {summary_file}")

    # Visualização: boxplot de lucro por método para cada objetivo
    _save_profit_boxplots(detail_df, output_prefix)
    _save_median_bar(summary_df, output_prefix)


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------

def _save_profit_boxplots(detail_df: pd.DataFrame, output_prefix: str) -> None:
    """Boxplot de lucro por método para cada objetivo (distribuição entre splits)."""
    objectives = detail_df["Objective"].unique()
    n_obj = len(objectives)
    fig, axes = plt.subplots(1, n_obj, figsize=(6 * n_obj, 5), squeeze=False)

    for col, objective in enumerate(sorted(objectives)):
        ax = axes[0][col]
        sub = detail_df[detail_df["Objective"] == objective].copy()
        sub["Config"] = sub["Method"] + "\n" + sub["ConstraintMode"]
        configs = sorted(sub["Config"].unique())
        data = [sub[sub["Config"] == c]["Profit"].values for c in configs]
        ax.boxplot(data, tick_labels=configs, vert=True)
        ax.set_title(f"{objective} - Lucro por split")
        ax.set_ylabel("Lucro semanal (EUR)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)

    plt.suptitle("Distribuicao do lucro por split (Backtest Fase 2)", fontsize=13)
    plt.tight_layout()
    out_file = f"{output_prefix}_boxplot.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Boxplot saved: {out_file}")


def _save_median_bar(summary_df: pd.DataFrame, output_prefix: str) -> None:
    """Gráfico de barras com mediana do lucro por objetivo e método."""
    fig, ax = plt.subplots(figsize=(12, 5))
    summary_df["Config"] = summary_df["Method"] + " (" + summary_df["ConstraintMode"] + ")"
    pivot = summary_df.pivot(index="Objective", columns="Config", values="Median_Profit")
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Mediana do lucro semanal por metodo - Backtest Fase 2")
    ax.set_ylabel("Lucro mediano (EUR)")
    ax.set_xlabel("Objetivo")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_file = f"{output_prefix}_median_bar.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Median bar chart saved: {out_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fase 2: Backtest de otimização sobre 12 splits de previsão"
    )
    parser.add_argument(
        "--iterations", type=int, default=300,
        help="Número de iterações de cada método por split (default: 300).",
    )
    parser.add_argument(
        "--n-runs", type=int, default=1,
        help="Execuções independentes por split/método (default: 1).",
    )
    parser.add_argument(
        "--output-prefix", default="csv/optimization/backtest",
        help="Prefixo dos ficheiros de saída.",
    )
    parser.add_argument(
        "--omega", type=float, default=DEFAULT_OMEGA,
        help="Peso do lucro no O3_weighted (default: 0.7).",
    )
    parser.add_argument(
        "--constraint-mode", choices=["repair", "penalty", "both"], default="repair",
        help="Modo de constraints para O2/O3 (default: repair).",
    )
    parser.add_argument(
        "--o3-mode", choices=["weighted", "pareto", "both", "none"], default="weighted",
        help="Variante(s) do O3 a incluir (default: weighted).",
    )
    parser.add_argument(
        "--sa-temp-initial", type=float, default=DEFAULT_SA_T_INITIAL,
        help="Temperatura inicial do SA.",
    )
    parser.add_argument(
        "--sa-temp-final", type=float, default=SA_T_FINAL,
        help="Temperatura final do SA.",
    )
    parser.add_argument(
        "--sa-cooling-rate", type=float, default=DEFAULT_SA_COOLING_RATE,
        help="Taxa de arrefecimento do SA.",
    )
    parser.add_argument(
        "--nsga-pop-size", type=int, default=30,
        help="Tamanho da populacao NSGA-II.",
    )
    parser.add_argument(
        "--ga-pop-size", type=int, default=DEFAULT_GA_POP_SIZE,
        help="Tamanho da populacao do GA (default: 40). Geracoes = iterations / ga-pop-size.",
    )
    parser.add_argument(
        "--seed-base", type=int, default=42,
        help="Seed base para reprodutibilidade.",
    )
    args = parser.parse_args()

    run_backtest(args)


if __name__ == "__main__":
    main()
