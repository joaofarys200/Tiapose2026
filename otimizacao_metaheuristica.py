"""
Otimização com Metaheurísticas: Monte Carlo, Hill Climbing, Simulated Annealing
Substitui DP anterior por heurísticas para O1, O2, O3
Vetor solução: 84 valores [PR, X, J] × 4 stores × 7 dias
"""

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================================
# CONSTANTES E PARÂMETROS
# ============================================================================

STORE_PARAMS = {
    "Baltimore": {"Fj": 1.00, "Fx": 1.15, "Ws": 700},
    "Lancaster": {"Fj": 1.05, "Fx": 1.20, "Ws": 730},
    "Philadelphia": {"Fj": 1.10, "Fx": 1.15, "Ws": 760},
    "Richmond": {"Fj": 1.15, "Fx": 1.25, "Ws": 800},
}

STORES_ORDERED = ["Baltimore", "Lancaster", "Philadelphia", "Richmond"]
DAYS_PER_WEEK = 7
VECTOR_SIZE = len(STORES_ORDERED) * DAYS_PER_WEEK * 3  # 84 valores

# Parâmetros Heurísticas
DEFAULT_MC_ITERATIONS = 1000
DEFAULT_HC_ITERATIONS = 1000
DEFAULT_SA_ITERATIONS = 1000
DEFAULT_SA_T_INITIAL = 50.0
SA_T_FINAL = 0.01
DEFAULT_SA_COOLING_RATE = 0.98
DEFAULT_OMEGA = 0.7  # peso do lucro no O3; restante peso vai para RH
NEIGHBOR_PERTURB_PCT = 0.20

PR_MIN_INT = 0
PR_MAX_INT = 30

UNITS_CAP = 10000
PENALTY_WEIGHT = 1000.0  # Penalidade por unidade excedida


def pr_from_int(pr_int: int) -> float:
    return max(0.0, min(0.3, pr_int / 100.0))


def discretize_pr(pr_value: float) -> float:
    return pr_from_int(int(round(pr_value * 100)))

# ============================================================================
# ESTRUTURAS DE DADOS
# ============================================================================


@dataclass(frozen=True)
class Group:
    """Informação de um dia×loja"""
    idx: int
    store: str
    date: pd.Timestamp
    horizon: int
    customers: int


@dataclass(frozen=True)
class Option:
    """Resultado de uma decisão (PR, X, J) para um grupo"""
    pr: float
    x: int
    j: int
    assisted_x: int
    assisted_j: int
    units_x: int
    units_j: int
    units_total: int
    sales_x: int
    sales_j: int
    hr_cost_x: int
    hr_cost_j: int
    daily_profit: int
    hr_total: int


@dataclass
class Solution:
    """Solução: vetor de 84 valores [PR, X, J, PR, X, J, ...]"""
    values: list[float]  # 84 elementos
    fitness_o1: float = -1e9
    fitness_o2: float = -1e9
    fitness_o3: float = -1e9
    total_units: int = 0
    total_hr: int = 0
    feasible: bool = False

    def copy(self):
        return Solution(
            values=self.values.copy(),
            fitness_o1=self.fitness_o1,
            fitness_o2=self.fitness_o2,
            fitness_o3=self.fitness_o3,
            total_units=self.total_units,
            total_hr=self.total_hr,
            feasible=self.feasible,
        )


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================


def is_weekend(date_value: pd.Timestamp) -> bool:
    return date_value.dayofweek >= 5


def wage_x(date_value: pd.Timestamp) -> int:
    return 95 if is_weekend(date_value) else 80


def wage_j(date_value: pd.Timestamp) -> int:
    return 70 if is_weekend(date_value) else 60


def round_units(factor: float, pr: float) -> int:
    if pr >= 2.0:
        pr = 1.99
    return int(round(factor * 10.0 / math.log(2.0 - pr)))


def build_day_option(group: Group, pr: float, x: int, j: int) -> Option:
    """Função eval do professor"""
    params = STORE_PARAMS[group.store]

    assisted_x = min(7 * x, group.customers)
    rem = group.customers - assisted_x
    assisted_j = min(6 * j, rem)

    units_per_x = round_units(params["Fx"], pr)
    units_per_j = round_units(params["Fj"], pr)

    units_x = assisted_x * units_per_x
    units_j = assisted_j * units_per_j
    units_total = units_x + units_j

    sales_x = int(round(units_x * (1.0 - pr) * 1.07))
    sales_j = int(round(units_j * (1.0 - pr) * 1.07))

    hr_cost_x = x * wage_x(group.date)
    hr_cost_j = j * wage_j(group.date)

    daily_profit = sales_x + sales_j - hr_cost_x - hr_cost_j

    return Option(
        pr=pr,
        x=x,
        j=j,
        assisted_x=assisted_x,
        assisted_j=assisted_j,
        units_x=units_x,
        units_j=units_j,
        units_total=units_total,
        sales_x=sales_x,
        sales_j=sales_j,
        hr_cost_x=hr_cost_x,
        hr_cost_j=hr_cost_j,
        daily_profit=daily_profit,
        hr_total=x + j,
    )


# ============================================================================
# VETOR SOLUÇÃO ↔ DECISÕES
# ============================================================================


def solution_vector_to_decisions(
    sol_vector: list[float], groups: list[Group]
) -> dict[int, tuple[float, int, int]]:
    """
    Converte vetor [PR, X, J, PR, X, J, ...] para dict {group_idx: (PR, X, J)}
    """
    decisions: dict[int, tuple[float, int, int]] = {}
    
    for i, group in enumerate(groups):
        base_idx = i * 3
        pr = sol_vector[base_idx]
        x = int(sol_vector[base_idx + 1])
        j = int(sol_vector[base_idx + 2])
        
        # Clamp valores
        pr = discretize_pr(pr)
        x_upper_bound = math.ceil(group.customers / 7) if group.customers > 0 else 0
        j_upper_bound = math.ceil(group.customers / 6) if group.customers > 0 else 0
        x = max(0, min(x_upper_bound, x))
        j = max(0, min(j_upper_bound, j))
        
        decisions[group.idx] = (pr, x, j)
    
    return decisions


def decisions_to_options(
    decisions: dict[int, tuple[float, int, int]], groups: list[Group]
) -> dict[int, Option]:
    """
    Converte decisões {group_idx: (PR, X, J)} para {group_idx: Option}
    """
    options: dict[int, Option] = {}
    for group in groups:
        pr, x, j = decisions[group.idx]
        options[group.idx] = build_day_option(group, pr, x, j)
    return options


def is_solution_invalid(sol: Solution, groups: list[Group], units_cap: int = UNITS_CAP) -> bool:
    """Verifica se a solução global S é inválida."""
    decisions = solution_vector_to_decisions(sol.values, groups)
    options = decisions_to_options(decisions, groups)
    total_units = sum(opt.units_total for opt in options.values())
    return total_units > units_cap


def repair_solution(sol: Solution, groups: list[Group], units_cap: int = UNITS_CAP) -> Solution:
    """Aplica repair em S reduzindo blocos (PR, X, J) até satisfazer o cap."""
    repaired = sol.copy()

    for _ in range(5000):
        decisions = solution_vector_to_decisions(repaired.values, groups)
        options = decisions_to_options(decisions, groups)
        total_units = sum(opt.units_total for opt in options.values())
        if total_units <= units_cap:
            break

        worst_group = max(groups, key=lambda g: options[g.idx].units_total)
        base_idx = worst_group.idx * 3

        x_val = int(repaired.values[base_idx + 1])
        j_val = int(repaired.values[base_idx + 2])
        pr_val = float(repaired.values[base_idx])

        if x_val >= j_val and x_val > 0:
            repaired.values[base_idx + 1] = x_val - 1
        elif j_val > 0:
            repaired.values[base_idx + 2] = j_val - 1
        else:
            repaired.values[base_idx] = discretize_pr(max(0.0, pr_val - 0.05))

    return repaired


def generate_neighbor_on_triplet(solution: Solution, groups: list[Group]) -> Solution:
    """Gera vizinho com perturbação multiplicativa e bounds válidos (gr=)."""
    neighbor = solution.copy()
    block_idx = random.randint(0, len(groups) - 1)
    base_idx = block_idx * 3
    group = groups[block_idx]

    x_upper_bound = math.ceil(group.customers / 7) if group.customers > 0 else 0
    j_upper_bound = math.ceil(group.customers / 6) if group.customers > 0 else 0

    # PR discreto em [0..30] com perturbação percentual multiplicativa.
    cur_pr_int = int(round(discretize_pr(float(neighbor.values[base_idx])) * 100))
    pr_scale = 1.0 + random.uniform(-NEIGHBOR_PERTURB_PCT, NEIGHBOR_PERTURB_PCT)
    new_pr_int = int(round(cur_pr_int * pr_scale))
    new_pr_int = max(PR_MIN_INT, min(PR_MAX_INT, new_pr_int))
    neighbor.values[base_idx] = pr_from_int(new_pr_int)

    # X e J também com perturbação multiplicativa percentual.
    cur_x = int(neighbor.values[base_idx + 1])
    cur_j = int(neighbor.values[base_idx + 2])
    x_scale = 1.0 + random.uniform(-NEIGHBOR_PERTURB_PCT, NEIGHBOR_PERTURB_PCT)
    j_scale = 1.0 + random.uniform(-NEIGHBOR_PERTURB_PCT, NEIGHBOR_PERTURB_PCT)

    new_x = int(round(cur_x * x_scale))
    new_j = int(round(cur_j * j_scale))
    neighbor.values[base_idx + 1] = max(0, min(x_upper_bound, new_x))
    neighbor.values[base_idx + 2] = max(0, min(j_upper_bound, new_j))

    return neighbor


# ============================================================================
# FUNÇÕES DE AVALIAÇÃO (FITNESS)
# ============================================================================


def evaluate_solution(
    sol: Solution,
    groups: list[Group],
    objective: str = "o1",
    omega: float = DEFAULT_OMEGA,
    constraint_mode: str = "repair",
) -> tuple[float, int, int, bool]:
    """Avalia S com modo de constraints: repair ou penalty (death penalty)."""
    hard_cap_objective = objective in {"o2", "o3_weighted", "o3_pareto"}
    candidate = sol
    if hard_cap_objective and constraint_mode == "repair" and is_solution_invalid(sol, groups):
        candidate = repair_solution(sol, groups)

    decisions = solution_vector_to_decisions(candidate.values, groups)
    options = decisions_to_options(decisions, groups)

    total_profit = 0
    total_units = 0
    total_hr = 0

    for group in groups:
        opt = options[group.idx]
        total_profit += opt.daily_profit
        total_units += opt.units_total
        total_hr += opt.hr_total

    if hard_cap_objective:
        feasible = total_units <= UNITS_CAP
    else:
        # O1 não impõe limite rígido global de unidades.
        feasible = True

    if feasible:
        penalty = 0.0
    elif hard_cap_objective and constraint_mode == "penalty":
        # Death penalty: qualquer solução inviável fica fortemente dominada.
        penalty = 1e12
    else:
        penalty = PENALTY_WEIGHT * (total_units - UNITS_CAP)

    if objective == "o1":
        # O1: maximizar lucro total.
        fitness = total_profit - penalty
    elif objective == "o2":
        # O2: maximizar O1 com limite rígido de 10k unidades.
        fitness = total_profit - penalty
    elif objective == "o3_weighted":
        # O3: maximizar O2 e minimizar RH (versão weighted).
        omega = max(0.0, min(1.0, omega))
        fitness = omega * total_profit - (1.0 - omega) * 100.0 * total_hr - penalty
    elif objective == "o3_pareto":
        # O3: aproximação Pareto local onde lucro domina e RH desempata.
        fitness = (total_profit * 1000.0) - total_hr - penalty
    else:
        fitness = total_profit - penalty

    return fitness, total_units, total_hr, feasible


# ============================================================================
# GERAÇÃO DE SOLUÇÕES ALEATÓRIAS
# ============================================================================


def generate_random_solution(groups: list[Group]) -> Solution:
    """Gera solução aleatória com bounds corretos"""
    values = []
    for group in groups:
        pr = pr_from_int(random.randint(PR_MIN_INT, PR_MAX_INT))
        x_upper_bound = math.ceil(group.customers / 7) if group.customers > 0 else 0
        j_upper_bound = math.ceil(group.customers / 6) if group.customers > 0 else 0
        x = random.randint(0, x_upper_bound)
        j = random.randint(0, j_upper_bound)
        values.extend([pr, x, j])
    
    return Solution(values=values)


def generate_random_feasible_solution(groups: list[Group], units_cap: int = UNITS_CAP) -> Solution:
    """Gera solução aleatória já viável, sem recorrer a repair."""
    values: list[float] = []
    accumulated_units = 0

    for group in groups:
        pr = pr_from_int(random.randint(PR_MIN_INT, PR_MAX_INT))
        x_upper_bound = math.ceil(group.customers / 7) if group.customers > 0 else 0
        j_upper_bound = math.ceil(group.customers / 6) if group.customers > 0 else 0

        feasible_choices: list[tuple[int, int]] = [(0, 0)]
        for x in range(x_upper_bound + 1):
            for j in range(j_upper_bound + 1):
                option = build_day_option(group, pr, x, j)
                if accumulated_units + option.units_total <= units_cap:
                    feasible_choices.append((x, j))

        x, j = random.choice(feasible_choices)
        accumulated_units += build_day_option(group, pr, x, j).units_total
        values.extend([pr, x, j])

    return Solution(values=values)


# ============================================================================
# MONTE CARLO
# ============================================================================


class MonteCarloOptimizer:
    def __init__(self, groups: list[Group], objective: str = "o1", omega: float = DEFAULT_OMEGA, constraint_mode: str = "repair", iterations: int = DEFAULT_MC_ITERATIONS):
        self.groups = groups
        self.objective = objective
        self.omega = omega
        self.constraint_mode = constraint_mode
        self.iterations = iterations
        self.best_solution = None
        self.convergence = []  # [(iteration, best_fitness), ...]

    def optimize(self) -> Solution:
        # Bootstrap: iniciar com solução conservadora (baixo PR, poucos workers)
        self.best_solution = Solution(values=[0.1, 2, 2] * len(self.groups))
        if self.constraint_mode == "penalty":
            self.best_solution = generate_random_feasible_solution(self.groups)
        elif self.constraint_mode == "repair":
            self.best_solution = repair_solution(self.best_solution, self.groups)
        fitness, units, hr, _ = evaluate_solution(
            self.best_solution, self.groups, self.objective, self.omega, self.constraint_mode
        )
        self.best_solution.fitness_o1 = fitness
        self.best_solution.total_units = units
        self.best_solution.total_hr = hr

        for iteration in range(self.iterations):
            # Estratégia: misturar aleatório puro com perturbação do melhor
            if random.random() < 0.3:
                # 30%: exploração pura (aleatório)
                if self.constraint_mode == "penalty":
                    candidate = generate_random_feasible_solution(self.groups)
                else:
                    candidate = generate_random_solution(self.groups)
            else:
                # 70%: perturbação controlada do melhor
                candidate = self.best_solution.copy()
                num_perturb = random.randint(2, 3)
                for _ in range(num_perturb):
                    candidate = generate_neighbor_on_triplet(candidate, self.groups)
            if self.constraint_mode == "repair":
                candidate = repair_solution(candidate, self.groups)

            fitness, units, hr, _ = evaluate_solution(
                candidate, self.groups, self.objective, self.omega, self.constraint_mode
            )
            
            if fitness > self.best_solution.fitness_o1:
                self.best_solution = candidate.copy()
                self.best_solution.fitness_o1 = fitness
                self.best_solution.total_units = units
                self.best_solution.total_hr = hr

            self.convergence.append((iteration, self.best_solution.fitness_o1))

        return self.best_solution


# ============================================================================
# HILL CLIMBING
# ============================================================================


class HillClimbingOptimizer:
    def __init__(self, groups: list[Group], objective: str = "o1", omega: float = DEFAULT_OMEGA, constraint_mode: str = "repair", iterations: int = DEFAULT_HC_ITERATIONS):
        self.groups = groups
        self.objective = objective
        self.omega = omega
        self.constraint_mode = constraint_mode
        self.iterations = iterations
        self.best_solution = None
        self.convergence = []

    def get_neighbor(self, solution: Solution) -> Solution:
        """Gera vizinho por bloco (PR, X, J)."""
        return generate_neighbor_on_triplet(solution, self.groups)

    def optimize(self) -> Solution:
        if self.constraint_mode == "penalty":
            self.best_solution = generate_random_feasible_solution(self.groups)
        else:
            self.best_solution = generate_random_solution(self.groups)
        if self.constraint_mode == "repair":
            self.best_solution = repair_solution(self.best_solution, self.groups)
        fitness, units, hr, _ = evaluate_solution(
            self.best_solution, self.groups, self.objective, self.omega, self.constraint_mode
        )
        self.best_solution.fitness_o1 = fitness
        self.best_solution.total_units = units
        self.best_solution.total_hr = hr

        for iteration in range(self.iterations):
            neighbor = self.get_neighbor(self.best_solution)
            if self.constraint_mode == "repair":
                neighbor = repair_solution(neighbor, self.groups)
            fitness, units, hr, _ = evaluate_solution(
                neighbor, self.groups, self.objective, self.omega, self.constraint_mode
            )
            
            if fitness > self.best_solution.fitness_o1:
                self.best_solution = neighbor.copy()
                self.best_solution.fitness_o1 = fitness
                self.best_solution.total_units = units
                self.best_solution.total_hr = hr

            self.convergence.append((iteration, self.best_solution.fitness_o1))

        return self.best_solution


# ============================================================================
# SIMULATED ANNEALING
# ============================================================================


class SimulatedAnnealingOptimizer:
    def __init__(
        self,
        groups: list[Group],
        objective: str = "o1",
        omega: float = DEFAULT_OMEGA,
        constraint_mode: str = "repair",
        iterations: int = DEFAULT_SA_ITERATIONS,
        t_initial: float = DEFAULT_SA_T_INITIAL,
        t_final: float = SA_T_FINAL,
        cooling_rate: float = DEFAULT_SA_COOLING_RATE,
    ):
        self.groups = groups
        self.objective = objective
        self.omega = omega
        self.constraint_mode = constraint_mode
        self.iterations = iterations
        self.t_initial = t_initial
        self.t_final = t_final
        self.cooling_rate = cooling_rate
        self.best_solution = None
        self.convergence = []

    def get_neighbor(self, solution: Solution) -> Solution:
        """Gera vizinho com perturbação por bloco (PR, X, J)."""
        return generate_neighbor_on_triplet(solution, self.groups)

    def optimize(self) -> Solution:
        if self.constraint_mode == "penalty":
            current = generate_random_feasible_solution(self.groups)
        else:
            current = generate_random_solution(self.groups)
        if self.constraint_mode == "repair":
            current = repair_solution(current, self.groups)
        fitness, units, hr, _ = evaluate_solution(
            current, self.groups, self.objective, self.omega, self.constraint_mode
        )
        current.fitness_o1 = fitness
        current.total_units = units
        current.total_hr = hr

        self.best_solution = current.copy()

        temperature = self.t_initial
        for iteration in range(self.iterations):
            neighbor = self.get_neighbor(current)
            if self.constraint_mode == "repair":
                neighbor = repair_solution(neighbor, self.groups)
            fitness_new, units_new, hr_new, _ = evaluate_solution(
                neighbor, self.groups, self.objective, self.omega, self.constraint_mode
            )
            neighbor.fitness_o1 = fitness_new
            neighbor.total_units = units_new
            neighbor.total_hr = hr_new

            # Critério de Metropolis
            delta = fitness_new - current.fitness_o1
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-8)):
                current = neighbor

            if current.fitness_o1 > self.best_solution.fitness_o1:
                self.best_solution = current.copy()

            # Resfriamento
            temperature *= self.cooling_rate
            temperature = max(temperature, self.t_final)

            self.convergence.append((iteration, self.best_solution.fitness_o1))

        return self.best_solution


# ============================================================================
# CARREGAMENTO DE DADOS
# ============================================================================


def load_forecast_groups(forecast_file: Path) -> list[Group]:
    """Carrega grupos de previsão (mesmo formato anterior)"""
    df = pd.read_csv(forecast_file)
    needed = {"Store", "Horizon", "ForecastDate", "Pred_Num_Customers"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Forecast file missing columns: {sorted(missing)}")

    df = df.copy()
    df["ForecastDate"] = pd.to_datetime(df["ForecastDate"])
    df["Pred_Num_Customers"] = (
        pd.to_numeric(df["Pred_Num_Customers"], errors="coerce").fillna(0.0).round().astype(int)
    )
    df["Pred_Num_Customers"] = df["Pred_Num_Customers"].clip(lower=0)

    df = df[df["Store"].isin(STORE_PARAMS.keys())]
    df = df.sort_values(["Store", "Horizon"]).reset_index(drop=True)

    groups: list[Group] = []
    idx = 0
    for store in STORES_ORDERED:
        store_grp = df[df["Store"] == store].copy()
        if len(store_grp) < 7:
            raise ValueError(f"Store {store} has fewer than 7 forecast rows.")
        first7 = store_grp.head(7)
        for _, row in first7.iterrows():
            groups.append(
                Group(
                    idx=idx,
                    store=store,
                    date=row["ForecastDate"],
                    horizon=int(row["Horizon"]),
                    customers=int(row["Pred_Num_Customers"]),
                )
            )
            idx += 1

    if len(groups) != 28:
        raise ValueError(f"Expected 28 rows (4 stores x 7 days). Got {len(groups)}.")

    return groups


# ============================================================================
# CONSTRUÇÃO DE PLANOS CSV
# ============================================================================


def solution_to_plan_df(
    objective: str, method: str, groups: list[Group], solution: Solution
) -> pd.DataFrame:
    """Converte solução para DataFrame de plano diário"""
    decisions = solution_vector_to_decisions(solution.values, groups)
    options = decisions_to_options(decisions, groups)

    rows = []
    for group in groups:
        opt = options[group.idx]
        rows.append(
            {
                "Objective": objective,
                "Method": method,
                "Store": group.store,
                "Date": group.date.date().isoformat(),
                "Horizon": group.horizon,
                "Pred_Customers": group.customers,
                "PR": opt.pr,
                "X": opt.x,
                "J": opt.j,
                "Assisted_X": opt.assisted_x,
                "Assisted_J": opt.assisted_j,
                "Units_X": opt.units_x,
                "Units_J": opt.units_j,
                "Units_Total": opt.units_total,
                "Sales_X": opt.sales_x,
                "Sales_J": opt.sales_j,
                "HR_Cost_X": opt.hr_cost_x,
                "HR_Cost_J": opt.hr_cost_j,
                "Daily_Profit": opt.daily_profit,
                "Daily_HR_Total": opt.hr_total,
                "IsWeekend": int(is_weekend(group.date)),
            }
        )

    df = pd.DataFrame(rows)
    return df


def save_convergence_plots(convergence_data: dict, output_prefix: str) -> list[str]:
    """Guarda curvas de convergência por objetivo (inclui modo de constraints)."""
    created_files: list[str] = []
    method_labels = {
        "monte_carlo": "Monte Carlo",
        "hill_climbing": "Hill Climbing",
        "simulated_annealing": "Simulated Annealing",
    }
    method_colors = {
        "monte_carlo": "tab:blue",
        "hill_climbing": "tab:green",
        "simulated_annealing": "tab:red",
    }

    objectives = sorted({key.rsplit("_", 2)[0] for key in convergence_data.keys()})
    for objective in objectives:
        plt.figure(figsize=(10, 6))
        has_series = False

        for method in ["monte_carlo", "hill_climbing", "simulated_annealing"]:
            for constraint_mode in ["repair", "penalty"]:
                key = f"{objective}_{method}_{constraint_mode}"
                if key not in convergence_data or not convergence_data[key]:
                    continue

                iterations = [point[0] for point in convergence_data[key]]
                scores = [point[1] for point in convergence_data[key]]
                linestyle = "-" if constraint_mode == "repair" else "--"
                plt.plot(
                    iterations,
                    scores,
                    label=f"{method_labels[method]} ({constraint_mode})",
                    color=method_colors[method],
                    linestyle=linestyle,
                    linewidth=2,
                )
                has_series = True

        if not has_series:
            plt.close()
            continue

        plt.title(f"Convergência - {objective.upper()}")
        plt.xlabel("Iteração")
        plt.ylabel("Best Fitness")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_file = f"{output_prefix}_{objective}_convergence.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        created_files.append(out_file)

    return created_files


def save_summary_visualization(summary: pd.DataFrame, output_prefix: str) -> str:
    """Guarda dashboard comparativo com medianas por objetivo/método/modo."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    viz_df = summary.copy()
    viz_df["Method_Config"] = viz_df["Method"] + "_" + viz_df["ConstraintMode"]

    profit_pivot = viz_df.pivot(index="Objective", columns="Method_Config", values="Median_Profit")
    hr_pivot = viz_df.pivot(index="Objective", columns="Method_Config", values="Median_HR")
    units_pivot = viz_df.pivot(index="Objective", columns="Method_Config", values="Median_Units")

    profit_pivot.plot(kind="bar", ax=axes[0], title="Lucro (mediana)")
    hr_pivot.plot(kind="bar", ax=axes[1], title="RH (mediana)")
    units_pivot.plot(kind="bar", ax=axes[2], title="Unidades (mediana)")

    axes[0].set_ylabel("Lucro")
    axes[1].set_ylabel("RH")
    axes[2].set_ylabel("Unidades")

    for ax in axes:
        ax.set_xlabel("Objetivo")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Comparação dos Métodos Metaheurísticos", fontsize=14)
    plt.tight_layout()

    out_file = f"{output_prefix}_dashboard.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Otimização com Metaheurísticas (MC, HC, SA) para O1, O2, O3"
    )
    parser.add_argument(
        "--forecast-file",
        default="csv/forecast/multivariate/multivariate_next7.csv",
        help="CSV com previsões",
    )
    parser.add_argument(
        "--output-prefix",
        default="csv/optimization/metaheuristica",
        help="Prefixo para ficheiros de saída",
    )
    parser.add_argument(
        "--save-all-plans",
        action="store_true",
        help="Guardar também os CSVs individuais do melhor run por cenário.",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=DEFAULT_OMEGA,
        help="Peso do lucro no O3, com valor entre 0 e 1.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Número de execuções independentes por método/objetivo.",
    )
    parser.add_argument(
        "--mc-iterations",
        type=int,
        default=DEFAULT_MC_ITERATIONS,
        help="Iterações do Monte Carlo.",
    )
    parser.add_argument(
        "--hc-iterations",
        type=int,
        default=DEFAULT_HC_ITERATIONS,
        help="Iterações do Hill Climbing.",
    )
    parser.add_argument(
        "--sa-iterations",
        type=int,
        default=DEFAULT_SA_ITERATIONS,
        help="Iterações do Simulated Annealing.",
    )
    parser.add_argument(
        "--sa-temp-initial",
        type=float,
        default=DEFAULT_SA_T_INITIAL,
        help="Temperatura inicial do Simulated Annealing.",
    )
    parser.add_argument(
        "--sa-temp-final",
        type=float,
        default=SA_T_FINAL,
        help="Temperatura final do Simulated Annealing.",
    )
    parser.add_argument(
        "--sa-cooling-rate",
        type=float,
        default=DEFAULT_SA_COOLING_RATE,
        help="Taxa de arrefecimento do Simulated Annealing.",
    )
    parser.add_argument(
        "--constraint-mode",
        choices=["repair", "penalty", "both"],
        default="both",
        help="Modo de constraints: repair, death penalty ou ambos.",
    )
    parser.add_argument(
        "--o3-mode",
        choices=["weighted", "pareto", "both"],
        default="both",
        help="Estratégia do O3: weighted, pareto, ou ambos.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Seed base para replicações.",
    )
    args = parser.parse_args()

    output_prefix_path = Path(args.output_prefix)
    output_prefix_path.parent.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_prefix_path)

    forecast_file = Path(args.forecast_file)
    if not forecast_file.exists():
        raise FileNotFoundError(f"Forecast file not found: {forecast_file}")

    groups = load_forecast_groups(forecast_file)

    objectives = ["o1", "o2"]
    if args.o3_mode in {"weighted", "both"}:
        objectives.append("o3_weighted")
    if args.o3_mode in {"pareto", "both"}:
        objectives.append("o3_pareto")

    constraint_modes = [args.constraint_mode] if args.constraint_mode != "both" else ["repair", "penalty"]
    methods = ["monte_carlo", "hill_climbing", "simulated_annealing"]
    plan_results: list[dict] = []
    run_rows: list[dict] = []
    scenario_rows: list[dict] = []
    convergence_data = {}

    print("=" * 70)
    print("OTIMIZAÇÃO COM METAHEURÍSTICAS")
    print("=" * 70)

    # Executar cenários com replicações por método/objetivo/modo.
    for objective in objectives:
        for constraint_mode in constraint_modes:
            for method in methods:
                label = f"{objective.upper()} × {method.upper()} × {constraint_mode.upper()}"
                print(f"\n[{label}]")

                per_run_results: list[dict] = []
                scenario_key = f"{objective}|{method}|{constraint_mode}"
                scenario_seed_offset = sum(ord(ch) for ch in scenario_key) * 1000
                for run_idx in range(args.n_runs):
                    random.seed(args.seed_base + scenario_seed_offset + run_idx)

                    if method == "monte_carlo":
                        optimizer = MonteCarloOptimizer(groups, objective, args.omega, constraint_mode, args.mc_iterations)
                    elif method == "hill_climbing":
                        optimizer = HillClimbingOptimizer(groups, objective, args.omega, constraint_mode, args.hc_iterations)
                    else:
                        optimizer = SimulatedAnnealingOptimizer(
                            groups,
                            objective,
                            args.omega,
                            constraint_mode,
                            args.sa_iterations,
                            args.sa_temp_initial,
                            args.sa_temp_final,
                            args.sa_cooling_rate,
                        )

                    solution = optimizer.optimize()
                    plan_df = solution_to_plan_df(objective.upper(), method.upper(), groups, solution)
                    total_profit = int(plan_df["Daily_Profit"].sum())
                    feasible = solution.total_units <= UNITS_CAP

                    per_run_results.append(
                        {
                            "run_idx": run_idx,
                            "solution": solution,
                            "plan_df": plan_df,
                            "fitness": float(solution.fitness_o1),
                            "total_units": int(solution.total_units),
                            "total_hr": int(solution.total_hr),
                            "total_profit": total_profit,
                            "feasible": int(feasible),
                            "convergence": optimizer.convergence,
                        }
                    )

                    run_rows.append(
                        {
                            "Objective": objective.upper(),
                            "ConstraintMode": constraint_mode.upper(),
                            "Method": method.upper(),
                            "Run": run_idx + 1,
                            "Best_Fitness": float(solution.fitness_o1),
                            "Total_Units": int(solution.total_units),
                            "Total_HR": int(solution.total_hr),
                            "Total_Profit": total_profit,
                            "Feasible": int(feasible),
                        }
                    )

                fitness_vals = [r["fitness"] for r in per_run_results]
                profit_vals = [r["total_profit"] for r in per_run_results]
                units_vals = [r["total_units"] for r in per_run_results]
                hr_vals = [r["total_hr"] for r in per_run_results]
                feasible_vals = [r["feasible"] for r in per_run_results]

                best_run = max(per_run_results, key=lambda r: r["fitness"])

                print(f"  Median Fitness: {float(pd.Series(fitness_vals).median()):.2f}")
                print(f"  Mean Fitness: {float(pd.Series(fitness_vals).mean()):.2f}")
                print(f"  Best Fitness: {best_run['fitness']:.2f}")
                print(f"  Feasible Rate: {float(pd.Series(feasible_vals).mean()):.2f}")

                if args.save_all_plans:
                    out_file = f"{output_prefix}_{objective}_{method}_{constraint_mode}_best_run_plan.csv"
                    best_run["plan_df"].to_csv(out_file, index=False)
                    print(f"  Plan: {out_file}")

                plan_results.append(
                    {
                        "Objective": objective.upper(),
                        "ConstraintMode": constraint_mode.upper(),
                        "Method": method.upper(),
                        "Fitness": best_run["fitness"],
                        "Total_Units": best_run["total_units"],
                        "Total_HR": best_run["total_hr"],
                        "Total_Profit": best_run["total_profit"],
                        "Feasible": best_run["feasible"],
                        "plan_df": best_run["plan_df"],
                    }
                )

                scenario_rows.append(
                    {
                        "Objective": objective.upper(),
                        "ConstraintMode": constraint_mode.upper(),
                        "Method": method.upper(),
                        "Median_Best_Fitness": float(pd.Series(fitness_vals).median()),
                        "Mean_Best_Fitness": float(pd.Series(fitness_vals).mean()),
                        "Median_Profit": float(pd.Series(profit_vals).median()),
                        "Median_Units": float(pd.Series(units_vals).median()),
                        "Median_HR": float(pd.Series(hr_vals).median()),
                        "Feasible_Rate": float(pd.Series(feasible_vals).mean()),
                    }
                )

                convergence_data[f"{objective}_{method}_{constraint_mode}"] = best_run["convergence"]

    # Guardar resultados por execução
    runs_file = f"{output_prefix}_runs.csv"
    pd.DataFrame(run_rows).to_csv(runs_file, index=False)
    print(f"Runs saved: {runs_file}")

    # Guardar convergência em JSON
    convergence_json = {
        key: [[int(it), float(fit)] for it, fit in conv]
        for key, conv in convergence_data.items()
    }
    conv_file = f"{output_prefix}_convergence.json"
    with open(conv_file, "w") as f:
        json.dump(convergence_json, f, indent=2)
    print(f"\nConvergence saved: {conv_file}")

    # Resumo global
    print("\n" + "=" * 70)
    print("RESUMO")
    print("=" * 70)
    summary = pd.DataFrame(scenario_rows).sort_values(["Objective", "ConstraintMode", "Method"])
    print("\n" + summary.to_string(index=False))

    # Guardar resumo principal
    summary_file = f"{output_prefix}_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"\nSummary saved: {summary_file}")

    # Guardar apenas o melhor plano de cada objetivo (com melhor run e configuração).
    best_plan_frames = []
    for objective in sorted({r["Objective"] for r in plan_results}):
        candidates = [r for r in plan_results if r["Objective"] == objective]
        if not candidates:
            continue
        best_result = max(candidates, key=lambda r: r["Fitness"])
        best_plan_frames.append(best_result["plan_df"])

    best_plans_df = pd.concat(best_plan_frames, ignore_index=True)
    best_plans_file = f"{output_prefix}_best_plans.csv"
    best_plans_df.to_csv(best_plans_file, index=False)
    print(f"Best plans saved: {best_plans_file}")

    # Guardar visualizações
    plot_files = save_convergence_plots(convergence_data, output_prefix)
    dashboard_file = save_summary_visualization(summary, output_prefix)
    print("\nVisualization files:")
    for file_name in plot_files:
        print(f" - {file_name}")
    print(f" - {dashboard_file}")


if __name__ == "__main__":
    main()
