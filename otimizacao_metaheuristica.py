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
DEFAULT_NSGA2_POP_SIZE = 40
DEFAULT_GA_POP_SIZE = 40       # populacao GA; geracoes = total_evals / pop_size
DEFAULT_GA_TOTAL_EVALS = 1000  # equivalente a 1000 iteracoes dos outros metodos
DEFAULT_GA_CROSSOVER_RATE = 0.8
DEFAULT_GA_MUTATION_RATE = 0.33  # mutationChance do professor (rbga genalg)
DEFAULT_GA_TOURNAMENT_K = 3
DEFAULT_OMEGA = 0.7  # peso do lucro no O3; restante peso vai para RH
NEIGHBOR_PERTURB_PCT = 0.20

PR_MIN_INT = 0
PR_MAX_INT = 30

UNITS_CAP = 10000


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
    """Aplica repair multiplicando s = s * 0.95 até satisfazer o cap (abordagem do professor)."""
    repaired = sol.copy()

    while True:
        decisions = solution_vector_to_decisions(repaired.values, groups)
        options = decisions_to_options(decisions, groups)
        total_units = sum(opt.units_total for opt in options.values())
        if total_units <= units_cap:
            break

        # Reduz X e J de todos os grupos por 0.95 (arredondado ao inteiro)
        changed = False
        for g in groups:
            base_idx = g.idx * 3
            new_x = int(repaired.values[base_idx + 1] * 0.95)  # truncamento garante 1->0
            new_j = int(repaired.values[base_idx + 2] * 0.95)
            if new_x != repaired.values[base_idx + 1] or new_j != repaired.values[base_idx + 2]:
                changed = True
            repaired.values[base_idx + 1] = new_x
            repaired.values[base_idx + 2] = new_j

        # Se nada mudou (todos X=0, J=0) não é possível reparar mais
        if not changed:
            break

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

    # X e J: perturbação multiplicativa percentual (análogo ao gr= do R/SANN).
    # Garantia de deslocamento mínimo de ±1 para evitar estagnação em inteiros pequenos
    # (ex: int(round(1 * 0.9)) = 1 sem este ajuste).
    cur_x = int(neighbor.values[base_idx + 1])
    cur_j = int(neighbor.values[base_idx + 2])
    x_scale = 1.0 + random.uniform(-NEIGHBOR_PERTURB_PCT, NEIGHBOR_PERTURB_PCT)
    j_scale = 1.0 + random.uniform(-NEIGHBOR_PERTURB_PCT, NEIGHBOR_PERTURB_PCT)

    raw_dx = cur_x * x_scale - cur_x  # deslocamento multiplicativo puro
    raw_dj = cur_j * j_scale - cur_j
    dx = math.copysign(max(1.0, abs(raw_dx)), raw_dx) if raw_dx != 0 else random.choice([-1, 1])
    dj = math.copysign(max(1.0, abs(raw_dj)), raw_dj) if raw_dj != 0 else random.choice([-1, 1])

    new_x = cur_x + int(round(dx))
    new_j = cur_j + int(round(dj))
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
    """
    Avalia S e devolve (fitness, total_units, total_hr, feasible).
    O2 e O3 têm cap de 10,000 unidades (O3 = 'Maximize O2 and Minimize HR').
    Repair aplicado a O2 e O3_weighted em modo repair; death penalty para ambos em modo penalty.
    O1 não tem cap.
    """
    has_cap = objective in ("o2", "o3_weighted", "o3_pareto")
    candidate = sol
    # Repair para O2 e O3 em modo repair.
    if has_cap and constraint_mode == "repair" and is_solution_invalid(sol, groups):
        candidate = repair_solution(sol, groups)

    decisions = solution_vector_to_decisions(candidate.values, groups)
    options = decisions_to_options(decisions, groups)

    total_profit = 0
    total_units = 0
    total_hr = 0

    # Acumula lucro diário por loja e subtrai custo fixo semanal Ws de cada loja.
    # Rs = sum_d(Rs,d) - Ws  (conforme enunciado, slide 14)
    store_daily_profit: dict[str, int] = {}
    for group in groups:
        opt = options[group.idx]
        store_daily_profit[group.store] = store_daily_profit.get(group.store, 0) + opt.daily_profit
        total_units += opt.units_total
        total_hr += opt.hr_total

    for store, daily_sum in store_daily_profit.items():
        ws = STORE_PARAMS[store]["Ws"]
        total_profit += daily_sum - ws

    # O2 e O3 têm restrição de cap de unidades (O3 = "Maximize O2 and Minimize HR").
    feasible = (total_units <= UNITS_CAP) if has_cap else True

    # Death penalty: solução inviável é imediatamente rejeitada com fitness -inf.
    if has_cap and not feasible and constraint_mode == "penalty":
        return -math.inf, total_units, total_hr, feasible

    if objective == "o1":
        # O1: maximizar lucro semanal total (sem restrição de cap).
        fitness = float(total_profit)
    elif objective == "o2":
        # O2: maximizar lucro com hard constraint de 10,000 unidades.
        fitness = float(total_profit)
    elif objective == "o3_weighted":
        # O3 weighted: "Maximize O2 and Minimize HR" via soma ponderada normalizada.
        # O cap é herdado de O2 (death penalty aplicada acima).
        # profit_ref calibrado para o lucro real máximo com cap de 10000 unidades (~500 após Ws).
        # hr_ref = número máximo de trabalhadores por semana (28 grupos × ~16 workers máx ≈ 450).
        omega = max(0.0, min(1.0, omega))
        profit_ref = 500.0    # lucro máximo real observado em O2 repair (após Ws)
        hr_ref = 450.0        # HR total máximo esperado
        fitness = (omega * (float(total_profit) / profit_ref)
                   - (1.0 - omega) * (float(total_hr) / hr_ref))
    elif objective == "o3_pareto":
        # O3 Pareto: NSGA-II usa evaluate_profit_units_hr diretamente; retorna lucro bruto.
        fitness = float(total_profit)
    else:
        fitness = float(total_profit)

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
        # Monte Carlo: pure random sampling (mcsearch do professor)
        # Gera N soluções independentes e mantém a melhor.
        has_cap = self.objective in ("o2", "o3_weighted")
        self.best_solution = generate_random_solution(self.groups)
        if has_cap and self.constraint_mode in ("repair", "penalty"):
            self.best_solution = repair_solution(self.best_solution, self.groups)
        fitness, units, hr, _ = evaluate_solution(
            self.best_solution, self.groups, self.objective, self.omega, self.constraint_mode
        )
        self.best_solution.fitness_o1 = fitness
        self.best_solution.total_units = units
        self.best_solution.total_hr = hr

        for iteration in range(self.iterations):
            candidate = generate_random_solution(self.groups)
            if has_cap and self.constraint_mode == "repair":
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
        has_cap = self.objective in ("o2", "o3_weighted")
        self.best_solution = generate_random_solution(self.groups)
        if has_cap and self.constraint_mode in ("repair", "penalty"):
            self.best_solution = repair_solution(self.best_solution, self.groups)
        fitness, units, hr, _ = evaluate_solution(
            self.best_solution, self.groups, self.objective, self.omega, self.constraint_mode
        )
        self.best_solution.fitness_o1 = fitness
        self.best_solution.total_units = units
        self.best_solution.total_hr = hr

        for iteration in range(self.iterations):
            neighbor = self.get_neighbor(self.best_solution)
            if has_cap and self.constraint_mode == "repair":
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
        has_cap = self.objective in ("o2", "o3_weighted")
        current = generate_random_solution(self.groups)
        if has_cap and self.constraint_mode in ("repair", "penalty"):
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
            if has_cap and self.constraint_mode == "repair":
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
# NSGA-II (O3 PARETO)
# ============================================================================


def evaluate_profit_units_hr(sol: Solution, groups: list[Group]) -> tuple[int, int, int]:
    """Avalia (lucro semanal total, unidades totais, HR total) para o NSGA-II.
    O lucro inclui a subtração do custo fixo semanal Ws de cada loja."""
    decisions = solution_vector_to_decisions(sol.values, groups)
    options = decisions_to_options(decisions, groups)
    store_daily: dict[str, int] = {}
    total_units = 0
    total_hr = 0
    for group in groups:
        opt = options[group.idx]
        store_daily[group.store] = store_daily.get(group.store, 0) + opt.daily_profit
        total_units += opt.units_total
        total_hr += opt.hr_total
    total_profit = sum(v - STORE_PARAMS[s]["Ws"] for s, v in store_daily.items())
    return int(total_profit), int(total_units), int(total_hr)


def dominates(a_profit: int, a_hr: int, b_profit: int, b_hr: int) -> bool:
    # Maximizamos lucro e minimizamos RH.
    return (a_profit >= b_profit and a_hr <= b_hr) and (a_profit > b_profit or a_hr < b_hr)


class NSGA2Optimizer:
    def __init__(self, groups: list[Group], generations: int = DEFAULT_SA_ITERATIONS, population_size: int = DEFAULT_NSGA2_POP_SIZE):
        self.groups = groups
        self.generations = generations
        self.population_size = max(10, population_size)
        self.best_solution = None
        self.convergence = []

    def _fast_non_dominated_sort(self, population: list[Solution], fitness: dict[int, tuple[int, int]]) -> list[list[int]]:
        domination_count = {i: 0 for i in range(len(population))}
        dominates_set = {i: [] for i in range(len(population))}
        fronts: list[list[int]] = [[]]

        for i in range(len(population)):
            p_profit, p_hr = fitness[i]
            for j in range(len(population)):
                if i == j:
                    continue
                q_profit, q_hr = fitness[j]
                if dominates(p_profit, p_hr, q_profit, q_hr):
                    dominates_set[i].append(j)
                elif dominates(q_profit, q_hr, p_profit, p_hr):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                fronts[0].append(i)

        k = 0
        while k < len(fronts) and fronts[k]:
            next_front: list[int] = []
            for i in fronts[k]:
                for j in dominates_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            k += 1

        return fronts

    def _crowding_distance(self, front: list[int], fitness: dict[int, tuple[int, int]]) -> dict[int, float]:
        if not front:
            return {}
        if len(front) <= 2:
            return {idx: float("inf") for idx in front}

        distance = {idx: 0.0 for idx in front}
        # Objective 1: lucro (max)
        by_profit = sorted(front, key=lambda i: fitness[i][0])
        distance[by_profit[0]] = float("inf")
        distance[by_profit[-1]] = float("inf")
        p_min, p_max = fitness[by_profit[0]][0], fitness[by_profit[-1]][0]
        p_range = max(1.0, float(p_max - p_min))
        for k in range(1, len(by_profit) - 1):
            prev_p = fitness[by_profit[k - 1]][0]
            next_p = fitness[by_profit[k + 1]][0]
            distance[by_profit[k]] += (next_p - prev_p) / p_range

        # Objective 2: RH (min)
        by_hr = sorted(front, key=lambda i: fitness[i][1])
        distance[by_hr[0]] = float("inf")
        distance[by_hr[-1]] = float("inf")
        h_min, h_max = fitness[by_hr[0]][1], fitness[by_hr[-1]][1]
        h_range = max(1.0, float(h_max - h_min))
        for k in range(1, len(by_hr) - 1):
            prev_h = fitness[by_hr[k - 1]][1]
            next_h = fitness[by_hr[k + 1]][1]
            distance[by_hr[k]] += (next_h - prev_h) / h_range

        return distance

    def _binary_tournament(self, population: list[Solution], rank: dict[int, int], crowd: dict[int, float]) -> Solution:
        i, j = random.sample(range(len(population)), 2)
        if rank[i] < rank[j]:
            return population[i]
        if rank[j] < rank[i]:
            return population[j]
        return population[i] if crowd.get(i, 0.0) >= crowd.get(j, 0.0) else population[j]

    def _crossover(self, parent_a: Solution, parent_b: Solution) -> Solution:
        child = parent_a.copy()
        for block in range(len(self.groups)):
            if random.random() < 0.5:
                base = block * 3
                child.values[base:base + 3] = parent_b.values[base:base + 3]
        return child

    def optimize(self) -> Solution:
        # O3 herda o cap de O2: inicializa com soluções feasible.
        population = [generate_random_feasible_solution(self.groups) for _ in range(self.population_size)]

        for generation in range(self.generations):
            fit = {i: evaluate_profit_units_hr(sol, self.groups)[::2] for i, sol in enumerate(population)}
            # fit[i] = (profit, hr)
            fronts = self._fast_non_dominated_sort(population, fit)
            rank: dict[int, int] = {}
            crowd: dict[int, float] = {}
            for f_idx, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = f_idx
                crowd.update(self._crowding_distance(front, fit))

            offspring: list[Solution] = []
            while len(offspring) < self.population_size:
                p1 = self._binary_tournament(population, rank, crowd)
                p2 = self._binary_tournament(population, rank, crowd)
                child = self._crossover(p1, p2)
                if random.random() < 0.7:
                    child = generate_neighbor_on_triplet(child, self.groups)
                # O3 tem cap de unidades: repair o offspring para manter feasibility.
                if is_solution_invalid(child, self.groups):
                    child = repair_solution(child, self.groups)
                offspring.append(child)

            combined = population + offspring
            fit_combined = {i: evaluate_profit_units_hr(sol, self.groups)[::2] for i, sol in enumerate(combined)}
            fronts_combined = self._fast_non_dominated_sort(combined, fit_combined)

            new_population: list[Solution] = []
            for front in fronts_combined:
                if len(new_population) + len(front) <= self.population_size:
                    new_population.extend([combined[i] for i in front])
                else:
                    crowd_front = self._crowding_distance(front, fit_combined)
                    ordered = sorted(front, key=lambda i: crowd_front.get(i, 0.0), reverse=True)
                    slots = self.population_size - len(new_population)
                    new_population.extend([combined[i] for i in ordered[:slots]])
                    break

            population = new_population
            best_profit = max(evaluate_profit_units_hr(sol, self.groups)[0] for sol in population)
            self.convergence.append((generation, float(best_profit)))

        # Seleção final para resumo: maior lucro; desempate por menor RH.
        best = max(population, key=lambda s: (evaluate_profit_units_hr(s, self.groups)[0], -evaluate_profit_units_hr(s, self.groups)[2]))
        best_profit, best_units, best_hr = evaluate_profit_units_hr(best, self.groups)
        best.fitness_o1 = float(best_profit)
        best.total_units = int(best_units)
        best.total_hr = int(best_hr)
        self.best_solution = best
        return self.best_solution


# ============================================================================
# ALGORITMO GENÉTICO (O1, O2, O3_WEIGHTED)
# ============================================================================


class GeneticAlgorithmOptimizer:
    """Algoritmo Genético com representação real (vetor de 84 valores).

    Para equivalência de avaliações com os outros métodos (1000 iterações):
        gerações = total_evals // pop_size   (ex: 1000 // 40 = 25)

    Operadores:
      - Seleção: torneio de tamanho k (default 3)
      - Cruzamento: uniforme por bloco de triplet (PR, X, J)
      - Mutação: perturbação multiplicativa (igual ao generate_neighbor_on_triplet)
      - Elitismo: o melhor indivíduo é sempre preservado
    """

    def __init__(
        self,
        groups: list[Group],
        objective: str = "o1",
        omega: float = DEFAULT_OMEGA,
        constraint_mode: str = "repair",
        total_evals: int = DEFAULT_GA_TOTAL_EVALS,
        pop_size: int = DEFAULT_GA_POP_SIZE,
        crossover_rate: float = DEFAULT_GA_CROSSOVER_RATE,
        mutation_rate: float = DEFAULT_GA_MUTATION_RATE,
        tournament_k: int = DEFAULT_GA_TOURNAMENT_K,
    ):
        self.groups = groups
        self.objective = objective
        self.omega = omega
        self.constraint_mode = constraint_mode
        self.pop_size = max(4, pop_size)
        self.generations = max(1, total_evals // self.pop_size)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = min(tournament_k, self.pop_size)
        self.best_solution: Solution | None = None
        self.convergence: list[tuple[int, float]] = []

    def _init_population(self) -> list[Solution]:
        has_cap = self.objective in ("o2", "o3_weighted")
        pop: list[Solution] = []
        for _ in range(self.pop_size):
            sol = generate_random_solution(self.groups)
            if has_cap and self.constraint_mode in ("repair", "penalty"):
                sol = repair_solution(sol, self.groups)
            pop.append(sol)
        return pop

    def _evaluate(self, sol: Solution) -> tuple[float, int, int]:
        fitness, units, hr, _ = evaluate_solution(
            sol, self.groups, self.objective, self.omega, self.constraint_mode
        )
        sol.fitness_o1 = fitness
        sol.total_units = units
        sol.total_hr = hr
        return fitness, units, hr

    def _tournament(self, population: list[Solution]) -> Solution:
        """Seleção por torneio: escolhe k candidatos e retorna o melhor."""
        contestants = random.sample(population, self.tournament_k)
        return max(contestants, key=lambda s: s.fitness_o1)

    def _crossover(self, parent_a: Solution, parent_b: Solution) -> tuple[Solution, Solution]:
        """Cruzamento uniforme por bloco de triplet (PR, X, J)."""
        if random.random() > self.crossover_rate:
            return parent_a.copy(), parent_b.copy()
        child_a = parent_a.copy()
        child_b = parent_b.copy()
        for block in range(len(self.groups)):
            if random.random() < 0.5:
                base = block * 3
                child_a.values[base:base + 3], child_b.values[base:base + 3] = (
                    parent_b.values[base:base + 3][:],
                    parent_a.values[base:base + 3][:],
                )
        return child_a, child_b

    def _mutate(self, sol: Solution) -> Solution:
        """Mutação: perturbação multiplicativa por bloco (igual ao Hill Climbing)."""
        if random.random() > self.mutation_rate:
            return sol
        return generate_neighbor_on_triplet(sol, self.groups)

    def optimize(self) -> Solution:
        has_cap = self.objective in ("o2", "o3_weighted")

        # Inicializar e avaliar população
        population = self._init_population()
        for sol in population:
            self._evaluate(sol)

        # Melhor de sempre (elitismo)
        self.best_solution = max(population, key=lambda s: s.fitness_o1).copy()

        eval_count = self.pop_size
        generation = 0

        for generation in range(self.generations):
            offspring: list[Solution] = []

            # Elitismo: preservar o melhor
            offspring.append(self.best_solution.copy())

            while len(offspring) < self.pop_size:
                # Seleção
                pa = self._tournament(population)
                pb = self._tournament(population)

                # Cruzamento
                ca, cb = self._crossover(pa, pb)

                # Mutação
                ca = self._mutate(ca)
                cb = self._mutate(cb)

                # Repair (se aplicável)
                for child in (ca, cb):
                    if has_cap and self.constraint_mode == "repair" and is_solution_invalid(child, self.groups):
                        child = repair_solution(child, self.groups)
                    self._evaluate(child)
                    offspring.append(child)
                    if len(offspring) >= self.pop_size:
                        break

            population = offspring
            eval_count += len(population)

            gen_best = max(population, key=lambda s: s.fitness_o1)
            if gen_best.fitness_o1 > self.best_solution.fitness_o1:
                self.best_solution = gen_best.copy()

            # Convergência registada como avaliação acumulada (equivalente a iteração)
            self.convergence.append((eval_count, self.best_solution.fitness_o1))

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
        "genetic_algorithm": "Genetic Algorithm",
        "nsga_ii": "NSGA-II",
    }
    method_colors = {
        "monte_carlo": "tab:blue",
        "hill_climbing": "tab:green",
        "simulated_annealing": "tab:red",
        "genetic_algorithm": "tab:purple",
        "nsga_ii": "tab:orange",
    }

    # Avoid splitting by underscores because method names (e.g., monte_carlo)
    # also contain underscores.
    known_objectives = ["o1", "o2", "o3_weighted", "o3_pareto"]
    objectives = [
        obj for obj in known_objectives if any(key.startswith(f"{obj}_") for key in convergence_data.keys())
    ]
    for objective in objectives:
        # Repair/none e penalty têm escalas incompatíveis — usar 2 subplots lado a lado.
        modes_groups = [("repair", "none"), ("penalty",)]
        has_objective_data = False
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Convergência - {objective.upper()}", fontsize=14)

        for ax, modes in zip(axes, modes_groups):
            has_series = False
            for method in ["monte_carlo", "hill_climbing", "simulated_annealing", "genetic_algorithm", "nsga_ii"]:
                for constraint_mode in modes:
                    key = f"{objective}_{method}_{constraint_mode}"
                    if key not in convergence_data or not convergence_data[key]:
                        continue

                    iterations = [point[0] for point in convergence_data[key]]
                    scores = [point[1] for point in convergence_data[key]]
                    linestyle = "-"
                    ax.plot(
                        iterations,
                        scores,
                        label=f"{method_labels[method]}",
                        color=method_colors[method],
                        linestyle=linestyle,
                        linewidth=2,
                    )
                    has_series = True
                    has_objective_data = True

            mode_label = "Repair / None" if "repair" in modes or "none" in modes else "Penalty"
            ax.set_title(mode_label)
            ax.set_xlabel("Iteração")
            ax.set_ylabel("Best Fitness")
            ax.grid(True, alpha=0.3)
            if has_series:
                ax.legend()
            else:
                ax.set_visible(False)

        if not has_objective_data:
            plt.close()
            continue

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
        "--nsga-pop-size",
        type=int,
        default=DEFAULT_NSGA2_POP_SIZE,
        help="Tamanho da populacao do NSGA-II no O3 Pareto.",
    )
    parser.add_argument(
        "--ga-pop-size",
        type=int,
        default=DEFAULT_GA_POP_SIZE,
        help="Tamanho da populacao do GA (default: 40).",
    )
    parser.add_argument(
        "--ga-total-evals",
        type=int,
        default=DEFAULT_GA_TOTAL_EVALS,
        help="Total de avaliacoes do GA; geracoes = ga-total-evals / ga-pop-size.",
    )
    parser.add_argument(
        "--ga-crossover-rate",
        type=float,
        default=DEFAULT_GA_CROSSOVER_RATE,
        help="Taxa de cruzamento do GA (default: 0.8).",
    )
    parser.add_argument(
        "--ga-mutation-rate",
        type=float,
        default=DEFAULT_GA_MUTATION_RATE,
        help="Taxa de mutacao do GA (default: 0.15).",
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

    default_constraint_modes = [args.constraint_mode] if args.constraint_mode != "both" else ["repair", "penalty"]
    base_methods = ["monte_carlo", "hill_climbing", "simulated_annealing", "genetic_algorithm"]
    plan_results: list[dict] = []
    run_rows: list[dict] = []
    scenario_rows: list[dict] = []
    convergence_data = {}

    print("=" * 70)
    print("OTIMIZAÇÃO COM METAHEURÍSTICAS")
    print("=" * 70)

    # Executar cenários com replicações por método/objetivo/modo.
    for objective in objectives:
        objective_constraint_modes = default_constraint_modes if objective in ("o2", "o3_weighted") else ["none"]
        objective_methods = ["nsga_ii"] if objective == "o3_pareto" else base_methods
        for constraint_mode in objective_constraint_modes:
            for method in objective_methods:
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
                    elif method == "simulated_annealing":
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
                    elif method == "genetic_algorithm":
                        optimizer = GeneticAlgorithmOptimizer(
                            groups,
                            objective,
                            args.omega,
                            constraint_mode,
                            total_evals=args.ga_total_evals,
                            pop_size=args.ga_pop_size,
                            crossover_rate=args.ga_crossover_rate,
                            mutation_rate=args.ga_mutation_rate,
                        )
                    else:
                        optimizer = NSGA2Optimizer(groups, generations=args.sa_iterations, population_size=args.nsga_pop_size)

                    solution = optimizer.optimize()
                    plan_df = solution_to_plan_df(objective.upper(), method.upper(), groups, solution)
                    total_profit = int(plan_df["Daily_Profit"].sum())
                    # Feasibility só se aplica ao O2; O1 e O3 são sempre válidos.
                    feasible = (solution.total_units <= UNITS_CAP) if objective in ("o2", "o3_weighted", "o3_pareto") else True

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

                feasible_runs = [r for r in per_run_results if r["feasible"] == 1]
                # For constrained objectives, prefer best feasible run; fallback keeps robustness.
                best_run = max(feasible_runs, key=lambda r: r["fitness"]) if feasible_runs else max(per_run_results, key=lambda r: r["fitness"])

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
    # Para O2, garante-se que só planos feasible (repair/penalty) são considerados.
    # Para O1 e O3, todas as soluções são válidas (sem restrição de cap).
    best_plan_frames = []
    print("\n" + "=" * 70)
    print("MELHORES PLANOS")
    print("=" * 70)
    for objective in sorted({r["Objective"] for r in plan_results}):
        candidates = [r for r in plan_results if r["Objective"] == objective and r["Feasible"] == 1]
        if not candidates:
            print(f"  {objective}: sem solucoes validas.")
            continue
        best_result = max(candidates, key=lambda r: r["Fitness"])
        best_plan_frames.append(best_result["plan_df"])
        print(
            f"  {objective}: Metodo={best_result['Method']}"
            f" | Modo={best_result['ConstraintMode']}"
            f" | Fitness={best_result['Fitness']:.4g}"
            f" | Unidades={best_result['Total_Units']}"
            f" | Feasible={best_result['Feasible']}"
        )

    best_plans_file = f"{output_prefix}_best_plans.csv"
    if best_plan_frames:
        best_plans_df = pd.concat(best_plan_frames, ignore_index=True)
        best_plans_df.to_csv(best_plans_file, index=False)
        print(f"\nBest plans saved: {best_plans_file}")
    else:
        print("\nBest plans not saved: nenhuma solução feasible encontrada em nenhum objetivo.")

    # Guardar visualizações
    plot_files = save_convergence_plots(convergence_data, output_prefix)
    dashboard_file = save_summary_visualization(summary, output_prefix)
    print("\nVisualization files:")
    for file_name in plot_files:
        print(f" - {file_name}")
    print(f" - {dashboard_file}")


if __name__ == "__main__":
    main()
