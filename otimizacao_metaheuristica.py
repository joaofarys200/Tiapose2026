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
ZERO_PERTURB_STD = 0.05

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
    total_profit: int = 0
    feasible: bool = False

    def copy(self):
        return Solution(
            values=self.values.copy(),
            fitness_o1=self.fitness_o1,
            fitness_o2=self.fitness_o2,
            fitness_o3=self.fitness_o3,
            total_units=self.total_units,
            total_hr=self.total_hr,
            total_profit=self.total_profit,
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
    """Reduz localmente os blocos menos eficientes até satisfazer o cap.

    Em vez de encolher todos os X/J por 0.95, remove capacidade apenas onde a
    perda marginal de lucro por unidade reduzida é menor.
    """
    repaired = sol.copy()

    while True:
        decisions = solution_vector_to_decisions(repaired.values, groups)
        options = decisions_to_options(decisions, groups)
        total_units = sum(opt.units_total for opt in options.values())
        if total_units <= units_cap:
            break

        overflow = total_units - units_cap
        best_move: tuple[float, int, int, int, int] | None = None

        for group in groups:
            base_idx = group.idx * 3
            current_pr = float(repaired.values[base_idx])
            current_x = int(repaired.values[base_idx + 1])
            current_j = int(repaired.values[base_idx + 2])
            current_opt = options[group.idx]

            if current_x > 0:
                candidate_opt = build_day_option(group, current_pr, current_x - 1, current_j)
                units_drop = current_opt.units_total - candidate_opt.units_total
                profit_drop = current_opt.daily_profit - candidate_opt.daily_profit
                if units_drop > 0:
                    score = profit_drop / units_drop
                    move = (score, -units_drop, base_idx + 1, current_x - 1, current_j)
                    best_move = move if best_move is None or move < best_move else best_move

            if current_j > 0:
                candidate_opt = build_day_option(group, current_pr, current_x, current_j - 1)
                units_drop = current_opt.units_total - candidate_opt.units_total
                profit_drop = current_opt.daily_profit - candidate_opt.daily_profit
                if units_drop > 0:
                    score = profit_drop / units_drop
                    move = (score, -units_drop, base_idx + 2, current_x, current_j - 1)
                    best_move = move if best_move is None or move < best_move else best_move

        if best_move is None:
            break

        _, _, move_idx, new_x, new_j = best_move
        base_idx = (move_idx // 3) * 3
        repaired.values[base_idx + 1] = new_x
        repaired.values[base_idx + 2] = new_j

        # Se o excesso for pequeno, permite sair logo após uma redução suficiente.
        if overflow > 0:
            decisions = solution_vector_to_decisions(repaired.values, groups)
            options = decisions_to_options(decisions, groups)
            total_units = sum(opt.units_total for opt in options.values())
            if total_units <= units_cap:
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

    # Professor: se x>0, usar perturbação multiplicativa x*p com p~N(1, 0.05);
    # se x==0, usar x+abs(p-1).
    # Para PR, o ramo x==0 funciona naturalmente em contínuo e depois é discretizado a 1%.
    cur_pr_int = int(round(discretize_pr(float(neighbor.values[base_idx])) * 100))
    cur_pr = cur_pr_int / 100.0
    if cur_pr > 0.0:
        pr_scale = random.gauss(1.0, ZERO_PERTURB_STD)
        new_pr = cur_pr * pr_scale
    else:
        new_pr = cur_pr + abs(random.gauss(1.0, ZERO_PERTURB_STD) - 1.0)
    new_pr_int = int(round(discretize_pr(new_pr) * 100))
    new_pr_int = max(PR_MIN_INT, min(PR_MAX_INT, new_pr_int))
    neighbor.values[base_idx] = pr_from_int(new_pr_int)

    # X e J: mesma lógica do quadro do professor.
    # Se x>0, usar x*p com p~N(1, 0.05); se x==0, usar x+abs(y-1), y~N(1, 0.05).
    # Como X e J são inteiros, o ramo x==0 arredonda por excesso para garantir saída de 0.
    cur_x = int(neighbor.values[base_idx + 1])
    cur_j = int(neighbor.values[base_idx + 2])
    if cur_x > 0:
        x_scale = random.gauss(1.0, ZERO_PERTURB_STD)
        new_x = int(round(cur_x * x_scale))
    else:
        new_x = int(math.ceil(cur_x + abs(random.gauss(1.0, ZERO_PERTURB_STD) - 1.0)))

    if cur_j > 0:
        j_scale = random.gauss(1.0, ZERO_PERTURB_STD)
        new_j = int(round(cur_j * j_scale))
    else:
        new_j = int(math.ceil(cur_j + abs(random.gauss(1.0, ZERO_PERTURB_STD) - 1.0)))

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
    Avalia S e devolve (fitness, total_units, total_hr, feasible, total_profit).
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

    # O2: penalização finita para manter gradiente útil perto do cap.
    if objective == "o2" and not feasible and constraint_mode == "penalty":
        overflow = float(total_units - UNITS_CAP)
        penalty = 2.0 * overflow + 0.01 * (overflow ** 2)
        fitness = float(total_profit) - penalty
        return fitness, total_units, total_hr, feasible, int(total_profit)

    # O3 mantém death penalty no modo penalty.
    if has_cap and not feasible and constraint_mode == "penalty":
        return -math.inf, total_units, total_hr, feasible, 0

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
        profit_ref = 500.0    # lucro líquido máximo com cap (gross ~3800 - Ws_total 2990 ≈ 825; arredondado a 500 para margem)
        hr_ref = 100.0        # HR máximo com cap de 10000 unidades (~93-97 observado; sem cap seria ~450)
        fitness = (omega * (float(total_profit) / profit_ref)
                   - (1.0 - omega) * (float(total_hr) / hr_ref))
    elif objective == "o3_pareto":
        # O3 Pareto: NSGA-II usa evaluate_profit_units_hr diretamente; retorna lucro bruto.
        fitness = float(total_profit)
    else:
        fitness = float(total_profit)

    return fitness, total_units, total_hr, feasible, int(total_profit)


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


def _best_o2_addition_move(
    sol: Solution,
    groups: list[Group],
    remaining_units: int,
) -> tuple[float, int, int, int, int] | None:
    decisions = solution_vector_to_decisions(sol.values, groups)
    options = decisions_to_options(decisions, groups)
    candidate_moves: list[tuple[float, int, int, int, int]] = []

    for group in groups:
        base_idx = group.idx * 3
        pr = float(sol.values[base_idx])
        x = int(sol.values[base_idx + 1])
        j = int(sol.values[base_idx + 2])
        current_opt = options[group.idx]

        x_upper_bound = math.ceil(group.customers / 7) if group.customers > 0 else 0
        j_upper_bound = math.ceil(group.customers / 6) if group.customers > 0 else 0

        if x < x_upper_bound:
            candidate_opt = build_day_option(group, pr, x + 1, j)
            units_gain = candidate_opt.units_total - current_opt.units_total
            profit_gain = candidate_opt.daily_profit - current_opt.daily_profit
            if 0 < units_gain <= remaining_units and profit_gain > 0:
                candidate_moves.append((profit_gain / units_gain, profit_gain, -units_gain, base_idx, x + 1, j))

        if j < j_upper_bound:
            candidate_opt = build_day_option(group, pr, x, j + 1)
            units_gain = candidate_opt.units_total - current_opt.units_total
            profit_gain = candidate_opt.daily_profit - current_opt.daily_profit
            if 0 < units_gain <= remaining_units and profit_gain > 0:
                candidate_moves.append((profit_gain / units_gain, profit_gain, -units_gain, base_idx, x, j + 1))

    if not candidate_moves:
        return None

    candidate_moves.sort(reverse=True)
    top_k = candidate_moves[:min(5, len(candidate_moves))]
    return random.choice(top_k)


def _fill_o2_slack(
    sol: Solution,
    groups: list[Group],
    units_cap: int = UNITS_CAP,
    max_steps: int | None = None,
) -> Solution:
    filled = sol.copy()
    steps = max_steps if max_steps is not None else max(8, len(groups) * 2)

    for _ in range(steps):
        decisions = solution_vector_to_decisions(filled.values, groups)
        options = decisions_to_options(decisions, groups)
        total_units = sum(opt.units_total for opt in options.values())
        remaining_units = units_cap - total_units
        if remaining_units <= 0:
            break

        move = _best_o2_addition_move(filled, groups, remaining_units)
        if move is None:
            break

        _, _, _, base_idx, new_x, new_j = move
        filled.values[base_idx + 1] = new_x
        filled.values[base_idx + 2] = new_j

    return filled


def generate_o2_seed_solution(groups: list[Group], units_cap: int = UNITS_CAP) -> Solution:
    """Gera solução do O2 já viável e tipicamente próxima do cap."""
    if random.random() < 0.35:
        seed = generate_random_feasible_solution(groups, units_cap)
    else:
        seed = repair_solution(generate_random_solution(groups), groups, units_cap)
    return _fill_o2_slack(seed, groups, units_cap)


def generate_neighbor_o2(
    solution: Solution,
    groups: list[Group],
    constraint_mode: str = "repair",
    units_cap: int = UNITS_CAP,
) -> Solution:
    """Gera vizinho do O2 e volta a empurrá-lo para a fronteira viável útil."""
    neighbor = solution.copy()
    n_moves = random.randint(1, 3)
    for _ in range(n_moves):
        neighbor = generate_neighbor_on_triplet(neighbor, groups)

    if constraint_mode == "repair":
        if is_solution_invalid(neighbor, groups, units_cap):
            neighbor = repair_solution(neighbor, groups, units_cap)
        neighbor = _fill_o2_slack(neighbor, groups, units_cap, max_steps=max(3, len(groups) // 2))
    elif not is_solution_invalid(neighbor, groups, units_cap):
        neighbor = _fill_o2_slack(neighbor, groups, units_cap, max_steps=3)

    return neighbor


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

    def _random_candidate(self) -> Solution:
        """Monte Carlo deve amostrar o espaço alvo, não colapsar tudo no mesmo repair.

        Em objetivos com cap e modo repair, gerar soluções já viáveis preserva
        diversidade; caso contrário, manter a amostragem original.
        """
        if self.objective == "o2" and self.constraint_mode == "repair":
            return generate_random_feasible_solution(self.groups)
        if self.objective == "o2":
            return generate_o2_seed_solution(self.groups)
        if self.objective in ("o2", "o3_weighted") and self.constraint_mode == "repair":
            return generate_random_feasible_solution(self.groups)
        return generate_random_solution(self.groups)

    def optimize(self) -> Solution:
        # Monte Carlo: pure random sampling (mcsearch do professor)
        # Gera N soluções independentes e mantém a melhor.
        has_cap = self.objective in ("o2", "o3_weighted")
        self.best_solution = self._random_candidate()
        if has_cap and self.constraint_mode in ("repair", "penalty") and is_solution_invalid(self.best_solution, self.groups):
            self.best_solution = repair_solution(self.best_solution, self.groups)
        fitness, units, hr, _, profit = evaluate_solution(
            self.best_solution, self.groups, self.objective, self.omega, self.constraint_mode
        )
        self.best_solution.fitness_o1 = fitness
        self.best_solution.total_units = units
        self.best_solution.total_hr = hr
        self.best_solution.total_profit = profit

        for iteration in range(self.iterations):
            candidate = self._random_candidate()
            if has_cap and self.constraint_mode == "repair" and is_solution_invalid(candidate, self.groups):
                candidate = repair_solution(candidate, self.groups)

            fitness, units, hr, _, profit = evaluate_solution(
                candidate, self.groups, self.objective, self.omega, self.constraint_mode
            )

            if fitness > self.best_solution.fitness_o1:
                self.best_solution = candidate.copy()
                self.best_solution.fitness_o1 = fitness
                self.best_solution.total_units = units
                self.best_solution.total_hr = hr
                self.best_solution.total_profit = profit

            self.convergence.append((iteration, self.best_solution.total_profit, self.best_solution.total_hr))

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
        if self.objective == "o2":
            return generate_neighbor_o2(solution, self.groups, self.constraint_mode)
        return generate_neighbor_on_triplet(solution, self.groups)

    def optimize(self) -> Solution:
        has_cap = self.objective in ("o2", "o3_weighted")
        self.best_solution = generate_o2_seed_solution(self.groups) if self.objective == "o2" else generate_random_solution(self.groups)
        if has_cap and self.constraint_mode in ("repair", "penalty"):
            self.best_solution = repair_solution(self.best_solution, self.groups)
        fitness, units, hr, _, profit = evaluate_solution(
            self.best_solution, self.groups, self.objective, self.omega, self.constraint_mode
        )
        self.best_solution.fitness_o1 = fitness
        self.best_solution.total_units = units
        self.best_solution.total_hr = hr
        self.best_solution.total_profit = profit

        for iteration in range(self.iterations):
            neighbor = self.get_neighbor(self.best_solution)
            if has_cap and self.constraint_mode == "repair":
                neighbor = repair_solution(neighbor, self.groups)
            fitness, units, hr, _, profit = evaluate_solution(
                neighbor, self.groups, self.objective, self.omega, self.constraint_mode
            )
            
            if fitness > self.best_solution.fitness_o1:
                self.best_solution = neighbor.copy()
                self.best_solution.fitness_o1 = fitness
                self.best_solution.total_units = units
                self.best_solution.total_hr = hr
                self.best_solution.total_profit = profit

            self.convergence.append((iteration, self.best_solution.total_profit, self.best_solution.total_hr))

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
        if self.objective == "o2":
            return generate_neighbor_o2(solution, self.groups, self.constraint_mode)
        return generate_neighbor_on_triplet(solution, self.groups)

    def optimize(self) -> Solution:
        has_cap = self.objective in ("o2", "o3_weighted")
        if self.objective == "o2" and self.constraint_mode == "repair":
            current = generate_random_feasible_solution(self.groups)
        elif self.objective == "o2":
            current = generate_o2_seed_solution(self.groups)
        else:
            current = generate_random_solution(self.groups)

        if has_cap and self.constraint_mode in ("repair", "penalty") and is_solution_invalid(current, self.groups):
            current = repair_solution(current, self.groups)
        fitness, units, hr, _, profit = evaluate_solution(
            current, self.groups, self.objective, self.omega, self.constraint_mode
        )
        current.fitness_o1 = fitness
        current.total_units = units
        current.total_hr = hr
        current.total_profit = profit

        self.best_solution = current.copy()

        temperature = self.t_initial
        for iteration in range(self.iterations):
            neighbor = self.get_neighbor(current)
            if has_cap and self.constraint_mode == "repair":
                neighbor = repair_solution(neighbor, self.groups)
            fitness_new, units_new, hr_new, _, profit_new = evaluate_solution(
                neighbor, self.groups, self.objective, self.omega, self.constraint_mode
            )
            neighbor.fitness_o1 = fitness_new
            neighbor.total_units = units_new
            neighbor.total_hr = hr_new
            neighbor.total_profit = profit_new

            # Critério de Metropolis
            delta = fitness_new - current.fitness_o1
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-8)):
                current = neighbor

            if current.fitness_o1 > self.best_solution.fitness_o1:
                self.best_solution = current.copy()

            # Resfriamento
            temperature *= self.cooling_rate
            temperature = max(temperature, self.t_final)

            self.convergence.append((iteration, self.best_solution.total_profit, self.best_solution.total_hr))

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
        self.pareto_front_points: list[tuple[int, int]] = []  # (profit, hr) da fronteira final
        self.pareto_front_history: list[tuple[int, list[tuple[int, int]]]] = []  # (geração, pontos)

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

    def _seed_solution(self) -> Solution:
        """Mistura seeds lucro-orientadas do O2 com feasible aleatório para diversidade."""
        if random.random() < 0.7:
            return generate_o2_seed_solution(self.groups)
        return generate_random_feasible_solution(self.groups)

    def _replacement_solution(self) -> Solution:
        """Reposição de filho inválido sem perder totalmente a diversidade da fronteira."""
        if random.random() < 0.5:
            return generate_o2_seed_solution(self.groups)
        return generate_random_feasible_solution(self.groups)

    def _mutate(self, solution: Solution) -> Solution:
        """NSGA-II precisa de exploração mais ampla que uma única vizinhança local.

        Aplicar 1-3 mutações por triplet aproxima melhor o comportamento de um
        operador real-valued do exemplo do professor, preservando a representação.
        """
        mutated = solution.copy()
        n_moves = random.randint(1, min(3, len(self.groups)))
        for _ in range(n_moves):
            if random.random() < 0.75:
                mutated = generate_neighbor_o2(mutated, self.groups, constraint_mode="repair")
            else:
                mutated = generate_neighbor_on_triplet(mutated, self.groups)
                if is_solution_invalid(mutated, self.groups):
                    mutated = repair_solution(mutated, self.groups)
        return mutated

    def optimize(self) -> Solution:
        # O3 herda o cap de O2: inicializa com soluções feasible.
        population = [self._seed_solution() for _ in range(self.population_size)]

        snap_interval = max(1, self.generations // 10)
        for generation in range(self.generations):
            fit = {i: evaluate_profit_units_hr(sol, self.groups)[::2] for i, sol in enumerate(population)}
            # fit[i] = (profit, hr)
            fronts = self._fast_non_dominated_sort(population, fit)
            # Gravar snapshot da fronteira para visualização da evolução.
            if generation % snap_interval == 0 or generation == self.generations - 1:
                self.pareto_front_history.append((generation, [fit[i] for i in fronts[0]]))
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
                    child = self._mutate(child)
                # O3 tem cap de unidades. Em vez de colapsar o filho por repair
                # global, reintroduzir diversidade com uma solução viável nova.
                if is_solution_invalid(child, self.groups):
                    child = self._replacement_solution()
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
            best_phr = max(
                (evaluate_profit_units_hr(sol, self.groups) for sol in population),
                key=lambda x: x[0],
            )
            self.convergence.append((generation, int(best_phr[0]), int(best_phr[2])))

        # Seleção final para resumo: maior lucro; desempate por menor RH.
        best = max(population, key=lambda s: (evaluate_profit_units_hr(s, self.groups)[0], -evaluate_profit_units_hr(s, self.groups)[2]))
        best_profit, best_units, best_hr = evaluate_profit_units_hr(best, self.groups)
        best.fitness_o1 = float(best_profit)
        best.total_units = int(best_units)
        best.total_hr = int(best_hr)
        best.total_profit = int(best_profit)
        self.best_solution = best

        # Extrair fronteira de Pareto final para visualização 2D (profit, hr).
        fit_final = {
            i: (evaluate_profit_units_hr(sol, self.groups)[0], evaluate_profit_units_hr(sol, self.groups)[2])
            for i, sol in enumerate(population)
        }
        fronts_final = self._fast_non_dominated_sort(population, fit_final)
        if fronts_final:
            self.pareto_front_points = [fit_final[i] for i in fronts_final[0]]

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
            sol = generate_o2_seed_solution(self.groups) if self.objective == "o2" else generate_random_solution(self.groups)
            if has_cap and self.constraint_mode in ("repair", "penalty"):
                sol = repair_solution(sol, self.groups)
            pop.append(sol)
        return pop

    def _evaluate(self, sol: Solution) -> tuple[float, int, int]:
        fitness, units, hr, _, profit = evaluate_solution(
            sol, self.groups, self.objective, self.omega, self.constraint_mode
        )
        sol.fitness_o1 = fitness
        sol.total_units = units
        sol.total_hr = hr
        sol.total_profit = profit
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
        if self.objective == "o2":
            return generate_neighbor_o2(sol, self.groups, self.constraint_mode)
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
            self.convergence.append((eval_count, self.best_solution.total_profit, self.best_solution.total_hr))

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


def _pareto_sort(pts: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    """Ordena pontos (profit, hr) por RH crescente; devolve (hrs, profits)."""
    s = sorted((h, p) for p, h in pts)
    return [x[0] for x in s], [x[1] for x in s]


def _hypervolume_2d(points: list[tuple[int, int]], ref_point: tuple[int, int]) -> float:
    """Hypervolume 2D para max profit / min RH com referência (profit_min, hr_max)."""
    if not points:
        return 0.0
    nd_points = list(dict.fromkeys(_filter_non_dominated(points)))
    ordered = sorted(nd_points, key=lambda x: x[1])
    ref_profit, ref_hr = ref_point
    hv = 0.0
    prev_profit = float(ref_profit)
    for profit, hr in ordered:
        width = max(0.0, float(ref_hr - hr))
        height = max(0.0, float(profit) - prev_profit)
        hv += width * height
        prev_profit = max(prev_profit, float(profit))
    return hv


def _filter_non_dominated(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Filtra pontos não-dominados (maximizar profit, minimizar hr)."""
    nd = []
    for p1, h1 in points:
        dominated = any(
            p2 >= p1 and h2 <= h1 and (p2 > p1 or h2 < h1)
            for p2, h2 in points
        )
        if not dominated:
            nd.append((p1, h1))
    return nd


def save_pareto_front_plot(
    pareto_points: list[tuple[int, int]],
    output_prefix: str,
    pareto_history: list[tuple[int, list[tuple[int, int]]]] | None = None,
) -> str:
    """Plota a fronteira de Pareto 2D do NSGA-II: X = RH total, Y = Lucro líquido.

    Se pareto_history for fornecido, mostra a evolução por geração (cinzento claro
    → azul escuro), à semelhança do opt-5-fes1.R da aula5.
    """
    if not pareto_points:
        return ""

    pareto_points = list(dict.fromkeys(_filter_non_dominated(pareto_points)))
    if not pareto_points:
        return ""

    fig, ax = plt.subplots(figsize=(8, 6))
    ref_point = (min(p for p, _ in pareto_points) - 1, max(h for _, h in pareto_points) + 1)
    hv = _hypervolume_2d(pareto_points, ref_point)

    # --- Evolução por geração (cinzento → azul, como no R) ---
    if pareto_history:
        n_steps = len(pareto_history)
        for step_idx, (gen, pts) in enumerate(pareto_history):
            nd_hist = list(dict.fromkeys(_filter_non_dominated(pts)))
            if not nd_hist:
                continue
            t = step_idx / max(1, n_steps - 1)          # 0 → 1
            gray = 0.80 - 0.65 * t                       # claro → escuro
            alpha = 0.20 + 0.55 * t
            col = (gray, gray, gray)
            sh, sp = _pareto_sort(nd_hist)
            ax.plot(sh, sp, color=col, alpha=alpha, linewidth=1.0)
            ax.scatter(sh, sp, color=col, s=12, alpha=alpha)

    # --- Fronteira final ---
    final_hrs, final_profits = _pareto_sort(pareto_points)
    ax.scatter(final_hrs, final_profits, c="tab:blue", s=60, zorder=5,
               label="Fronteira final (não dominadas)")
    ax.plot(final_hrs, final_profits, color="tab:blue", linewidth=2.0, zorder=4)

    best_profit_pt = max(pareto_points, key=lambda p: p[0])
    min_hr_pt      = min(pareto_points, key=lambda p: p[1])
    ax.scatter([best_profit_pt[1]], [best_profit_pt[0]], c="darkgreen", s=120, zorder=6,
               label=f"Max Lucro ({best_profit_pt[0]:,} €)")
    ax.scatter([min_hr_pt[1]], [min_hr_pt[0]], c="darkred", s=120, zorder=6,
               label=f"Min RH ({min_hr_pt[1]})")

    ax.set_xlabel("RH Total (trabalhador-dias)", fontsize=11)
    ax.set_ylabel("Lucro Líquido Semanal (€)", fontsize=11)
    ax.set_title(
        f"Fronteira de Pareto — O3 (NSGA-II) | HV={hv:.0f} | pontos={len(pareto_points)}\n"
        "Maximizar Lucro  ×  Minimizar RH",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.axvline(ref_point[1], color="gray", linewidth=1, alpha=0.6)
    ax.axhline(ref_point[0], color="gray", linewidth=1, alpha=0.6)
    ax.scatter([ref_point[1]], [ref_point[0]], color="gray", s=35, zorder=3)
    plt.tight_layout()
    out_file = f"{output_prefix}_o3_pareto_front.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file


def save_weighted_sweep_pareto_plot(
    sweep_points: list[tuple[int, int]],
    output_prefix: str,
    nsga2_points: list[tuple[int, int]] | None = None,
) -> str:
    """Plota a frente de Pareto aproximada via soma ponderada (sweep de omegas),
    comparada com a fronteira do NSGA-II.
    X = RH total, Y = Lucro líquido semanal.
    """
    if not sweep_points:
        return ""

    nd_points = _filter_non_dominated(sweep_points)
    if not nd_points:
        return ""

    fig, ax = plt.subplots(figsize=(8, 6))
    all_points = list(nd_points)
    if nsga2_points:
        all_points.extend(nsga2_points)
    ref_point = (min(p for p, _ in all_points) - 1, max(h for _, h in all_points) + 1)
    hv_nsga = None
    hv_weighted = _hypervolume_2d(nd_points, ref_point)

    # Frente NSGA-II (se disponível)
    if nsga2_points:
        nd_nsga = list(dict.fromkeys(_filter_non_dominated(nsga2_points)))
        if nd_nsga:
            hv_nsga = _hypervolume_2d(nd_nsga, ref_point)
            nsga_hrs, nsga_profits = _pareto_sort(nd_nsga)
            ax.plot(
                nsga_hrs,
                nsga_profits,
                color="tab:blue",
                linewidth=2.0,
                label=f"NSGA-II (HV={hv_nsga:.0f}, n={len(nd_nsga)})",
            )
            ax.scatter(nsga_hrs, nsga_profits, c="tab:blue", s=50, zorder=5)

    # Frente por soma ponderada
    ws_hrs, ws_profits = _pareto_sort(nd_points)
    ax.plot(ws_hrs, ws_profits, color="tab:orange", linewidth=2.0,
            linestyle="--", label=f"GA ponderado (HV={hv_weighted:.0f}, n={len(nd_points)})")
    ax.scatter(ws_hrs, ws_profits, c="tab:orange", s=60, zorder=5)

    ax.set_xlabel("RH Total (trabalhador-dias)", fontsize=11)
    ax.set_ylabel("Lucro Líquido Semanal (€)", fontsize=11)
    ax.set_title(
        "Frente de Pareto — NSGA-II vs GA ponderado (sweep ω)\nMaximizar Lucro  ×  Minimizar RH",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.axvline(ref_point[1], color="gray", linewidth=1, alpha=0.6)
    ax.axhline(ref_point[0], color="gray", linewidth=1, alpha=0.6)
    ax.scatter([ref_point[1]], [ref_point[0]], color="gray", s=35, zorder=3)
    plt.tight_layout()
    out_file = f"{output_prefix}_o3_weighted_sweep_front.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file


def save_convergence_plots(convergence_data: dict, output_prefix: str) -> list[str]:
    """Guarda curvas de convergência por objetivo.

    Cada ponto é um 3-tuplo (step, profit, hr).
    - O1/O2: X = step (iteração), Y = lucro  (melhor encontrado até ao momento)
    - O3_WEIGHTED/O3_PARETO: X = RH, Y = lucro  (espaço bi-objetivo)
    """
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

    known_objectives = ["o1", "o2", "o3_weighted", "o3_pareto"]
    objectives = [
        obj for obj in known_objectives if any(key.startswith(f"{obj}_") for key in convergence_data.keys())
    ]
    for objective in objectives:
        use_rh_axes = objective in ("o3_pareto",)
        modes_groups = [("repair", "none"), ("penalty",)]
        has_objective_data = False
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        title_suffix = " (Lucro vs RH)" if use_rh_axes else " (Lucro por Iteração)"
        fig.suptitle(f"Convergência - {objective.upper()}{title_suffix}", fontsize=14)

        for ax, modes in zip(axes, modes_groups):
            has_series = False
            for method in ["monte_carlo", "hill_climbing", "simulated_annealing", "genetic_algorithm", "nsga_ii"]:
                for constraint_mode in modes:
                    key = f"{objective}_{method}_{constraint_mode}"
                    if key not in convergence_data or not convergence_data[key]:
                        continue

                    raw_points = convergence_data[key]  # [(step, profit, hr), ...]

                    if use_rh_axes:
                        # Remover pontos consecutivos duplicados (profit, hr) para clareza.
                        deduped = [raw_points[0]]
                        for p in raw_points[1:]:
                            if (p[1], p[2]) != (deduped[-1][1], deduped[-1][2]):
                                deduped.append(p)
                        explored_points = [(int(p[1]), int(p[2])) for p in deduped]
                        frontier_points = list(dict.fromkeys(_filter_non_dominated(explored_points)))
                        frontier_hr, frontier_profit = _pareto_sort(frontier_points)

                        ax.scatter(
                            [p[1] for p in explored_points],
                            [p[0] for p in explored_points],
                            color=method_colors[method],
                            s=22,
                            alpha=0.35,
                        )
                        ax.plot(
                            frontier_hr,
                            frontier_profit,
                            label=method_labels[method],
                            color=method_colors[method],
                            linewidth=2,
                            marker="o",
                            markersize=4,
                        )
                    else:
                        x_vals = [p[0] for p in raw_points]   # iteração
                        y_vals = [p[1] for p in raw_points]   # lucro (best so far)
                        ax.plot(x_vals, y_vals, label=method_labels[method],
                                color=method_colors[method], linewidth=2)

                    has_series = True
                    has_objective_data = True

            mode_label = "Repair / None" if "repair" in modes or "none" in modes else "Penalty"
            ax.set_title(mode_label)
            ax.set_xlabel("RH Total (trabalhadores)" if use_rh_axes else "Iteração")
            ax.set_ylabel("Lucro (€)")
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
        default="csv/optimization/normal/metaheuristica",
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
        "--o3-omega-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Executa O3_WEIGHTED com múltiplos omegas para aproximar a frente de Pareto (default: ativo; desativar com --no-o3-omega-sweep).",
    )
    parser.add_argument(
        "--o3-omega-values",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Valores de omega para o sweep (default: 0.1 a 0.9).",
    )
    parser.add_argument(
        "--o3-omega-sweep-runs",
        type=int,
        default=1,
        help="Número de runs por omega no sweep (default: 1; não precisa de robustez estatística).",
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
    pareto_data: dict[str, list] = {}
    weighted_sweep_points: list[tuple[int, int]] = []  # (profit, hr) para frente Pareto via soma ponderada

    print("=" * 70)
    print("OTIMIZAÇÃO COM METAHEURÍSTICAS")
    print("=" * 70)

    # Executar cenários com replicações por método/objetivo/modo.
    for objective in objectives:
        objective_constraint_modes = default_constraint_modes if objective in ("o2", "o3_weighted") else ["none"]
        if objective == "o3_pareto":
            objective_methods = ["nsga_ii"]
        elif objective == "o3_weighted":
            objective_methods = ["genetic_algorithm"]
        else:
            objective_methods = base_methods
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
                        # Parâmetros calibrados por objetivo (grid search 5 runs × 40 combinações).
                        # O2/O3: fitness em escala de centenas → precisa de T_initial muito maior que O1.
                        SA_PARAMS_PER_OBJECTIVE = {
                            "o2":           {"repair": (1000, 0.990), "penalty": (1000, 0.993)},
                            "o3_weighted":  {"repair": (1000, 0.990), "penalty": (1000, 0.993)},
                        }
                        obj_params = SA_PARAMS_PER_OBJECTIVE.get(objective, {})
                        sa_t_init  = obj_params.get(constraint_mode, (args.sa_temp_initial, args.sa_cooling_rate))[0]
                        sa_c_rate  = obj_params.get(constraint_mode, (args.sa_temp_initial, args.sa_cooling_rate))[1]
                        optimizer = SimulatedAnnealingOptimizer(
                            groups,
                            objective,
                            args.omega,
                            constraint_mode,
                            args.sa_iterations,
                            sa_t_init,
                            args.sa_temp_final,
                            sa_c_rate,
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
                    total_profit = int(getattr(solution, "total_profit", 0))
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
                            "pareto_front": getattr(optimizer, "pareto_front_points", []),
                            "pareto_history": getattr(optimizer, "pareto_front_history", []),
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
                if method == "nsga_ii":
                    pareto_data[objective] = {
                        "points": best_run.get("pareto_front", []),
                        "history": best_run.get("pareto_history", []),
                    }

                # --- Omega sweep para O3_WEIGHTED: como no professor, usar apenas GA ponderado ---
                if args.o3_omega_sweep and objective == "o3_weighted" and method == "genetic_algorithm":
                    # Incluir resultado da corrida actual (omega = args.omega)
                    if best_run["feasible"]:
                        weighted_sweep_points.append((best_run["total_profit"], best_run["total_hr"]))

                    # Correr os restantes omegas do sweep
                    for sw_omega in args.o3_omega_values:
                        if abs(sw_omega - args.omega) < 1e-9:
                            continue  # já foi executado acima
                        sw_per_run: list[dict] = []
                        sw_key = f"sweep_{sw_omega:.2f}|{method}|{constraint_mode}"
                        sw_seed_offset = sum(ord(ch) for ch in sw_key) * 1000
                        for sw_run in range(args.o3_omega_sweep_runs):
                            random.seed(args.seed_base + sw_seed_offset + sw_run)
                            sw_opt = GeneticAlgorithmOptimizer(
                                groups,
                                objective,
                                sw_omega,
                                constraint_mode,
                                total_evals=args.ga_total_evals,
                                pop_size=args.ga_pop_size,
                                crossover_rate=args.ga_crossover_rate,
                                mutation_rate=args.ga_mutation_rate,
                            )
                            sw_sol = sw_opt.optimize()
                            sw_feasible = (sw_sol.total_units <= UNITS_CAP)
                            sw_per_run.append({"feasible": sw_feasible, "profit": int(sw_sol.total_profit), "hr": int(sw_sol.total_hr), "fitness": float(sw_sol.fitness_o1)})

                        sw_feasible_runs = [r for r in sw_per_run if r["feasible"]]
                        if sw_feasible_runs:
                            sw_best = max(sw_feasible_runs, key=lambda r: r["fitness"])
                            weighted_sweep_points.append((sw_best["profit"], sw_best["hr"]))
                    print(f"  Sweep omega concluído: {len(weighted_sweep_points)} pontos acumulados.")

    # Guardar resultados por execução
    runs_file = f"{output_prefix}_runs.csv"
    pd.DataFrame(run_rows).to_csv(runs_file, index=False)
    print(f"Runs saved: {runs_file}")

    # Guardar convergência em JSON
    convergence_json = {
        key: [[int(step), int(profit), int(hr)] for step, profit, hr in conv]
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
    pareto_plot_files: list[str] = []
    for obj_key, pdata in pareto_data.items():
        pts  = pdata.get("points", []) if isinstance(pdata, dict) else pdata
        hist = pdata.get("history") if isinstance(pdata, dict) else None
        pf = save_pareto_front_plot(pts, output_prefix, hist)
        if pf:
            pareto_plot_files.append(pf)

    # Gráfico comparativo: soma ponderada (sweep omegas) vs NSGA-II
    if args.o3_omega_sweep and weighted_sweep_points:
        nsga2_pts = pareto_data.get("o3_pareto", {}).get("points", []) if pareto_data else []
        wsf = save_weighted_sweep_pareto_plot(weighted_sweep_points, output_prefix, nsga2_pts or None)
        if wsf:
            pareto_plot_files.append(wsf)
            print(f"Weighted sweep Pareto front saved: {wsf}")

    print("\nVisualization files:")
    for file_name in plot_files:
        print(f" - {file_name}")
    print(f" - {dashboard_file}")
    for file_name in pareto_plot_files:
        print(f" - {file_name}")


if __name__ == "__main__":
    main()
