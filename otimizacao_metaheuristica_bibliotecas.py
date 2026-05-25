"""Versao separada com otimizadores de biblioteca sobre a logica do ficheiro base.

Mantem:
- a representacao de 84 valores [PR, X, J] x 4 lojas x 7 dias
- a avaliacao do `otimizacao_metaheuristica.py`
- a perturbacao multiplicativa e o repair do ficheiro base

Troca apenas os metodos de procura por wrappers de bibliotecas:
- nevergrad: Monte Carlo e Hill Climbing
- scipy: Simulated Annealing
- pymoo: Genetic Algorithm e NSGA-II
"""

from __future__ import annotations

import random
import sys
import warnings

import nevergrad as ng
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from scipy.optimize import dual_annealing

import otimizacao_metaheuristica as base


warnings.filterwarnings("ignore", message="Bounds are 2.0 sigma away from each other")


def _vector_bounds(groups: list[base.Group]) -> tuple[np.ndarray, np.ndarray]:
    lower: list[float] = []
    upper: list[float] = []
    for group in groups:
        x_upper_bound = float(np.ceil(group.customers / 7.0)) if group.customers > 0 else 0.0
        j_upper_bound = float(np.ceil(group.customers / 6.0)) if group.customers > 0 else 0.0
        lower.extend([float(base.PR_MIN_INT), 0.0, 0.0])
        upper.extend([float(base.PR_MAX_INT), x_upper_bound, j_upper_bound])
    return np.array(lower, dtype=float), np.array(upper, dtype=float)


def _vector_to_solution(values: np.ndarray | list[float], groups: list[base.Group]) -> base.Solution:
    vector = np.asarray(values, dtype=float)
    repaired: list[float] = []
    for block_idx, idx in enumerate(range(0, len(vector), 3)):
        group = groups[block_idx]
        pr_int = int(round(float(vector[idx])))
        x_val = int(round(float(vector[idx + 1])))
        j_val = int(round(float(vector[idx + 2])))
        x_upper_bound = int(np.ceil(group.customers / 7.0)) if group.customers > 0 else 0
        j_upper_bound = int(np.ceil(group.customers / 6.0)) if group.customers > 0 else 0
        repaired.extend(
            [
                base.pr_from_int(max(base.PR_MIN_INT, min(base.PR_MAX_INT, pr_int))),
                max(0, min(x_upper_bound, x_val)),
                max(0, min(j_upper_bound, j_val)),
            ]
        )
    return base.Solution(values=repaired)


def _solution_to_vector(solution: base.Solution) -> np.ndarray:
    vector: list[float] = []
    for idx in range(0, len(solution.values), 3):
        vector.extend(
            [
                float(round(solution.values[idx] * 100)),
                float(solution.values[idx + 1]),
                float(solution.values[idx + 2]),
            ]
        )
    return np.array(vector, dtype=float)


def _seed_solution(groups: list[base.Group], objective: str, constraint_mode: str) -> base.Solution:
    if objective == "o2" and constraint_mode == "repair":
        return base.generate_random_feasible_solution(groups)
    if objective in ("o2", "o3_weighted", "o3_pareto"):
        if random.random() < 0.7:
            return base.generate_o2_seed_solution(groups)
        return base.generate_random_feasible_solution(groups)
    return base.generate_random_solution(groups)


def _mutate_solution(
    solution: base.Solution,
    groups: list[base.Group],
    objective: str,
    constraint_mode: str,
) -> base.Solution:
    if objective in ("o2", "o3_weighted", "o3_pareto"):
        mode = "repair" if objective == "o3_pareto" else constraint_mode
        return base.generate_neighbor_o2(solution, groups, mode)

    mutated = solution.copy()
    n_moves = random.randint(1, min(3, len(groups)))
    for _ in range(n_moves):
        mutated = base.generate_neighbor_on_triplet(mutated, groups)
    return mutated


def _repair_solution(
    solution: base.Solution,
    groups: list[base.Group],
    objective: str,
    constraint_mode: str,
) -> base.Solution:
    repaired = solution.copy()
    if objective in ("o2", "o3_weighted", "o3_pareto"):
        mode = "repair" if objective == "o3_pareto" else constraint_mode
        if mode == "repair":
            invalid = base.is_solution_invalid(repaired, groups)
            if invalid:
                repaired = base.repair_solution(repaired, groups)
            if objective != "o2":
                repaired = base._fill_o2_slack(repaired, groups, max_steps=max(3, len(groups) // 2))
    return repaired


def _evaluate_candidate(
    vector: np.ndarray | list[float],
    groups: list[base.Group],
    objective: str,
    omega: float,
    constraint_mode: str,
) -> tuple[float, int, int, bool, int, base.Solution]:
    solution = _vector_to_solution(vector, groups)
    solution = _repair_solution(solution, groups, objective, constraint_mode)

    if objective == "o3_pareto":
        profit, units, hr = base.evaluate_profit_units_hr(solution, groups)
        feasible = units <= base.UNITS_CAP
        solution.fitness_o1 = float(profit)
        solution.total_units = int(units)
        solution.total_hr = int(hr)
        solution.total_profit = int(profit)
        solution.feasible = feasible
        return float(profit), int(units), int(hr), feasible, int(profit), solution

    fitness, units, hr, feasible, profit = base.evaluate_solution(
        solution, groups, objective, omega, constraint_mode
    )
    solution.fitness_o1 = float(fitness)
    solution.total_units = int(units)
    solution.total_hr = int(hr)
    solution.total_profit = int(profit)
    solution.feasible = bool(feasible)
    return float(fitness), int(units), int(hr), bool(feasible), int(profit), solution


def _key_for_best(item: tuple[float, int, int, bool, int, base.Solution], objective: str) -> tuple[float, float]:
    if objective == "o3_pareto":
        return float(item[4]), -float(item[2])
    return float(item[0]), -float(item[2])


def _best_evaluated_from_vectors(
    vectors: list[np.ndarray],
    groups: list[base.Group],
    objective: str,
    omega: float,
    constraint_mode: str,
) -> base.Solution:
    evaluated = [
        _evaluate_candidate(vector, groups, objective, omega, constraint_mode)
        for vector in vectors
    ]
    best = max(evaluated, key=lambda item: _key_for_best(item, objective))
    return best[5]


def _loss_from_fitness(fitness: float, units: int, hr: int) -> float:
    if np.isfinite(fitness):
        return -float(fitness)
    overflow = max(0, int(units) - int(base.UNITS_CAP))
    return float(1e12 + overflow * 1e6 + max(0, int(hr)))


def _vectors_from_result(result, fallback: list[np.ndarray]) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    if getattr(result, "X", None) is not None:
        xs = np.atleast_2d(result.X)
        vectors.extend(np.array(x, dtype=float) for x in xs)
    elif getattr(result, "pop", None) is not None:
        try:
            xs = result.pop.get("X")
            if xs is not None:
                vectors.extend(np.array(x, dtype=float) for x in np.atleast_2d(xs))
        except Exception:
            pass
    if not vectors:
        vectors.extend(np.array(x, dtype=float) for x in fallback)
    return vectors


def _history_best_progress(
    history: list,
    groups: list[base.Group],
    objective: str,
    omega: float,
    constraint_mode: str,
    step_scale: int = 1,
) -> list[tuple[int, int, int]]:
    progress: list[tuple[int, int, int]] = []
    best_item: tuple[float, int, int, bool, int, base.Solution] | None = None
    for idx, entry in enumerate(history):
        xs = entry.pop.get("X") if getattr(entry, "pop", None) is not None else None
        if xs is None:
            continue
        evaluated = [
            _evaluate_candidate(x, groups, objective, omega, constraint_mode)
            for x in np.atleast_2d(xs)
        ]
        gen_best = max(evaluated, key=lambda item: _key_for_best(item, objective))
        if best_item is None or _key_for_best(gen_best, objective) > _key_for_best(best_item, objective):
            best_item = gen_best
        best_sol = best_item[5]
        progress.append((int((idx + 1) * step_scale), int(best_sol.total_profit), int(best_sol.total_hr)))
    return progress


class _RetailSampling(Sampling):
    def __init__(self, groups: list[base.Group], objective: str, constraint_mode: str):
        super().__init__()
        self.groups = groups
        self.objective = objective
        self.constraint_mode = constraint_mode

    def _do(self, problem, n_samples, **kwargs):
        return np.array(
            [
                _solution_to_vector(_seed_solution(self.groups, self.objective, self.constraint_mode))
                for _ in range(n_samples)
            ],
            dtype=float,
        )


class _RetailMutation(Mutation):
    def __init__(self, groups: list[base.Group], objective: str, constraint_mode: str, rate: float = 1.0):
        super().__init__()
        self.groups = groups
        self.objective = objective
        self.constraint_mode = constraint_mode
        self.rate = rate

    def _do(self, problem, X, **kwargs):
        mutated = np.array(X, copy=True)
        for idx in range(len(mutated)):
            if random.random() > self.rate:
                continue
            sol = _vector_to_solution(mutated[idx], self.groups)
            sol = _mutate_solution(sol, self.groups, self.objective, self.constraint_mode)
            sol = _repair_solution(sol, self.groups, self.objective, self.constraint_mode)
            mutated[idx] = _solution_to_vector(sol)
        return mutated


class _RetailRepair(Repair):
    def __init__(self, groups: list[base.Group], objective: str, constraint_mode: str):
        super().__init__()
        self.groups = groups
        self.objective = objective
        self.constraint_mode = constraint_mode

    def _do(self, problem, X, **kwargs):
        repaired = np.array(X, copy=True)
        for idx in range(len(repaired)):
            sol = _vector_to_solution(repaired[idx], self.groups)
            sol = _repair_solution(sol, self.groups, self.objective, self.constraint_mode)
            repaired[idx] = _solution_to_vector(sol)
        return repaired


class _RetailSingleObjectiveProblem(ElementwiseProblem):
    def __init__(self, groups: list[base.Group], objective: str, omega: float, constraint_mode: str):
        xl, xu = _vector_bounds(groups)
        n_ieq = 1 if objective in ("o2", "o3_weighted") and constraint_mode == "repair" else 0
        super().__init__(n_var=len(xl), n_obj=1, n_ieq_constr=n_ieq, xl=xl, xu=xu)
        self.groups = groups
        self.objective = objective
        self.omega = omega
        self.constraint_mode = constraint_mode

    def _evaluate(self, x, out, *args, **kwargs):
        fitness, units, hr, _feasible, _profit, _solution = _evaluate_candidate(
            x, self.groups, self.objective, self.omega, self.constraint_mode
        )
        out["F"] = np.array([_loss_from_fitness(fitness, units, hr)], dtype=float)
        if self.n_ieq_constr:
            out["G"] = np.array([float(units - base.UNITS_CAP)], dtype=float)


class _RetailParetoProblem(ElementwiseProblem):
    def __init__(self, groups: list[base.Group]):
        xl, xu = _vector_bounds(groups)
        super().__init__(n_var=len(xl), n_obj=2, n_ieq_constr=1, xl=xl, xu=xu)
        self.groups = groups

    def _evaluate(self, x, out, *args, **kwargs):
        profit, units, hr, _feasible, _net_profit, _solution = _evaluate_candidate(
            x, self.groups, "o3_pareto", base.DEFAULT_OMEGA, "repair"
        )
        out["F"] = np.array([-profit, hr], dtype=float)
        out["G"] = np.array([float(units - base.UNITS_CAP)], dtype=float)


class MonteCarloOptimizer:
    def __init__(self, groups: list[base.Group], objective: str = "o1", omega: float = base.DEFAULT_OMEGA, constraint_mode: str = "repair", iterations: int = base.DEFAULT_MC_ITERATIONS):
        self.groups = groups
        self.objective = objective
        self.omega = omega
        self.constraint_mode = constraint_mode
        self.iterations = iterations
        self.best_solution: base.Solution | None = None
        self.convergence: list[tuple[int, int, int]] = []

    def optimize(self) -> base.Solution:
        init = _solution_to_vector(_seed_solution(self.groups, self.objective, self.constraint_mode))
        xl, xu = _vector_bounds(self.groups)
        parametrization = ng.p.Array(init=init).set_bounds(xl, xu)
        optimizer = ng.optimizers.RandomSearch(parametrization=parametrization, budget=max(1, self.iterations), num_workers=1)

        best_solution: base.Solution | None = None
        best_key: tuple[float, float] | None = None
        for iteration in range(max(1, self.iterations)):
            candidate = optimizer.ask()
            evaluated = _evaluate_candidate(candidate.value, self.groups, self.objective, self.omega, self.constraint_mode)
            optimizer.tell(candidate, _loss_from_fitness(evaluated[0], evaluated[1], evaluated[2]))
            key = _key_for_best(evaluated, self.objective)
            if best_solution is None or key > best_key:
                best_solution = evaluated[5].copy()
                best_key = key
            self.convergence.append((iteration, int(best_solution.total_profit), int(best_solution.total_hr)))

        self.best_solution = best_solution if best_solution is not None else _best_evaluated_from_vectors(
            [init], self.groups, self.objective, self.omega, self.constraint_mode
        )
        return self.best_solution


class HillClimbingOptimizer:
    def __init__(self, groups: list[base.Group], objective: str = "o1", omega: float = base.DEFAULT_OMEGA, constraint_mode: str = "repair", iterations: int = base.DEFAULT_HC_ITERATIONS):
        self.groups = groups
        self.objective = objective
        self.omega = omega
        self.constraint_mode = constraint_mode
        self.iterations = iterations
        self.best_solution: base.Solution | None = None
        self.convergence: list[tuple[int, int, int]] = []

    def optimize(self) -> base.Solution:
        init = _solution_to_vector(_seed_solution(self.groups, self.objective, self.constraint_mode))
        xl, xu = _vector_bounds(self.groups)
        parametrization = ng.p.Array(init=init).set_bounds(xl, xu)
        optimizer = ng.optimizers.OnePlusOne(parametrization=parametrization, budget=max(1, self.iterations), num_workers=1)

        best_solution: base.Solution | None = None
        best_key: tuple[float, float] | None = None
        for iteration in range(max(1, self.iterations)):
            candidate = optimizer.ask()
            evaluated = _evaluate_candidate(candidate.value, self.groups, self.objective, self.omega, self.constraint_mode)
            optimizer.tell(candidate, _loss_from_fitness(evaluated[0], evaluated[1], evaluated[2]))
            key = _key_for_best(evaluated, self.objective)
            if best_solution is None or key > best_key:
                best_solution = evaluated[5].copy()
                best_key = key
            self.convergence.append((iteration, int(best_solution.total_profit), int(best_solution.total_hr)))

        self.best_solution = best_solution if best_solution is not None else _best_evaluated_from_vectors(
            [init], self.groups, self.objective, self.omega, self.constraint_mode
        )
        return self.best_solution


class SimulatedAnnealingOptimizer:
    def __init__(
        self,
        groups: list[base.Group],
        objective: str = "o1",
        omega: float = base.DEFAULT_OMEGA,
        constraint_mode: str = "repair",
        iterations: int = base.DEFAULT_SA_ITERATIONS,
        t_initial: float = base.DEFAULT_SA_T_INITIAL,
        t_final: float = base.SA_T_FINAL,
        cooling_rate: float = base.DEFAULT_SA_COOLING_RATE,
    ):
        self.groups = groups
        self.objective = objective
        self.omega = omega
        self.constraint_mode = constraint_mode
        self.iterations = iterations
        self.t_initial = t_initial
        self.t_final = t_final
        self.cooling_rate = cooling_rate
        self.best_solution: base.Solution | None = None
        self.convergence: list[tuple[int, int, int]] = []

    def optimize(self) -> base.Solution:
        init = _solution_to_vector(_seed_solution(self.groups, self.objective, self.constraint_mode))
        xl, xu = _vector_bounds(self.groups)
        bounds = list(zip(xl, xu))
        seen_vectors: list[np.ndarray] = []

        def objective_fn(x: np.ndarray) -> float:
            seen_vectors.append(np.array(x, dtype=float))
            fitness, units, hr, _feasible, _profit, _solution = _evaluate_candidate(
                x, self.groups, self.objective, self.omega, self.constraint_mode
            )
            return _loss_from_fitness(fitness, units, hr)

        eval_budget = max(10, self.iterations)
        maxiter = max(1, min(self.iterations, 50))
        visit = min(2.95, max(1.01, 1.5 + (1.0 - min(0.999, self.cooling_rate))))
        result = dual_annealing(
            objective_fn,
            bounds=bounds,
            x0=init,
            maxiter=maxiter,
            maxfun=eval_budget,
            initial_temp=max(self.t_final + 1e-6, self.t_initial),
            restart_temp_ratio=max(1e-6, min(0.99, self.t_final / max(self.t_initial, 1e-6))),
            visit=visit,
            no_local_search=True,
            seed=123,
        )

        if result.x is not None:
            seen_vectors.append(np.array(result.x, dtype=float))
        if not seen_vectors:
            seen_vectors.append(init)

        self.best_solution = _best_evaluated_from_vectors(
            seen_vectors, self.groups, self.objective, self.omega, self.constraint_mode
        )
        self.convergence = _history_best_progress(
            [type("Entry", (), {"pop": type("Pop", (), {"get": lambda _self, _key: np.atleast_2d(v)})()}) for v in seen_vectors],
            self.groups,
            self.objective,
            self.omega,
            self.constraint_mode,
        )
        return self.best_solution


class GeneticAlgorithmOptimizer:
    def __init__(
        self,
        groups: list[base.Group],
        objective: str = "o1",
        omega: float = base.DEFAULT_OMEGA,
        constraint_mode: str = "repair",
        total_evals: int = base.DEFAULT_GA_TOTAL_EVALS,
        pop_size: int = base.DEFAULT_GA_POP_SIZE,
        crossover_rate: float = base.DEFAULT_GA_CROSSOVER_RATE,
        mutation_rate: float = base.DEFAULT_GA_MUTATION_RATE,
        tournament_k: int = base.DEFAULT_GA_TOURNAMENT_K,
    ):
        self.groups = groups
        self.objective = objective
        self.omega = omega
        self.constraint_mode = constraint_mode
        self.total_evals = max(1, total_evals)
        self.pop_size = max(4, pop_size)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.best_solution: base.Solution | None = None
        self.convergence: list[tuple[int, int, int]] = []

    def optimize(self) -> base.Solution:
        problem = _RetailSingleObjectiveProblem(self.groups, self.objective, self.omega, self.constraint_mode)
        n_gens = max(1, int(np.ceil(self.total_evals / self.pop_size)))
        sampling = _RetailSampling(self.groups, self.objective, self.constraint_mode)
        mutation = _RetailMutation(self.groups, self.objective, self.constraint_mode, rate=self.mutation_rate)
        repair = _RetailRepair(self.groups, self.objective, self.constraint_mode) if self.constraint_mode == "repair" else None
        algorithm = GA(
            pop_size=self.pop_size,
            sampling=sampling,
            mutation=mutation,
            repair=repair,
            eliminate_duplicates=True,
        )
        result = minimize(problem, algorithm, termination=("n_gen", n_gens), verbose=False, save_history=True)

        fallback = [_solution_to_vector(_seed_solution(self.groups, self.objective, self.constraint_mode)) for _ in range(self.pop_size)]
        vectors = _vectors_from_result(result, fallback)
        self.best_solution = _best_evaluated_from_vectors(vectors, self.groups, self.objective, self.omega, self.constraint_mode)
        self.convergence = _history_best_progress(
            result.history or [],
            self.groups,
            self.objective,
            self.omega,
            self.constraint_mode,
            step_scale=self.pop_size,
        )
        if not self.convergence:
            self.convergence = [(int((idx + 1) * self.pop_size), int(self.best_solution.total_profit), int(self.best_solution.total_hr)) for idx in range(n_gens)]
        return self.best_solution


class NSGA2Optimizer:
    def __init__(self, groups: list[base.Group], generations: int = base.DEFAULT_SA_ITERATIONS, population_size: int = base.DEFAULT_NSGA2_POP_SIZE):
        self.groups = groups
        self.generations = max(1, generations)
        self.population_size = max(10, population_size)
        self.best_solution: base.Solution | None = None
        self.convergence: list[tuple[int, int, int]] = []
        self.pareto_front_points: list[tuple[int, int]] = []
        self.pareto_front_history: list[tuple[int, list[tuple[int, int]]]] = []

    def optimize(self) -> base.Solution:
        problem = _RetailParetoProblem(self.groups)
        sampling = _RetailSampling(self.groups, "o3_pareto", "repair")
        mutation = _RetailMutation(self.groups, "o3_pareto", "repair", rate=0.8)
        algorithm = NSGA2(
            pop_size=self.population_size,
            sampling=sampling,
            mutation=mutation,
            repair=_RetailRepair(self.groups, "o3_pareto", "repair"),
            eliminate_duplicates=True,
        )
        result = minimize(problem, algorithm, termination=("n_gen", self.generations), verbose=False, save_history=True)

        fallback = [_solution_to_vector(_seed_solution(self.groups, "o3_pareto", "repair")) for _ in range(self.population_size)]
        vectors = _vectors_from_result(result, fallback)
        evaluated_final = [
            _evaluate_candidate(vector, self.groups, "o3_pareto", base.DEFAULT_OMEGA, "repair")
            for vector in vectors
        ]
        final_points = list(dict.fromkeys(base._filter_non_dominated([(item[4], item[2]) for item in evaluated_final])))
        self.pareto_front_points = final_points
        if result.history:
            snap_interval = max(1, len(result.history) // 10)
            best_item: tuple[float, int, int, bool, int, base.Solution] | None = None
            for gen_idx, entry in enumerate(result.history):
                xs = entry.pop.get("X") if getattr(entry, "pop", None) is not None else None
                if xs is None:
                    continue
                evaluated = [
                    _evaluate_candidate(x, self.groups, "o3_pareto", base.DEFAULT_OMEGA, "repair")
                    for x in np.atleast_2d(xs)
                ]
                gen_points = list(dict.fromkeys(base._filter_non_dominated([(item[4], item[2]) for item in evaluated])))
                gen_best = max(evaluated, key=lambda item: _key_for_best(item, "o3_pareto"))
                if best_item is None or _key_for_best(gen_best, "o3_pareto") > _key_for_best(best_item, "o3_pareto"):
                    best_item = gen_best
                best_sol = best_item[5]
                self.convergence.append((gen_idx, int(best_sol.total_profit), int(best_sol.total_hr)))
                if gen_idx % snap_interval == 0 or gen_idx == len(result.history) - 1:
                    self.pareto_front_history.append((gen_idx, gen_points))

        self.best_solution = max(evaluated_final, key=lambda item: _key_for_best(item, "o3_pareto"))[5]
        if not self.convergence:
            self.convergence = [(gen, int(self.best_solution.total_profit), int(self.best_solution.total_hr)) for gen in range(self.generations)]
        return self.best_solution


base.MonteCarloOptimizer = MonteCarloOptimizer
base.HillClimbingOptimizer = HillClimbingOptimizer
base.SimulatedAnnealingOptimizer = SimulatedAnnealingOptimizer
base.GeneticAlgorithmOptimizer = GeneticAlgorithmOptimizer
base.NSGA2Optimizer = NSGA2Optimizer


def main() -> None:
    argv = sys.argv[1:]
    has_output_prefix = any(arg == "--output-prefix" or arg.startswith("--output-prefix=") for arg in argv)
    if not has_output_prefix:
        sys.argv = [sys.argv[0], "--output-prefix", "csv/optimization/bibliotecas/metaheuristica_bibliotecas", *argv]
    base.main()


if __name__ == "__main__":
    main()