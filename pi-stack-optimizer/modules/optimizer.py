"""Generic optimizer module housing PSO and future algorithms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from modules.constants import IDX, PARAM_NAMES


# Parameter bounds to prevent optimizer from exploring unphysical regions
# These keep the search space reasonable for molecular stacking
PARAM_BOUNDS = {
    "cos_like": (-1.0, 1.0),
    "sin_like": (-1.0, 1.0),
    "Tx": (-10.0, 10.0),
    "Ty": (-10.0, 10.0),
    "Tz": (2.0, 8.0),
    "Cx": (-10.0, 10.0),
    "Cy": (-10.0, 10.0),
}

SUPPORTED_METHODS: Dict[str, str] = {
    "pso": "Particle Swarm Optimization",
    "ga": "Genetic Algorithm",
    "gwo": "Grey Wolf Optimizer",
    "pso-nm": "PSO + Nelder-Mead",
}


def list_supported_methods() -> List[str]:
    """Return optimizer method keys that callers can expose via CLI selections."""
    return sorted(SUPPORTED_METHODS.keys())


def describe_method(method: str) -> str:
    """Return a human-readable method label if available."""
    return SUPPORTED_METHODS.get(method.lower(), method)


def _clamp_positions(P: np.ndarray, dim: int) -> np.ndarray:
    """Enforce parameter bounds to keep optimizer in physically reasonable space."""
    for param_name, (low, high) in PARAM_BOUNDS.items():
        if param_name in IDX and IDX[param_name] < dim:
            idx = IDX[param_name]
            P[:, idx] = np.clip(P[:, idx], low, high)
    # Clamp torsion angles (parameters beyond standard transform parameters) to [-2π, 2π]
    if dim > len(PARAM_NAMES):
        torsion_start = len(PARAM_NAMES)
        P[:, torsion_start:] = np.clip(P[:, torsion_start:], -2*np.pi, 2*np.pi)
    return P


def _initialize_positions(rng: np.random.Generator, count: int, dim: int) -> np.ndarray:
    P = np.zeros((count, dim))
    if IDX["cos_like"] < dim:
        P[:, IDX["cos_like"]] = rng.normal(0.0, 1.0, count)
    if IDX["sin_like"] < dim:
        P[:, IDX["sin_like"]] = rng.normal(0.0, 1.0, count)
    for name, sd in [("Tx", 2.0), ("Ty", 2.0), ("Cx", 2.0), ("Cy", 2.0)]:
        if IDX[name] < dim:
            P[:, IDX[name]] = rng.normal(0.0, sd, count)
    if IDX["Tz"] < dim:
        P[:, IDX["Tz"]] = rng.normal(3.5, 1.0, count)
    if dim > len(PARAM_NAMES):
        extra = dim - len(PARAM_NAMES)
        P[:, len(PARAM_NAMES) :] = rng.normal(0.0, 0.5, (count, extra))
    # Clamp initial positions to bounds
    P = _clamp_positions(P, dim)
    return P


@dataclass
class PSOConfig:
    swarm_size: int = 60
    max_iters: int = 300
    inertia: float = 0.73
    cognitive: float = 1.50
    social: float = 1.50
    seed: Optional[int] = None
    verbose_every: int = 1
    tol: float = 0.01
    patience: int = 20
    print_trajectories: bool = False


class PSO:
    def __init__(self, cfg: PSOConfig, dim: int):
        self.cfg = cfg
        self.dim = dim
        self.rng = np.random.default_rng(cfg.seed)
        self.trajectory_writer = None
        self.last_patience_delta: Optional[float] = None
        self._header_printed = False

    def optimize(self, objective_fn) -> Tuple[np.ndarray, float]:
        P = _initialize_positions(self.rng, self.cfg.swarm_size, self.dim)
        V = self.rng.normal(0.0, 0.5, P.shape)
        fitness = objective_fn.batch_evaluate(P)
        pbest = P.copy()
        pbest_val = fitness.copy()
        g_idx = np.argmin(pbest_val)
        gbest = pbest[g_idx].copy()
        gbest_val = pbest_val[g_idx]
        patience_counter = 0
        best_history = [gbest_val]
        iter_delta = None
        for it in range(1, self.cfg.max_iters + 1):
            r1 = self.rng.random(P.shape)
            r2 = self.rng.random(P.shape)
            V = (
                self.cfg.inertia * V
                + self.cfg.cognitive * r1 * (pbest - P)
                + self.cfg.social * r2 * (gbest - P)
            )
            P = P + V
            # Clamp positions to bounds after update
            P = _clamp_positions(P, self.dim)
            fitness = objective_fn.batch_evaluate(P)
            improved = fitness < pbest_val
            pbest[improved] = P[improved]
            pbest_val[improved] = fitness[improved]
            g_idx = np.argmin(pbest_val)
            delta = gbest_val - pbest_val[g_idx]
            if delta > self.cfg.tol:
                gbest_val = pbest_val[g_idx]
                gbest = pbest[g_idx].copy()
                patience_counter = 0
            else:
                patience_counter += 1
            best_history.append(gbest_val)
            if len(best_history) >= self.cfg.patience:
                ref_val = best_history[-self.cfg.patience]
                iter_delta = abs(gbest_val - ref_val)
            else:
                ref_val = best_history[0]
                iter_delta = abs(gbest_val - ref_val)
            if self.cfg.verbose_every and (it % self.cfg.verbose_every == 0):
                if not self._header_printed:
                    print("[PSO]  iter    best          Δ_patience")
                    self._header_printed = True
                print(
                    f"[PSO]  {it:4d}  {gbest_val:12.6f}  {iter_delta:12.6f}"
                )
                if self.cfg.print_trajectories and self.trajectory_writer:
                    for i in range(self.cfg.swarm_size):
                        row = [it, i, False] + P[i].tolist() + [fitness[i]]
                        self.trajectory_writer.writerow(row)
                    g_idx = np.argmin(fitness)
                    row = [it, "global_best", True] + gbest.tolist() + [gbest_val]
                    self.trajectory_writer.writerow(row)
            if patience_counter >= self.cfg.patience:
                if len(best_history) >= self.cfg.patience:
                    improvement = abs(best_history[-self.cfg.patience] - gbest_val)
                    if improvement < self.cfg.tol:
                        print(
                            f"[PSO] Early stopping at iter {it}: no improvement for {self.cfg.patience} iterations"
                        )
                        break
        if best_history:
            ref_index = -self.cfg.patience if len(best_history) >= self.cfg.patience else 0
            reference_val = best_history[ref_index]
            self.last_patience_delta = abs(gbest_val - reference_val)
        else:
            self.last_patience_delta = None
        return gbest, gbest_val


@dataclass
class GAConfig:
    population_size: int = 100
    max_generations: int = 500
    mutation_rate: float = 0.10
    mutation_sigma: float = 0.30
    crossover_rate: float = 0.80
    elite_fraction: float = 0.05
    tournament_size: int = 3
    seed: Optional[int] = None
    verbose_every: int = 1
    tol: float = 0.01
    patience: int = 20


class GA:
    def __init__(self, cfg: GAConfig, dim: int):
        self.cfg = cfg
        self.dim = dim
        self.rng = np.random.default_rng(cfg.seed)
        self.last_patience_delta: Optional[float] = None
        self._header_printed = False

    def optimize(self, objective_fn) -> Tuple[np.ndarray, float]:
        population = _initialize_positions(
            self.rng, self.cfg.population_size, self.dim
        )
        fitness = objective_fn.batch_evaluate(population)
        best_idx = int(np.argmin(fitness))
        best_params = population[best_idx].copy()
        best_val = float(fitness[best_idx])
        elite_fraction = min(max(self.cfg.elite_fraction, 0.0), 1.0)
        elite_count = max(1, int(elite_fraction * self.cfg.population_size))
        elite_count = min(elite_count, self.cfg.population_size)
        
        patience_counter = 0
        best_history = [best_val]
        iter_delta = None

        for generation in range(1, self.cfg.max_generations + 1):
            elite_indices = np.argsort(fitness)[:elite_count]
            elites = population[elite_indices]
            new_population = np.empty_like(population)
            new_population[:elite_count] = elites

            idx = elite_count
            while idx < self.cfg.population_size:
                parent_a = self._tournament_select(population, fitness)
                parent_b = self._tournament_select(population, fitness)
                child = self._crossover(parent_a, parent_b)
                child = self._mutate(child)
                new_population[idx] = child
                idx += 1

            population = new_population
            fitness = objective_fn.batch_evaluate(population)
            current_best_idx = int(np.argmin(fitness))
            current_best_val = float(fitness[current_best_idx])
            
            delta = best_val - current_best_val
            if delta > self.cfg.tol:
                best_val = current_best_val
                best_params = population[current_best_idx].copy()
                patience_counter = 0
            else:
                if current_best_val < best_val:
                    best_val = current_best_val
                    best_params = population[current_best_idx].copy()
                patience_counter += 1
            
            best_history.append(best_val)
            
            if len(best_history) >= self.cfg.patience:
                ref_val = best_history[-self.cfg.patience]
                iter_delta = abs(best_val - ref_val)
            else:
                ref_val = best_history[0]
                iter_delta = abs(best_val - ref_val)

            if self.cfg.verbose_every and generation % self.cfg.verbose_every == 0:
                if not self._header_printed:
                    print("[GA]   gen    best          Δ_patience")
                    self._header_printed = True
                print(f"[GA]  {generation:4d}  {best_val:12.6f}  {iter_delta:12.6f}")
            
            if patience_counter >= self.cfg.patience:
                if len(best_history) >= self.cfg.patience:
                    improvement = abs(best_history[-self.cfg.patience] - best_val)
                    if improvement < self.cfg.tol:
                        print(
                            f"[GA] Early stopping at gen {generation}: no improvement for {self.cfg.patience} generations"
                        )
                        break
        
        if best_history:
            ref_index = -self.cfg.patience if len(best_history) >= self.cfg.patience else 0
            reference_val = best_history[ref_index]
            self.last_patience_delta = abs(best_val - reference_val)
        else:
            self.last_patience_delta = None

        return best_params, best_val

    def _tournament_select(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> np.ndarray:
        size = min(self.cfg.tournament_size, self.cfg.population_size)
        size = max(1, size)
        participants = self.rng.choice(
            self.cfg.population_size, size=size, replace=False
        )
        best_idx = participants[np.argmin(fitness[participants])]
        return population[best_idx]

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.cfg.crossover_rate:
            return parent_a.copy()
        if self.dim <= 1:
            return parent_a.copy()
        point = self.rng.integers(1, self.dim)
        return np.concatenate((parent_a[:point], parent_b[point:]))

    def _mutate(self, child: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.dim) < self.cfg.mutation_rate
        if np.any(mask):
            child = child.copy()
            child[mask] += self.rng.normal(0.0, self.cfg.mutation_sigma, mask.sum())
        return child


@dataclass
class GWOConfig:
    pack_size: int = 50
    max_iters: int = 300
    a_start: float = 2.0
    a_end: float = 0.0
    seed: Optional[int] = None
    verbose_every: int = 1
    tol: float = 0.01
    patience: int = 20


class GWO:
    def __init__(self, cfg: GWOConfig, dim: int):
        self.cfg = cfg
        self.dim = dim
        self.rng = np.random.default_rng(cfg.seed)
        self.last_patience_delta: Optional[float] = None
        self._header_printed = False

    def optimize(self, objective_fn) -> Tuple[np.ndarray, float]:
        pack_size = max(1, self.cfg.pack_size)
        wolves = _initialize_positions(self.rng, pack_size, self.dim)
        fitness = objective_fn.batch_evaluate(wolves)
        alpha, beta, delta, alpha_val = self._update_leaders(wolves, fitness)
        
        patience_counter = 0
        best_history = [alpha_val]
        iter_delta = None

        for iteration in range(1, self.cfg.max_iters + 1):
            a = self._compute_a(iteration)
            wolves = self._update_positions(wolves, alpha, beta, delta, a)
            fitness = objective_fn.batch_evaluate(wolves)
            prev_alpha_val = alpha_val
            alpha, beta, delta, alpha_val = self._update_leaders(wolves, fitness)
            
            delta = prev_alpha_val - alpha_val
            if delta > self.cfg.tol:
                patience_counter = 0
            else:
                patience_counter += 1
            
            best_history.append(alpha_val)
            
            if len(best_history) >= self.cfg.patience:
                ref_val = best_history[-self.cfg.patience]
                iter_delta = abs(alpha_val - ref_val)
            else:
                ref_val = best_history[0]
                iter_delta = abs(alpha_val - ref_val)

            if self.cfg.verbose_every and iteration % self.cfg.verbose_every == 0:
                if not self._header_printed:
                    print("[GWO]  iter    best          Δ_patience")
                    self._header_printed = True
                print(f"[GWO]  {iteration:4d}  {alpha_val:12.6f}  {iter_delta:12.6f}")
            
            if patience_counter >= self.cfg.patience:
                if len(best_history) >= self.cfg.patience:
                    improvement = abs(best_history[-self.cfg.patience] - alpha_val)
                    if improvement < self.cfg.tol:
                        print(
                            f"[GWO] Early stopping at iter {iteration}: no improvement for {self.cfg.patience} iterations"
                        )
                        break
        
        if best_history:
            ref_index = -self.cfg.patience if len(best_history) >= self.cfg.patience else 0
            reference_val = best_history[ref_index]
            self.last_patience_delta = abs(alpha_val - reference_val)
        else:
            self.last_patience_delta = None

        return alpha.copy(), float(alpha_val)

    def _compute_a(self, iteration: int) -> float:
        if self.cfg.max_iters <= 1:
            return self.cfg.a_end
        frac = (iteration - 1) / (self.cfg.max_iters - 1)
        return self.cfg.a_start + frac * (self.cfg.a_end - self.cfg.a_start)

    def _update_positions(
        self,
        wolves: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        delta: np.ndarray,
        a: float,
    ) -> np.ndarray:
        pack_size = wolves.shape[0]
        alpha_vec = alpha[np.newaxis, :]
        beta_vec = beta[np.newaxis, :]
        delta_vec = delta[np.newaxis, :]

        A1 = 2 * a * self.rng.random((pack_size, self.dim)) - a
        C1 = 2 * self.rng.random((pack_size, self.dim))
        X1 = alpha_vec - A1 * np.abs(C1 * alpha_vec - wolves)

        A2 = 2 * a * self.rng.random((pack_size, self.dim)) - a
        C2 = 2 * self.rng.random((pack_size, self.dim))
        X2 = beta_vec - A2 * np.abs(C2 * beta_vec - wolves)

        A3 = 2 * a * self.rng.random((pack_size, self.dim)) - a
        C3 = 2 * self.rng.random((pack_size, self.dim))
        X3 = delta_vec - A3 * np.abs(C3 * delta_vec - wolves)

        return (X1 + X2 + X3) / 3.0

    def _update_leaders(
        self, wolves: np.ndarray, fitness: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        sorted_idx = np.argsort(fitness)
        alpha_idx = sorted_idx[0]
        beta_idx = sorted_idx[1] if len(sorted_idx) > 1 else alpha_idx
        delta_idx = sorted_idx[2] if len(sorted_idx) > 2 else beta_idx
        alpha = wolves[alpha_idx].copy()
        beta = wolves[beta_idx].copy()
        delta = wolves[delta_idx].copy()
        return alpha, beta, delta, float(fitness[alpha_idx])


@dataclass
class PSONMConfig:
    pso: PSOConfig = field(default_factory=PSOConfig)
    nm_max_iters: int = 200
    nm_alpha: float = 1.0
    nm_gamma: float = 2.0
    nm_rho: float = 0.50
    nm_sigma: float = 0.50
    nm_initial_step: float = 0.20
    nm_tol: float = 0.01


class PSONM:
    def __init__(self, cfg: PSONMConfig, dim: int):
        self.cfg = cfg
        self.dim = dim
        self.pso = PSO(cfg.pso, dim)
        self.trajectory_writer = None
        self.last_patience_delta: Optional[float] = None

    def optimize(self, objective_fn) -> Tuple[np.ndarray, float]:
        self.pso.trajectory_writer = self.trajectory_writer
        best_params, best_val = self.pso.optimize(objective_fn)
        self.last_patience_delta = getattr(self.pso, "last_patience_delta", None)
        refined_params, refined_val = self._nelder_mead(best_params, objective_fn)
        if refined_val < best_val:
            return refined_params, refined_val
        return best_params, best_val

    def _nelder_mead(self, start: np.ndarray, objective_fn) -> Tuple[np.ndarray, float]:
        dim = len(start)
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = start.copy()
        step = self.cfg.nm_initial_step or 1e-1
        for i in range(dim):
            vertex = start.copy()
            vertex[i] = vertex[i] + step if vertex[i] != 0 else step
            simplex[i + 1] = vertex
        values = np.array([self._eval_point(objective_fn, v) for v in simplex])

        for _ in range(max(1, self.cfg.nm_max_iters)):
            order = np.argsort(values)
            simplex = simplex[order]
            values = values[order]
            best = simplex[0]
            best_val = values[0]
            worst = simplex[-1]
            second_worst = simplex[-2]

            if np.std(values) < self.cfg.nm_tol:
                break

            centroid = simplex[:-1].mean(axis=0)
            reflected = centroid + self.cfg.nm_alpha * (centroid - worst)
            refl_val = self._eval_point(objective_fn, reflected)

            if refl_val < best_val:
                expanded = centroid + self.cfg.nm_gamma * (reflected - centroid)
                exp_val = self._eval_point(objective_fn, expanded)
                if exp_val < refl_val:
                    simplex[-1] = expanded
                    values[-1] = exp_val
                else:
                    simplex[-1] = reflected
                    values[-1] = refl_val
                continue

            if refl_val < values[-2]:
                simplex[-1] = reflected
                values[-1] = refl_val
                continue

            contracted = centroid + self.cfg.nm_rho * (worst - centroid)
            con_val = self._eval_point(objective_fn, contracted)
            if con_val < values[-1]:
                simplex[-1] = contracted
                values[-1] = con_val
                continue

            simplex[1:] = simplex[0] + self.cfg.nm_sigma * (simplex[1:] - simplex[0])
            values[1:] = [self._eval_point(objective_fn, v) for v in simplex[1:]]

        best_idx = int(np.argmin(values))
        return simplex[best_idx].copy(), float(values[best_idx])

    @staticmethod
    def _eval_point(objective_fn, point: np.ndarray) -> float:
        arr = np.asarray(point, dtype=float).reshape(1, -1)
        return float(objective_fn.batch_evaluate(arr)[0])


@dataclass
class OptimizerRequest:
    """Container describing which optimizer the caller wants and its params."""

    method: str = "pso"
    pso: PSOConfig = field(default_factory=PSOConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    gwo: GWOConfig = field(default_factory=GWOConfig)
    pso_nm: PSONMConfig = field(default_factory=PSONMConfig)


@dataclass
class OptimizerHandle:
    """Return value with optimizer instance and metadata for logging."""

    method: str
    optimizer: object
    config: object
    supports_trajectories: bool = False


def create_optimizer(request: OptimizerRequest, dim: int) -> OptimizerHandle:
    """Instantiate the requested optimizer and expose metadata."""

    method_key = request.method.lower()
    if method_key == "pso":
        optimizer = PSO(request.pso, dim)
        return OptimizerHandle(
            method="pso",
            optimizer=optimizer,
            config=request.pso,
            supports_trajectories=request.pso.print_trajectories,
        )
    if method_key == "ga":
        optimizer = GA(request.ga, dim)
        return OptimizerHandle(
            method="ga",
            optimizer=optimizer,
            config=request.ga,
            supports_trajectories=False,
        )
    if method_key == "gwo":
        optimizer = GWO(request.gwo, dim)
        return OptimizerHandle(
            method="gwo",
            optimizer=optimizer,
            config=request.gwo,
            supports_trajectories=False,
        )
    if method_key == "pso-nm":
        optimizer = PSONM(request.pso_nm, dim)
        return OptimizerHandle(
            method="pso-nm",
            optimizer=optimizer,
            config=request.pso_nm,
            supports_trajectories=request.pso_nm.pso.print_trajectories,
        )
    supported = ", ".join(list_supported_methods())
    raise ValueError(f"Unsupported optimizer '{request.method}'. Supported: {supported}")
