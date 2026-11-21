from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

import modules.common.TypeContracts as tc
from modules.optimizers.AdaptiveQuantumDirectionSampler import (
    AdaptiveQuantumDirectionSampler,
)
from modules.optimizers.QuantumDirectionCOBYLA import (
    QuantumDirectionCOBYLA,
    _DirectionObjective,
)


class AdaptiveQuantumDirectionCOBYLA(QuantumDirectionCOBYLA):
    """Coordinate-wise adaptive Ry sampler with one-dimensional COBYLA."""

    @tc.validate_types
    def __init__(
        self,
        steps: int = 1,
        direction_cobyla_evaluations: int = 20,
        cobyla_radius: float = 0.1,
        seed: int = 42,
        adaptive_learning_rate: float = 0.1,
        theta_group_size: int | None = 1,
        target_loss_meter: object | None = None,
    ) -> None:
        super().__init__(
            steps=steps,
            direction_cobyla_evaluations=direction_cobyla_evaluations,
            cobyla_radius=cobyla_radius,
            seed=seed,
            target_loss_meter=target_loss_meter,
        )
        self.sampler = AdaptiveQuantumDirectionSampler(
            learning_rate=adaptive_learning_rate,
            theta_group_size=theta_group_size,
        )
        self.adaptation_history: list[dict] = []

    @tc.validate_types
    def prepare(self, model: object) -> None:
        super().prepare(model)
        self.sampler.prepare(sum(p.numel() for p in self.params))

    @tc.validate_types
    def optimize(self, evaluate: Callable[[], float]) -> tc.Payload:
        if not self.params or self.sampler is None:
            raise RuntimeError("prepare(model) must be called before optimize()")

        evaluations = 0
        status = "completed"
        count = sum(p.numel() for p in self.params)
        for step in range(self.steps):
            directions = self.create_directions(count)
            sampled = np.sign(np.concatenate([
                direction.detach().cpu().reshape(-1).numpy()
                for direction in directions
            ])).astype(np.int8)
            baseline = [p.detach().clone() for p in self.params]
            objective = _DirectionObjective(
                self.params, baseline, directions, self._measure_evaluate(evaluate)
            )
            try:
                # COBYLA normally starts at x0=0. Cache that first evaluation,
                # so baseline measurement does not consume an extra objective call.
                baseline_loss = objective(np.array([0.0]))
                cached = True

                def cached_objective(alpha: np.ndarray) -> float:
                    nonlocal cached
                    if cached and float(alpha[0]) == 0.0:
                        cached = False
                        return baseline_loss
                    cached = False
                    return objective(alpha)

                result = minimize(
                    cached_objective,
                    np.array([0.0]),
                    method="COBYLA",
                    options={
                        "maxiter": self.budget,
                        "rhobeg": self.radius,
                        "tol": 1e-5,
                    },
                )
            except BaseException:
                objective.assign(0.0)
                raise

            if objective.best_loss < baseline_loss:
                objective.assign(objective.best_alpha)
                improvement = baseline_loss - objective.best_loss
            else:
                objective.assign(0.0)
                improvement = 0.0

            # Negative alpha reverses the accepted parameter displacement.
            accepted_signs = (
                sampled * np.sign(objective.best_alpha).astype(np.int8)
                if improvement > 0 and objective.best_alpha != 0
                else sampled
            )
            record = self.sampler.update(accepted_signs, improvement)
            evaluations += objective.evaluations
            record.update({
                "step": step,
                "baseline_loss": baseline_loss,
                "best_loss": objective.best_loss,
                "best_alpha": objective.best_alpha,
                "optimization_evaluations": objective.evaluations,
            })
            self.adaptation_history.append(record)
            status = str(result.message)

        return {
            "optimization_evaluations": evaluations,
            "quantum_shots": count * self.steps,
            "quantum_sampling_calls": self.steps,
            "status": status,
            "steps_done": self.steps,
            "theta_group_size": self.sampler.theta_group_size,
            "adaptive_theta": np.asarray(self.sampler.theta).tolist(),
            "adaptive_probability_plus": self.sampler.probability_plus,
            "adaptive_probabilities_plus": self.sampler.probabilities_plus.tolist(),
            "adaptation_history": self.adaptation_history,
        }
