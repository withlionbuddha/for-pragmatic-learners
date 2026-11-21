from __future__ import annotations

import math
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
    """Independent adaptive Ry sampler + one-dimensional COBYLA optimizer."""

    @tc.validate_types
    def __init__(
        self,
        steps: int = 1,
        direction_cobyla_evaluations: int = 20,
        cobyla_radius: float = 0.1,
        seed: int = 42,
        adaptive_learning_rate: float = 0.1,
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
            learning_rate=adaptive_learning_rate
        )
        self.adaptation_history: list[dict] = []

    @tc.validate_types
    def optimize(self, evaluate: Callable[[], float]) -> tc.Payload:
        if not self.params or self.sampler is None:
            raise RuntimeError("prepare(model) must be called before optimize()")

        evaluations = 0
        status = "completed"
        count = sum(p.numel() for p in self.params)
        for step in range(self.steps):
            directions = self.create_directions(count)
            sampled = np.concatenate(
                [
                    direction.detach().cpu().reshape(-1).numpy()
                    for direction in directions
                ]
            )
            baseline = [p.detach().clone() for p in self.params]
            objective = _DirectionObjective(
                self.params, baseline, directions, self._measure_evaluate(evaluate)
            )
            try:
                # An extra objective call measures the baseline for a valid reward.
                baseline_loss = float(self._measure_evaluate(evaluate)())
                evaluations += 1
                if not math.isfinite(baseline_loss):
                    raise FloatingPointError("Non-finite baseline loss")
                result = minimize(
                    objective,
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

            # Keep the original weights when the search did not improve the loss.
            if objective.best_loss < baseline_loss:
                objective.assign(objective.best_alpha)
                improvement = baseline_loss - objective.best_loss
            else:
                objective.assign(0.0)
                improvement = 0.0
            evaluations += objective.evaluations
            record = self.sampler.update(sampled, improvement)
            record.update(
                {
                    "step": step,
                    "baseline_loss": baseline_loss,
                    "best_loss": objective.best_loss,
                    "best_alpha": objective.best_alpha,
                    "optimization_evaluations": objective.evaluations + 1,
                }
            )
            self.adaptation_history.append(record)
            status = str(result.message)

        return {
            "optimization_evaluations": evaluations,
            "quantum_shots": count * self.steps,
            "quantum_sampling_calls": self.steps,
            "status": status,
            "steps_done": self.steps,
            "adaptive_theta": self.sampler.theta,
            "adaptive_probability_plus": self.sampler.probability_plus,
            "adaptation_history": self.adaptation_history,
        }
