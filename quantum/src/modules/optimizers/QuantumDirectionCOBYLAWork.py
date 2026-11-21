from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
import math

import numpy as np
import torch
from scipy.optimize import minimize

import modules.common.TypeContracts as tc
from modules.optimizers.QuantumDirectionSampler import QuantumDirectionSampler


class _DirectionObjective:
    """One COBYLA step's parameter assignment and objective state."""

    @tc.validate_types
    def __init__(
        self,
        params: list[torch.nn.Parameter],
        baseline: list[torch.Tensor],
        directions: list[torch.Tensor],
        evaluate: Callable[[], float],
    ) -> None:
        self.params = params
        self.baseline = baseline
        self.directions = directions
        self.evaluate = evaluate
        self.evaluations = 0
        self.best_alpha = 0.0
        self.best_loss = float("inf")

    @torch.no_grad()
    @tc.validate_types
    def assign(self, alpha: float) -> None:
        for param, initial, direction in zip(
            self.params, self.baseline, self.directions
        ):
            param.copy_(initial + float(alpha) * direction)

    @tc.validate_types
    def __call__(self, alpha_vector: np.ndarray) -> float:
        alpha = float(alpha_vector[0])
        self.assign(alpha)
        value = float(self.evaluate())
        self.evaluations += 1

        if not math.isfinite(value):
            raise FloatingPointError("Non-finite COBYLA objective")

        if value < self.best_loss:
            self.best_loss = value
            self.best_alpha = alpha
        return value


class QuantumDirectionCOBYLAWork:
    @tc.validate_types
    def __init__(
        self,
        steps: int = 1,
        direction_cobyla_evaluations: int = 20,
        cobyla_radius: float = 0.1,
        seed: int = 42,
        target_loss_meter: object | None = None,
    ) -> None:
        if steps < 1 or direction_cobyla_evaluations < 3 or cobyla_radius <= 0:
            raise ValueError("invalid COBYLA configuration")

        self.steps = steps
        self.budget = direction_cobyla_evaluations
        self.radius = cobyla_radius
        self.rng = np.random.default_rng(seed + 2000)
        self.sampler: QuantumDirectionSampler | None = None
        self.params: list[torch.nn.Parameter] = []
        self.target_loss_meter = target_loss_meter

    @tc.validate_types
    def prepare(self, model: object) -> None:
        if not callable(getattr(model, "parameters", None)):
            raise TypeError("model must provide parameters()")
        self.params = [p for p in model.parameters() if p.requires_grad]
        if not self.params:
            raise ValueError("model has no trainable parameters")
        if self.sampler is None:
            self.sampler = QuantumDirectionSampler()

    @tc.validate_types
    def _measure_evaluate(self, evaluate: Callable[[], float]) -> Callable[[], float]:
        if self.target_loss_meter is None:
            return evaluate
        return lambda: self.target_loss_meter.evaluate(evaluate)

    @tc.validate_types
    def optimize(self, evaluate: Callable[[], float]) -> tc.Payload:
        if not self.params or self.sampler is None:
            raise RuntimeError("prepare(model) must be called before optimize()")

        count = sum(p.numel() for p in self.params)
        evaluations = 0
        status = "completed"

        for _ in range(self.steps):
            directions = self.create_directions(count)
            baseline = [p.detach().clone() for p in self.params]
            objective = _DirectionObjective(
                self.params, baseline, directions, self._measure_evaluate(evaluate)
            )

            try:
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

            objective.assign(objective.best_alpha)
            evaluations += objective.evaluations
            status = str(result.message)

        return {
            "optimization_evaluations": evaluations,
            "quantum_shots": count * self.steps,
            "quantum_sampling_calls": self.steps,
            "status": status,
            "steps_done": self.steps,
        }

    @tc.validate_types
    def create_directions(self, count: int) -> list[torch.Tensor]:
        if self.sampler is None:
            raise RuntimeError("prepare(model) must be called before sampling")

        bits = self.sampler.sample(count, int(self.rng.integers(0, 2**31 - 1)))
        direction_scale = 1.0 / math.sqrt(count)
        directions = []
        offset = 0
        for param in self.params:
            size = param.numel()
            directions.append(
                torch.as_tensor(
                    bits[offset : offset + size], dtype=param.dtype, device=param.device
                ).reshape(param.shape) * direction_scale
            )
            offset += size
        return directions

    def initialize_parameters(self, model: object) -> None:
        """Compatibility entry point for QuantumAdapter."""
        self.prepare(model)

    def run_objective(self, evaluate: Callable[[], float]) -> tc.Payload:
        """Compatibility entry point for QuantumAdapter."""
        return self.optimize(evaluate)
