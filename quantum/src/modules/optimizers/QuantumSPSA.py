"""Quantum-sampled SPSA; loss evaluations and updates remain classical."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

import modules.common.TypeContracts as tc
from modules.optimizers.QuantumDirectionSampler import QuantumDirectionSampler


class QuantumSPSA:
    @tc.validate_types
    def __init__(
        self,
        steps: int = 1,
        epsilon: float = 0.1,
        spsa_lr: float = 0.01,
        seed: int = 42,
    ) -> None:

        if steps < 1 or epsilon <= 0 or spsa_lr <= 0:
            raise ValueError("steps, epsilon and spsa_lr must be positive")

        self.steps = steps
        self.epsilon = epsilon
        self.spsa_lr = spsa_lr
        self.rng = np.random.default_rng(seed + 1000)
        self.sampler: QuantumDirectionSampler | None = None
        self.params: list[torch.nn.Parameter] = []

    @tc.validate_types
    def prepare(self, model: object) -> None:
        if not callable(getattr(model, "parameters", None)):
            raise TypeError("model must provide parameters()")

        self.params = [p for p in model.parameters() if p.requires_grad]

        if not self.params:
            raise ValueError("model has no trainable parameters")

        if self.sampler is None:
            self.sampler = QuantumDirectionSampler()

    @torch.no_grad()
    @tc.validate_types
    def _perturb(self, bits: np.ndarray, amount: float) -> None:
        offset = 0

        for param in self.params:
            size = param.numel()
            direction = torch.as_tensor(
                bits[offset : offset + size], dtype=param.dtype, device=param.device
            ).reshape(param.shape)
            param.add_(direction, alpha=float(amount))
            offset += size

    @tc.validate_types
    def optimize(self, evaluate: Callable[[], float]) -> tc.Payload:

        if not self.params or self.sampler is None:
            raise RuntimeError("prepare(model) must be called before optimize()")

        count = sum(p.numel() for p in self.params)
        # Unit-norm directions keep perturbations comparable across model sizes.
        direction_scale = 1.0 / math.sqrt(count)

        for step in range(1, self.steps + 1):
            bits = self.sampler.sample(count, int(self.rng.integers(0, 2**31 - 1)))
            epsilon = self.epsilon / step**0.101
            rate = self.spsa_lr / step**0.602
            position = 0.0

            try:
                self._perturb(bits, epsilon * direction_scale)
                position = epsilon
                plus = float(evaluate())
                self._perturb(bits, -2 * epsilon * direction_scale)
                position = -epsilon
                minus = float(evaluate())
            finally:
                if position:
                    self._perturb(bits, -position * direction_scale)

            if not math.isfinite(plus) or not math.isfinite(minus):
                raise FloatingPointError(
                    f"Non-finite SPSA loss: step={step}, "
                    f"loss_plus={plus}, loss_minus={minus}, "
                    f"epsilon={epsilon}, rate={rate}"
                )

            slope = (plus - minus) / (2 * epsilon)

            if not math.isfinite(slope):
                raise FloatingPointError(
                    f"Non-finite SPSA slope: step={step}, "
                    f"loss_plus={plus}, loss_minus={minus}, epsilon={epsilon}, "
                    f"rate={rate}, slope={slope}"
                )

            self._perturb(bits, -rate * slope * direction_scale)

        return {
            "optimization_evaluations": 2 * self.steps,
            "quantum_shots": count * self.steps,
            "quantum_sampling_calls": self.steps,
            "status": "completed",
            "steps_done": self.steps,
        }

    def initialize_parameters(self, model: object) -> None:
        """Compatibility entry point for QuantumAdapter."""
        self.prepare(model)

    def run_objective(self, evaluate: Callable[[], float]) -> tc.Payload:
        """Compatibility entry point for QuantumAdapter."""
        return self.optimize(evaluate)
