from __future__ import annotations

import math

import numpy as np

import modules.common.TypeContracts as tc


class AdaptiveQuantumDirectionSampler:
    """Independent one-qubit Ry probabilities for each parameter coordinate.

    The statevector sampler is a simulation, not evidence of quantum advantage.
    A shared scalar reward does not identify causal per-weight contributions.
    """

    @tc.validate_types
    def __init__(
        self,
        theta: float = math.pi / 2,
        learning_rate: float = 0.1,
        min_probability: float = 0.05,
        theta_group_size: int | None = 1,
    ) -> None:
        if not (0 < min_probability < 0.5) or learning_rate <= 0:
            raise ValueError("invalid adaptive sampler configuration")
        if theta_group_size is not None and (type(theta_group_size) is not int or theta_group_size < 1):
            raise ValueError("theta_group_size must be a positive integer or None")
        self.theta_group_size = theta_group_size
        self.parameter_count: int | None = None
        self.learning_rate = float(learning_rate)
        self.min_probability = float(min_probability)
        self.theta = float(theta)
        self._clip_theta()

    @tc.validate_types
    def _clip_theta(self) -> None:
        lower = 2 * math.asin(math.sqrt(self.min_probability))
        self.theta = np.clip(self.theta, lower, math.pi - lower)
        if np.ndim(self.theta) == 0:
            self.theta = float(self.theta)

    def prepare(self, count: int) -> None:
        if count < 1:
            raise ValueError("count must be positive")
        if self.parameter_count is not None and self.parameter_count != count:
            raise ValueError("parameter count changed")
        self.parameter_count = count
        size = count if self.theta_group_size is None else self.theta_group_size
        groups = (count + size - 1) // size
        if np.ndim(self.theta) == 0:
            self.theta = np.full(groups, float(self.theta))
        elif len(self.theta) != groups:
            raise ValueError("parameter count changed")
        self._clip_theta()

    @property
    def probabilities_plus(self) -> np.ndarray:
        return np.sin(np.asarray(self.theta) / 2) ** 2

    def _group_indices(self, count: int) -> np.ndarray:
        size = count if self.theta_group_size is None else self.theta_group_size
        return np.arange(count) // size

    @property
    def probability_plus(self) -> float:
        return float(np.mean(self.probabilities_plus))

    @tc.validate_types
    def sample(self, count: int, seed: int) -> np.ndarray:
        if count < 1:
            raise ValueError("count must be positive")
        self.prepare(count)
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        # Each coordinate has its own Ry(theta). Sampling is independent.
        rng = np.random.default_rng(seed)
        values = np.empty(count, dtype=np.int8)
        for i, group in enumerate(self._group_indices(count)):
            angle = self.theta[group]
            circuit = QuantumCircuit(1)
            circuit.ry(float(angle), 0)
            state = Statevector.from_instruction(circuit)
            state.seed(int(rng.integers(0, 2**31 - 1)))
            values[i] = 1 if state.sample_memory(shots=1)[0] == "1" else -1
        return values

    @tc.validate_types
    def update(self, directions: np.ndarray, improvement: float) -> dict:
        """Coordinate-wise score update with a shared, bounded reward.

        The signed directions are the actual accepted displacement signs.
        This is a heuristic: the reward is not a per-coordinate attribution.
        """
        directions = np.asarray(directions)
        if directions.ndim != 1 or directions.size == 0 or not np.all(
            np.isin(directions, [-1, 1])
        ):
            raise ValueError("directions must contain +/-1")
        reward = float(improvement)
        if not math.isfinite(reward):
            raise ValueError("improvement must be finite")
        self.prepare(int(directions.size))
        previous_theta = self.theta.copy()
        previous_p = self.probabilities_plus.copy()
        groups = self._group_indices(directions.size)
        plus_counts = np.bincount(groups, weights=(directions == 1).astype(float), minlength=len(self.theta))
        group_counts = np.bincount(groups, minlength=len(self.theta))
        empirical_plus = plus_counts / group_counts
        score = 2 * (empirical_plus - previous_p) / np.sin(self.theta)
        self.theta += self.learning_rate * float(np.clip(reward, -1.0, 1.0)) * score
        self._clip_theta()
        return {
            "theta_before": previous_theta.tolist(),
            "theta_after": self.theta.tolist(),
            "probability_plus_before": previous_p.tolist(),
            "probability_plus_after": self.probabilities_plus.tolist(),
            "reward": reward,
            "score": score.tolist(),
        }
