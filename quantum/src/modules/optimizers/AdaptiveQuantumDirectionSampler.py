from __future__ import annotations

import math

import numpy as np

import modules.common.TypeContracts as tc


class AdaptiveQuantumDirectionSampler:
    """One-qubit Ry(theta) direction sampler with bounded REINFORCE updates.

    This is a statevector simulation, not evidence of quantum advantage.
    """

    @tc.validate_types
    def __init__(
        self,
        theta: float = math.pi / 2,
        learning_rate: float = 0.1,
        min_probability: float = 0.05,
    ) -> None:
        if not (0 < min_probability < 0.5) or learning_rate <= 0:
            raise ValueError("invalid adaptive sampler configuration")
        self.learning_rate = float(learning_rate)
        self.min_probability = float(min_probability)
        self.theta = float(theta)
        self._clip_theta()

    @tc.validate_types
    def _clip_theta(self) -> None:
        lower = 2 * math.asin(math.sqrt(self.min_probability))
        self.theta = float(np.clip(self.theta, lower, math.pi - lower))

    @property
    @tc.validate_types
    def probability_plus(self) -> float:
        return math.sin(self.theta / 2) ** 2

    @tc.validate_types
    def sample(self, count: int, seed: int) -> np.ndarray:
        if count < 1:
            raise ValueError("count must be positive")
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        circuit = QuantumCircuit(1)
        circuit.ry(self.theta, 0)
        state = Statevector.from_instruction(circuit)
        state.seed(int(seed))
        memory = state.sample_memory(shots=int(count))
        return np.fromiter(
            (1 if bit == "1" else -1 for bit in memory), dtype=np.int8, count=count
        )

    @tc.validate_types
    def update(self, directions: np.ndarray, improvement: float) -> dict:
        """Update theta using reward * score of Bernoulli samples.

        The reward is baseline_loss - best_loss; it is NOT a per-weight
        contribution estimate. The update is intentionally bounded.
        """
        directions = np.asarray(directions)
        if directions.size == 0 or not np.all(np.isin(directions, [-1, 1])):
            raise ValueError("directions must contain +/-1")
        previous_theta = self.theta
        p = self.probability_plus
        empirical_plus = float(np.mean(directions == 1))
        # d log P(bits|theta)/d theta = count*(mean(bit)-p)*2/sin(theta)
        # Divide by count to avoid width-dependent step sizes.
        score = 2 * (empirical_plus - p) / math.sin(self.theta)
        reward = float(improvement)
        if not math.isfinite(reward):
            raise ValueError("improvement must be finite")
        self.theta += self.learning_rate * float(np.clip(reward, -1.0, 1.0)) * score
        self._clip_theta()
        return {
            "theta_before": previous_theta,
            "theta_after": self.theta,
            "probability_plus_before": p,
            "probability_plus_after": self.probability_plus,
            "reward": reward,
            "score": score,
        }
