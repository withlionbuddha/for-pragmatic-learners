from __future__ import annotations

import modules.common.TypeContracts as tc
from modules.training.results.EpochResult import EpochResult


class QuantumEpochResult(EpochResult):
    """EpochResult with additive quantum optimization diagnostics."""

    @tc.validate_types
    def __init__(self) -> None:
        super().__init__()
        self.quantum_sampling_calls = 0
        self.quantum_shots = 0
        self.optimization_evaluations = 0
        self.optimizer_statuses = []

    @tc.validate_types
    def add(self, batch_result: tc.Payload) -> None:
        super().add(batch_result)
        status = (batch_result.get("optimizer_result") or {}).get("status")
        if status is not None:
            self.optimizer_statuses.append(str(status))
        for key in (
            "quantum_sampling_calls",
            "quantum_shots",
            "optimization_evaluations",
        ):
            value = batch_result.get("optimizer_result", {}).get(key, batch_result.get(key, 0))
            if value is not None:
                setattr(self, key, getattr(self, key) + value)

    @tc.validate_types
    def _quantum_metrics(self) -> tc.Payload:
        return {
            "optimizer_result": {
                "quantum_sampling_calls": self.quantum_sampling_calls,
                "quantum_shots": self.quantum_shots,
                "optimization_evaluations": self.optimization_evaluations,
                "statuses": list(self.optimizer_statuses),
            }
        }

    @tc.validate_types
    def to_train_dict(self, accuracy_calculator: tc.AccuracyAverageLike, loss_calculator: tc.LossAverageLike) -> tc.Payload:
        return {
            **super().to_train_dict(accuracy_calculator, loss_calculator),
            **self._quantum_metrics(),
        }

    @tc.validate_types
    def to_valid_dict(self, accuracy_calculator: tc.AccuracyAverageLike, loss_calculator: tc.LossAverageLike) -> tc.Payload:
        return {
            **super().to_valid_dict(accuracy_calculator, loss_calculator),
            **self._quantum_metrics(),
        }
