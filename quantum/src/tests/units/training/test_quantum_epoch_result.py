import pytest
from torch import nn

from modules.calculators.AccuracyCalculator import AccuracyCalculator
from modules.calculators.LossCalculator import LossCalculator
from modules.training.results.QuantumEpochResult import QuantumEpochResult


def test_quantum_epoch_result_aggregates_batch_metrics():
    result = QuantumEpochResult()
    for shots in (3, 5):
        result.add(
            {
                "loss": 2.0,
                "correct": 1,
                "total": 2,
                "optimizer_result": {
                    "quantum_shots": shots,
                    "quantum_sampling_calls": 1,
                    "optimization_evaluations": 2,
                    "status": f"completed-{shots}",
                },
            }
        )
    metrics = result.to_train_dict(
        AccuracyCalculator(), LossCalculator(nn.CrossEntropyLoss())
    )
    assert metrics["optimizer_result"] == {
        "quantum_shots": 8,
        "quantum_sampling_calls": 2,
        "optimization_evaluations": 4,
        "statuses": ["completed-3", "completed-5"],
    }
    assert metrics["train_loss"] == pytest.approx(2.0)
    assert metrics["checkpoint_loss"] == pytest.approx(2.0)


def test_quantum_epoch_result_validation_payload():
    result = QuantumEpochResult()
    result.add({"loss": 1.0, "correct": 2, "total": 2, "quantum_shots": 4})
    metrics = result.to_valid_dict(
        AccuracyCalculator(), LossCalculator(nn.CrossEntropyLoss())
    )
    assert metrics["valid_loss"] == pytest.approx(1.0)
    assert metrics["optimizer_result"]["quantum_shots"] == 4
