import pytest
import torch
from torch import nn

from modules.calculators.AccuracyCalculator import AccuracyCalculator
from modules.calculators.LossCalculator import LossCalculator
from modules.training.runners.QuantumTrainEpochRunner import QuantumTrainEpochRunner


class FakeTrainBatch:
    requires_grad = False

    def __init__(self):
        self.calls = 0
        self.grad_modes = []

    def run_train_batch(self, x_batch, y_batch, optimizer, profiler=None):
        self.calls += 1
        self.grad_modes.append(torch.is_grad_enabled())
        return {
            "loss": 1.0,
            "correct": 1,
            "total": 2,
            "optimizer_result": {
                "quantum_shots": 2,
                "quantum_sampling_calls": 1,
                "optimization_evaluations": 2,
            },
        }

    def get_accuracy_calculator(self):
        return AccuracyCalculator()

    def get_loss_calculator(self):
        return LossCalculator(nn.CrossEntropyLoss())


def test_quantum_epoch_runner_disables_grad_and_aggregates():
    model = nn.Linear(2, 2)
    batch = FakeTrainBatch()
    runner = QuantumTrainEpochRunner(model, batch)
    result = runner.run(
        [(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long))] * 2, object()
    )
    assert model.training
    assert batch.calls == 2
    assert batch.grad_modes == [False, False]
    assert result["optimizer_result"]["quantum_shots"] == 4
    assert result["train_loss"] == pytest.approx(1.0)
    assert result["checkpoint_loss"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "loader,optimizer,message",
    [
        (None, object(), "train_loader"),
        ([], None, "optimizer"),
    ],
)
def test_quantum_epoch_runner_rejects_missing_dependencies(loader, optimizer, message):
    runner = QuantumTrainEpochRunner(nn.Linear(2, 2), FakeTrainBatch())
    with pytest.raises(RuntimeError, match=message):
        runner.run(loader, optimizer)
