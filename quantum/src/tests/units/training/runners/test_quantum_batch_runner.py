import torch
from torch import nn

from modules.builders.ModelBuilder import ModelBuilder
from modules.calculators.AccuracyCalculator import AccuracyCalculator
from modules.calculators.LossCalculator import LossCalculator
from modules.training.profilers.TrainingProfiler import TrainingProfiler
from modules.training.runners.QuantumBatchRunner import QuantumBatchRunner


class FakeOptimizer:
    def __init__(self):
        self.model = None
        self.calls = 0

    def prepare(self, model):
        self.model = model

    def optimize(self, evaluate):
        self.calls += 1
        loss = evaluate()
        return {
            "optimization_evaluations": 1,
            "quantum_shots": 2,
            "quantum_sampling_calls": 1,
            "evaluated_loss": loss,
        }


def test_quantum_batch_runner_train_and_validation():
    model = ModelBuilder(device_name="cpu").add(nn.Linear(2, 2)).build()
    runner = QuantumBatchRunner(
        model, LossCalculator(nn.CrossEntropyLoss()), AccuracyCalculator()
    )
    optimizer = FakeOptimizer()
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([0, 1])
    train = runner.run_train_batch(x, y, optimizer)
    valid = runner.run_valid_batch(x, y, TrainingProfiler())
    assert optimizer.model is model
    assert optimizer.calls == 1
    assert train["optimizer_result"]["optimization_evaluations"] == 1
    assert train["total"] == valid["total"] == 2
    assert train["loss"] >= 0 and valid["loss"] >= 0
