"""Unit tests for QuantumDirectionCOBYLA."""
import numpy as np
import pytest
import torch
from torch import nn
from modules.optimizers.QuantumDirectionCOBYLA import QuantumDirectionCOBYLA


class FakeSampler:
    def sample(self, count: int, seed: int) -> np.ndarray:
        return np.ones(count, dtype=np.int8)


def test_quantum_work_updates_and_reports_metrics():
    work = QuantumDirectionCOBYLA(steps=1, direction_cobyla_evaluations=12)
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    work.sampler = FakeSampler()
    work.prepare(model)
    def evaluate():
        with torch.no_grad():
            return float((model.weight - 2).square().sum().item())
    before = evaluate()
    metrics = work.optimize(evaluate)
    assert evaluate() < before
    assert metrics["quantum_sampling_calls"] == 1
    assert metrics["quantum_shots"] == 1
    assert metrics["optimization_evaluations"] >= 2
    assert metrics["steps_done"] == 1


def test_optimize_requires_prepare():
    work = QuantumDirectionCOBYLA()
    with pytest.raises(RuntimeError, match="prepare"):
        work.optimize(lambda: 1.0)


def test_cobyla_restores_parameters_when_objective_fails():
    work = QuantumDirectionCOBYLA(steps=1)
    work.sampler = FakeSampler()
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.25)
    work.prepare(model)
    baseline = model.weight.detach().clone()

    def failing_evaluate():
        raise RuntimeError("objective failed")

    with pytest.raises(RuntimeError, match="objective failed"):
        work.optimize(failing_evaluate)
    assert torch.equal(model.weight.detach(), baseline)
