"""Deterministic tests for quantum-assisted gradient-free optimizers."""
import numpy as np
import pytest
import torch
from torch import nn

from modules.optimizers.AdaptiveQuantumDirectionCOBYLA import AdaptiveQuantumDirectionCOBYLA
from modules.optimizers.QuantumSPSA import QuantumSPSA
from modules.optimizers.QuantumDirectionCOBYLA import QuantumDirectionCOBYLA


class FakeSampler:
    def sample(self, count: int, seed: int) -> np.ndarray:
        return np.ones(count, dtype=np.int8)


@pytest.mark.parametrize("work", [
    QuantumSPSA(steps=1, epsilon=0.1, spsa_lr=0.1),
    QuantumDirectionCOBYLA(steps=1, direction_cobyla_evaluations=12),
])
def test_quantum_work_updates_and_reports_metrics(work):
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


@pytest.mark.parametrize("work", [QuantumSPSA(), QuantumDirectionCOBYLA()])
def test_optimize_requires_prepare(work):
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


def test_spsa_restores_parameters_when_objective_fails():
    work = QuantumSPSA(steps=1)
    work.sampler = FakeSampler()
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.25)
    work.prepare(model)
    baseline = model.weight.detach().clone()

    with pytest.raises(RuntimeError, match="objective failed"):
        work.optimize(lambda: (_ for _ in ()).throw(RuntimeError("objective failed")))
    assert torch.equal(model.weight.detach(), baseline)


def test_adaptive_cobyla_updates_with_normalized_multi_parameter_directions():
    """The adaptive sampler must receive signs, not 1/sqrt(N)-scaled vectors."""
    model = nn.Linear(2, 1, bias=True)  # Three parameters; normalized values are not +/-1.
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = AdaptiveQuantumDirectionCOBYLA(
        steps=1, direction_cobyla_evaluations=12, seed=7
    )
    optimizer.prepare(model)
    optimizer.sampler.sample = FakeSampler().sample

    def evaluate():
        with torch.no_grad():
            return float((model.weight - 2).square().sum() + (model.bias - 2).square().sum())

    metrics = optimizer.optimize(evaluate)
    assert metrics["steps_done"] == 1
    assert metrics["quantum_shots"] == 3
    assert metrics["optimization_evaluations"] >= 2
    assert len(metrics["adaptation_history"]) == 1
    assert 0.0 <= metrics["adaptive_probability_plus"] <= 1.0
