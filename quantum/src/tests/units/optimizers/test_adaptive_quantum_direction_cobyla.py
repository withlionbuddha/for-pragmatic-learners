"""Unit tests for AdaptiveQuantumDirectionCOBYLA."""
import numpy as np
import pytest
import torch
from torch import nn
from modules.optimizers.AdaptiveQuantumDirectionCOBYLA import AdaptiveQuantumDirectionCOBYLA


class FakeSampler:
    def sample(self, count: int, seed: int) -> np.ndarray:
        return np.ones(count, dtype=np.int8)


def test_optimize_requires_prepare():
    work = AdaptiveQuantumDirectionCOBYLA()
    with pytest.raises(RuntimeError, match="prepare"):
        work.optimize(lambda: 1.0)

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


def test_adaptive_cobyla_sampler_receives_only_finite_signed_directions(monkeypatch):
    """Expose sampled values if the adaptive update receives invalid directions."""
    model = nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = AdaptiveQuantumDirectionCOBYLA(
        steps=1, direction_cobyla_evaluations=12, seed=7
    )
    optimizer.prepare(model)
    optimizer.sampler.sample = FakeSampler().sample
    original_update = optimizer.sampler.update
    observed = []

    def checked_update(directions, improvement):
        values = np.asarray(directions)
        diagnostics = (
            f"dtype={values.dtype}, unique={np.unique(values)}, "
            f"shape={values.shape}, finite={np.all(np.isfinite(values))}, "
            f"valid={np.all(np.isin(values, [-1, 1]))}"
        )
        assert values.shape == (3,), diagnostics
        assert np.all(np.isfinite(values)), diagnostics
        assert np.all(np.isin(values, [-1, 1])), diagnostics
        observed.append(values.copy())
        return original_update(directions, improvement)

    monkeypatch.setattr(optimizer.sampler, "update", checked_update)

    def evaluate():
        with torch.no_grad():
            return float(
                (model.weight - 2).square().sum().item()
                + (model.bias - 2).square().sum().item()
            )

    optimizer.optimize(evaluate)
    assert len(observed) == 1


@pytest.mark.xfail(strict=True, reason="Pending: reward must credit the actual signed COBYLA displacement")
def test_adaptation_credits_actual_displacement_when_alpha_is_negative(monkeypatch):
    """A negative alpha reverses the sampled direction used for adaptation."""
    import modules.optimizers.AdaptiveQuantumDirectionCOBYLA as adaptive_module

    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = AdaptiveQuantumDirectionCOBYLA(steps=1, direction_cobyla_evaluations=4)
    optimizer.prepare(model)
    optimizer.sampler.sample = FakeSampler().sample
    observed = []
    optimizer.sampler.update = lambda directions, improvement: observed.append(
        (np.asarray(directions).copy(), improvement)
    ) or {}

    def fake_minimize(objective, x0, method, options):
        objective(np.array([0.0]))
        objective(np.array([-1.0]))
        return type("Result", (), {"message": "done"})()

    monkeypatch.setattr(adaptive_module, "minimize", fake_minimize)

    def evaluate():
        with torch.no_grad():
            return float((model.weight + 1).square().sum())

    optimizer.optimize(evaluate)
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0][0], np.array([-1], dtype=np.int8))
    assert observed[0][1] > 0


@pytest.mark.xfail(strict=True, reason="Pending: reuse baseline evaluation to preserve COBYLA search budget")
def test_adaptive_uses_full_cobyla_budget_with_same_total_evaluations(monkeypatch):
    """Baseline measurement must not reduce the COBYLA search budget."""
    import modules.optimizers.AdaptiveQuantumDirectionCOBYLA as adaptive_module

    model = nn.Linear(1, 1, bias=False)
    optimizer = AdaptiveQuantumDirectionCOBYLA(steps=1, direction_cobyla_evaluations=12)
    optimizer.prepare(model)
    optimizer.sampler.sample = FakeSampler().sample
    budgets = []

    def fake_minimize(objective, x0, method, options):
        budgets.append(options["maxiter"])
        objective(np.array([0.0]))
        return type("Result", (), {"message": "done"})()

    monkeypatch.setattr(adaptive_module, "minimize", fake_minimize)
    optimizer.optimize(lambda: 1.0)
    assert budgets == [12]
    # Once baseline is reused, its objective evaluation is counted only once.
    assert optimizer.adaptation_history[0]["optimization_evaluations"] == 1
