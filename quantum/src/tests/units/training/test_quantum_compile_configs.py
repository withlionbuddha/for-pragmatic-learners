"""QuantumCompileConfigs constructor, caching and optimizer protocol tests."""

import pytest
from torch import nn

from modules.training.QuantumCompileConfigs import QuantumCompileConfigs


class FakeQuantumOptimizer:
    def __init__(self, *, step_size: float) -> None:
        self.step_size = step_size
        self.model = None

    def prepare(self, model: object) -> None:
        self.model = model

    def optimize(self, evaluate) -> dict:
        return {"optimization_evaluations": 1, "loss": evaluate()}


def test_constructs_without_parameters_and_caches_instance():
    model = nn.Linear(2, 1)
    config = QuantumCompileConfigs(
        optimizer_class=FakeQuantumOptimizer,
        optimizer_kwargs={"step_size": 0.2},
        criterion=nn.MSELoss(),
        profile_configs={},
    )
    optimizer = config.get_optimizer(model)
    assert isinstance(optimizer, FakeQuantumOptimizer)
    assert optimizer.step_size == pytest.approx(0.2)
    assert config.get_optimizer(model) is optimizer
    assert optimizer.model is None


def test_criterion_is_preserved():
    criterion = nn.MSELoss()
    config = QuantumCompileConfigs(
        FakeQuantumOptimizer, {"step_size": 0.1}, criterion, profile_configs={}
    )
    assert config.get_criterion() is criterion


def test_missing_model_is_rejected_before_creation():
    config = QuantumCompileConfigs(
        FakeQuantumOptimizer, {"step_size": 0.1}, nn.MSELoss(), profile_configs={}
    )
    with pytest.raises(RuntimeError, match="model"):
        config.get_optimizer(None)


def test_invalid_optimizer_kwargs_are_propagated():
    config = QuantumCompileConfigs(
        FakeQuantumOptimizer, {"unexpected": 1}, nn.MSELoss(), profile_configs={}
    )
    with pytest.raises(TypeError):
        config.get_optimizer(nn.Linear(2, 1))
