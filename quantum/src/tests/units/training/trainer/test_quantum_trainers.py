from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from torch import nn

from modules.builders.ModelBuilder import ModelBuilder
from modules.training.profilers.TrainingProfiler import TrainingProfiler
from modules.training.QuantumCompileConfigs import QuantumCompileConfigs
from modules.training.runners.QuantumTrainEpochRunner import QuantumTrainEpochRunner
from modules.training.runners.TrainValidEpochRunner import TrainValidEpochRunner
from modules.training.trainers.QuantumTrainOnlyTrainer import QuantumTrainOnlyTrainer
from modules.training.trainers.QuantumTrainWithValidationTrainer import (
    QuantumTrainWithValidationTrainer,
)


class FakeOptimizer:
    def prepare(self, model):
        pass

    def optimize(self, evaluate):
        return {"optimization_evaluations": 1}


def make_trainer(cls):
    model = ModelBuilder(device_name="cpu").add(nn.Linear(2, 2)).build()
    trainer = cls(model)
    trainer.set_compile_configs(
        QuantumCompileConfigs(
            FakeOptimizer, {}, nn.CrossEntropyLoss(), profile_configs={}
        )
    )
    trainer.set_data_loaders(SimpleNamespace(train=[("x", "y")], valid=[("x", "y")]))
    trainer.set_saver_manager(Mock())
    trainer.set_training_profiler(TrainingProfiler())
    return trainer


def test_quantum_train_only_trainer_creates_quantum_runner():
    trainer = make_trainer(QuantumTrainOnlyTrainer)
    assert isinstance(trainer.create_runner(), QuantumTrainEpochRunner)
    runner = Mock()
    runner.run.return_value = {"train_loss": 1.0}
    trainer.run(runner)
    runner.run.assert_called_once_with(
        trainer.data_loaders.train,
        trainer.compile_configs.get_optimizer(trainer.model),
        trainer.get_profiler(),
    )


def test_quantum_train_with_validation_trainer_passes_loaders_and_profiler():
    trainer = make_trainer(QuantumTrainWithValidationTrainer)
    assert isinstance(trainer.create_runner(), TrainValidEpochRunner)
    runner = Mock()
    runner.run.return_value = {"train_loss": 1.0}
    trainer.run(runner)
    runner.run.assert_called_once_with(
        trainer.data_loaders.train,
        trainer.data_loaders.valid,
        trainer.compile_configs.get_optimizer(trainer.model),
        trainer.get_profiler(),
    )


def test_quantum_train_with_validation_requires_valid_loader():
    trainer = make_trainer(QuantumTrainWithValidationTrainer)
    trainer.set_data_loaders(SimpleNamespace(train=[], valid=None))
    with pytest.raises(RuntimeError, match="valid"):
        trainer._validate()
