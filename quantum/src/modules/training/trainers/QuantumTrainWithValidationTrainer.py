from __future__ import annotations
from modules.calculators.AccuracyCalculator import AccuracyCalculator
from modules.calculators.LossCalculator import LossCalculator
import modules.common.TypeContracts as tc
from modules.training.runners.QuantumBatchRunner import QuantumBatchRunner
from modules.training.runners.QuantumTrainEpochRunner import QuantumTrainEpochRunner
from modules.training.trainers.BaseTrainTrainer import BaseTrainTrainer
from modules.training.runners.TrainValidEpochRunner import TrainValidEpochRunner
from modules.training.runners.ValidEpochRunner import ValidEpochRunner


class QuantumTrainWithValidationTrainer(BaseTrainTrainer):
    """
        학습 및 검증 Epoch를 함께 실행한다.
    """

    @tc.validate_types
    def _validate(self) -> None:
        super()._validate()
        if getattr(self.data_loaders, "valid", None) is None:
            raise RuntimeError("valid data_loader가 없습니다.")

    @tc.validate_types
    def create_runner(self) -> tc.RunnerLike:
        return TrainValidEpochRunner(
            QuantumTrainEpochRunner(self.model, self.batch_runner()),
            ValidEpochRunner(self.model, self.batch_runner()),
        )

    @tc.validate_types
    def run(self, epoch_runner: tc.RunnerLike) -> tc.Payload:
        return epoch_runner.run(
            self.data_loaders.train,
            self.data_loaders.valid,
            self.compile_configs.get_optimizer(self.model)
            , self.get_profiler()
        )
    
    @tc.validate_types
    def create_batch_runner(self) -> tc.BatchRunnerLike:
        return QuantumBatchRunner(
            model=self.model,
            loss_calculator=LossCalculator(self.compile_configs.get_criterion()),
            accuracy_calculator=AccuracyCalculator()
        )