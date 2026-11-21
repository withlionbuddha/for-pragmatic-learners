from __future__ import annotations
from modules.calculators.AccuracyCalculator import AccuracyCalculator
from modules.calculators.LossCalculator import LossCalculator
import modules.common.TypeContracts as tc
from modules.training.runners.QuantumBatchRunner import QuantumBatchRunner
from modules.training.runners.QuantumTrainEpochRunner import QuantumTrainEpochRunner
from modules.training.trainers.BaseTrainTrainer import BaseTrainTrainer


class QuantumTrainOnlyTrainer(BaseTrainTrainer):
    @tc.validate_types
    def __init__(self, model: tc.nn.Module | tc.DeviceModel) -> None:
        super(QuantumTrainOnlyTrainer,self).__init__(model)

    @tc.validate_types
    def create_runner(self) -> tc.RunnerLike:
        return QuantumTrainEpochRunner(self.model, self.batch_runner())
    
    @tc.validate_types
    def run(self, epoch_runner: tc.RunnerLike) -> tc.Payload:
        return epoch_runner.run(
            self.data_loaders.train,
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


