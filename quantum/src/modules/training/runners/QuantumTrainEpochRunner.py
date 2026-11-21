from __future__ import annotations
import modules.common.TypeContracts as tc
import torch

from modules.training.profilers.TrainingProfiler import TrainingProfiler
from modules.training.results.QuantumEpochResult import QuantumEpochResult


"""
    1 epoch 의 train 실행
    
    전체 학습 데이터(train_loader)의 모든 batch에 대해서 학습(train)
"""
class QuantumTrainEpochRunner:
    @tc.validate_types
    def __init__(self, model: tc.TrainModeLike, batch_runner: tc.TrainBatchLike) -> None:
        self.model = model
        self.batch_runner = batch_runner

    @tc.validate_types
    def run(self
            , data_loader: tc.Iterable[object] | None
            , optimizer: object | None
            , profiler: TrainingProfiler | None = None) -> tc.Payload:

        if data_loader is None:
            raise RuntimeError("train_loader가 없습니다.")
        
        if optimizer is None:
            raise RuntimeError("optimizer가 없습니다.")

        self.model.train()

        with torch.set_grad_enabled(
            self.batch_runner.requires_grad
        ):

            epoch_result = QuantumEpochResult()
                
            for x_batch, y_batch in data_loader:
                batch_result = self.batch_runner.run_train_batch(
                    x_batch
                    ,y_batch
                    ,optimizer
                    , profiler
                )
                epoch_result.add(batch_result)

        return epoch_result.to_train_dict(
              self.batch_runner.get_accuracy_calculator()
            , self.batch_runner.get_loss_calculator()
        )
