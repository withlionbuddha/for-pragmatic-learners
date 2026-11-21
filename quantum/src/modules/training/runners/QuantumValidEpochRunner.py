from __future__ import annotations
import modules.common.TypeContracts as tc
from modules.training.profilers.TrainingProfiler import TrainingProfiler
from modules.training.results.EpochResult import EpochResult
import torch

"""
    1 epoch 의 valid 실행
    
    전체 검증 데이터(valid_laoder)의 모든 batch에 대해서 검증(valid) 실행

"""
class QuantumValidEpochRunner:
    @tc.validate_types
    def __init__(self, model: tc.EvalModeLike, batch_runner: tc.ValidBatchLike) -> None:
        self.model = model
        self.batch_runner = batch_runner

    @tc.validate_types
    def run(self
            , valid_loader: tc.Iterable[object] | None
            , profiler: TrainingProfiler | None = None) -> tc.Payload:

        if valid_loader is None:
            raise RuntimeError("valid_loader가 없습니다.")
        
        with torch.no_grad():
            self.model.eval()

            epoch_result = EpochResult()
            for x_batch, y_batch in valid_loader:
                batch_result = self.batch_runner.run_valid_batch(
                    x_batch
                    , y_batch
                    , profiler
                )
                epoch_result.add(batch_result)

        return epoch_result.to_valid_dict(
            self.batch_runner.get_accuracy_calculator()
            , self.batch_runner.get_loss_calculator()
        )
