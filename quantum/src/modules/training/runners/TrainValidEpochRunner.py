"""
    1 epoch 의 train 와 valid 실행
    
    전체 학습 데이터(train_laoder)의 모든 batch에 대해서 학습(train) 실행    
    전체 검증 데이터(valid_laoder)의 모든 batch에 대해서 검증(valid) 실행

"""
from __future__ import annotations
import modules.common.TypeContracts as tc
from modules.training.profilers.TrainingProfiler import TrainingProfiler
class TrainValidEpochRunner:
    @tc.validate_types
    def __init__(self, train_epoch_runner: tc.RunnerLike, valid_epoch_runner: tc.RunnerLike) -> None:
        self.train_epoch_runner = train_epoch_runner
        self.valid_epoch_runner = valid_epoch_runner

    @tc.validate_types
    def run(self
            , train_loader: object
            , valid_loader: object
            , optimizer: object | None
            , profiler: TrainingProfiler | None = None) -> tc.Payload:

        train_epoch_result = self.train_epoch_runner.run(train_loader, optimizer, profiler) 
        valid_epoch_result = self.valid_epoch_runner.run(valid_loader, profiler) 

        result = {
            "train_loss": train_epoch_result["train_loss"],
            "train_acc": train_epoch_result["train_acc"],
            "valid_loss": valid_epoch_result["valid_loss"],
            "valid_acc": valid_epoch_result["valid_acc"],
            "checkpoint_loss": valid_epoch_result["valid_loss"],
        }
        if "gradient_diagnostics" in train_epoch_result:
            result["gradient_diagnostics"] = train_epoch_result["gradient_diagnostics"]
        if "optimizer_result" in train_epoch_result:
            result["optimizer_result"] = train_epoch_result["optimizer_result"]
        for source in (train_epoch_result, valid_epoch_result):
            if "measurement" in source:
                result["measurement"] = {**result.get("measurement", {}), **source["measurement"]}
        return result
