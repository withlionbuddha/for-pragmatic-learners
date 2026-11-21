from __future__ import annotations

import modules.common.TypeContracts as tc
from modules.calculators.AccuracyCalculator import AccuracyCalculator
from modules.calculators.LossCalculator import LossCalculator
from modules.callbacks.EarlyStopping import EarlyStopping
from modules.training.logging.EpochLogger import EpochLogger
from modules.training.runners.QuantumBatchRunner import QuantumBatchRunner
from modules.training.trainers.BaseTrainer import BaseTrainer


class BaseTrainTrainer(BaseTrainer):
    """
    학습 Trainer의 공통 Epoch 반복, 저장 및 종료 생명주기.
    """

    @tc.validate_types
    def __init__(self, model: tc.nn.Module | tc.DeviceModel) -> None:
        super().__init__(model)

    @tc.validate_types
    def _validate(self) -> None:
        super()._validate()
        if self.compile_configs.get_optimizer(self.model) is None:
            raise RuntimeError("optimizer가 없습니다.")
        if getattr(self.data_loaders, "train", None) is None:
            raise RuntimeError("train data_loader가 없습니다.")
        if self.saver_manager is None:
            raise RuntimeError("saver가 없습니다.")

    @tc.validate_types
    def train(
        self, epochs: int = 100, patience: int | None = 5, log_interval: int = 5
    ) -> tc.PathManagerLike | None:
        self._validate()
        path_manager = None

        try:
            self._attach_hooks()

            epoch_runner = self.create_runner()
            early_stopping = EarlyStopping(patience) if patience is not None else None
            logger = EpochLogger()

            self.saver_manager.save_start(
                model=self.model,
                compile_configs=self.compile_configs,
                epochs=epochs,
                patience=patience,
                log_interval=log_interval,
            )

            for epoch in range(1, epochs + 1):
                epoch_result = self.run(epoch_runner)
                logger.print_epoch_result(epoch, log_interval, epoch_result)

                is_improved = self.saver_manager.save_epoch(
                    model=self.model,
                    compile_configs=self.compile_configs,
                    epoch=epoch,
                    epoch_result=epoch_result,
                )

                if early_stopping is not None:
                    early_stopping.update(is_improved, epoch_result["checkpoint_loss"])
                if early_stopping is not None and early_stopping.is_early_stopping():
                    break

            self.saver_manager.save_end()
        except Exception as exc:
            try:
                self.saver_manager.save_error(exc)
            except Exception as save_exc:
                print(f"Failed to save training error: {save_exc}")
            raise
        finally:
            try:
                path_manager = self.saver_manager.save_finish()
                profiler = self.get_profiler()
                if profiler is not None and hasattr(profiler, "save_records"):
                    profiler.save_records(
                        path_manager.get_run_dir() / "training_profiler.json"
                    )
            finally:
                self._detach_hooks()
        return path_manager

    @tc.validate_types
    def create_batch_runner(self) -> tc.BatchRunnerLike:
        return QuantumBatchRunner(
            model=self.model,
            loss_calculator=LossCalculator(self.compile_configs.get_criterion()),
            accuracy_calculator=AccuracyCalculator(),
        )

    @tc.validate_types
    def get_profiler(self) -> object | None:
        return self.training_profiler
