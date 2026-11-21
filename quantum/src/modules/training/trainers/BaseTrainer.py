from __future__ import annotations

from abc import ABC, abstractmethod

import modules.common.TypeContracts as tc


class BaseTrainer(ABC):
    """학습 및 검증 Trainer의 공통 설정과 Runner 인터페이스."""

    @tc.validate_types
    def __init__(self, model: tc.nn.Module | tc.DeviceModel) -> None:
        self.model = model
        self.compile_configs = None
        self.data_loaders = None
        self.device_manager = None
        self.saver_manager = None
        self.training_profiler = None
        # Compatibility with trainers that still access self.profiler.
        self.profiler = None

    @tc.validate_types
    def set_compile_configs(
        self, compile_configs: tc.CriterionProvider | None = None
    ) -> None:
        self.compile_configs = compile_configs

    @tc.validate_types
    def set_data_loaders(self, data_loaders: tc.TrainLoadersLike | None = None) -> None:
        self.data_loaders = data_loaders

    @tc.validate_types
    def set_device_manager(
        self, device_manager: tc.DeviceManagerLike | None = None
    ) -> None:
        self.device_manager = device_manager

    @tc.validate_types
    def set_saver_manager(self, saver_manager: tc.SaverLike | None = None) -> None:
        self.saver_manager = saver_manager

    @tc.validate_types
    def set_training_profiler(self, training_profiler: object | None) -> None:
        self.training_profiler = training_profiler
        self.profiler = training_profiler

    @tc.validate_types
    def _validate(self) -> None:
        if self.compile_configs is None:
            raise RuntimeError("compile_configs가 없습니다.")
        if self.compile_configs.get_criterion() is None:
            raise RuntimeError("criterion이 없습니다.")
        if self.data_loaders is None:
            raise RuntimeError("data_loaders가 없습니다.")

    @tc.validate_types
    def batch_runner(self) -> tc.BatchRunnerLike:
        batch_runner = self.create_batch_runner()
        return batch_runner

    @tc.validate_types
    def _attach_hooks(self) -> None:
        if self.device_manager is not None:
            self.device_manager.attach_model_hooks(self.model)
        if self.training_profiler is not None:
            self.training_profiler.attach(self.model)

    @tc.validate_types
    def _detach_hooks(self) -> None:
        if self.device_manager is not None:
            self.device_manager.detach_model_hooks()
        if self.training_profiler is not None:
            self.training_profiler.detach()

    @abstractmethod
    @tc.validate_types
    def create_runner(self) -> tc.RunnerLike:
        """실행 방식에 맞는 EpochRunner를 생성한다."""

    @abstractmethod
    @tc.validate_types
    def create_batch_runner(self) -> tc.BatchRunnerLike:
        """실행 방식에 맞는 BatchRunner를 생성한다."""

    @abstractmethod
    @tc.validate_types
    def run(self, epoch_runner: tc.RunnerLike) -> tc.Payload:
        """EpochRunner를 실행하고 결과를 반환한다."""
