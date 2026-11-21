from __future__ import annotations
import modules.common.TypeContracts as tc
from .FilePathManager import FilePathManager
from .HistorySaver import HistorySaver
from .MetadataSaver import MetadataSaver
from modules.checkpointing.CheckpointSaver import CheckpointSaver


class SaverManager:
    """학습 실행의 History, Checkpoint, Metadata 저장을 총괄한다."""

    @tc.validate_types
    def __init__(self, enable_checkpointSaver: bool = False,
                 enable_historySaver: bool = True,
                 enable_metadataSaver: bool = False) -> None:
        """실행별 저장 경로와 각 산출물 저장 객체를 초기화한다."""
        self.path_manager = FilePathManager()
        self.checkpoint_saver = CheckpointSaver(enable_checkpointSaver, self.path_manager)
        self.history_saver = HistorySaver(enable_historySaver, self.path_manager)
        self.metadata_saver = MetadataSaver(enable_metadataSaver, self.path_manager)

    @tc.validate_types
    def save_start(self, model: object, compile_configs: tc.CompileLike | None, epochs: int, patience: int | None, log_interval: int) -> None:
        """학습 설정을 수집하여 시작 상태의 Metadata를 기록한다.

        Args:
            model: 학습 대상 모델.
            compile_configs: Optimizer와 Criterion을 제공하는 설정 객체.
            epochs: 최대 학습 Epoch 수.
            patience: Early Stopping 대기 Epoch 수. None이면 비활성화.
            log_interval: Epoch 로그 출력 간격.

        Raises:
            RuntimeError: 설정 조회 또는 Metadata 기록에 실패한 경우.
        """
        if not self.metadata_saver.enable_saver:
            return
        try:
            optimizer = compile_configs.get_optimizer(model)
            criterion = compile_configs.get_criterion()
            config = {
                'model': type(model).__name__,
                'optimizer': type(optimizer).__name__,
                'criterion': self.metadata_saver.criterion_config(criterion),
                'learning_rate': [group.get('lr') for group in getattr(optimizer, 'param_groups', [])],
                'epochs': epochs,
                'patience': patience,
                'log_interval': log_interval,
            }
            self.metadata_saver.start(config)
        except Exception as exc:
            raise RuntimeError('SaverManager.save_start() 실패') from exc

    @tc.validate_types
    def save_epoch(self, model: object, compile_configs: tc.CompileLike | None, epoch: int, epoch_result: tc.Payload) -> bool:
        """Epoch 결과를 CSV History에 기록하고 최적 Checkpoint를 저장한다.

        Args:
            model: 저장할 모델.
            compile_configs: Optimizer를 제공하는 설정 객체.
            epoch: 현재 Epoch 번호(1부터 시작).
            epoch_result: 학습·검증 지표 및 checkpoint_loss를 담은 딕셔너리.

        Returns:
            bool: 기존 최저 Loss보다 개선되어 Checkpoint를 저장했으면 True.
        """
        # History is retained in memory for reporting even when CSV output is disabled.
        self.history_saver.add(epoch, epoch_result)
        if self.history_saver.enable_saver:
            self.history_saver.save()

        if self.checkpoint_saver.enable_saver:
            return self.checkpoint_saver.save(
                model=model,
                optimizer=compile_configs.get_optimizer(model),
                epoch=epoch,
                current_loss=epoch_result['checkpoint_loss'],
            )
        return False

    @tc.validate_types
    def _create_summary(self) -> tc.Payload:
        """누적 History와 최적 Checkpoint 정보를 Metadata 요약으로 구성한다.

        Checkpoint 경로는 JSON 직렬화를 위해 문자열로 변환한다.
        저장된 Checkpoint가 없으면 경로와 최적 정확도는 None일 수 있다.

        Returns:
            dict: 완료 Epoch 수, 최적 지표, Checkpoint·History 경로.
        """
        history = self.history_saver.get_history()

        if self.checkpoint_saver.enable_saver:
            best_epoch = self.checkpoint_saver.get_best_epoch()
            best_loss = self.checkpoint_saver.get_best_loss()
            best_path = self.checkpoint_saver.get_best_checkpoint_path()
        else:
            best_epoch = best_loss = best_path = None

        best_row = next(
            (row for row in history if row["epoch"] == best_epoch),
            None,
        )

        return {
            "completed_epochs": len(history),
            "best_epoch": best_epoch,
            "best_valid_loss": best_loss,
            "best_valid_acc": (
                best_row.get("valid_acc")
                if best_row is not None
                else None
            ),
            "best_checkpoint_path": (
                str(best_path)
                if best_path is not None
                else None
            ),
            "history_path": str(
                self.path_manager.get_history_path()
            ),
        }

    @tc.validate_types
    def save_end(self) -> None:
        """정상 종료 시 학습 요약을 포함한 완료 상태의 Metadata를 기록한다."""
        if self.metadata_saver.enable_saver:
            self.metadata_saver.finish(self._create_summary())

    @tc.validate_types
    def save_error(self, error: BaseException) -> None:
        """실패 원인과 가능한 범위의 학습 요약을 Metadata에 기록한다.

        Args:
            error: 학습 중 발생한 예외 객체.

        요약 생성에 실패해도 오류 정보는 기록하도록 시도한다.
        """
        if not self.metadata_saver.enable_saver:
            return
        try:
            summary = self._create_summary()
        except Exception:
            summary = None
        self.metadata_saver.error(error=error, summary=summary)

    @tc.validate_types
    def save_finish(self) -> tc.PathManagerLike:
        """학습 종료 시 공통 후처리를 위한 확장 지점이다.

        저장 경로 관리 객체를 반환한다.
        """
        return self.path_manager

