from __future__ import annotations
import modules.common.TypeContracts as tc
import torch


class CheckpointSaver:
    @tc.validate_types
    def __init__(self, enable_saver:bool, path_manager: tc.PathManagerLike) -> None:
        self.path_manager = path_manager
        self.best_loss = float("inf")  # 양의 무한대
        self.best_epoch = None
        self.best_checkpoint_path = None
        self.enable_saver = enable_saver

    @tc.validate_types
    def _is_improved(self, current_loss: float) -> bool:
        """현재 Loss가 기존 최적 Loss보다 낮은지 판단한다."""
        return current_loss < self.best_loss

    @tc.validate_types
    def _get_model_class(self, model: tc.ModelLike | None) -> str:
        """모델 클래스의 모듈 경로를 반환한다."""
        return f"{model.__class__.__module__}." f"{model.__class__.__qualname__}"

    @tc.validate_types
    def _get_model_config(self, model: tc.ModelLike | None) -> tc.Payload | None:
        """모델 생성에 필요한 설정을 반환한다."""
        get_config = getattr(model, "get_config", None)
        return get_config() if callable(get_config) else None

    @tc.validate_types
    def _get_optimizer_class(self, optimizer: tc.OptimizerLike | None) -> str | None:
        """Optimizer 클래스의 모듈 경로를 반환한다."""
        if optimizer is None:
            return None

        return (
            f"{optimizer.__class__.__module__}." f"{optimizer.__class__.__qualname__}"
        )

    @tc.validate_types
    def _get_optimizer_config(self, optimizer: tc.OptimizerLike | None) -> tc.Payload | None:
        """Optimizer 생성에 필요한 설정을 반환한다."""
        if optimizer is None:
            return None

        return optimizer.defaults.copy()

    @tc.validate_types
    def _get_optimizer_state_dict(self, optimizer: tc.OptimizerLike | None) -> tc.Payload | None:
        """Optimizer의 내부 학습 상태를 반환한다."""
        if optimizer is None:
            return None

        return optimizer.state_dict()

    @staticmethod
    @tc.validate_types
    def _get_optimizer_param_names(model: tc.ModelLike | None, optimizer: tc.OptimizerLike | None) -> list[list[str]] | None:
        if optimizer is None:
            return None
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        try:
            return [
                [names[id(parameter)] for parameter in group["params"]]
                for group in optimizer.param_groups
            ]
        except KeyError as exc:
            raise ValueError(
                "Optimizer contains a parameter outside the model"
            ) from exc

    @tc.validate_types
    def _build_checkpoint(self, model: tc.ModelLike | None, optimizer: tc.OptimizerLike | None, epoch: int, current_loss: float) -> tc.Payload:
        """모델 및 학습 상태를 Checkpoint 데이터로 구성한다."""

        checkpoint = {
            "epoch": epoch,
            "loss": current_loss,
            # Model
            "model_class": self._get_model_class(model),
            "model_config": self._get_model_config(model),
            "model_state_dict": model.state_dict(),
            # Optimizer
            "optimizer_class": self._get_optimizer_class(optimizer),
            "optimizer_config": self._get_optimizer_config(optimizer),
            "optimizer_state_dict": self._get_optimizer_state_dict(optimizer),
            "optimizer_param_names": self._get_optimizer_param_names(model, optimizer),
        }

        return checkpoint

    @tc.validate_types
    def save(self, model: tc.ModelLike | None, optimizer: tc.OptimizerLike | None, epoch: int, current_loss: float) -> bool:
        if not self._is_improved(current_loss):
            return False

        checkpoint = self._build_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            current_loss=current_loss,
        )
        save_path = self.path_manager.get_checkpoint_path(
            loss=current_loss,
            epoch=epoch,
        )

        torch.save(checkpoint, save_path)

        self.best_loss = current_loss
        self.best_epoch = epoch
        self.best_checkpoint_path = save_path

        print(f"{current_loss} 개선됨. best model 저장: {save_path}")
        return True

    @tc.validate_types
    def get_best_loss(self) -> float:
        return self.best_loss

    @tc.validate_types
    def get_best_epoch(self) -> int | None:
        return self.best_epoch

    @tc.validate_types
    def get_best_checkpoint_path(self) -> tc.PathValue | None:
        return self.best_checkpoint_path
