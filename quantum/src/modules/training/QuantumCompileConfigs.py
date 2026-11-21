from __future__ import annotations

from torch import nn

import modules.common.TypeContracts as tc


class QuantumCompileConfigs:
    """Configuration for gradient-free quantum optimizers.

    Unlike CompileConfigs, the optimizer is constructed without model parameters.
    QuantumBatchRunner passes the model to optimizer.prepare(model) before
    optimization.
    """

    @tc.validate_types
    def __init__(
        self,
        optimizer_class: tc.Callable[..., tc.QuantumOptimizerLike],
        optimizer_kwargs: dict,
        criterion: nn.Module,
        profile_configs: dict,
    ) -> None:
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion = criterion
        self.optimizer: tc.QuantumOptimizerLike | None = None
        self.profile_configs = profile_configs

    @tc.validate_types
    def get_optimizer(self, model: tc.nn.Module | None) -> tc.QuantumOptimizerLike:
        if self.optimizer is None:
            if model is None:
                raise RuntimeError("Optimizer 생성에 필요한 model이 없습니다.")
            self.optimizer = self.optimizer_class(**self.optimizer_kwargs)
        return self.optimizer

    @tc.validate_types
    def get_criterion(self) -> tc.Callable[..., tc.torch.Tensor] | None:
        return self.criterion

    @tc.validate_types
    def get_profile_configs(self) -> dict:
        return self.profile_configs
