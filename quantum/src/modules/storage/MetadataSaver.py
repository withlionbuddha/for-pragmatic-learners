"""Training metadata persisted as JSON."""
from __future__ import annotations
import modules.common.TypeContracts as tc
import json
import inspect

import torch
from datetime import datetime, timezone


class MetadataSaver:
    @tc.validate_types
    def __init__(self, enable_saver: bool, path_manager: tc.PathManagerLike) -> None:
        self.path_manager = path_manager
        self.metadata = {}
        self.enable_saver = enable_saver

    @staticmethod
    @tc.validate_types
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @tc.validate_types
    def _save(self) -> str:
        if not self.enable_saver:
            return ""
        path = self.path_manager.get_metadata_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as stream:
            json.dump(self.metadata, stream, ensure_ascii=False, indent=4)
        return str(path)


    @staticmethod
    @tc.validate_types
    def _serialize_value(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return {
                "__type__": "tensor",
                "dtype": str(value.dtype).removeprefix("torch."),
                "data": value.detach().cpu().tolist(),
            }
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [MetadataSaver._serialize_value(item) for item in value]
        raise ValueError(f"직렬화할 수 없는 Criterion 인자: {type(value).__name__}")

    @staticmethod
    @tc.validate_types
    def criterion_config(criterion: tc.Callable[..., tc.torch.Tensor] | None) -> tc.Payload:
        kwargs = {}
        signature = inspect.signature(type(criterion).__init__)
        for name, parameter in signature.parameters.items():
            if name == "self" or parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ) or name in ("size_average", "reduce"):
                continue
            if hasattr(criterion, name):
                value = getattr(criterion, name)
            elif parameter.default is not inspect.Parameter.empty:
                value = parameter.default
            else:
                raise ValueError(f"Criterion 인자를 확인할 수 없습니다: {name}")
            kwargs[name] = MetadataSaver._serialize_value(value)
        return {"class_name": type(criterion).__name__, "kwargs": kwargs}

    @tc.validate_types
    def start(self, config: tc.Payload) -> str:
        self.metadata = {
            'status': 'running',
            'started_at': self._now(),
            'configuration': config,
        }
        return self._save()

    @tc.validate_types
    def finish(self, summary: tc.Payload | None) -> str:
        self.metadata['status'] = 'completed'
        self.metadata['finished_at'] = self._now()
        self.metadata['summary'] = summary
        self.metadata.pop('error', None)
        return self._save()

    @tc.validate_types
    def error(self, error: BaseException, summary: tc.Payload | None=None) -> str:
        self.metadata['status'] = 'failed'
        self.metadata['finished_at'] = self._now()
        if summary is not None:
            self.metadata['summary'] = summary
        self.metadata['error'] = {
            'type': type(error).__name__,
            'message': str(error),
        }
        return self._save()

