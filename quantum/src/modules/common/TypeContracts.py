"""Runtime checking for repository-authored function signatures.

Structural protocols preserve supported adapters/test doubles. Tensor shape and
value constraints remain the responsibility of the individual implementation.
"""

from __future__ import annotations

import collections.abc as cabc
import functools
import inspect
import os
import types
import typing
from typing import (
    Any,
    Callable,
    Iterator,
    Protocol,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)
from typing import (
    Iterable as Iterable,
)
from typing import (
    KeysView as KeysView,
)

import torch
from torch import nn

Payload = dict[str, Any]
PathValue = str | os.PathLike[str]
DeviceValue = str | torch.device | None
Metric = tuple[float, int, int]


class ModelLike(Protocol):
    def parameters(self) -> Iterator[nn.Parameter]: ...
    def modules(self) -> Iterator[nn.Module]: ...


class DeviceModel(Protocol):
    def get_device(self) -> str | torch.device: ...
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class AccuracyLike(Protocol):
    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> Metric: ...
    def calculate(self, correct: int, total: int) -> float: ...
    def get_accuracy_average(self, total_correct: int, total_count: int) -> float: ...


class LossLike(Protocol):
    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...
    def get_loss_average(self, sum_loss: float, num_batches: int) -> float: ...


class RunnerLike(Protocol):
    def run(self, *args: object, **kwargs: object) -> Payload: ...


class BatchRunnerLike(Protocol):
    def run_train_batch(self, *args: object) -> Payload: ...
    def run_valid_batch(self, *args: object) -> Payload: ...
    def get_loss_calculator(self) -> object: ...
    def get_accuracy_calculator(self) -> object: ...


class CompileLike(Protocol):
    def get_optimizer(self, model: object) -> object: ...
    def get_criterion(self) -> object: ...


class PathManagerLike(Protocol):
    def get_run_dir(self) -> PathValue: ...


class RegistryLike(Protocol):
    def get(self, module: nn.Module) -> list[object]: ...


class IndexerLike(Protocol):
    def get_neuron_count(self, module: nn.Module) -> int: ...


class SaverLike(Protocol):
    def save_start(self, **kwargs: object) -> None: ...
    def save_epoch(self, **kwargs: object) -> bool: ...
    def save_end(self) -> None: ...
    def save_finish(self) -> PathManagerLike: ...


F = TypeVar("F", bound=Callable[..., Any])


def matches_type(value: object, expected: object) -> bool:
    """Check collections completely without consuming iterators or moving tensors."""
    if expected in (Any, object, inspect.Signature.empty):
        return True
    if expected is None or expected is type(None):
        return value is None
    origin = get_origin(expected)
    args = get_args(expected)
    if origin in (typing.Union, types.UnionType):
        return any(matches_type(value, item) for item in args)
    if origin is typing.Annotated:
        return matches_type(value, args[0])
    if origin is typing.Literal:
        return any(type(value) is type(item) and value == item for item in args)
    if isinstance(expected, TypeVar):
        if expected.__constraints__:
            return any(matches_type(value, item) for item in expected.__constraints__)
        return matches_type(value, expected.__bound__) if expected.__bound__ else True
    if origin is type:
        return isinstance(value, type) and (
            not args or args[0] is Any or issubclass(value, args[0])
        )
    if origin in (list, set, frozenset):
        return isinstance(value, origin) and (
            not args or all(matches_type(item, args[0]) for item in value)
        )
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if len(args) == 2 and args[1] is Ellipsis:
            return all(matches_type(item, args[0]) for item in value)
        return not args or (
            len(value) == len(args)
            and all(matches_type(item, hint) for item, hint in zip(value, args))
        )
    if origin in (dict, cabc.Mapping, cabc.MutableMapping):
        return isinstance(value, origin) and (
            not args
            or all(
                matches_type(k, args[0]) and matches_type(v, args[1])
                for k, v in value.items()
            )
        )
    if origin is cabc.Callable:
        return callable(value)
    if origin in (cabc.Sequence, cabc.MutableSequence, cabc.Set, cabc.Collection):
        return isinstance(value, origin) and (
            not args or all(matches_type(item, args[0]) for item in value)
        )
    if origin is not None:
        # Iterator values are deliberately not consumed by a type check.
        return isinstance(value, origin)
    if getattr(expected, "_is_protocol", False):
        required = set()
        methods = set()
        for cls in expected.__mro__:
            if cls in (Protocol, typing.Generic, object):
                continue
            required.update(getattr(cls, "__annotations__", {}))
            methods.update(
                name
                for name, member in vars(cls).items()
                if callable(member) and (not name.startswith("_") or name == "__call__")
            )
        return all(hasattr(value, name) for name in required) and all(
            callable(getattr(value, name, None)) for name in methods
        )
    if expected is float:
        return isinstance(value, (float, int)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(expected, type):
        return isinstance(value, expected)
    raise TypeError(f"Unsupported type annotation: {expected!r}")


def validate_types(function: F) -> F:
    """Validate annotated arguments before execution and the return afterwards."""
    signature = inspect.signature(function)
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if name not in ("self", "cls")
        and parameter.annotation is inspect.Signature.empty
    ]
    if missing or signature.return_annotation is inspect.Signature.empty:
        raise TypeError(
            f"{function.__qualname__}: input and return annotations are required"
        )
    hints: dict[str, Any] | None = None

    @functools.wraps(function)
    def checked(*args: Any, **kwargs: Any) -> Any:
        nonlocal hints
        if hints is None:
            hints = get_type_hints(function)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            if name in ("self", "cls"):
                continue
            expected = hints.get(name, inspect.Signature.empty)
            parameter = signature.parameters[name]
            values = (
                value
                if parameter.kind is inspect.Parameter.VAR_POSITIONAL
                else (
                    value.values()
                    if parameter.kind is inspect.Parameter.VAR_KEYWORD
                    else (value,)
                )
            )
            if not all(matches_type(item, expected) for item in values):
                raise TypeError(
                    f"{function.__qualname__}: {name} must match "
                    f"{type_name(expected)}; "
                    f"got {type(value).__name__}"
                )
        result = function(*args, **kwargs)
        expected = hints.get("return", inspect.Signature.empty)
        if not matches_type(result, expected):
            raise TypeError(
                f"{function.__qualname__}: return must match {type_name(expected)}; "
                f"got {type(result).__name__}"
            )
        return result

    checked.__type_contract__ = True
    return typing.cast(F, checked)


class OptimizerLike(Protocol):
    def zero_grad(self, *args: object, **kwargs: object) -> None: ...
    def step(self, *args: object, **kwargs: object) -> object: ...


class QuantumOptimizerLike(Protocol):
    def prepare(self, model: object) -> None: ...
    def optimize(self, evaluate: Callable[[], float]) -> Payload: ...


class DataLoadersLike(Protocol):
    train: object
    valid: object


class DeviceManagerLike(Protocol):
    def attach_model_hooks(self, model: object) -> None: ...
    def detach_model_hooks(self) -> None: ...


class AnalyzerLike(Protocol):
    def report(self) -> Payload: ...
    def bytes_to_megabytes(self, value: int) -> float: ...


class CriterionProvider(Protocol):
    def get_criterion(self) -> object: ...


class TrainModeLike(Protocol):
    def train(self) -> object: ...


class EvalModeLike(Protocol):
    def eval(self) -> object: ...


class ITrainBatch(Protocol):
    def run_train_batch(
        self, x_batch: object, y_batch: torch.Tensor, optimizer: object
    ) -> dict[str, Any]: ...
    def get_loss_calculator(self) -> Callable[..., torch.Tensor]: ...
    def get_accuracy_calculator(self) -> Callable[..., tuple[float, int, int]]: ...


class IValidBatch(Protocol):
    def run_valid_batch(
        self, x_batch: object, y_batch: torch.Tensor
    ) -> dict[str, Any]: ...
    def get_loss_calculator(self) -> Callable[..., torch.Tensor]: ...
    def get_accuracy_calculator(self) -> Callable[..., tuple[float, int, int]]: ...


# Backward-compatible names for the shared batch interfaces.
TrainBatchLike = ITrainBatch
ValidBatchLike = IValidBatch


class AccuracyAverageLike(Protocol):
    def get_accuracy_average(self, correct: int, total: int) -> float: ...


class LossAverageLike(Protocol):
    def get_loss_average(self, total: float, count: int) -> float: ...


class NeuronInspectorLike(Protocol):
    def supports(self, module: object) -> bool: ...
    def inspect_neuron(self, module: nn.Module, neuron_index: int) -> object: ...


class SupportsModule(Protocol):
    def supports(self, module: object) -> bool: ...


class TrainLoadersLike(Protocol):
    train: object


def type_name(expected: object) -> str:
    if isinstance(expected, type) and issubclass(expected, nn.Module):
        return f"nn.{expected.__name__}"
    return str(expected)


class ContainerResolverLike(Protocol):
    def resolve(self, model: nn.Module) -> list[object]: ...
    def get_layer_count(self) -> int: ...
    def get_neuron_count(self, layer_index: int) -> int: ...


class PrimaryResolverLike(Protocol):
    def resolve(self, module: nn.Module, name: str) -> object: ...
    def get_neuron_count(self, module: nn.Module) -> int: ...


C = TypeVar("C", bound=type)


def validate_dataclass(cls: C) -> C:
    """Validate generated dataclass constructor arguments and initialized fields."""
    import dataclasses

    original = cls.__init__
    signature = inspect.signature(original)
    hints: dict[str, Any] | None = None

    @functools.wraps(original)
    def initialize(self: object, *args: Any, **kwargs: Any) -> None:
        nonlocal hints
        if hints is None:
            hints = get_type_hints(cls)
        bound = signature.bind(self, *args, **kwargs)
        # Unspecified default_factory parameters are checked after construction;
        # the dataclass's internal factory sentinel is not a user input value.
        for name, value in bound.arguments.items():
            if name == "self":
                continue
            expected = hints[name]
            if not matches_type(value, expected):
                raise TypeError(
                    f"{cls.__name__}: {name} must match {type_name(expected)}"
                )
        original(self, *args, **kwargs)
        for field in dataclasses.fields(cls):
            if not matches_type(getattr(self, field.name), hints[field.name]):
                raise TypeError(
                    f"{cls.__name__}: initialized field {field.name} violates its type"
                )

    initialize.__type_contract__ = True
    cls.__init__ = initialize
    return cls
