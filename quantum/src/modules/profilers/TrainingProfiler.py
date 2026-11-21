from __future__ import annotations

import json
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch

import modules.common.TypeContracts as tc

DEFAULT_CONFIGS = {
    "enable_memory": False,
    "enable_times": False,
    "enable_forward_count": False,
    "enable_backward_count": False,
    "enable_stage_recorder": False,
    "enable_gradient_diagnostics": False,
}

RUN_CONFIGS = {
    "enable_memory": True,
    "enable_times": True,
    "enable_forward_count": True,
    "enable_backward_count": True,
    "enable_stage_recorder": True,
}

COUNT_ONLY_STAGES = {"objective_evaluation"}


class TrainingProfiler:
    @tc.validate_types
    def __init__(
        self,
        configs: dict | None = None,
        *,
        device: tc.DeviceValue = "cpu",
        model: torch.nn.Module | None = None,
        sample_interval: float = 0.001,
    ) -> None:

        unknown = set(configs or {}) - set(DEFAULT_CONFIGS)
        if unknown:
            raise ValueError(f"Unknown profiler options: {sorted(unknown)}")

        self.configs = {**DEFAULT_CONFIGS, **(configs or {})}
        if any(type(v) is not bool for v in self.configs.values()):
            raise TypeError("All enable_* options must be bool")

        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive")

        self.device = torch.device(device)
        self.api = (
            getattr(torch, self.device.type, None)
            if self.device.type in ("cuda", "xpu")
            else None
        )
        self.sample_interval = sample_interval
        self.forward_count = Counter()
        self.backward_count = Counter()
        self.stage_count = Counter()
        self.rows = []
        self.elapsed_seconds_by_stage = Counter()
        self._active_memory_stages = 0
        self._current_stage = None
        self._handle = None
        self._process = None
        self._gradient_stats = {}

        if self.configs["enable_memory"]:
            import psutil

            self._process = psutil.Process()

        if self.configs["enable_forward_count"] and model is not None:
            self.attach(model)

    @tc.validate_types
    def attach(self, model: torch.nn.Module) -> TrainingProfiler:
        self.detach()
        if self.configs["enable_forward_count"]:
            self._handle = model.register_forward_pre_hook(self._on_forward)
        return self

    @tc.validate_types
    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @tc.validate_types
    def _on_forward(self, module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        self.forward_count[self._current_stage or "unscoped"] += 1

    @tc.validate_types
    def _sync(self) -> None:
        if self.api is not None:
            self.api.synchronize(self.device)

    @tc.validate_types
    def _snapshot(self) -> tc.Payload:
        mib = 1024**2
        rss = self._process.memory_info().rss / mib if self._process else None
        return {
            "rss_mib": rss,
            "device_allocated_mib": self.api.memory_allocated(self.device) / mib
            if self.api
            else None,
            "device_reserved_mib": self.api.memory_reserved(self.device) / mib
            if self.api
            else None,
        }

    @contextmanager
    @tc.validate_types
    def __call__(self, stage: str) -> tc.Iterator[None]:
        enabled = any(self.configs.values())

        if not enabled:
            yield
            return

        previous = self._current_stage
        self._current_stage = stage
        self.stage_count[stage] += 1

        if stage == "backward" and self.configs["enable_backward_count"]:
            self.backward_count[stage] += 1

        COUNT_ONLY = stage in COUNT_ONLY_STAGES
        memory = self.configs["enable_memory"] and not COUNT_ONLY
        timing = self.configs["enable_times"] and not COUNT_ONLY

        self._sync() if (memory or timing) else None
        before = self._snapshot() if memory else {}
        # Allocator peaks are global to the device: do not overlap profiled stages
        # or run a concurrent whole-training allocator-peak meter.

        reset_device_peak = (
            memory and self.api is not None and self._active_memory_stages == 0
        )
        if reset_device_peak:
            self.api.reset_peak_memory_stats(self.device)
        if memory:
            self._active_memory_stages += 1

        peak_rss = [before.get("rss_mib")]
        stop = threading.Event()
        worker = None

        if memory:

            @tc.validate_types
            def poll() -> None:
                while not stop.wait(self.sample_interval):
                    try:
                        value = self._process.memory_info().rss / (1024**2)
                        peak_rss[0] = max(peak_rss[0], value)
                    except Exception:
                        return

            worker = threading.Thread(target=poll, daemon=True)
            worker.start()

        start = time.perf_counter() if timing else None
        error = None

        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            try:
                self._sync() if (memory or timing) else None
                elapsed = time.perf_counter() - start if timing else None
                if elapsed is not None:
                    self.elapsed_seconds_by_stage[stage] += elapsed

                if worker is not None:
                    stop.set()
                    worker.join()

                after = self._snapshot() if memory else {}

                if memory:
                    peak_rss[0] = max(peak_rss[0], after["rss_mib"])

                if self.configs["enable_stage_recorder"]:
                    row = {"stage": stage, "elapsed_seconds": elapsed, "error": error}
                    if memory:
                        row.update(
                            {
                                "rss_before_mib": before["rss_mib"],
                                "rss_after_mib": after["rss_mib"],
                                "rss_peak_sampled_mib": peak_rss[0],
                                "device_allocated_before_mib": before[
                                    "device_allocated_mib"
                                ],
                                "device_allocated_after_mib": after[
                                    "device_allocated_mib"
                                ],
                                "device_reserved_before_mib": before[
                                    "device_reserved_mib"
                                ],
                                "device_reserved_after_mib": after[
                                    "device_reserved_mib"
                                ],
                                "device_allocated_peak_mib": (
                                    self.api.max_memory_allocated(self.device)
                                    / (1024**2)
                                    if self.api
                                    else None
                                ),
                                "device_reserved_peak_mib": (
                                    self.api.max_memory_reserved(self.device)
                                    / (1024**2)
                                    if self.api
                                    else None
                                ),
                            }
                        )
                    self.rows.append(row)
            finally:
                if memory:
                    self._active_memory_stages -= 1
                self._current_stage = previous

    @tc.validate_types
    def reset_gradient_epoch(self) -> None:
        """Discard previous epoch's online gradient statistics."""
        self._gradient_stats = {}

    @torch.no_grad()
    @tc.validate_types
    def record_hidden_gradients(self, model: torch.nn.Module) -> None:
        """Observe hidden Linear gradients between backward and optimizer.step."""
        if not self.configs["enable_gradient_diagnostics"]:
            return
        layers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
        ]
        for name, layer in layers[:-1]:
            grad = layer.weight.grad
            if grad is None:
                continue
            rms = float(grad.detach().float().square().mean().sqrt().item())
            stats = self._gradient_stats.setdefault(
                name,
                {
                    "count": 0,
                    "sum": 0.0,
                    "min": float("inf"),
                    "max": 0.0,
                },
            )
            stats["count"] += 1
            stats["sum"] += rms
            stats["min"] = min(stats["min"], rms)
            stats["max"] = max(stats["max"], rms)

    @tc.validate_types
    def gradient_epoch_summary(self) -> tc.Payload:
        """Return epoch statistics; ratio compares first/last hidden Linear weights."""
        layers = {}
        for name, stats in self._gradient_stats.items():
            if stats["count"]:
                layers[name] = {
                    "mean_rms": stats["sum"] / stats["count"],
                    "min_rms": stats["min"],
                    "max_rms": stats["max"],
                    "batches_observed": stats["count"],
                }
        values = list(layers.values())
        ratio = (
            values[0]["mean_rms"] / values[-1]["mean_rms"]
            if len(values) >= 2 and values[-1]["mean_rms"] != 0
            else None
        )
        return {
            "hidden_layer_gradient_rms": layers,
            "first_last_hidden_gradient_ratio": ratio,
            "gradient_scope": "epoch_batch_mean" if layers else "unavailable",
        }

    @tc.validate_types
    def summary(self) -> tc.Payload:
        return {
            "configs": dict(self.configs),
            "stage_count": dict(self.stage_count),
            "forward_count": dict(self.forward_count),
            "backward_count": dict(self.backward_count),
            "stages": list(self.rows),
            "elapsed_seconds_by_stage": dict(self.elapsed_seconds_by_stage),
        }

    @tc.validate_types
    def save_records(self, output_path: tc.PathValue) -> Path:
        """Persist stage measurements and counters as a UTF-8 JSON report.

        Writes atomically so a partial file is not mistaken for a valid report.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            temporary.write_text(
                json.dumps(self.summary(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    @tc.validate_types
    def epoch_summary(self) -> tc.Payload:
        """Summarize forward counts by training, monitoring, and validation role."""
        counts = self.forward_count
        optimizer = sum(
            value
            for stage, value in counts.items()
            if stage not in ("training_monitor", "validation", "unscoped")
        )
        result = {
            "optimizer": optimizer,
            "training_monitor": counts["training_monitor"],
            "validation": counts["validation"],
        }
        return {
            "forward_counts": result,
            "training_forward_total": optimizer + result["training_monitor"],
        }
