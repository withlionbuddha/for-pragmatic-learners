"""Run one theta-group experiment in an isolated Python process."""
import json
import math
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from modules.optimizers.AdaptiveQuantumDirectionCOBYLA import AdaptiveQuantumDirectionCOBYLA
from modules.profilers.TrainingProfiler import TrainingProfiler

ROUNDS = 8
EVALS_PER_ROUND = 12
MAX_PARAMS = 20000
SCALES = {"S": (8, 1), "M": (64, 2), "L": (256, 6), "XL": (1024, 24)}
torch.set_num_threads(min(4, torch.get_num_threads()))

class Residual(nn.Module):
    def __init__(self, w, d):
        super().__init__()
        self.inp = nn.Linear(8, w)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Linear(w, w))
            for _ in range(d)
        ])
        self.out = nn.Linear(w, 1)

    def forward(self, x):
        z = self.inp(x)
        for block in self.blocks:
            z = z + block(z) / math.sqrt(z.shape[-1])
        return self.out(z)


class Transformer(nn.Module):
    def __init__(self, w, d):
        super().__init__()
        self.embed = nn.Linear(1, w)
        layer = nn.TransformerEncoderLayer(
            d_model=w, nhead=2, dim_feedforward=4 * w,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=d, enable_nested_tensor=False,
        )
        self.out = nn.Linear(w, 1)

    def forward(self, x):
        return self.out(self.encoder(self.embed(x.unsqueeze(-1))).mean(1))


def model_for(family, scale):
    w, d = SCALES[scale]
    if family == "linear":
        return nn.Linear(8, 1)
    if family == "mlp":
        layers = [nn.Linear(8, w), nn.Tanh()]
        for _ in range(d - 1):
            layers.extend([nn.Linear(w, w), nn.Tanh()])
        return nn.Sequential(*layers, nn.Linear(w, 1))
    if family == "residual":
        return Residual(w, d)
    if family == "transformer":
        return Transformer(w, d)
    raise ValueError(family)


def nparams(model):
    return sum(p.numel() for p in model.parameters())


def dataset(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(256, 8)).astype("float32")
    y = (
        0.7 * np.sin(x[:, 0]) + 0.4 * x[:, 1] ** 2
        - 0.5 * x[:, 2] * x[:, 3] + 0.2 * x[:, 4]
        + rng.normal(0, 0.05, 256)
    ).astype("float32")
    order = rng.permutation(256)
    return [
        torch.from_numpy(v)
        for v in (
            x[order[:192]], y[order[:192], None],
            x[order[192:]], y[order[192:], None],
        )
    ]


class DirectionRecordingAdaptiveCOBYLA(AdaptiveQuantumDirectionCOBYLA):
    """Record sampled signs before COBYLA changes the direction scale."""
    def create_directions(self, count):
        directions = super().create_directions(count)
        self.last_sampled_signs = np.concatenate([
            np.sign(direction.detach().cpu().reshape(-1).numpy()).astype(np.int8)
            for direction in directions
        ])
        return directions


def run_one(family, scale, seed, group_size):
    profiler = TrainingProfiler(configs={"enable_memory": True, "enable_stage_recorder": True}, device="cpu")
    with profiler("experiment"):
        torch.manual_seed(seed + 1000)
        model = model_for(family, scale)
        if nparams(model) > MAX_PARAMS:
            raise ValueError("parameter limit exceeded")
        xt, yt, xv, yv = dataset(seed)
        optimizer = DirectionRecordingAdaptiveCOBYLA(
            steps=1, direction_cobyla_evaluations=EVALS_PER_ROUND,
            adaptive_learning_rate=0.1, theta_group_size=group_size,
            cobyla_radius=0.1, seed=seed,
        )
        optimizer.prepare(model)
    calls = 0

    def objective():
        nonlocal calls
        calls += 1
        with torch.no_grad():
            return float(nn.functional.mse_loss(model(xt), yt))

    def mse(x, y):
        with torch.no_grad():
            return float(nn.functional.mse_loss(model(x), y))

    def theta_diagnostics(round_index):
        theta = np.asarray(optimizer.sampler.theta, dtype=float)
        probabilities = optimizer.sampler.probabilities_plus
        return {
            "round": round_index,
            "mean_abs_theta_shift": float(np.mean(np.abs(theta - np.pi / 2))),
            "max_abs_theta_shift": float(np.max(np.abs(theta - np.pi / 2))),
            "std_probability_plus": float(np.std(probabilities)),
            "mean_probability_plus": float(np.mean(probabilities)),
        }

    history = [{"round": 0, "train_mse": mse(xt, yt), "val_mse": mse(xv, yv)}]
    theta_history = [theta_diagnostics(0)]
    direction_history = []
    start = time.perf_counter()
    with profiler("experiment_optimization"):
        for round_index in range(1, ROUNDS + 1):
            before = calls
            info = optimizer.optimize(objective)
            assert calls - before == info["optimization_evaluations"]
            assert calls - before <= EVALS_PER_ROUND
            direction_history.append({
                "round": round_index,
                "signs": optimizer.last_sampled_signs.astype(int).tolist(),
            })
            theta_history.append(theta_diagnostics(round_index))
            history.append({
                "round": round_index, "train_mse": mse(xt, yt),
                "val_mse": mse(xv, yv),
            })
    memory = profiler.summary()["stages"][0]
    return {
        "family": family, "scale": scale, "seed": seed,
        "theta_group_size": group_size,
        "label": "shared" if group_size is None else str(group_size),
        "parameters": nparams(model),
        "theta_count": len(optimizer.sampler.theta),
        "objective_calls": calls,
        "theta_bytes": int(optimizer.sampler.theta.nbytes),
        "rss_before_mib": memory["rss_before_mib"],
        "rss_after_mib": memory["rss_after_mib"],
        "rss_peak_sampled_mib": memory["rss_peak_sampled_mib"],
        "rss_peak_delta_mib": memory["rss_peak_sampled_mib"] - memory["rss_before_mib"],
        "elapsed_seconds": time.perf_counter() - start,
        "history": history,
        "theta_history": theta_history,
        "direction_history": direction_history,
    }


if __name__ == "__main__":
    family, scale, seed, group = json.loads(sys.argv[1])
    result = run_one(family, scale, seed, group)
    with open(sys.argv[2], "w", encoding="utf-8") as output:
        json.dump(result, output, allow_nan=False)
