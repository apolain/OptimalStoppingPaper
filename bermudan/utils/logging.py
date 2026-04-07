from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class RunConfig:
    """Metadata for a single experiment run.

    Passed to the logger before training starts.
    """

    method: str  # "PG", "DOS", "LSMC"
    case: str  # "A", "B_sym", "B_asym", "C"
    d: int = 1  # asset dimension
    S0: float | list[float] = 0.0  # initial spot(s)
    N_S: int = 0  # number of exercise dates
    N: int = 0  # simulation grid steps
    hyperparams: dict[str, Any] = field(default_factory=dict)

    @property
    def tag(self) -> str:
        """Short identifier for filenames."""
        s0 = self.S0 if isinstance(self.S0, (int, float)) else self.S0[0]
        return f"{self.method}_{self.case}_d{self.d}_S{s0}_N{self.N_S}"


class ExperimentLogger:
    """Central logger for numerical experiments.

    Parameters
    ----------
    log_dir : str or Path
        Root directory for all logs.  Created if it doesn't exist.
    save_models : bool
        If True, save network state_dicts after training.

    Usage
    -----
    >>> logger = ExperimentLogger("logs/run_001")
    >>> logger.start_run(RunConfig(method="PG", case="A", d=1, S0=36, N_S=50))
    >>> for epoch in range(n_epochs):
    ...     # ... training ...
    ...     logger.log_epoch(epoch=epoch, loss=loss, reward_mean=r, ...)
    >>> logger.end_run(price=4.48, std=0.01, train_time=60.0, eval_time=2.0, n_params=4481)
    >>> logger.save_model(policy.state_dict(), "policy")
    """

    def __init__(self, log_dir: str | Path, save_models: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_models = save_models

        self._summary_path = self.log_dir / "summary.csv"
        self._summary_header_written = self._summary_path.exists()

        self._current_run: RunConfig | None = None
        self._epoch_writer = None
        self._epoch_file = None
        self._epoch_fields: list[str] = []
        self._run_dir: Path | None = None

    def start_run(self, config: RunConfig) -> Path:
        """Begin a new experiment run.

        Returns the run-specific directory path.
        """
        self._current_run = config
        self._run_dir = self.log_dir / config.tag
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Save config as JSON
        cfg_dict = asdict(config)
        with open(self._run_dir / "config.json", "w") as f:
            json.dump(cfg_dict, f, indent=2, default=str)

        # Prepare epoch CSV (opened lazily on first log_epoch)
        self._epoch_file = None
        self._epoch_writer = None
        self._epoch_fields = []

        return self._run_dir

    def log_epoch(self, **metrics) -> None:
        """Log metrics for one training epoch.

        Any keyword arguments are written as CSV columns.
        The header is inferred from the first call.

        Common fields:
            epoch, loss, reward_mean, reward_std, entropy_mean,
            grad_norm, lr, date (for DOS)
        """
        if self._run_dir is None:
            return

        if self._epoch_file is None:
            self._epoch_fields = list(metrics.keys())
            path = self._run_dir / "epochs.csv"
            self._epoch_file = open(path, "w", newline="")
            self._epoch_writer = csv.DictWriter(
                self._epoch_file, fieldnames=self._epoch_fields
            )
            self._epoch_writer.writeheader()

        # Convert tensors to floats
        row = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                row[k] = v.item()
            else:
                row[k] = v
        self._epoch_writer.writerow(row)

    def end_run(
        self,
        price: float,
        std: float,
        train_time: float,
        eval_time: float = 0.0,
        n_params: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Finalise the current run and append to summary CSV."""
        # Close epoch file
        if self._epoch_file is not None:
            self._epoch_file.close()
            self._epoch_file = None
            self._epoch_writer = None

        if self._current_run is None:
            return

        cfg = self._current_run
        s0 = cfg.S0 if isinstance(cfg.S0, (int, float)) else cfg.S0[0]

        row = {
            "method": cfg.method,
            "case": cfg.case,
            "d": cfg.d,
            "S0": s0,
            "N_S": cfg.N_S,
            "price": f"{price:.6f}",
            "std": f"{std:.6f}",
            "train_time": f"{train_time:.2f}",
            "eval_time": f"{eval_time:.2f}",
            "n_params": n_params,
            "hyperparams": json.dumps(cfg.hyperparams, default=str),
        }
        if extra:
            row.update(extra)

        # Append to summary
        write_header = not self._summary_header_written
        with open(self._summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._summary_header_written = True
            writer.writerow(row)

        # Save result JSON in run dir
        if self._run_dir is not None:
            result = dict(row)
            with open(self._run_dir / "result.json", "w") as f:
                json.dump(result, f, indent=2)

        self._current_run = None

    def save_model(
        self,
        state_dict: dict,
        name: str = "model",
    ) -> Path | None:
        """Save a network state_dict as .pt file.

        Returns the saved path, or None if save_models is False.
        """
        if not self.save_models or self._run_dir is None:
            return None
        path = self._run_dir / f"{name}.pt"
        torch.save(state_dict, path)
        return path

    def save_models_dict(
        self,
        models: dict[str, dict],
    ) -> None:
        """Save multiple models (e.g. DOS networks keyed by date index)."""
        if not self.save_models or self._run_dir is None:
            return
        models_dir = self._run_dir / "models"
        models_dir.mkdir(exist_ok=True)
        for name, sd in models.items():
            torch.save(sd, models_dir / f"{name}.pt")
