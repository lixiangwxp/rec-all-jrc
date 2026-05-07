from __future__ import annotations

import importlib.metadata as metadata
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return to_plain(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list | set):
        return [to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    return value


def git_metadata(project_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=project_root, text=True).strip()

    status = git("status", "--short")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_short": status,
    }


def dependency_versions(project_root: Path) -> dict[str, Any]:
    package_names = {"wandb"}
    requirements_path = project_root / "requirements.txt"
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for marker in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            if marker in line:
                line = line.split(marker, 1)[0]
                break
        package_names.add(line.strip())

    installed = {dist.metadata["Name"].lower(): dist.version for dist in metadata.distributions() if dist.metadata["Name"]}
    versions: dict[str, str] = {}
    missing: list[str] = []
    for package_name in sorted(package_names):
        version = installed.get(package_name.lower())
        if version is None:
            missing.append(package_name)
        else:
            versions[package_name] = version
    return {"versions": versions, "missing": missing}


def runtime_metadata() -> dict[str, str]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
    }


def device_metadata(device: Any) -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "selected": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_devices"] = [
            {
                "name": torch.cuda.get_device_name(index),
                "capability": ".".join(str(part) for part in torch.cuda.get_device_capability(index)),
                "total_memory_gb": round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 2),
            }
            for index in range(torch.cuda.device_count())
        ]
    return info


def model_metadata(model: Any) -> dict[str, Any]:
    parameters = list(model.parameters())
    return {
        "class_name": model.__class__.__name__,
        "total_parameters": int(sum(param.numel() for param in parameters)),
        "trainable_parameters": int(sum(param.numel() for param in parameters if param.requires_grad)),
    }


class WandbRunLogger:
    def __init__(
        self,
        experiment_config: Any,
        scenario: Any,
        model: Any,
        device: Any,
        output_path: Path,
    ):
        import wandb

        self.wandb = wandb
        self.config = experiment_config.wandb
        self.output_path = output_path
        self.scenario_name = scenario.name
        self.run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=self.config.run_name,
            mode=self.config.mode,
            group=self.config.group,
            tags=list(self.config.tags),
            config={
                "experiment": to_plain(experiment_config),
                "scenario": to_plain(scenario),
                "git": git_metadata(experiment_config.paths.project_root),
                "runtime": runtime_metadata(),
                "device": device_metadata(device),
                "dependencies": dependency_versions(experiment_config.paths.project_root),
                "model": model_metadata(model),
                "scheduler": {"name": "none"},
            },
        )
        self.wandb.define_metric("train/global_step")
        self.wandb.define_metric("train/*", step_metric="train/global_step")
        self.wandb.define_metric("epoch")
        self.wandb.define_metric("val/*", step_metric="epoch")
        self.wandb.define_metric("best/*", step_metric="epoch")

        if self.config.watch_model:
            self.wandb.watch(model, log=self.config.watch_log, log_freq=self.config.watch_log_freq)
        self.log_model_structure(model)

    def should_log_step(self, global_step: int) -> bool:
        return global_step % self.config.log_every_n_steps == 0

    def log_optimizer(self, optimizer: Any, optimizer_name: str, loss_name: str) -> None:
        self.run.config.update(
            {
                "optimizer": {
                    "name": optimizer_name,
                    "param_groups": [
                        {
                            "lr": group.get("lr"),
                            "weight_decay": group.get("weight_decay"),
                        }
                        for group in optimizer.param_groups
                    ],
                },
                "loss": {"name": loss_name},
                "scheduler": {"name": "none"},
            },
            allow_val_change=True,
        )

    def log_model_structure(self, model: Any) -> None:
        self.output_path.mkdir(parents=True, exist_ok=True)
        path = self.output_path / f"{self.scenario_name}_model_structure.txt"
        path.write_text(str(model), encoding="utf-8")
        artifact = self.wandb.Artifact(f"{self.run.id}-{self.scenario_name}-model-structure", type="model-structure")
        artifact.add_file(str(path), name=path.name)
        self.run.log_artifact(artifact)

    def log_train_step(self, metrics: dict[str, float], global_step: int) -> None:
        self.wandb.log(metrics, step=global_step)

    def log_epoch(self, row: dict[str, float], global_step: int) -> None:
        metrics: dict[str, float] = {"epoch": row["epoch"]}
        for key, value in row.items():
            if key == "epoch":
                continue
            if key == "elapsed_sec":
                metrics["train/epoch_elapsed_sec"] = value
            elif key == "total_loss":
                metrics["train/epoch_loss"] = value
            elif key.startswith("val_"):
                metrics[f"val/{key.removeprefix('val_')}"] = value
            elif key.startswith(("full_", "hit_")):
                metrics[f"val/{key}"] = value
            else:
                metrics[f"train/epoch_{key}"] = value
        self.wandb.log(metrics, step=global_step)

    def update_best(self, best_epoch: int, best_metric: float, row: dict[str, float]) -> None:
        self.run.summary["best/epoch"] = best_epoch
        self.run.summary["best/metric"] = best_metric
        for key in ("full_mrr", "full_ndcg", "hit_mrr", "hit_ndcg", "val_loss"):
            if key in row:
                self.run.summary[f"best/{key}"] = row[key]

    def log_prediction_artifacts(self, files: dict[str, Path], metadata: dict[str, Any]) -> None:
        if not self.config.log_prediction_artifacts:
            return
        artifact = self.wandb.Artifact(f"{self.run.id}-{self.scenario_name}-prediction-tables", type="predictions", metadata=metadata)
        for path in files.values():
            artifact.add_file(str(path), name=path.name)
        self.run.log_artifact(artifact)

    def log_model_weights(self, model: Any) -> None:
        import torch

        self.output_path.mkdir(parents=True, exist_ok=True)
        path = self.output_path / f"{self.scenario_name}_model_state_dict.pt"
        torch.save(model.state_dict(), path)
        artifact = self.wandb.Artifact(f"{self.run.id}-{self.scenario_name}-model", type="model")
        artifact.add_file(str(path), name=path.name)
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        self.wandb.finish()
