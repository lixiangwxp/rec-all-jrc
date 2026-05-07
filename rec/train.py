from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rec.config import DEFAULT_SEED, TrainConfig
from rec.data import PreparedRankingData, make_loader, move_batch_to_device
from rec.evaluate import RankingEvaluator, raw_predict_outputs
from rec.losses import build_loss


@dataclass
class TrainingResult:
    """训练结束后返回给 pipeline 的结果包。

    Debug 时可以在 fit 返回前展开这个对象：
    history 记录每个 epoch 的平均 loss/指标；best_eval 保存最佳验证结果；
    best_state_dict 是当时模型参数的拷贝，最后会被加载回 model。
    """

    history: list[dict[str, float]]
    best_epoch: int | None
    best_metric: float | None
    best_eval: dict[str, Any] | None
    best_state_dict: dict[str, torch.Tensor] | None


def set_seed(seed: int) -> None:
    """固定随机种子，减少每次 debug 时 batch 顺序和初始化差异。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device(device_name: str | None = None) -> torch.device:
    """选择训练设备。

    如果 launch.json 或命令行传了 --device，会直接使用指定设备；
    否则按 cuda -> mps -> cpu 的顺序自动选择。单步调试时 cpu 通常最容易观察。
    """

    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RankMixerTrainer:
    def __init__(self, config: TrainConfig, loss_config: Any):
        """保存训练配置，并创建评估器。

        self.device 会影响后面 x_batch/y_batch 被移动到哪里；
        self.evaluator 负责每个 epoch 结束后的验证集排序指标。
        """

        self.config = config
        self.loss_config = loss_config
        self.device = default_device(config.device)
        self.evaluator = RankingEvaluator(topk=config.topk)

    def fit(
        self,
        model: torch.nn.Module,
        data: PreparedRankingData,
        loss_name: str,
        head_type: str,
        score_mode: str,
        wandb_logger=None,
    ) -> TrainingResult:
        """执行完整训练循环。

        建议在这里打断点并用 F10/F11 观察主线：
        DataLoader 产出 x_batch/y_batch -> model(x_batch) forward 得到 output ->
        loss_fn 计算 total loss -> backward 写入梯度 -> optimizer.step 更新参数 ->
        epoch 结束后 raw_predict_outputs + evaluate_raw 做验证集排序评估。
        """

        set_seed(DEFAULT_SEED)
        model = model.to(self.device)
        loss_fn = build_loss(loss_name, self.loss_config)
        optimizer_cls = torch.optim.AdamW if loss_name in {"jrc_bpr", "listwise_bpr_bce"} else torch.optim.Adam
        optimizer = optimizer_cls(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        if wandb_logger is not None:
            wandb_logger.log_optimizer(optimizer, optimizer_cls.__name__, loss_name)
        # train_loader 把 prepare_ranking_data 产出的 numpy 数组包装成 batch。
        # group_by_query=True 表示同一个 query 的候选尽量一起出现，方便 listwise/pairwise loss 使用。
        train_loader = make_loader(
            data.x_train,
            data.y_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            group_by_query=True,
            seed=DEFAULT_SEED,
        )

        history: list[dict[str, float]] = []
        best_metric = float("-inf")
        best_epoch: int | None = None
        best_eval: dict[str, Any] | None = None
        best_state_dict: dict[str, torch.Tensor] | None = None
        global_step = 0

        for epoch in range(1, self.config.epochs + 1):
            # 一个 epoch 会完整遍历一次 train_loader。started_at 用来统计本轮耗时。
            started_at = time.time()
            # train() 会打开 dropout/batch norm 等训练行为；如果模型里没有这些层，也保持标准写法。
            model.train()
            # epoch_parts 先收集每个 batch 的 loss，epoch 结束再求平均写入 history。
            epoch_parts: dict[str, list[float]] = {"total_loss": []}

            for batch_idx, (x_batch, y_batch) in enumerate(train_loader, start=1):
                # x_batch 是一个特征字典，key 对应各个 sparse/dense 特征名；
                # y_batch 是这一批候选的标签。单步时可重点看每个 tensor 的 shape。
                x_batch = move_batch_to_device(x_batch, self.device)
                y_batch = y_batch.to(self.device)
                # 清空上一批次留下的梯度，避免梯度累加影响当前 batch。
                optimizer.zero_grad()
                # forward：模型把特征字典转换成 raw output，形状由 head_type 决定。
                output = model(x_batch)
                # loss_fn 同时拿 output、标签和原始 batch 特征，返回 total 和若干可记录的分项。
                loss_output = loss_fn(output, y_batch, x_batch)
                # backward：从 total loss 反向传播，把梯度写到 model.parameters() 上。
                loss_output.total.backward()
                if self.config.grad_clip_norm:
                    # 梯度裁剪只限制梯度范数，不改变 loss 定义；debug 时可观察更新前梯度是否过大。
                    grad_norm_pre_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip_norm).detach().cpu())
                else:
                    grad_norm_pre_clip = None
                grad_norm = parameter_grad_norm(model)
                # optimizer.step 根据刚才的梯度更新参数，完成一个 batch 的训练闭环。
                should_log_step = wandb_logger is not None and wandb_logger.should_log_step(global_step + 1)
                params_before = parameter_snapshot(model) if should_log_step and wandb_logger.config.log_parameter_updates else None
                optimizer.step()
                global_step += 1

                # detach/cpu/float 只用于日志记录，避免把训练计算图继续挂在 history 上。
                epoch_parts["total_loss"].append(float(loss_output.total.detach().cpu()))
                for name, value in loss_output.parts.items():
                    epoch_parts.setdefault(name, []).append(float(value.detach().cpu()))
                if should_log_step:
                    step_metrics = {
                        "train/global_step": float(global_step),
                        "epoch": float(epoch),
                        "train/batch_idx": float(batch_idx),
                        "train/batch_size": float(y_batch.size(0)),
                        "train/loss": float(loss_output.total.detach().cpu()),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "train/grad_norm": grad_norm,
                        "train/param_norm": parameter_norm(model),
                    }
                    if grad_norm_pre_clip is not None:
                        step_metrics["train/grad_norm_pre_clip"] = grad_norm_pre_clip
                    if params_before is not None:
                        update_norm = parameter_update_norm(model, params_before)
                        step_metrics["train/update_norm"] = update_norm
                        step_metrics["train/update_to_param_ratio"] = update_norm / step_metrics["train/param_norm"]
                    for name, value in loss_output.parts.items():
                        step_metrics[f"train/{name}"] = float(value.detach().cpu())
                    wandb_logger.log_train_step(step_metrics, global_step)

            # 把 batch 级别的 loss 列表压缩成 epoch 级别的一行日志。
            row = {name: float(np.mean(values)) for name, values in epoch_parts.items()}
            row["epoch"] = float(epoch)
            row["elapsed_sec"] = float(time.time() - started_at)

            if data.x_val is not None and data.y_val is not None and data.val_frame is not None:
                row.update(evaluate_validation_loss(model, loss_fn, data, self.config.batch_size, self.device))
                # 验证阶段不再 backward，只用当前模型对验证特征做 raw output 预测。
                raw_output = raw_predict_outputs(model, data.x_val, self.config.batch_size, self.device)
                # evaluate_raw 会把 raw output 转成排序分数，再分别计算 full/hit 指标。
                eval_result = self.evaluator.evaluate_raw(
                    raw_output,
                    data.y_val,
                    data.val_frame,
                    head_type=head_type,
                    score_mode=score_mode,
                    hit_mask=data.val_hit_mask,
                )
                row.update(
                    {
                        "full_mrr": float(eval_result.get("full_mrr", 0.0)),
                        "full_ndcg": float(eval_result.get("full_ndcg", 0.0)),
                        "hit_mrr": float(eval_result.get("hit_mrr", 0.0)),
                        "hit_ndcg": float(eval_result.get("hit_ndcg", 0.0)),
                    }
                )
                # 以 full_mrr 为主选择最佳 epoch；如果 full_mrr 不可用，再退到 hit_mrr。
                metric = row["full_mrr"] if not np.isnan(row["full_mrr"]) else row["hit_mrr"]
                if metric > best_metric:
                    best_metric = metric
                    best_epoch = epoch
                    # best_eval 保存当时的指标和预测明细；state_dict 深拷贝保存当时的模型参数。
                    best_eval = eval_result
                    best_state_dict = copy.deepcopy(model.state_dict())
                    if wandb_logger is not None:
                        wandb_logger.update_best(epoch, float(metric), row)

            history.append(row)
            if wandb_logger is not None:
                wandb_logger.log_epoch(row, global_step)
            print(format_epoch_log(row))

        if best_state_dict is not None:
            # 训练结束后把模型恢复到验证指标最好的 epoch，pipeline 返回的 model 就是最佳版本。
            model.load_state_dict(best_state_dict)

        return TrainingResult(
            history=history,
            best_epoch=best_epoch,
            best_metric=None if best_metric == float("-inf") else float(best_metric),
            best_eval=best_eval,
            best_state_dict=best_state_dict,
        )


def format_epoch_log(row: dict[str, float]) -> str:
    """把 history 中的一行格式化成控制台日志，便于边训练边观察指标变化。"""

    pieces = [
        f"epoch={int(row['epoch'])}",
        f"loss={row['total_loss']:.4f}",
        f"time={row['elapsed_sec']:.1f}s",
    ]
    for key in ("ce_loss", "ge_loss", "listwise_loss", "bpr_loss", "bce_aux_loss", "val_loss", "full_mrr", "hit_mrr"):
        if key in row:
            pieces.append(f"{key}={row[key]:.5f}")
    return " | ".join(pieces)


def parameter_snapshot(model: torch.nn.Module) -> list[torch.Tensor]:
    return [param.detach().clone() for param in model.parameters() if param.requires_grad]


def parameter_norm(model: torch.nn.Module) -> float:
    total = sum(float(param.detach().float().square().sum().cpu()) for param in model.parameters() if param.requires_grad)
    return float(total**0.5)


def parameter_grad_norm(model: torch.nn.Module) -> float:
    total = sum(
        float(param.grad.detach().float().square().sum().cpu())
        for param in model.parameters()
        if param.requires_grad and param.grad is not None
    )
    return float(total**0.5)


def parameter_update_norm(model: torch.nn.Module, params_before: list[torch.Tensor]) -> float:
    total = 0.0
    for param, before in zip((param for param in model.parameters() if param.requires_grad), params_before):
        total += float((param.detach() - before).float().square().sum().cpu())
    return float(total**0.5)


def evaluate_validation_loss(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data: PreparedRankingData,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    loader = make_loader(data.x_val, data.y_val, batch_size=batch_size, shuffle=False, group_by_query=True)
    parts: dict[str, list[float]] = {"val_loss": []}
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = move_batch_to_device(x_batch, device)
            y_batch = y_batch.to(device)
            loss_output = loss_fn(model(x_batch), y_batch, x_batch)
            parts["val_loss"].append(float(loss_output.total.detach().cpu()))
            for name, value in loss_output.parts.items():
                parts.setdefault(f"val_{name}", []).append(float(value.detach().cpu()))
    return {name: float(np.mean(values)) for name, values in parts.items()}
