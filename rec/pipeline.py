from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import pandas as pd

from rec.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig, WandbConfig
from rec.features import build_feature_preset, build_token_specs


@dataclass(frozen=True)
class Scenario:
    """一次实验的最小配置。

    Debug 时可以把它理解成“本次要跑哪种 loss、输出头如何解释”。
    run_scenario 会按这个对象里的字段逐步组装特征、模型、训练器和评估逻辑。
    """

    name: str
    loss_name: str
    head_type: str
    score_mode: str


DEFAULT_SCENARIOS = {
    "balanced_ge": Scenario("balanced_ge", "balanced_ge", "two_logit", "click"),
    "old_ge": Scenario("old_ge", "old_ge", "two_logit", "diff"),
    "jrc_bpr": Scenario("jrc_bpr", "jrc_bpr", "two_logit", "diff"),
    "listwise_bpr_bce": Scenario("listwise_bpr_bce", "listwise_bpr_bce", "single_logit", "scalar"),
    "query_softmax_ce": Scenario("query_softmax_ce", "query_softmax_ce", "single_logit", "scalar"),
}


def build_model(scenario: Scenario, data, model_config: ModelConfig, article_svd_dim: int):
    """创建固定的 RankMixer 排序模型。

    所有 scenario 都使用 RankMixerRanker，并把文章 SVD 维度传给模型。
    """

    from rec.models.rankmixer import RankMixerRanker

    return RankMixerRanker(
        data.sparse_vocab_sizes,
        data.feature_preset,
        model_config,
        article_svd_dim=article_svd_dim,
        head_type=scenario.head_type,
    )


def preview_project(config: ExperimentConfig = ExperimentConfig()) -> None:
    """打印项目结构和可选 scenario，适合先用断点熟悉入口。

    这里不会加载训练数据，也不会训练模型；如果命令行没有显式传 --train，
    cli_main 会走到这个函数然后 return，方便初学者先确认配置和实验列表。
    """

    preset = build_feature_preset(article_svd_dim=config.data.article_svd_dim)
    rows = [
        {
            "scenario": scenario.name,
            "loss": scenario.loss_name,
            "head": scenario.head_type,
            "score": scenario.score_mode,
        }
        for scenario in DEFAULT_SCENARIOS.values()
    ]
    print("RankMixer project layout")
    print("  data:      rec/data.py")
    print("  features:  rec/features.py")
    print("  models:    rec/models/rankmixer.py")
    print("  losses:    rec/losses.py")
    print("  train:     rec/train.py")
    print("  eval:      rec/evaluate.py")
    print("  pipeline:  rec/pipeline.py")
    print()
    print(f"sparse features: {len(preset.sparse_features)}")
    print(f"dense features: {len(preset.dense_features)}")
    print(f"rankmixer tokens: {len(build_token_specs(preset, config.data.article_svd_dim))}")
    print(pd.DataFrame(rows).to_string(index=False))


def run_scenario(scenario: Scenario, config: ExperimentConfig) -> dict[str, object]:
    """运行单个推荐实验的主入口。

    这是调试 pipeline 最重要的函数。建议在第一行打断点，然后用 F10 观察执行顺序：
    1. build_feature_preset：根据 scenario 和 data config 定义要用哪些特征；
    2. load_feature_frames：从磁盘读取原始/中间特征表；
    3. merge_history_frames：把用户历史、候选文章等表合并到一起；
    4. prepare_ranking_data：转换成训练/验证需要的 numpy 数组、DataLoader 输入和验证 frame；
    5. build_model：创建 RankMixer 模型；
    6. RankMixerTrainer.fit：进入训练循环，里面会做 DataLoader -> forward -> loss -> backward -> eval；
    7. save_result：把 history、验证预测和 summary 写到 output 目录。

    Debug 小贴士：命令行传 --limit-users 可以只保留少量用户的数据，让 DataLoader、
    loss 和评估都更快跑完。它只改变数据量，适合单步调试变量形状和中间结果。
    """

    from rec.data import load_feature_frames, merge_history_frames, prepare_ranking_data
    from rec.models.rankmixer import count_parameters
    from rec.train import RankMixerTrainer
    #根据配置组装本次实验要用的特征清单。
    preset = build_feature_preset(
        name=scenario.name,
        article_svd_dim=config.data.article_svd_dim,
    )
    loaded = load_feature_frames(config.paths, config.data)
    merged = merge_history_frames(loaded)
    prepared = prepare_ranking_data(merged, preset, config.data)
    model = build_model(scenario, prepared, config.model, config.data.article_svd_dim)
    print(f"scenario={scenario.name} | model=rankmixer | params={count_parameters(model):,}")
    trainer = RankMixerTrainer(config.train, config.loss)
    wandb_logger = None
    if config.wandb.enabled:
        from rec.wandb_utils import WandbRunLogger
        wandb_logger = WandbRunLogger(config, scenario, model, trainer.device, config.paths.output_path)
    try:
        result = trainer.fit(model, prepared, scenario.loss_name, scenario.head_type, scenario.score_mode, wandb_logger)
        if wandb_logger is not None and config.wandb.log_model:
            wandb_logger.log_model_weights(model)
        save_result(config.paths.output_path, scenario, result, topk=config.train.topk, wandb_logger=wandb_logger)
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()
    return {"scenario": scenario, "result": result, "model": model, "data": prepared}


def save_result(output_path: Path, scenario: Scenario, result, topk: int, wandb_logger=None) -> None:
    """保存一次 scenario 的训练结果。

    单步到这里时，result.history 是每个 epoch 的 loss/指标列表；
    result.best_eval 里通常包含验证集预测明细，可用于之后检查排序分数。
    """

    output_path.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(result.history)
    history_path = output_path / f"{scenario.name}_history.csv"
    history_df.to_csv(history_path, index=False)
    artifact_files = {"history": history_path}
    if result.best_eval and result.best_eval.get("val_pred_df") is not None:
        val_predictions_path = output_path / f"{scenario.name}_val_predictions.csv"
        result.best_eval["val_pred_df"].to_csv(val_predictions_path, index=False)
        artifact_files["val_predictions"] = val_predictions_path
        if {"label", "pred_rank"}.issubset(result.best_eval["val_pred_df"].columns):
            bad_case_path = output_path / f"{scenario.name}_bad_case_positive_rows.csv"
            bad_cases = result.best_eval["val_pred_df"][
                (result.best_eval["val_pred_df"]["label"] > 0) & (result.best_eval["val_pred_df"]["pred_rank"] > topk)
            ]
            bad_cases.to_csv(bad_case_path, index=False)
            artifact_files["bad_case_positive_rows"] = bad_case_path
    if result.best_eval and result.best_eval.get("hit_pred_df") is not None:
        hit_predictions_path = output_path / f"{scenario.name}_hit_predictions.csv"
        result.best_eval["hit_pred_df"].to_csv(hit_predictions_path, index=False)
        artifact_files["hit_predictions"] = hit_predictions_path
    summary = pd.DataFrame(
        [
            {
                "scenario": scenario.name,
                "best_epoch": result.best_epoch,
                "best_metric": result.best_metric,
            }
        ]
    )
    summary_path = output_path / f"{scenario.name}_summary.csv"
    summary.to_csv(summary_path, index=False)
    artifact_files["summary"] = summary_path
    if wandb_logger is not None:
        wandb_logger.log_prediction_artifacts(
            artifact_files,
            metadata={"scenario": scenario.name, "best_epoch": result.best_epoch, "topk": topk},
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """定义命令行参数，也是 VSCode launch.json 常用参数的来源。

    调试时常用组合：
    --train --scenario jrc_bpr --epochs 1 --limit-users 20
    其中 --limit-users 的作用是缩小用户数，让断点不会被完整数据集拖慢。
    """

    parser = argparse.ArgumentParser(description="RankMixer fair ablation v2 score project runner.")
    parser.add_argument("--scenario", default="jrc_bpr", choices=sorted(DEFAULT_SCENARIOS))
    parser.add_argument("--preview", action="store_true", help="Show project layout and available scenarios.")
    parser.add_argument("--train", action="store_true", help="Load data and train the selected scenario.")
    parser.add_argument("--train-variant", default="top64", choices=("baseline", "top64", "top100"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--limit-users", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-article-svd", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Log training and validation metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default="funrec-rankmixer")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-log-every-n-steps", type=int, default=1)
    parser.add_argument("--wandb-watch", action="store_true")
    parser.add_argument("--wandb-watch-log", default="gradients", choices=("gradients", "parameters", "all"))
    parser.add_argument("--wandb-watch-log-freq", type=int, default=100)
    parser.add_argument("--wandb-log-model", action="store_true")
    parser.add_argument("--wandb-prediction-artifacts", action=argparse.BooleanOptionalAction, default=True)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """把 argparse 的扁平参数整理成项目内部的配置对象。

    F10 经过这里时，可以重点看 data.limit_users / data.limit_rows 是否符合预期；
    后续 load/prepare 数据时会使用这些限制，训练逻辑本身不会特殊处理它们。
    """

    data = DataConfig(
        train_variant=args.train_variant,
        article_svd_dim=0 if args.no_article_svd else 16,
        limit_rows=args.limit_rows,
        limit_users=args.limit_users,
    )
    train = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        topk=args.topk,
        device=args.device,
    )
    model = ModelConfig()
    if args.no_article_svd:
        model = replace(model, token_dim=64)
    tags = tuple(tag.strip() for tag in args.wandb_tags.split(",") if tag.strip())
    wandb = WandbConfig(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        mode=args.wandb_mode,
        group=args.wandb_group,
        tags=tags,
        log_every_n_steps=args.wandb_log_every_n_steps,
        watch_model=args.wandb_watch,
        watch_log=args.wandb_watch_log,
        watch_log_freq=args.wandb_watch_log_freq,
        log_model=args.wandb_log_model,
        log_prediction_artifacts=args.wandb_prediction_artifacts,
    )
    return ExperimentConfig(data=data, train=train, model=model, wandb=wandb)


def cli_main(argv: Sequence[str] | None = None) -> None:
    """命令行入口。

    执行顺序很短：解析参数 -> 生成 config -> preview 或 run_scenario。
    初学者可以从这里开始 F11 进入 run_scenario，理解一次训练从入口到评估的完整链路。
    """

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    if args.preview or not args.train:
        preview_project(config)
        return
    run_scenario(DEFAULT_SCENARIOS[args.scenario], config)


if __name__ == "__main__":
    cli_main()
