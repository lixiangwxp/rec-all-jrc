from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.getenv("FUNREC_PROJECT_ROOT", PACKAGE_ROOT.parent)).resolve()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = Path(os.getenv("FUNREC_RAW_DATA_PATH", DATA_DIR / "raw")).resolve()
DATA_PATH = RAW_DATA_DIR if RAW_DATA_DIR.name == "news_recommendation" else RAW_DATA_DIR / "news_recommendation"

PROCESSED_DATA_DIR = Path(os.getenv("FUNREC_PROCESSED_DATA_PATH", DATA_DIR / "processed")).resolve()
SAVE_PATH = Path(os.getenv("FUNREC_SAVE_PATH", PROCESSED_DATA_DIR / "temp_results")).resolve()
TEMP_RESULTS_DIR = SAVE_PATH

OUTPUT_PATH = PROJECT_ROOT / "outputs"
OUTPUTS_DIR = OUTPUT_PATH

TRAIN_FEATURE_FILES = {
    "baseline": "trn_user_item_feats_df_all.csv",
    "top64": "trn_user_item_feats_df_all_rankmixer_v2_top64.csv",
    "top100": "trn_user_item_feats_df_all_rankmixer_v2_top100.csv",
}

DEFAULT_SEED = 42


@dataclass(frozen=True)
class PathConfig:
    project_root: Path = PROJECT_ROOT
    data_path: Path = DATA_PATH
    save_path: Path = SAVE_PATH
    output_path: Path = OUTPUT_PATH


@dataclass(frozen=True)
class DataConfig:
    mode: str = "offline"
    train_variant: str = "top64"
    max_history_len: int = 50
    article_svd_dim: int = 41
    limit_rows: int | None = None
    limit_users: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    emb_dim: int = 8
    token_dim: int = 64
    num_layers: int = 2
    expansion_ratio: int = 4
    dropout_rate: float = 0.1
    pooling: str = "mean"
    logit_head_units: tuple[int, ...] = (64,)
    attention_hidden_units: tuple[int, ...] = (64, 32)


@dataclass(frozen=True)
class LossConfig:
    alpha: float = 0.5
    listwise_tau: float = 0.8
    bpr_weight: float = 0.25
    bce_aux_weight: float = 0.10
    jrc_bpr_weight: float = 0.25
    max_neg_per_query: int = 80


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    topk: int = 5
    device: str | None = None


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = "funrec-rankmixer"
    entity: str | None = None
    run_name: str | None = None
    mode: str = "online"
    group: str | None = None
    tags: tuple[str, ...] = ()
    log_every_n_steps: int = 1
    log_parameter_updates: bool = True
    watch_model: bool = False
    watch_log: str = "gradients"
    watch_log_freq: int = 100
    log_model: bool = False
    log_prediction_artifacts: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathConfig = PathConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig = TrainConfig()
    wandb: WandbConfig = WandbConfig()


SERVICE_ALIASES = {
    "4.recall_all": "rankmixer_score",
    "recall": "rankmixer_score",
    "recall_all": "rankmixer_score",
    "5.feature_engineering": "rankmixer_score",
    "feature_engineering": "rankmixer_score",
    "5.feature_engineering_all": "rankmixer_score",
    "features": "rankmixer_score",
    "feature_engineering_all": "rankmixer_score",
    "6.ranking": "rankmixer_score",
    "ranking": "rankmixer_score",
    "ranking_baselines": "rankmixer_score",
    "jrc-ranking_all": "rankmixer_score",
    "jrc_ranking": "rankmixer_score",
    "rankmixer": "rankmixer_score",
    "rankmixer_direct": "rankmixer_score",
    "rankmixer_fair_ablation": "rankmixer_score",
    "rankmixer_fair_ablation_v2": "rankmixer_score",
    "rankmixer_ablation_v2": "rankmixer_score",
    "rankmixer_fair_ablation_v2_score": "rankmixer_score",
    "rankmixer_v2_score": "rankmixer_score",
    "rankmixer_score": "rankmixer_score",
    "rankmixerclean": "rankmixer_score",
    "rankmixer_clean": "rankmixer_score",
}

SERVICE_MODULES = {
    "rankmixer_score": "rec.services.rankmixer_score",
}


def resolve_service_name(name: str) -> str:
    normalized = name.removesuffix(".py").removesuffix(".ipynb")
    return SERVICE_ALIASES.get(normalized, normalized)


def train_feature_name(variant: str = "top64") -> str:
    return TRAIN_FEATURE_FILES[variant]


def recall_dict_name(mode: str = "offline") -> str:
    return f"{mode}_all_final_recall_items_dict.pkl"


def recall_csv_name(mode: str = "offline") -> str:
    return f"{mode}_all_final_recall_items.csv"
