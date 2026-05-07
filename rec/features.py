from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from rec.config import DEFAULT_SEED


HISTORY_FEATURE = "hist_click_article_id"
META_INT_FEATURES = ("query_id",)
META_FLOAT_FEATURES = ("sample_prob",)

BASE_SPARSE_FEATURES = (
    "click_article_id",
    "category_id",
    "click_environment",
    "click_deviceGroup",
    "click_os",
    "click_country",
    "click_region",
    "click_referrer_type",
)

RANK_SCORE_SPARSE_FEATURES = ("rank_bucket",)

BASE_DENSE_FEATURES = (
    "sim0",
    "time_diff0",
    "word_diff0",
    "sim_max",
    "sim_min",
    "sim_sum",
    "sim_mean",
    "score",
    "rank",
    "click_size",
    "time_diff_mean",
    "active_level",
    "created_at_ts",
    "user_time_hob1",
    "user_time_hob2",
    "words_hbo",
    "words_count",
    "is_cat_hab",
    "article_hot_level",
    "article_user_num",
    "article_time_diff_mean",
)

RANK_SCORE_DENSE_FEATURES = (
    "rank_recip",
    "rank_log1p",
    "rank_pct200",
    "score_user_z",
    "score_user_minmax",
    "score_gap_top",
    "time_diff0_log1p",
    "word_diff0_log1p",
    "time_diff_mean_log1p",
    "article_time_diff_mean_log1p",
)

SPARSE_TOKEN_ORDER = (
    "click_article_id",
    "category_id",
    "click_environment",
    "click_deviceGroup",
    "click_os",
    "click_country",
    "click_region",
    "click_referrer_type",
    "rank_bucket",
)

BASE_DENSE_TOKEN_GROUPS = OrderedDict(
    [
        (
            "recall_similarity_dense",
            (
                "sim0",
                "time_diff0",
                "word_diff0",
                "sim_max",
                "sim_min",
                "sim_sum",
                "sim_mean",
                "time_diff0_log1p",
                "word_diff0_log1p",
                "time_diff_mean",
                "time_diff_mean_log1p",
                "article_time_diff_mean",
                "article_time_diff_mean_log1p",
            ),
        ),
        (
            "rank_score_dense",
            (
                "score",
                "rank",
                "rank_recip",
                "rank_log1p",
                "rank_pct200",
                "score_user_z",
                "score_user_minmax",
                "score_gap_top",
            ),
        ),
        ("ranking_meta_dense", ("click_size", "active_level", "created_at_ts")),
        (
            "user_article_profile_dense",
            (
                "user_time_hob1",
                "user_time_hob2",
                "words_hbo",
                "words_count",
                "is_cat_hab",
                "article_hot_level",
                "article_user_num",
            ),
        ),
    ]
)


@dataclass(frozen=True)
class FeaturePreset:
    """一次训练要使用的特征清单。

    sparse_features 会先映射成离散 id，再进入 embedding；dense_features 是连续数值特征。
    meta_features 是给训练/评估逻辑使用的辅助列，例如 query_id 只负责把同一用户的候选样本分组，
    不作为 user_id 语义特征喂给模型。
    """

    name: str
    sparse_features: tuple[str, ...]
    dense_features: tuple[str, ...]
    meta_features: tuple[str, ...] = META_INT_FEATURES


def dedupe(values: Sequence[str]) -> tuple[str, ...]:
    """按原顺序去重；dict.fromkeys 会保留第一次出现的位置。"""

    return tuple(dict.fromkeys(values))


def article_svd_features(dim: int) -> tuple[str, ...]:
    """生成文章内容 embedding 降维后的列名，例如 article_svd_0。"""

    return tuple(f"article_svd_{index}" for index in range(dim))


def build_feature_preset(
    name: str = "rankmixer_main",
    article_svd_dim: int = 41,
) -> FeaturePreset:
    """组合默认的稀疏特征、稠密特征和文章 SVD 特征。"""

    sparse = BASE_SPARSE_FEATURES + RANK_SCORE_SPARSE_FEATURES
    dense = BASE_DENSE_FEATURES + RANK_SCORE_DENSE_FEATURES + article_svd_features(article_svd_dim)
    return FeaturePreset(name=name, sparse_features=dedupe(sparse), dense_features=dedupe(dense))


def build_dense_token_groups(preset: FeaturePreset, article_svd_dim: int = 41) -> OrderedDict[str, tuple[str, ...]]:
    """把稠密特征按业务含义分组，后续模型会把每组当成一个 dense token。"""

    available = set(preset.dense_features)
    groups: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    used: set[str] = set()
    for token_name, features in BASE_DENSE_TOKEN_GROUPS.items():
        selected = tuple(feature for feature in features if feature in available and feature not in used)
        if selected:
            groups[token_name] = selected
            used.update(selected)
    svd_group = tuple(feature for feature in article_svd_features(article_svd_dim) if feature in available)
    if svd_group:
        groups["article_content_dense"] = svd_group
        used.update(svd_group)
    residual = tuple(feature for feature in preset.dense_features if feature not in used)
    if residual:
        groups["dense_residual"] = residual
    return groups


def build_token_specs(preset: FeaturePreset, article_svd_dim: int = 41) -> list[dict[str, object]]:
    """生成模型 token 配置：每个稀疏列一个 token，稠密列按组打包，历史序列单独成 token。"""

    specs: list[dict[str, object]] = []
    remaining_sparse = set(preset.sparse_features)
    for feature in SPARSE_TOKEN_ORDER:
        if feature in remaining_sparse:
            specs.append({"name": feature, "kind": "sparse", "features": (feature,)})
            remaining_sparse.remove(feature)
    for feature in sorted(remaining_sparse):
        specs.append({"name": feature, "kind": "sparse", "features": (feature,)})
    for token_name, features in build_dense_token_groups(preset, article_svd_dim).items():
        specs.append({"name": token_name, "kind": "dense", "features": features})
    specs.append({"name": "history_token", "kind": "history", "features": (HISTORY_FEATURE,)})
    return specs


def ensure_feature_columns(df: pd.DataFrame, preset: FeaturePreset) -> pd.DataFrame:
    """补齐并规范化训练所需列。

    pd.to_numeric(..., errors="coerce") 会把无法转成数字的值变成 NaN；
    后面的 fillna/astype 再把缺失值落到稳定的默认值和 dtype，方便后续转 numpy。
    """

    df = df.copy()

    # 对 preset.sparse_features 里的离散特征：
    # 比如 click_article_id、category_id、rank_bucket，补缺失列，然后转成 int64，因为后面要进 embedding。
    for feature in preset.sparse_features:
        if feature not in df.columns:
            df[feature] = 0
        df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0).astype("int64")

    # 对 preset.dense_features 里的连续特征，补缺失列，然后转成 float32，方便后续转 numpy。
    for feature in preset.dense_features:
        if feature not in df.columns:
            df[feature] = 0.0
        df[feature] = (
            pd.to_numeric(df[feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype("float32")
        )

    #这个是采样概率，后面的 loss 可能会用它做校正。
    if "sample_prob" not in df.columns:
        df["sample_prob"] = 1.0
    df["sample_prob"] = (
        pd.to_numeric(df["sample_prob"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(1e-4, 1.0)
        .astype("float32")
    )
    return df


def add_rank_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """基于召回分数和同一用户内的排序位置，追加 ranking 相关特征。

    这里的 user_id 只用来做同用户候选集合内的统计和排序，不会直接作为模型输入特征。
    groupby("user_id") 表示按用户分组计算；cumcount() 是每个分组内部从 0 开始的行号。
    """

    df = df.copy()
    score = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    if "rank" in df.columns:
        rank = pd.to_numeric(df["rank"], errors="coerce").fillna(0.0)
    else:
        work = pd.DataFrame({"user_id": df["user_id"], "score": score, "row": np.arange(len(df))})
        work = work.sort_values(["user_id", "score", "row"], ascending=[True, False, True], kind="mergesort")
        work["rank"] = work.groupby("user_id", sort=False).cumcount()
        rank = pd.Series(work.sort_values("row", kind="mergesort")["rank"].to_numpy(), index=df.index)

    rank = rank.clip(lower=0.0)
    #把 rank 限制为不小于 0。
    df["rank"] = rank.astype("float32")
    # 这些是 rank 的不同数值表达：倒数、log、比例和分桶，帮助模型感知候选在用户内的位置。
    df["rank_recip"] = (1.0 / (rank + 1.0)).astype("float32")
    df["rank_log1p"] = np.log1p(rank).astype("float32")
    df["rank_pct200"] = (rank / 199.0).astype("float32")
    df["rank_bucket"] = (
        pd.cut(rank, bins=[-1, 0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 10**12], labels=False)
        .fillna(0)
        .astype("int64")
    )

    score_frame = pd.DataFrame({"user_id": df["user_id"], "score": score}, index=df.index)

    # agg(["mean", "std", "min", "max"]) 会为每个用户算一组统计量，join(..., on="user_id") 再贴回每条候选样本。
    stats = score_frame.groupby("user_id", sort=False)["score"].agg(["mean", "std", "min", "max"])
    df = df.join(stats, on="user_id", rsuffix="_user")
    df["score_user_z"] = ((score - df["mean"]) / (df["std"].fillna(0.0) + 1e-6)).astype("float32")

    #用户内 min-max 归一化。
    df["score_user_minmax"] = ((score - df["min"]) / (df["max"] - df["min"] + 1e-6)).astype("float32")
    #和该用户最高召回分数的差距。
    df["score_gap_top"] = (df["max"] - score).astype("float32")
    df = df.drop(columns=["mean", "std", "min", "max"])
     
    #最后是几个时间/词数差异特征的 log 变换：
    log_sources = {
        "time_diff0": "time_diff0_log1p",
        "word_diff0": "word_diff0_log1p",
        "time_diff_mean": "time_diff_mean_log1p",
        "article_time_diff_mean": "article_time_diff_mean_log1p",
    }
    for source, target in log_sources.items():
        values = pd.to_numeric(df[source], errors="coerce").fillna(0.0) if source in df.columns else 0.0
        df[target] = np.log1p(np.clip(values, 0.0, None)).astype("float32")
    return df


def load_article_embedding(data_path: Path, save_path: Path) -> pd.DataFrame:
    """读取文章内容向量，并统一整理成 click_article_id + article_emb_* 的 DataFrame。"""

    pkl_path = save_path / "item_content_emb_all.pkl"
    if pkl_path.exists():
        import pickle

        with pkl_path.open("rb") as file:
            embedding_dict = pickle.load(file)
        item_ids = np.asarray(list(embedding_dict.keys()), dtype=np.int64)
        matrix = np.asarray([np.asarray(value, dtype=np.float32).reshape(-1) for value in embedding_dict.values()])
        df = pd.DataFrame(matrix, columns=[f"article_emb_{index}" for index in range(matrix.shape[1])])
        df.insert(0, "click_article_id", item_ids)
        return df

    df = pd.read_csv(data_path / "articles_emb.csv")
    id_col = "click_article_id" if "click_article_id" in df.columns else "article_id"
    df = df.rename(columns={id_col: "click_article_id"})
    numeric_cols = [column for column in df.select_dtypes(include=[np.number]).columns if column != "click_article_id"]
    emb_cols = [column for column in numeric_cols if "emb" in column.lower()] or numeric_cols
    matrix = df[emb_cols].to_numpy(dtype=np.float32)
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    out = pd.DataFrame(matrix, columns=[f"article_emb_{index}" for index in range(matrix.shape[1])])
    out.insert(0, "click_article_id", pd.to_numeric(df["click_article_id"], errors="coerce").fillna(0).astype("int64"))
    return out


def add_article_svd_features(
    frames: Sequence[pd.DataFrame | None],
    data_path: Path,
    save_path: Path,
    n_components: int,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame | None, ...]:
    """把高维文章 embedding 用 TruncatedSVD 降维后 merge 到候选样本。

    merge(..., on="click_article_id", how="left") 表示按文章 id 左连接：
    候选样本一行不丢，找不到 embedding 的文章后面用 0.0 补齐。
    """

    if n_components <= 0:
        return tuple(frames)
    article_embedding = load_article_embedding(data_path, save_path)
    emb_cols = [column for column in article_embedding.columns if column.startswith("article_emb_")]
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    matrix = svd.fit_transform(article_embedding[emb_cols].to_numpy(dtype=np.float32)).astype("float32")
    svd_df = pd.DataFrame(matrix, columns=list(article_svd_features(n_components)))
    svd_df.insert(0, "click_article_id", article_embedding["click_article_id"].astype("int64").to_numpy())

    merged_frames: list[pd.DataFrame | None] = []
    for frame in frames:
        if frame is None:
            merged_frames.append(None)
            continue
        merged = frame.copy()
        if "article_id" in merged.columns and "click_article_id" not in merged.columns:
            merged = merged.rename(columns={"article_id": "click_article_id"})
        merged = merged.merge(svd_df, on="click_article_id", how="left")
        merged[list(article_svd_features(n_components))] = merged[list(article_svd_features(n_components))].fillna(0.0)
        merged_frames.append(merged)
    return tuple(merged_frames)
