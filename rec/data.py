from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from rec.config import DataConfig, PathConfig, train_feature_name
from rec.features import (
    HISTORY_FEATURE,
    META_FLOAT_FEATURES,
    FeaturePreset,
    add_article_svd_features,
    add_rank_score_features,
    ensure_feature_columns,
)


PAD_IDX = 0
OOV_IDX = 1


@dataclass
class LoadedFrames:
    """从磁盘读入的原始特征表集合。

    train/val/test 是候选样本表，history 是用户历史点击表；此阶段还没有把历史序列拼到候选样本上。
    """

    train: pd.DataFrame
    val: pd.DataFrame | None
    test: pd.DataFrame
    history: pd.DataFrame


@dataclass
class MergedFrames:
    """已经补好 query_id 和历史点击序列的样本表集合。"""

    train: pd.DataFrame
    val: pd.DataFrame | None
    test: pd.DataFrame
    val_hit_mask: pd.Series | None
    val_hit_frame: pd.DataFrame | None


@dataclass
class PreparedRankingData:
    """训练前的最终数据包。

    *_frame 保留 pandas 版本，便于调试和评估；x_* 是已经转成 numpy 的模型输入字典；
    y_* 是 label 数组；sparse_maps/scalers 记录离散映射和连续特征归一化器，供验证/测试复用。
    """

    train_frame: pd.DataFrame
    val_frame: pd.DataFrame | None
    test_frame: pd.DataFrame
    val_hit_mask: pd.Series | None
    val_hit_frame: pd.DataFrame | None
    x_train: dict[str, np.ndarray]
    y_train: np.ndarray
    x_val: dict[str, np.ndarray] | None
    y_val: np.ndarray | None
    x_test: dict[str, np.ndarray] | None
    sparse_maps: dict[str, dict[int, int]]
    sparse_vocab_sizes: dict[str, int]
    dense_scalers: dict[str, MinMaxScaler]
    feature_preset: FeaturePreset


class RankingDataset(Dataset):
    """把 numpy input dict 包成 PyTorch Dataset。

    x_dict 的每个 key 是一个特征名，每个 value 的第 0 维都对应样本行号。
    """

    def __init__(self, x_dict: dict[str, np.ndarray], y: np.ndarray | None = None):
        self.x_dict = x_dict
        self.y = y
        self.length = len(next(iter(x_dict.values())))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        sample = {key: value[index] for key, value in self.x_dict.items()}
        if self.y is None:
            return sample
        return sample, self.y[index]


class QueryBatchSampler:
    """按 query_id 组织 batch，使用 numpy 边界避免为每行创建 Python list。"""

    def __init__(self, query_ids: Sequence[int], batch_size: int, shuffle: bool=True, seed: int | None=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        query_ids = np.asarray(query_ids)
        if query_ids.size and np.all(query_ids[1:] >= query_ids[:-1]):
            self.order = np.arange(query_ids.size, dtype=np.int64)
            sorted_query_ids = query_ids
        else:
            self.order = np.argsort(query_ids, kind='mergesort')
            sorted_query_ids = query_ids[self.order]

        if sorted_query_ids.size == 0:
            self.starts = np.asarray([], dtype=np.int64)
            self.ends = np.asarray([], dtype=np.int64)
        else:
            boundaries = np.flatnonzero(sorted_query_ids[1:] != sorted_query_ids[:-1]) + 1
            self.starts = np.concatenate(([0], boundaries)).astype(np.int64)
            self.ends = np.concatenate((boundaries, [sorted_query_ids.size])).astype(np.int64)
        self.block_sizes = self.ends - self.starts

    def _block(self, block_index: int) -> np.ndarray:
        return self.order[self.starts[block_index]:self.ends[block_index]]

    def _block_order(self) -> np.ndarray:
        block_order = np.arange(len(self.starts), dtype=np.int64)
        if self.shuffle and block_order.size:
            rng = np.random.default_rng(None if self.seed is None else self.seed + self.epoch)
            rng.shuffle(block_order)
        return block_order

    def __iter__(self):
        block_order = self._block_order()
        self.epoch += 1

        batch_parts: list[np.ndarray] = []
        batch_size = 0
        for block_index in block_order:
            block = self._block(int(block_index))
            block_size = int(block.size)
            if block_size >= self.batch_size:
                if batch_parts:
                    yield np.concatenate(batch_parts).tolist()
                    batch_parts = []
                    batch_size = 0
                yield block.tolist()
                continue
            if batch_size + block_size > self.batch_size:
                yield np.concatenate(batch_parts).tolist()
                batch_parts = [block]
                batch_size = block_size
            else:
                batch_parts.append(block)
                batch_size += block_size
        if batch_parts:
            yield np.concatenate(batch_parts).tolist()

    def __len__(self) -> int:
        count = 0
        size = 0
        for block_index in self._block_order():
            block_size = int(self.block_sizes[int(block_index)])
            if block_size >= self.batch_size:
                count += 1 + int(size > 0)
                size = 0
            elif size + block_size > self.batch_size:
                count += 1
                size = block_size
            else:
                size += block_size
        return count + int(size > 0)

def tensor_dtype(array: np.ndarray) -> torch.dtype:
    return torch.long if np.issubdtype(array.dtype, np.integer) else torch.float32


def collate_features(batch: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """把 Dataset 取出的多条样本拼成一个 batch tensor 字典。"""

    return {
        key: torch.tensor(np.stack([sample[key] for sample in batch]), dtype=tensor_dtype(np.asarray(batch[0][key])))
        for key in batch[0]
    }


def collate_with_labels(batch: Sequence[tuple[dict[str, Any], Any]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    features, labels = zip(*batch)
    return collate_features(features), torch.tensor(np.asarray(labels), dtype=torch.float32)


def make_loader(
    x_dict: dict[str, np.ndarray],
    y: np.ndarray | None = None,
    batch_size: int = 256,
    shuffle: bool = False,
    group_by_query: bool = False,
    seed: int | None = None,
) -> DataLoader:
    """创建 PyTorch DataLoader；训练时可选择按 query_id 保持候选集合分组。"""

    dataset = RankingDataset(x_dict, y)
    if y is None:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_features)
    if group_by_query:
        sampler = QueryBatchSampler(x_dict["query_id"], batch_size=batch_size, shuffle=shuffle, seed=seed)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_with_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_with_labels)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


def collect_first_user_ids(path: Path, limit_users: int, chunksize: int = 200_000) -> set[int]:
    """从文件开头按出现顺序收集前 limit_users 个用户 id。

    注意 limit_users 限制的是用户数，不是样本行数；后续会保留这些用户的完整候选集合，
    这样同一个用户下的正负样本不会被截断，适合 debug 小数据集。
    """

    users: list[int] = []
    seen: set[int] = set()
    for chunk in pd.read_csv(path, usecols=["user_id"], chunksize=chunksize):
        for user_id in pd.unique(chunk["user_id"]):
            user_id = int(user_id)
            if user_id in seen:
                continue
            seen.add(user_id)
            users.append(user_id)
            if len(users) >= limit_users:
                return set(users)
    return set(users)


def read_csv_for_users(path: Path, user_ids: set[int], chunksize: int = 200_000) -> pd.DataFrame:
    """分块读取 CSV，只保留指定用户的所有行。

    chunk["user_id"].isin(user_ids) 会生成布尔 mask；chunk[mask] 就是 pandas 的按行筛选。
    """

    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        selected = chunk[chunk["user_id"].isin(user_ids)]
        if len(selected):
            frames.append(selected)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.read_csv(path, nrows=0)


def read_feature_split(path: Path, config: DataConfig) -> tuple[pd.DataFrame, set[int]]:
    """读取一个 train/val/test split，并返回本 split 覆盖到的用户集合。"""

    if config.limit_users:
        user_ids = collect_first_user_ids(path, config.limit_users)
        return read_csv_for_users(path, user_ids), user_ids
    frame = read_csv(path, config.limit_rows)
    return frame, set(pd.to_numeric(frame["user_id"], errors="coerce").dropna().astype("int64").tolist())


def load_feature_frames(paths: PathConfig, config: DataConfig) -> LoadedFrames:
    """加载候选特征、历史点击和文章内容特征。

    如果配置了 limit_users，会先抽用户，再读取这些用户的完整候选集合和历史点击；
    这比简单 limit_rows 更适合调试排序任务，因为每个用户的候选列表仍然完整。
    """

    train, train_users = read_feature_split(paths.save_path / train_feature_name(config.train_variant), config)
    val_path = paths.save_path / "val_user_item_feats_df_all.csv"
    test_path = paths.save_path / "tst_user_item_feats_df_all.csv"
    history_path = paths.save_path / "click_hist_all.csv"
    val, val_users = read_feature_split(val_path, config) if config.mode == "offline" and val_path.exists() else (None, set())
    test, test_users = read_feature_split(test_path, config) if test_path.exists() else (pd.DataFrame(), set())
    sampled_users = train_users | val_users | test_users
    history = read_csv_for_users(history_path, sampled_users) if config.limit_users else read_csv(history_path, config.limit_rows)

    frames = []
    for frame in (train, val, test):
        if len(frame):
            frames.append(add_rank_score_features(frame))
        else:
            frames.append(frame)

    train, val, test = add_article_svd_features(frames, paths.data_path, paths.save_path, config.article_svd_dim)
    return LoadedFrames(train=train, val=val, test=test, history=history)


def add_query_id(df: pd.DataFrame) -> pd.DataFrame:
    """补 query_id 列。

    query_id 当前取自 user_id，只用于 DataLoader/排序损失的“同组候选”标识。
    user_id 本身不会进入模型特征；这里也不会把它加入 FeaturePreset。
    """

    df = df.copy()
    if "article_id" in df.columns and "click_article_id" not in df.columns:
        df = df.rename(columns={"article_id": "click_article_id"})
    df["query_id"] = pd.to_numeric(df["user_id"], errors="coerce").fillna(0).astype("int64")
    return df


def build_history_sequences(history: pd.DataFrame) -> pd.DataFrame:
    """把历史点击明细压成每个用户一条点击序列。

    sort_values 先按用户和时间排序；groupby("user_id")["click_article_id"].agg(list)
    会把同一用户的文章 id 收集成 Python list，作为历史行为序列特征。
    """

    sort_cols = ["user_id"] + (["click_timestamp"] if "click_timestamp" in history.columns else [])
    history = history.sort_values(sort_cols, kind="mergesort")
    sequences = history.groupby("user_id", sort=False)["click_article_id"].agg(list).reset_index()
    sequences[HISTORY_FEATURE] = sequences["click_article_id"]
    return sequences[["user_id", HISTORY_FEATURE]]


def merge_history_frames(frames: LoadedFrames) -> MergedFrames:
    """把历史点击序列拼到 train/val/test 候选样本上。

    merge(history, on="user_id", how="left") 表示按用户左连接：每条候选样本保留，
    同一个用户会拿到同一条历史点击序列；没有历史的用户用空 list。
    """

    history = build_history_sequences(frames.history)

    def merge(frame: pd.DataFrame | None) -> pd.DataFrame | None:
        if frame is None:
            return None
        merged = add_query_id(frame).merge(history, on="user_id", how="left")
        merged[HISTORY_FEATURE] = merged[HISTORY_FEATURE].apply(lambda value: value if isinstance(value, list) else [])
        return merged

    train = merge(frames.train)
    val = merge(frames.val)
    test = merge(frames.test) if len(frames.test) else add_query_id(frames.test.copy())
    # groupby("query_id")["label"].transform("max") 会把“本组是否有正样本”的结果广播回组内每一行。
    # eq(1) 得到布尔 mask，只保留至少命中过一次的验证用户，避免评估时出现没有正例的 query。
    val_hit_mask = val.groupby("query_id")["label"].transform("max").eq(1) if val is not None and "label" in val else None
    val_hit_frame = val[val_hit_mask].copy() if val_hit_mask is not None else None
    return MergedFrames(train=train, val=val, test=test, val_hit_mask=val_hit_mask, val_hit_frame=val_hit_frame)


def normalize_dense_features(
    train: pd.DataFrame,
    val: pd.DataFrame | None,
    test: pd.DataFrame,
    dense_features: Sequence[str],
) -> dict[str, MinMaxScaler]:
    """用训练集拟合连续特征归一化器，并复用到 val/test。
    fit_transform 只在 train 上学习 min/max；val/test 只 transform，避免验证/测试信息泄漏。
    scalers = normalize_dense_features(train, val, test, preset.dense_features)
    """

    scalers: dict[str, MinMaxScaler] = {}
    for feature in dense_features:
        scaler = MinMaxScaler()
        train[feature] = scaler.fit_transform(train[[feature]]).astype("float32")
        if val is not None:
            val[feature] = scaler.transform(val[[feature]]).astype("float32")
        if len(test):
            test[feature] = scaler.transform(test[[feature]]).astype("float32")
        scalers[feature] = scaler
    return scalers


def build_sparse_maps(
    frames: Sequence[pd.DataFrame],
    sparse_features: Sequence[str],
    history_feature: str = HISTORY_FEATURE,
) -> dict[str, dict[int, int]]:
    """为每个稀疏特征建立“原始取值 -> 连续 id”的映射。
    click_article_id 同时出现在候选文章和历史序列里，所以要把历史里的文章 id 也加入词表。
    0 预留给 PAD，1 预留给 OOV，方便处理补齐和未知文章。
    """

    sparse_maps: dict[str, dict[int, int]] = {}
    for feature in sparse_features:
        values = pd.concat(
            [pd.to_numeric(frame[feature], errors="coerce").fillna(0).astype("int64") for frame in frames],
            ignore_index=True,
        )
        #转成 Python set，去重，并且去掉 0。
        unique_values = set(int(value) for value in values.tolist() if int(value) != 0)
        if feature == "click_article_id":
            for frame in frames:
                for sequence in frame[history_feature].tolist():
                    #因为文章 ID 不只出现在当前候选文章列里，也出现在用户历史点击序列里：
                    unique_values.update(int(value) for value in sequence if int(value) != 0)
            sparse_maps[feature] = {value: index for index, value in enumerate(sorted(unique_values), start=OOV_IDX + 1)}
            #0 = PAD，1 = OOV，所以真实文章 ID 从 2 开始编号。
        else:
            sparse_maps[feature] = {value: index for index, value in enumerate(sorted(unique_values), start=1)}
    return sparse_maps


def sparse_vocab_sizes(sparse_maps: dict[str, dict[int, int]], sparse_features: Sequence[str]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for feature in sparse_features:
        minimum = OOV_IDX + 1 if feature == "click_article_id" else 1
        sizes[feature] = max(max(sparse_maps[feature].values(), default=minimum - 1) + 1, minimum)
    return sizes


def map_sparse(series: pd.Series, mapping: dict[int, int], feature: str) -> np.ndarray:
    """把 pandas Series 中的原始稀疏取值映射成 numpy int64 id 数组。"""

    values = pd.to_numeric(series, errors="coerce").fillna(0).astype("int64")
    default = OOV_IDX if feature == "click_article_id" else PAD_IDX
    return values.map(mapping).fillna(default).astype("int64").to_numpy()


def pad_sequences(sequences: Sequence[Sequence[int]], maxlen: int) -> np.ndarray:
    """把不等长历史序列补齐成二维数组，只保留最近 maxlen 个点击。"""

    padded = np.zeros((len(sequences), maxlen), dtype=np.int64)
    for index, sequence in enumerate(sequences):
        recent = list(sequence)[-maxlen:]
        if recent:
            padded[index, -len(recent) :] = np.asarray(recent, dtype=np.int64)
    return padded


def map_history(sequences: Sequence[Sequence[int]], item_mapping: dict[int, int], maxlen: int) -> np.ndarray:
    """把历史文章 id 序列映射成 embedding id，并做 padding。"""

    mapped_sequences: list[list[int]] = []
    for sequence in sequences:
        mapped_sequences.append([item_mapping.get(int(item_id), OOV_IDX) for item_id in sequence if int(item_id) != 0])
    return pad_sequences(mapped_sequences, maxlen)


def build_model_input(
    frame: pd.DataFrame,
    sparse_maps: dict[str, dict[int, int]],
    preset: FeaturePreset,
    max_history_len: int,
) -> dict[str, np.ndarray]:
    """把一个 DataFrame split 转成模型需要的 numpy input dict。

    这里是从 pandas 到模型输入的关键转换：
    稀疏列先过 sparse_maps 变成 int64 id，连续列直接转 float32 numpy，
    query_id/sample_prob 等 meta 列也放进字典供训练流程使用；历史序列转成 [样本数, max_history_len]。
    """

    x: dict[str, np.ndarray] = {}
    #数值大小本身没有连续意义。要映射成 embedding 下标。
    # 因为类别 ID 的数字大小没有意义。10 不比 3 大三倍，它只是另一个类别。所以模型不应该把它当连续数值，而应该给每个类别学一个向量。
    for feature in preset.sparse_features:
        x[feature] = map_sparse(frame[feature], sparse_maps[feature], feature)
    #连续特征直接转成 float32 numpy 数组。
    for feature in preset.dense_features:
        x[feature] = frame[feature].astype("float32").to_numpy()
    # query_id/sample_prob 这些 meta 列也转成 numpy，供训练流程使用，但它们不会进模型特征。
    for feature in preset.meta_features:
        x[feature] = frame[feature].astype("int64").to_numpy()
    # 不是直接给模型学习语义的主特征，而是训练/评估流程要用的辅助字段。
    for feature in META_FLOAT_FEATURES:
        x[feature] = frame[feature].astype("float32").to_numpy()
    x[HISTORY_FEATURE] = map_history(frame[HISTORY_FEATURE].tolist(), sparse_maps["click_article_id"], max_history_len)
    return x


def prepare_ranking_data(frames: MergedFrames, preset: FeaturePreset, config: DataConfig) -> PreparedRankingData:
    """完成训练前的数据准备：补列、归一化、建词表、转 numpy 输入。"""

    train = ensure_feature_columns(frames.train, preset)
    val = ensure_feature_columns(frames.val, preset) if frames.val is not None else None
    test = ensure_feature_columns(frames.test, preset) if len(frames.test) else frames.test.copy()

    #连续特征进入模型前的最后清洗/归一化
    scalers = normalize_dense_features(train, val, test, preset.dense_features)
    
    # 只用训练集建稀疏词表，避免把 val/test 的未来类别分布提前暴露给模型。
    vocab_frames = [train]
    
    maps = build_sparse_maps(vocab_frames, preset.sparse_features)
    vocab_sizes = sparse_vocab_sizes(maps, preset.sparse_features)

    x_train = build_model_input(train, maps, preset, config.max_history_len)
    # label 保持为一维 float32 numpy 数组；x_train/x_val/x_test 则是“特征名 -> numpy 数组”的字典。
    y_train = train["label"].astype("float32").to_numpy()
    x_val = build_model_input(val, maps, preset, config.max_history_len) if val is not None else None
    y_val = val["label"].astype("float32").to_numpy() if val is not None else None
    x_test = build_model_input(test, maps, preset, config.max_history_len) if len(test) else None

    return PreparedRankingData(
        train_frame=train,
        val_frame=val,
        test_frame=test,
        val_hit_mask=frames.val_hit_mask,
        val_hit_frame=frames.val_hit_frame,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        sparse_maps=maps,
        sparse_vocab_sizes=vocab_sizes,
        dense_scalers=scalers,
        feature_preset=preset,
    )
