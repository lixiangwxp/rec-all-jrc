from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ndcg_score, roc_auc_score

from rec.data import make_loader, move_batch_to_device


def output_to_scores(raw_output: np.ndarray, head_type: str, score_mode: str) -> np.ndarray:
    """把模型 raw output 转成“越大越应该排前面”的一维分数。

    Debug 时从 evaluate_raw 进入这里，重点看 raw_output 的形状：
    - two_logit + click：把两个 logit 做 softmax，取点击类概率；
    - two_logit + diff：用正类 logit 减负类 logit 当排序分；
    - single_logit：模型已经输出一个标量，直接拉平成一维。
    """

    raw_output = np.asarray(raw_output)
    if head_type == "two_logit":
        if score_mode == "click":
            shifted = raw_output - raw_output.max(axis=1, keepdims=True)
            probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            return probs[:, 1].astype("float32")
        return (raw_output[:, 1] - raw_output[:, 0]).astype("float32")
    return raw_output.reshape(-1).astype("float32")


def safe_auc(labels: Sequence[float], preds: Sequence[float]) -> float:
    """计算 AUC；如果当前标签只有一类，返回 nan 避免 sklearn 报错。

    这里不影响排序指标，只是让日志在小样本 debug 时仍能继续生成。
    """

    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, preds))


def ndcg_at_k(labels: Sequence[float], preds: Sequence[float], query_ids: Sequence[int], k: int) -> float:
    """按 query 分组计算 NDCG@k。

    单步调试时可以观察 group_scores/group_labels：
    每个 query_id 下保存的是同一次推荐请求里的候选文章分数和标签。
    """

    group_scores: dict[int, list[float]] = defaultdict(list)
    group_labels: dict[int, list[float]] = defaultdict(list)
    for label, pred, query_id in zip(labels, preds, query_ids):
        group_scores[int(query_id)].append(float(pred))
        group_labels[int(query_id)].append(float(label))
    values = []
    for query_id in group_scores:
        y_true = np.array([group_labels[query_id]])
        y_score = np.array([group_scores[query_id]])
        values.append(0.0 if y_true.sum() == 0 else float(ndcg_score(y_true, y_score, k=k)))
    return float(np.mean(values)) if values else 0.0


def mrr_hit_at_k(frame: pd.DataFrame, score_col: str, k: int) -> dict[str, float]:
    """按预测分数排序后计算 MRR 和 Hit Rate。

    ranked 是真正用于指标的排序结果：同一个 query 内 pred_score 越大越靠前。
    labels 中第一个 1 的位置就是命中文章的排名，排名不超过 k 才计入 hit。
    """

    ranked = frame.sort_values(["query_id", score_col], ascending=[True, False], kind="mergesort")
    mrr_sum = 0.0
    hit_sum = 0.0
    query_count = 0
    pos_query_count = 0
    for _, group in ranked.groupby("query_id", sort=False):
        query_count += 1
        labels = group["label"].to_numpy()
        if labels.sum() == 0:
            continue
        pos_query_count += 1
        rank = int(np.where(labels == 1)[0][0]) + 1
        if rank <= k:
            mrr_sum += 1.0 / rank
            hit_sum += 1.0
    denom = max(query_count, 1)
    return {
        "mrr": float(mrr_sum / denom),
        "hit_rate": float(hit_sum / denom),
        "query_count": int(query_count),
        "pos_query_count": int(pos_query_count),
    }


def prediction_frame(base_df: pd.DataFrame, scores: Sequence[float]) -> pd.DataFrame:
    """把验证集原始 frame 和模型分数拼成可落盘检查的预测明细。

    Debug 时可以打开返回的 frame：
    pred_score/rankmixer_score 是模型排序分，其他列来自验证候选数据，方便对照原始召回分和标签。
    """

    keep_cols = [
        column
        for column in (
            "query_id",
            "user_id",
            "click_article_id",
            "label",
            "score",
            "rank",
            "rank_recip",
            "category_id",
            "article_hot_level",
            "article_user_num",
        )
        if column in base_df.columns
    ]
    frame = base_df[keep_cols].copy()
    frame["pred_score"] = np.asarray(scores, dtype="float32").reshape(-1)
    frame["rankmixer_score"] = frame["pred_score"]
    frame["pred_rank"] = frame.groupby("query_id")["pred_score"].rank(method="first", ascending=False).astype("int32")
    if "label" in frame.columns:
        frame["label"] = pd.to_numeric(frame["label"], errors="coerce").fillna(0).astype("int8")
    return frame


@dataclass
class RankingEvaluator:
    """推荐排序评估器。

    topk 控制 MRR/Hit/NDCG 只关注前多少个候选；训练时由 TrainConfig.topk 传入。
    """

    topk: int = 5

    def evaluate_scores(self, frame: pd.DataFrame, labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
        """在已经有一维排序分数的前提下计算指标。

        这里会先生成 pred_df，再基于同一批 scores 计算 AUC、NDCG、MRR 和 Hit Rate。
        """

        pred_df = prediction_frame(frame, scores)
        mrr_hit = mrr_hit_at_k(pred_df, "pred_score", self.topk)
        return {
            "auc": safe_auc(labels, scores),
            "ndcg": ndcg_at_k(labels, scores, pred_df["query_id"].to_numpy(), self.topk),
            "mrr": mrr_hit["mrr"],
            "hit_rate": mrr_hit["hit_rate"],
            "query_count": mrr_hit["query_count"],
            "pos_query_count": mrr_hit["pos_query_count"],
            "pred_df": pred_df,
        }

    def evaluate_raw(
        self,
        raw_output: np.ndarray,
        labels: np.ndarray,
        frame: pd.DataFrame,
        head_type: str,
        score_mode: str,
        hit_mask: pd.Series | None = None,
    ) -> dict[str, Any]:
        """从模型原始输出开始完成验证评估。

        这是训练循环里每个 epoch 后调用的评估入口。建议 F11 跟进观察三步：
        1. output_to_scores：把 raw_output 转成一维排序分 scores；
        2. evaluate_scores(full)：在完整验证候选上计算 full_auc/full_mrr/full_ndcg 等指标；
        3. 如果有 hit_mask，再只保留命中召回子集计算 hit_mrr/hit_ndcg 等指标。

        返回值里的 val_pred_df 是完整验证集预测明细，hit_pred_df 是 hit 子集预测明细。
        """

        scores = output_to_scores(raw_output, head_type, score_mode)
        # full 指标覆盖整个验证候选集合，反映模型在所有候选上的排序表现。
        full = self.evaluate_scores(frame, labels, scores)
        result = {f"full_{key}": value for key, value in full.items() if key != "pred_df"}
        result["val_pred_df"] = full["pred_df"]

        if hit_mask is not None:
            # hit 指标只看 hit_mask=True 的候选子集，常用于观察召回命中范围内的重排效果。
            hit_mask_array = hit_mask.to_numpy(dtype=bool)
            hit = self.evaluate_scores(frame.loc[hit_mask_array], labels[hit_mask_array], scores[hit_mask_array])
            result.update({f"hit_{key}": value for key, value in hit.items() if key != "pred_df"})
            result["hit_pred_df"] = hit["pred_df"]
        return result


def raw_predict_outputs(
    model: torch.nn.Module,
    x_data: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """用当前模型批量生成验证集 raw output。

    这里和训练 DataLoader 很像，但只做 forward：
    model.eval() 切到评估模式，torch.no_grad() 关闭梯度记录，最后把每个 batch 的输出拼回 numpy。
    """

    loader = make_loader(x_data, batch_size=batch_size)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)
