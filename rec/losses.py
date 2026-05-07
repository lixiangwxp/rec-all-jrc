from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossOutput:
    """训练步骤统一返回的 loss 容器。

    total 是真正用于 backward 的标量 tensor；parts 存放各个子 loss，方便日志和调试。
    """

    total: torch.Tensor
    parts: dict[str, torch.Tensor]


def ranking_scores(output: torch.Tensor, head_type: str, score_mode: str) -> torch.Tensor:
    """把模型输出转换成排序分数，返回 shape `[B]`。

    `two_logit` 表示模型输出 `[B, 2]`：第 0 列是未点击 logit，第 1 列是点击 logit。
    - score_mode="click"：先 softmax，再取点击概率 `P(click)`，适合看概率意义。
    - score_mode="diff"：返回 `click_logit - non_click_logit`，适合 pairwise/BPR 比较相对优势。
    `single_logit` 时 output 本身就是一个标量分数，score_mode 通常写成 "scalar"。
    """

    if head_type == "two_logit":
        if score_mode == "click":
            return torch.softmax(output, dim=1)[:, 1]
        return output[:, 1] - output[:, 0]
    return output.reshape(-1)


class ContextGELoss(nn.Module):
    """按 query 上下文计算的 GE loss。

    batch 中同一个 query_id 下的候选样本会被放在一起比较。logits shape 是 `[B, 2]`，
    labels shape 是 `[B]`，query_ids shape 是 `[B]`。balanced=True 时会分别按正负样本数归一。
    """

    def __init__(self, balanced: bool = True, pos_weight: float = 1.0, neg_weight: float = 1.0):
        super().__init__()
        self.balanced = balanced
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, query_ids: torch.Tensor) -> torch.Tensor:
        labels_long = labels.long()
        # labels_onehot: `[B, 2]`，第 0 列是负样本/未点击，第 1 列是正样本/点击。
        labels_onehot = torch.stack([1 - labels_long, labels_long], dim=1).float()
        batch_size = logits.size(0)
        # mask[i, j] 为 True 表示第 i、j 个样本来自同一个 query，可以放在同一上下文里归一化。
        mask = query_ids.unsqueeze(0).eq(query_ids.unsqueeze(1))
        logits_tile = logits.unsqueeze(1).expand(-1, batch_size, -1).masked_fill(~mask.unsqueeze(-1), -1e9)
        labels_tile = labels_onehot.unsqueeze(1).expand(-1, batch_size, -1) * mask.unsqueeze(-1).float()

        # 分别对负类 logit 和正类 logit 在同 query 上下文中做 log_softmax。
        log_softmax_neg = F.log_softmax(logits_tile[:, :, 0], dim=0)
        log_softmax_pos = F.log_softmax(logits_tile[:, :, 1], dim=0)
        y_neg = labels_tile[:, :, 0]
        y_pos = labels_tile[:, :, 1]
        loss_neg = -(y_neg * log_softmax_neg).sum(dim=0)
        loss_pos = -(y_pos * log_softmax_pos).sum(dim=0)

        if self.balanced:
            neg_count = y_neg.sum(dim=0).clamp_min(1.0)
            pos_count = y_pos.sum(dim=0).clamp_min(1.0)
            return (self.neg_weight * loss_neg / neg_count + self.pos_weight * loss_pos / pos_count).mean()
        context_size = mask.float().sum(dim=0).clamp_min(1.0)
        return ((loss_neg + loss_pos) / context_size).mean()


class BalancedGELoss(nn.Module):
    """交叉熵 + balanced GE 的 two_logit loss。

    用 CE 保持逐样本分类监督，用 GE 加强同一个 query 下候选集合的上下文约束。
    total = alpha * CE + (1-alpha) * GE。
    """

    name = "balanced_ge"
    head_type = "two_logit"

    def __init__(self, config: Any):
        super().__init__()
        self.alpha = config.alpha
        self.ge = ContextGELoss(balanced=True)

    def forward(self, output: torch.Tensor, labels: torch.Tensor, batch: dict[str, torch.Tensor]) -> LossOutput:
        # output: `[B, 2]` two_logit；labels: `[B]`，取值 0/1。
        ce = F.cross_entropy(output, labels.long())
        ge = self.ge(output, labels, batch["query_id"])
        total = self.alpha * ce + (1.0 - self.alpha) * ge
        return LossOutput(total=total, parts={"ce_loss": ce, "ge_loss": ge})


class OldGELoss(nn.Module):
    """旧版 GE 组合 loss。

    与 BalancedGELoss 一样是 CE + GE，但 GE 内部不按正负样本数分别平衡。
    """

    name = "old_ge"
    head_type = "two_logit"

    def __init__(self, config: Any):
        super().__init__()
        self.alpha = config.alpha
        self.ge = ContextGELoss(balanced=False)

    def forward(self, output: torch.Tensor, labels: torch.Tensor, batch: dict[str, torch.Tensor]) -> LossOutput:
        ce = F.cross_entropy(output, labels.long())
        ge = self.ge(output, labels, batch["query_id"])
        total = self.alpha * ce + (1.0 - self.alpha) * ge
        return LossOutput(total=total, parts={"ce_loss": ce, "ge_loss": ge})


class QuerySoftmaxCELoss(nn.Module):
    """按 query 分组的 listwise softmax CE。

    适用于 single_logit 输出：每个候选样本一个分数 `[B]`。同一个 query 下，
    正样本希望在该候选列表的 softmax 里获得更高概率。
    """

    name = "query_softmax_ce"
    head_type = "single_logit"

    def __init__(self, config: Any, tau: float | None = None, use_sample_prob: bool = False):
        super().__init__()
        self.tau = config.listwise_tau if tau is None else tau
        self.use_sample_prob = use_sample_prob

    def forward(self, output: torch.Tensor, labels: torch.Tensor, batch: dict[str, torch.Tensor]) -> LossOutput:
        # single_logit 的 ranking score 就是模型输出本身；这里用 "scalar" 表示标量分数。
        scores = ranking_scores(output, "single_logit", "scalar")
        loss = query_softmax_ce(scores, labels, batch["query_id"], batch.get("sample_prob") if self.use_sample_prob else None, self.tau)
        return LossOutput(total=loss, parts={"query_softmax_ce": loss})


class RankWeightedBPRLoss(nn.Module):
    """按 query 构造正负样本对的 BPR loss。

    scores shape 为 `[B]`。对每个 query，取所有正样本分数 pos_scores 和负样本分数 neg_scores，
    优化 `pos_score > neg_score`。neg_weight 可用于给负样本对加权，例如 rank_recip。
    """

    def __init__(self, max_neg_per_query: int = 80):
        super().__init__()
        self.max_neg_per_query = max_neg_per_query

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        query_ids: torch.Tensor,
        neg_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        weights = neg_weight.reshape(-1) if neg_weight is not None else None
        for query_id in torch.unique(query_ids):
            # 每次只处理一个 query 下的候选列表，避免跨 query 生成没有业务意义的正负对。
            mask = query_ids.eq(query_id)
            group_scores = scores[mask]
            group_labels = labels[mask]
            pos_scores = group_scores[group_labels > 0.5]
            neg_scores = group_scores[group_labels <= 0.5]
            if pos_scores.numel() == 0 or neg_scores.numel() == 0:
                continue
            neg_weights = weights[mask][group_labels <= 0.5].clamp_min(1e-4) if weights is not None else None
            if self.max_neg_per_query > 0 and neg_scores.numel() > self.max_neg_per_query:
                # 负样本太多时随机截断，控制 pairwise 矩阵大小和训练耗时。
                sample_idx = torch.randperm(neg_scores.numel(), device=neg_scores.device)[: self.max_neg_per_query]
                neg_scores = neg_scores[sample_idx]
                neg_weights = neg_weights[sample_idx] if neg_weights is not None else None
            # pair_loss shape `[num_pos, num_neg]`，每个元素对应一个正负样本对。
            pair_loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0))
            if neg_weights is None:
                losses.append(pair_loss.mean())
            else:
                pair_weight = neg_weights.unsqueeze(0).expand_as(pair_loss)
                losses.append((pair_loss * pair_weight).sum() / pair_weight.sum().clamp_min(1e-8))
        return torch.stack(losses).mean() if losses else scores.sum() * 0.0


class JRCBPRLoss(nn.Module):
    """JRC 风格的 two_logit 组合 loss：BalancedGE + BPR。

    base 部分仍然用 two_logit 的 CE/GE 监督“点不点击”；BPR 部分使用
    `click_logit - non_click_logit` 作为排序分数，让同 query 内正样本排在负样本前。
    total = base.total + jrc_bpr_weight * bpr。
    """

    name = "jrc_bpr"
    head_type = "two_logit"

    def __init__(self, config: Any):
        super().__init__()
        self.base = BalancedGELoss(config)#CE + GE
        self.bpr = RankWeightedBPRLoss(config.max_neg_per_query)# pairwise ranking loss
        self.bpr_weight = config.jrc_bpr_weight

    def forward(self, output: torch.Tensor, labels: torch.Tensor, batch: dict[str, torch.Tensor]) -> LossOutput:
        base = self.base(output, labels, batch)
        # diff 分数体现点击 logit 相对未点击 logit 的优势，shape `[B]`。
        scores = ranking_scores(output, "two_logit", "diff")
        bpr = self.bpr(scores, labels, batch["query_id"], batch.get("rank_recip"))
        total = base.total + self.bpr_weight * bpr
        parts = dict(base.parts)
        parts["bpr_loss"] = bpr
        return LossOutput(total=total, parts=parts)


class ListwiseBPRBCELoss(nn.Module):
    """single_logit 的组合 loss：listwise softmax CE + BPR + BCE 辅助项。

    listwise 负责同 query 候选列表内的整体排序；BPR 负责正负样本 pair 的相对顺序；
    BCE 辅助项保留逐样本点击/未点击的二分类监督。total =
    listwise + bpr_weight * bpr + bce_aux_weight * bce。
    """

    name = "listwise_bpr_bce"
    head_type = "single_logit"

    def __init__(self, config: Any):
        super().__init__()
        self.listwise = QuerySoftmaxCELoss(config, tau=config.listwise_tau, use_sample_prob=True)
        self.bpr = RankWeightedBPRLoss(config.max_neg_per_query)
        self.bpr_weight = config.bpr_weight
        self.bce_aux_weight = config.bce_aux_weight

    def forward(self, output: torch.Tensor, labels: torch.Tensor, batch: dict[str, torch.Tensor]) -> LossOutput:
        # scalar 分数来自 single_logit 输出，既用于 listwise/BPR，也用于 BCEWithLogits。
        scores = ranking_scores(output, "single_logit", "scalar")
        listwise = self.listwise(output, labels, batch).total
        bpr = self.bpr(scores, labels, batch["query_id"], batch.get("rank_recip"))
        bce = F.binary_cross_entropy_with_logits(scores, labels.float())
        total = listwise + self.bpr_weight * bpr + self.bce_aux_weight * bce
        return LossOutput(total=total, parts={"listwise_loss": listwise, "bpr_loss": bpr, "bce_aux_loss": bce})


def query_softmax_ce(
    scores: torch.Tensor,
    labels: torch.Tensor,
    query_ids: torch.Tensor,
    sample_prob: torch.Tensor | None = None,
    tau: float = 1.0,
) -> torch.Tensor:
    """同 query 候选列表内的 softmax 交叉熵。

    scores/labels/query_ids 都是 `[B]`。函数按 query_id 分组，只对含正样本的 query 计算；
    若传入 sample_prob，会先做采样概率校正；tau 是 softmax 温度，越小分布越尖锐。
    """

    corrected_scores = scores.reshape(-1)
    if sample_prob is not None:
        # 采样概率越小，校正后分数越高，用于减轻负采样分布带来的偏差。
        corrected_scores = corrected_scores - torch.log(sample_prob.reshape(-1).clamp_min(1e-4))
    corrected_scores = corrected_scores / tau

    losses: list[torch.Tensor] = []
    for query_id in torch.unique(query_ids):
        mask = query_ids.eq(query_id)
        group_labels = labels[mask]
        pos_idx = torch.nonzero(group_labels > 0.5, as_tuple=False).flatten()
        if pos_idx.numel() == 0:
            continue
        # log_probs 是当前 query 的候选列表 softmax 后的 log 概率，只惩罚正样本概率不够高。
        log_probs = F.log_softmax(corrected_scores[mask], dim=0)
        losses.append(-log_probs[pos_idx].mean())
    return torch.stack(losses).mean() if losses else corrected_scores.sum() * 0.0


LOSS_REGISTRY = {
    "balanced_ge": BalancedGELoss,
    "old_ge": OldGELoss,
    "query_softmax_ce": QuerySoftmaxCELoss,
    "jrc_bpr": JRCBPRLoss,
    "listwise_bpr_bce": ListwiseBPRBCELoss,
}


def build_loss(name: str, config: Any) -> nn.Module:
    return LOSS_REGISTRY[name](config)
