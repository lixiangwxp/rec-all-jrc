from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from rec.features import HISTORY_FEATURE, FeaturePreset, build_token_specs


class MLP(nn.Module):
    """通用多层感知机。

    业务上用于把拼好的特征表示继续压缩/变换；输入通常是 `[B, D]`，
    其中 `B` 是 batch 内样本数，`D` 是当前表示维度。
    """

    def __init__(self, input_dim: int, hidden_units: Sequence[int], dropout_rate: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for units in hidden_units:
            layers.extend([nn.Linear(prev_dim, units), nn.ReLU()])
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = units
        self.layers = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DINAttentionPooling(nn.Module):
    """DIN 风格的历史行为注意力池化。

    query_emb 表示当前候选文章，shape 为 `[B, E]`；hist_emb 表示用户历史点击文章序列，
    shape 为 `[B, H, E]`；mask 标记哪些历史位置是真实文章而不是 padding，shape 为 `[B, H]`。
    输出是每个用户历史序列围绕当前候选文章聚合后的兴趣向量，shape 为 `[B, E]`。
    """

    def __init__(self, emb_dim: int, hidden_units: Sequence[int]):
        super().__init__()
        self.att_mlp = MLP(emb_dim * 4, hidden_units)
        self.out = nn.Linear(self.att_mlp.output_dim, 1)

    def forward(self, query_emb: torch.Tensor, hist_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 把候选文章向量复制到每个历史位置，便于逐个比较 query 与 history item。
        query = query_emb.unsqueeze(1).expand_as(hist_emb)
        # DIN 常用的交互特征：[query, history, query-history, query*history]，最后一维变为 4E。
        att_input = torch.cat([query, hist_emb, query - hist_emb, query * hist_emb], dim=-1)
        # scores 是每个历史点击对当前候选文章的相关性分数，shape `[B, H]`。
        scores = self.out(self.att_mlp(att_input)).squeeze(-1)
        # padding 的历史位置不应参与 softmax，所以填成很小的数。
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1) * mask.float()
        # 重新归一化是为了处理 mask 后的权重和；全 padding 时 clamp 避免除零。
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        # 按注意力权重对历史文章 embedding 加权求和，得到用户对当前候选文章的兴趣表示。
        return torch.bmm(weights.unsqueeze(1), hist_emb).squeeze(1)


class FeatureInputEncoder(nn.Module):
    """把原始 batch 特征编码成模型可用的张量表示。

    sparse 特征会查 embedding 表得到 `[B, E]`；dense 特征会 stack 成 `[B, D_dense]`；
    history 特征先用文章 embedding 得到 `[B, H, E]`，再通过 DINAttentionPooling 变成 `[B, E]`。
    forward 返回的 flat_input 是给普通 MLP 用的扁平特征，shape 为
    `[B, len(sparse)*E + D_dense + E]`。
    """

    def __init__(self, sparse_vocab_sizes: dict[str, int], dense_features: Sequence[str], model_config: Any):
        super().__init__()
        self.sparse_features = tuple(sparse_vocab_sizes)
        self.dense_features = tuple(dense_features)
        self.emb_dim = model_config.emb_dim
        self.sparse_embeddings = nn.ModuleDict()
        for feature, vocab_size in sparse_vocab_sizes.items():
            padding_idx = 0 if feature == "click_article_id" else None
            self.sparse_embeddings[feature] = nn.Embedding(vocab_size, self.emb_dim, padding_idx=padding_idx)
        self.din_pooling = DINAttentionPooling(self.emb_dim, model_config.attention_hidden_units)
        self.flat_output_dim = len(self.sparse_features) * self.emb_dim + len(self.dense_features) + self.emb_dim

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        # sparse_embeddings: 每个离散特征一个 `[B, E]` embedding，例如用户/文章/类别等 ID。
        sparse_embeddings = {feature: self.sparse_embeddings[feature](batch[feature]) for feature in self.sparse_features}

        # dense_tensor: 连续数值特征按列拼成 `[B, D_dense]`，后续可直接进入 MLP 或 wide 分支。
        dense_tensor = torch.stack([batch[feature] for feature in self.dense_features], dim=1)

        # target_emb 是当前候选文章 embedding，作为 DIN attention 的 query。
        target_emb = sparse_embeddings["click_article_id"]
        hist_ids = batch[HISTORY_FEATURE]
        # hist_ids 中 0 是 padding；embedding 后得到用户历史点击序列 `[B, H, E]`。
        hist_emb = self.sparse_embeddings["click_article_id"](hist_ids)
        # hist_repr 是“当前候选文章相关”的用户历史兴趣表示，而不是简单平均历史。
        hist_repr = self.din_pooling(target_emb, hist_emb, hist_ids.ne(0))
        #[B,E]
        flat_input = torch.cat([sparse_embeddings[feature] for feature in self.sparse_features] + [dense_tensor, hist_repr], dim=1)
        return {
            "sparse_embeddings": sparse_embeddings,
            "dense_tensor": dense_tensor,
            "hist_repr": hist_repr,
            "flat_input": flat_input,
        }


class FieldwiseTokenization(nn.Module):
    """把字段组切成 RankMixer 的 token 序列。

    每个 token 代表一组业务字段，例如若干 sparse 字段、一组 dense 字段，或 history 表示。
    输入来自 FeatureInputEncoder，输出 tokens 的 shape 为 `[B, T, token_dim]`，
    其中 `T` 是字段组数量，后续 RankMixer block 会在 token 维度做重排混合。
    """

    def __init__(self, emb_dim: int, token_dim: int, preset: FeaturePreset, article_svd_dim: int):
        super().__init__()
        self.token_specs = build_token_specs(preset, article_svd_dim)
        self.projections = nn.ModuleDict()
        for spec in self.token_specs:
            features = spec["features"]
            kind = spec["kind"]
            input_dim = len(features) * emb_dim if kind == "sparse" else len(features)
            if kind == "history":
                input_dim = emb_dim
            self.projections[spec["name"]] = nn.Linear(input_dim, token_dim)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        sparse_embeddings: dict[str, torch.Tensor],
        hist_repr: torch.Tensor,
    ) -> torch.Tensor:
        """把各种字段特征变成 RankMixer 可以处理的 token 序列，shape 为 `[B, T, token_dim]`。"""

        tokens: list[torch.Tensor] = []
        for spec in self.token_specs:
            features = spec["features"]
            kind = spec["kind"]
            if kind == "sparse":
                token_input = torch.cat([sparse_embeddings[feature] for feature in features], dim=1)
            elif kind == "dense":
                token_input = torch.stack([batch[feature] for feature in features], dim=1)
            else:
                # history token 使用 DIN 聚合后的 hist_repr，shape `[B, E]`。
                token_input = hist_repr
            tokens.append(self.projections[spec["name"]](token_input))
        # stack 后多出 token 维度：`[B, T, token_dim]`。
        return torch.stack(tokens, dim=1)


def round_up_to_multiple(value: int, multiple: int) -> int:
    """返回不小于 value 的最小 multiple 倍数。"""

    return ((value + multiple - 1) // multiple) * multiple


class MultiHeadTokenMixing(nn.Module):
    """原论文式 Multi-head Token Mixing。

    输入 `x` 的 shape 是 `[B, T, D]`。RankMixer 把每个 token 沿隐藏维切成 `T` 个 head，
    再交换 token 维和 head 维，最后重新拼回 `[B, T, D]`。这一步只有 reshape/transpose，
    没有 Q/K/V、attention score，也没有可学习参数。
    """

    def __init__(self, token_count: int, token_dim: int):
        super().__init__()
        self.token_count = token_count
        self.head_dim = token_dim // token_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.reshape(batch_size, self.token_count, self.token_count, self.head_dim).transpose(1, 2).reshape(batch_size, self.token_count, -1)


class RankMixerBlock(nn.Module):
    """RankMixer block：Multi-head Token Mixing + Per-token FFN。"""

    def __init__(self, token_count: int, token_dim: int, expansion_ratio: int, dropout_rate: float):
        super().__init__()
        self.token_mixing = MultiHeadTokenMixing(token_count, token_dim)
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * expansion_ratio),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(token_dim * expansion_ratio, token_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.token_mixing(x))
        return self.norm2(x + self.ffn(x))


class RankMixerBackbone(nn.Module):
    """RankMixer 主干网络：字段 token 化 + 多层 RankMixer block + 池化。

    输入是 batch 的原始特征、已算好的 sparse embedding 和 hist_repr；
    输出是样本级表示 `[B, token_dim]`，还没有变成点击 logit。
    """

    def __init__(self, emb_dim: int, preset: FeaturePreset, model_config: Any, article_svd_dim: int):
        super().__init__()
        self.pooling = model_config.pooling
        base_token_count = len(build_token_specs(preset, article_svd_dim))
        self.token_count = base_token_count + (1 if self.pooling == "cls" else 0)
        self.token_dim = round_up_to_multiple(model_config.token_dim, self.token_count)
        self.tokenization = FieldwiseTokenization(emb_dim, self.token_dim, preset, article_svd_dim)
        self.layers = nn.ModuleList(
            RankMixerBlock(
                token_count=self.token_count,
                token_dim=self.token_dim,
                expansion_ratio=model_config.expansion_ratio,
                dropout_rate=model_config.dropout_rate,
            )
            for _ in range(model_config.num_layers)
        )
        self.output_norm = nn.LayerNorm(self.token_dim)
        self.output_dim = self.token_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_dim)) if self.pooling == "cls" else None

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        sparse_embeddings: dict[str, torch.Tensor],
        hist_repr: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.tokenization(batch, sparse_embeddings, hist_repr)
        if self.cls_token is not None:
            # cls 模式下新增一个可学习 token，最后取它作为整条样本的汇总表示。
            tokens = torch.cat([self.cls_token.expand(tokens.size(0), -1, -1), tokens], dim=1)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.output_norm(tokens)
        # cls pooling 返回第 0 个 token；mean pooling 则平均所有字段 token。
        return tokens[:, 0] if self.cls_token is not None else tokens.mean(dim=1)


class OutputHead(nn.Module):
    """把样本表示映射成最终打分。

    `single_logit` 输出 `[B]`，一个数越大表示越倾向点击；
    `two_logit` 输出 `[B, 2]`，第 0 列是未点击 logit，第 1 列是点击 logit。
    """

    def __init__(self, input_dim: int, hidden_units: Sequence[int], head_type: str, dropout_rate: float):
        super().__init__()
        self.head_type = head_type
        self.mlp = MLP(input_dim, hidden_units, dropout_rate)
        self.out = nn.Linear(self.mlp.output_dim, 2 if head_type == "two_logit" else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.out(self.mlp(x))
        return output.squeeze(-1) if self.head_type == "single_logit" else output


class RankMixerRanker(nn.Module):
    """完整排序模型：FeatureInputEncoder + RankMixerBackbone + 输出头。

    forward 输入 batch 字典，输出由 `head_type` 决定：
    `two_logit` 时返回 `[B, 2]`，含未点击/点击两个 logit，训练时可用于交叉熵或 diff 打分；
    `single_logit` 时返回 `[B]`，直接作为点击倾向分数。这里的输出仍是 logit，不是概率。
    """

    def __init__(
        self,
        sparse_vocab_sizes: dict[str, int],
        preset: FeaturePreset,
        model_config: Any,
        article_svd_dim: int,
        head_type: str = "two_logit",
    ):
        super().__init__()
        self.head_type = head_type
        self.input_encoder = FeatureInputEncoder(sparse_vocab_sizes, preset.dense_features, model_config)
        self.backbone = RankMixerBackbone(model_config.emb_dim, preset, model_config, article_svd_dim)
        self.dense_shortcut = MLP(len(preset.dense_features), (256, 128), model_config.dropout_rate)
        self.wide_out = nn.Linear(len(preset.dense_features), 1)
        final_dim = self.backbone.output_dim + self.dense_shortcut.output_dim
        self.output_head = OutputHead(final_dim, model_config.logit_head_units, head_type, model_config.dropout_rate)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # encoded 同时服务于 token 主干和 dense shortcut。
        encoded = self.input_encoder(batch)
        # repr_tensor 是 RankMixer 主干输出的样本表示，shape `[B, token_dim]`。
        repr_tensor = self.backbone(batch, encoded["sparse_embeddings"], encoded["hist_repr"])
        # dense_shortcut 给连续特征一条深层旁路；wide_out 是线性 wide 分数，保留数值特征的直接贡献。
        repr_tensor = torch.cat([repr_tensor, self.dense_shortcut(encoded["dense_tensor"])], dim=1)
        wide_score = self.wide_out(encoded["dense_tensor"]).squeeze(-1)
        output = self.output_head(repr_tensor)
        if self.head_type == "single_logit":
            return output + wide_score
        output = output.clone()
        # two_logit 下 wide_score 只加到“点击”logit 上，相当于提高或降低点击类别的相对优势。
        output[:, 1] = output[:, 1] + wide_score
        return output


RankMixer = RankMixerBackbone


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
