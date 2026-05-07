from __future__ import annotations

import gc
import html
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "rankmixerclean_full_val.ipynb"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK_PATH.read_text())


def save_notebook(nb: dict[str, Any]) -> None:
    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


def stream_output(text: str) -> dict[str, Any]:
    return {"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}


def error_output(error: BaseException) -> dict[str, Any]:
    return {
        "output_type": "error",
        "ename": error.__class__.__name__,
        "evalue": str(error),
        "traceback": traceback.format_exception(type(error), error, error.__traceback__),
    }


def table_output(title: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "output_type": "display_data",
        "metadata": {},
        "data": {
            "text/plain": f"{title}\n{frame.to_string(index=False)}",
            "text/html": f"<h4>{html.escape(title)}</h4>{frame.to_html(index=False)}",
        },
    }


def write_cell(cell_index: int, outputs: list[dict[str, Any]], execution_count: int) -> None:
    nb = load_notebook()
    nb["cells"][cell_index]["execution_count"] = execution_count
    nb["cells"][cell_index]["outputs"] = outputs
    save_notebook(nb)


def write_status(stage: str, exit_code: int | None = None) -> None:
    status_path = os.getenv("RANKMIXER_STATUS_PATH")
    if not status_path:
        return
    lines = [f"stage={stage}", f"updated_at={time.strftime('%Y-%m-%d %H:%M:%S %Z')}"]
    if exit_code is not None:
        lines.append(f"exit_code={exit_code}")
    Path(status_path).write_text("\n".join(lines) + "\n")


def clear_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def iter_user_complete_chunks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    carry = pd.DataFrame()
    for chunk in pd.read_csv(path, chunksize=chunksize):
        frame = pd.concat([carry, chunk], ignore_index=True) if len(carry) else chunk
        if frame.empty:
            continue
        last_user = frame["user_id"].iloc[-1]
        ready_mask = frame["user_id"].ne(last_user)
        ready = frame[ready_mask].copy()
        carry = frame[~ready_mask].copy()
        if len(ready):
            yield ready
    if len(carry):
        yield carry


def build_svd_frame(data_path: Path, save_path: Path, n_components: int, seed: int):
    from rec.features import article_svd_features, load_article_embedding

    article_embedding = load_article_embedding(data_path, save_path)
    emb_cols = [column for column in article_embedding.columns if column.startswith("article_emb_")]
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    matrix = svd.fit_transform(article_embedding[emb_cols].to_numpy(dtype=np.float32)).astype("float32")
    svd_df = pd.DataFrame(matrix, columns=list(article_svd_features(n_components)))
    svd_df.insert(0, "click_article_id", article_embedding["click_article_id"].astype("int64").to_numpy())
    return svd_df


def merge_svd(frame: pd.DataFrame, svd_df: pd.DataFrame) -> pd.DataFrame:
    if "article_id" in frame.columns and "click_article_id" not in frame.columns:
        frame = frame.rename(columns={"article_id": "click_article_id"})
    merged = frame.merge(svd_df, on="click_article_id", how="left")
    svd_cols = [column for column in svd_df.columns if column != "click_article_id"]
    merged[svd_cols] = merged[svd_cols].fillna(0.0).astype("float32")
    return merged


def add_history(frame: pd.DataFrame, history_sequences: pd.DataFrame) -> pd.DataFrame:
    from rec.data import add_query_id
    from rec.features import HISTORY_FEATURE

    merged = add_query_id(frame).merge(history_sequences, on="user_id", how="left")
    merged[HISTORY_FEATURE] = merged[HISTORY_FEATURE].apply(lambda value: value if isinstance(value, list) else [])
    return merged


def build_sparse_maps_from_sets(sparse_values: dict[str, set[int]], sparse_features: tuple[str, ...]) -> dict[str, dict[int, int]]:
    maps: dict[str, dict[int, int]] = {}
    for feature in sparse_features:
        start = 2 if feature == "click_article_id" else 1
        maps[feature] = {value: index for index, value in enumerate(sorted(sparse_values[feature]), start=start)}
    return maps


@dataclass
class MetricAccumulator:
    topk: int
    full_query_count: int = 0
    full_pos_query_count: int = 0
    full_mrr_sum: float = 0.0
    full_hit_sum: float = 0.0
    full_ndcg_sum: float = 0.0
    hit_mrr_sum: float = 0.0
    hit_hit_sum: float = 0.0
    hit_ndcg_sum: float = 0.0
    labels: list[np.ndarray] | None = None
    scores: list[np.ndarray] | None = None
    hit_labels: list[np.ndarray] | None = None
    hit_scores: list[np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.labels = []
        self.scores = []
        self.hit_labels = []
        self.hit_scores = []

    def update(self, pred_df: pd.DataFrame) -> None:
        self.labels.append(pred_df["label"].to_numpy(dtype=np.float32))
        self.scores.append(pred_df["pred_score"].to_numpy(dtype=np.float32))
        for _, group in pred_df.groupby("query_id", sort=False):
            labels = group["label"].to_numpy(dtype=np.float32)
            scores = group["pred_score"].to_numpy(dtype=np.float32)
            self.full_query_count += 1
            order = np.argsort(-scores, kind="mergesort")
            ranked_labels = labels[order]
            pos_count = int(labels.sum())
            ndcg_value = binary_ndcg_at_k(ranked_labels, pos_count, self.topk)
            self.full_ndcg_sum += ndcg_value
            if pos_count == 0:
                continue
            self.full_pos_query_count += 1
            self.hit_labels.append(labels)
            self.hit_scores.append(scores)
            self.hit_ndcg_sum += ndcg_value
            pos_positions = np.flatnonzero(ranked_labels == 1)
            if pos_positions.size:
                rank = int(pos_positions[0]) + 1
                if rank <= self.topk:
                    self.full_mrr_sum += 1.0 / rank
                    self.full_hit_sum += 1.0
                    self.hit_mrr_sum += 1.0 / rank
                    self.hit_hit_sum += 1.0

    def result(self) -> dict[str, float]:
        labels = np.concatenate(self.labels) if self.labels else np.asarray([])
        scores = np.concatenate(self.scores) if self.scores else np.asarray([])
        hit_labels = np.concatenate(self.hit_labels) if self.hit_labels else np.asarray([])
        hit_scores = np.concatenate(self.hit_scores) if self.hit_scores else np.asarray([])
        return {
            "full_auc": safe_auc(labels, scores),
            "full_mrr": self.full_mrr_sum / max(self.full_query_count, 1),
            "full_ndcg": self.full_ndcg_sum / max(self.full_query_count, 1),
            "full_hit_rate": self.full_hit_sum / max(self.full_query_count, 1),
            "full_query_count": self.full_query_count,
            "full_pos_query_count": self.full_pos_query_count,
            "hit_auc": safe_auc(hit_labels, hit_scores),
            "hit_mrr": self.hit_mrr_sum / max(self.full_pos_query_count, 1),
            "hit_ndcg": self.hit_ndcg_sum / max(self.full_pos_query_count, 1),
            "hit_hit_rate": self.hit_hit_sum / max(self.full_pos_query_count, 1),
            "hit_query_count": self.full_pos_query_count,
            "hit_pos_query_count": self.full_pos_query_count,
        }


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def binary_ndcg_at_k(ranked_labels: np.ndarray, pos_count: int, k: int) -> float:
    if pos_count == 0:
        return 0.0
    gains = ranked_labels[:k]
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float(np.sum(gains * discounts))
    ideal_len = min(pos_count, k)
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_len + 2))
    idcg = float(np.sum(ideal_discounts))
    return 0.0 if idcg == 0 else dcg / idcg


def make_prediction_frame(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
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
        if column in frame.columns
    ]
    pred_df = frame[keep_cols].copy()
    pred_df["pred_score"] = scores.astype("float32").reshape(-1)
    pred_df["rankmixer_score"] = pred_df["pred_score"]
    pred_df["pred_rank"] = pred_df.groupby("query_id")["pred_score"].rank(method="first", ascending=False).astype("int32")
    pred_df["label"] = pd.to_numeric(pred_df["label"], errors="coerce").fillna(0).astype("int8")
    return pred_df


def main() -> int:
    os.chdir(PROJECT_ROOT)
    from rec.config import DEFAULT_SEED, ExperimentConfig, train_feature_name
    from rec.data import build_history_sequences, build_model_input, map_sparse, sparse_vocab_sizes
    from rec.features import HISTORY_FEATURE, META_FLOAT_FEATURES, add_rank_score_features, build_feature_preset, ensure_feature_columns
    from rec.models.rankmixer import count_parameters
    from rec.pipeline import DEFAULT_SCENARIOS, build_model
    from rec.train import default_device

    config = ExperimentConfig()
    scenarios = ["old_ge", "query_softmax_ce", "listwise_bpr_bce"]
    chunk_size = int(os.getenv("RANKMIXER_FULL_VAL_CHUNK_SIZE", "200000"))
    device = default_device(config.train.device)
    paths = config.paths
    train_path = paths.save_path / train_feature_name(config.data.train_variant)
    val_path = paths.save_path / "val_user_item_feats_df_all.csv"
    history_path = paths.save_path / "click_hist_all.csv"

    config_text = "\n".join(
        [
            f"project_root={paths.project_root}",
            f"save_path={paths.save_path}",
            f"output_path={paths.output_path}",
            f"device={device}",
            f"chunk_size={chunk_size}",
            f"scenarios={scenarios}",
        ]
    )
    print(config_text, flush=True)
    write_cell(1, [stream_output(config_text + "\n")], 1)

    write_status("building_article_svd")
    preset = build_feature_preset(name="full_val_eval_streaming", article_svd_dim=config.data.article_svd_dim)
    svd_df = build_svd_frame(paths.data_path, paths.save_path, config.data.article_svd_dim, DEFAULT_SEED)
    history = pd.read_csv(history_path)
    history_sequences = build_history_sequences(history)

    write_status("streaming_train_vocab_and_scaler")
    sparse_values = {feature: set() for feature in preset.sparse_features}
    train_users: set[int] = set()
    scaler = MinMaxScaler()
    train_rows = 0
    train_started_at = time.time()
    for chunk_idx, chunk in enumerate(iter_user_complete_chunks(train_path, chunk_size), start=1):
        train_rows += len(chunk)
        train_users.update(pd.to_numeric(chunk["user_id"], errors="coerce").dropna().astype("int64").tolist())
        frame = add_history(merge_svd(add_rank_score_features(chunk), svd_df), history_sequences)
        frame = ensure_feature_columns(frame, preset)
        scaler.partial_fit(frame[list(preset.dense_features)])
        for feature in preset.sparse_features:
            values = pd.to_numeric(frame[feature], errors="coerce").fillna(0).astype("int64")
            sparse_values[feature].update(int(value) for value in values.unique() if int(value) != 0)
        if chunk_idx == 1 or chunk_idx % 10 == 0:
            print(f"train chunk {chunk_idx} | rows={train_rows:,}", flush=True)
            write_status(f"streaming_train_vocab_and_scaler_chunk_{chunk_idx}")
        del frame, chunk
        clear_cache()

    hist_clicks = history.loc[history["user_id"].isin(train_users), "click_article_id"]
    sparse_values["click_article_id"].update(int(value) for value in pd.to_numeric(hist_clicks, errors="coerce").dropna().astype("int64").unique() if int(value) != 0)
    sparse_maps = build_sparse_maps_from_sets(sparse_values, preset.sparse_features)
    vocab_sizes = sparse_vocab_sizes(sparse_maps, preset.sparse_features)
    prepared_stub = SimpleNamespace(sparse_vocab_sizes=vocab_sizes, feature_preset=preset)
    prep_df = pd.DataFrame(
        [
            {"name": "train_rows_streamed", "value": train_rows},
            {"name": "train_users", "value": len(train_users)},
            {"name": "val_rows_file", "value": sum(1 for _ in open(val_path)) - 1},
            {"name": "sparse_vocab_click_article_id", "value": vocab_sizes["click_article_id"]},
            {"name": "prepare_elapsed_sec", "value": round(time.time() - train_started_at, 1)},
        ]
    )
    write_cell(3, [table_output("streaming preparation", prep_df)], 2)
    write_cell(5, [stream_output("streaming full-val helper functions loaded\n")], 3)

    def build_val_input(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        frame = ensure_feature_columns(frame, preset)
        frame[list(preset.dense_features)] = scaler.transform(frame[list(preset.dense_features)]).astype("float32")
        return build_model_input(frame, sparse_maps, preset, config.data.max_history_len)

    def predict_scores(model: torch.nn.Module, x_dict: dict[str, np.ndarray], scenario) -> np.ndarray:
        from rec.evaluate import output_to_scores
        from rec.data import make_loader, move_batch_to_device

        loader = make_loader(x_dict, batch_size=config.train.batch_size)
        raw_parts: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                raw_parts.append(model(batch).detach().cpu().numpy())
        raw_output = np.concatenate(raw_parts, axis=0)
        return output_to_scores(raw_output, scenario.head_type, scenario.score_mode)

    run_outputs: list[dict[str, Any]] = []
    all_summary_frames: list[pd.DataFrame] = []
    all_artifact_rows: list[dict[str, str]] = []
    for scenario_name in scenarios:
        write_status(f"evaluating_{scenario_name}")
        scenario = DEFAULT_SCENARIOS[scenario_name]
        state_path = paths.output_path / f"{scenario_name}_best_state.pt"
        model = build_model(scenario, prepared_stub, config.model, config.data.article_svd_dim).to(device)
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
        print(f"\n=== full val eval: {scenario_name} | params={count_parameters(model):,} ===", flush=True)
        pred_path = paths.output_path / f"{scenario_name}_full_val_predictions.csv"
        hit_pred_path = paths.output_path / f"{scenario_name}_full_val_hit_predictions.csv"
        for path in (pred_path, hit_pred_path):
            if path.exists():
                path.unlink()
        acc = MetricAccumulator(topk=config.train.topk)
        scenario_started_at = time.time()
        val_rows = 0
        for chunk_idx, chunk in enumerate(iter_user_complete_chunks(val_path, chunk_size), start=1):
            frame = add_history(merge_svd(add_rank_score_features(chunk), svd_df), history_sequences)
            labels = frame["label"].astype("float32").to_numpy()
            x_dict = build_val_input(frame)
            scores = predict_scores(model, x_dict, scenario)
            pred_df = make_prediction_frame(frame, scores)
            acc.update(pred_df)
            write_header = not pred_path.exists()
            pred_df.to_csv(pred_path, mode="a", header=write_header, index=False)
            hit_pred_df = pred_df.groupby("query_id", sort=False).filter(lambda group: group["label"].sum() > 0)
            hit_pred_df.to_csv(hit_pred_path, mode="a", header=not hit_pred_path.exists(), index=False)
            val_rows += len(frame)
            if chunk_idx == 1 or chunk_idx % 10 == 0:
                print(f"{scenario_name} val chunk {chunk_idx} | rows={val_rows:,}", flush=True)
                write_status(f"evaluating_{scenario_name}_chunk_{chunk_idx}")
            del chunk, frame, labels, x_dict, scores, pred_df, hit_pred_df
            clear_cache()

        row = {
            "scenario": scenario_name,
            "protocol": f"{scenario.loss_name} + {scenario.head_type} + {scenario.score_mode}",
            "eval_scope": "full_validation_streaming",
            "val_user_sample_size": None,
            "elapsed_sec": time.time() - scenario_started_at,
            **acc.result(),
        }
        sampled_path = paths.output_path / f"{scenario_name}_summary.csv"
        if sampled_path.exists():
            sampled = pd.read_csv(sampled_path).tail(1).iloc[0].to_dict()
            for key in ("best_epoch", "best_metric", "full_mrr", "hit_mrr"):
                if key in sampled:
                    row[f"sampled_{key}"] = sampled[key]
        summary_df = pd.DataFrame([row])
        summary_path = paths.output_path / f"{scenario_name}_full_val_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        artifacts = {"summary": summary_path, "predictions": pred_path, "hit_predictions": hit_pred_path}
        artifact_df = pd.DataFrame({"artifact": list(artifacts), "path": [str(path) for path in artifacts.values()]})
        run_outputs.extend([table_output(f"{scenario_name} summary", summary_df), table_output(f"{scenario_name} artifacts", artifact_df)])
        all_summary_frames.append(summary_df)
        for artifact, path in artifacts.items():
            all_artifact_rows.append({"scenario": scenario_name, "artifact": artifact, "path": str(path)})
        write_cell(7, run_outputs, 4)
        del model, acc
        clear_cache()

    full_val_summary = pd.concat(all_summary_frames, ignore_index=True)
    full_val_artifacts = pd.DataFrame(all_artifact_rows)
    combined_summary_path = paths.output_path / "rankmixerclean_full_val_summary.csv"
    combined_artifacts_path = paths.output_path / "rankmixerclean_full_val_artifacts.csv"
    full_val_summary.to_csv(combined_summary_path, index=False)
    full_val_artifacts.to_csv(combined_artifacts_path, index=False)
    run_outputs.extend(
        [
            table_output("full_val_summary", full_val_summary),
            table_output("full_val_artifacts", full_val_artifacts),
            stream_output(f"combined_summary={combined_summary_path}\ncombined_artifacts={combined_artifacts_path}\n"),
        ]
    )
    write_cell(7, run_outputs, 4)
    metric_cols = [
        "scenario",
        "protocol",
        "sampled_best_epoch",
        "sampled_full_mrr",
        "full_mrr",
        "sampled_hit_mrr",
        "hit_mrr",
        "full_ndcg",
        "hit_ndcg",
        "elapsed_sec",
    ]
    metric_df = full_val_summary[[column for column in metric_cols if column in full_val_summary.columns]]
    write_cell(9, [table_output("core metrics", metric_df)], 5)
    write_status("completed", exit_code=0)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        traceback.print_exc()
        try:
            write_cell(7, [error_output(exc)], 4)
        except Exception:
            pass
        write_status("failed", exit_code=1)
        raise
