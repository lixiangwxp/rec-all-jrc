from __future__ import annotations

import gc
import html
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


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


def write_status(text: str, exit_code: int | None = None) -> None:
    status_path = os.getenv("RANKMIXER_STATUS_PATH")
    if not status_path:
        return
    lines = [
        f"stage={text}",
        f"updated_at={time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
    ]
    if exit_code is not None:
        lines.append(f"exit_code={exit_code}")
    Path(status_path).write_text("\n".join(lines) + "\n")


def clear_eval_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def raw_predict_outputs_with_progress(model: torch.nn.Module, x_data: dict[str, np.ndarray], batch_size: int, device: torch.device) -> np.ndarray:
    from rec.data import make_loader, move_batch_to_device

    loader = make_loader(x_data, batch_size=batch_size)
    total_batches = len(loader)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = move_batch_to_device(batch, device)
            outputs.append(model(batch).detach().cpu().numpy())
            if batch_idx == 1 or batch_idx % 200 == 0 or batch_idx == total_batches:
                print(f"predict batch {batch_idx}/{total_batches}", flush=True)
    return np.concatenate(outputs, axis=0)


def main() -> int:
    os.chdir(PROJECT_ROOT)
    pd.set_option("display.max_columns", 120)
    pd.set_option("display.width", 180)

    from rec.config import ExperimentConfig, train_feature_name
    from rec.data import (
        LoadedFrames,
        build_model_input,
        build_sparse_maps,
        merge_history_frames,
        normalize_dense_features,
        sparse_vocab_sizes,
    )
    from rec.evaluate import RankingEvaluator
    from rec.features import add_article_svd_features, add_rank_score_features, build_feature_preset, ensure_feature_columns
    from rec.models.rankmixer import count_parameters
    from rec.pipeline import DEFAULT_SCENARIOS, build_model
    from rec.train import default_device

    config = ExperimentConfig()
    scenarios = ["old_ge", "query_softmax_ce", "listwise_bpr_bce"]
    device = default_device(config.train.device)

    config_text = "\n".join(
        [
            f"project_root={config.paths.project_root}",
            f"save_path={config.paths.save_path}",
            f"output_path={config.paths.output_path}",
            f"device={device}",
            f"scenarios={scenarios}",
        ]
    )
    print(config_text, flush=True)
    write_cell(1, [stream_output(config_text + "\n")], 1)

    write_status("preparing_full_validation_data")
    started_at = time.time()
    preset = build_feature_preset(name="full_val_eval", article_svd_dim=config.data.article_svd_dim)
    train_path = config.paths.save_path / train_feature_name(config.data.train_variant)
    val_path = config.paths.save_path / "val_user_item_feats_df_all.csv"
    history_path = config.paths.save_path / "click_hist_all.csv"
    train_frame = add_rank_score_features(pd.read_csv(train_path))
    val_frame = add_rank_score_features(pd.read_csv(val_path))
    train_frame, val_frame = add_article_svd_features(
        (train_frame, val_frame),
        config.paths.data_path,
        config.paths.save_path,
        config.data.article_svd_dim,
    )
    loaded_frames = LoadedFrames(
        train=train_frame,
        val=val_frame,
        test=pd.DataFrame(),
        history=pd.read_csv(history_path),
    )
    loaded_shape_df = pd.DataFrame(
        [
            {"split": "train", "shape": loaded_frames.train.shape},
            {"split": "val", "shape": None if loaded_frames.val is None else loaded_frames.val.shape},
            {"split": "test", "shape": loaded_frames.test.shape},
            {"split": "history", "shape": loaded_frames.history.shape},
        ]
    )
    merged_frames = merge_history_frames(loaded_frames)
    train = ensure_feature_columns(merged_frames.train, preset)
    val = ensure_feature_columns(merged_frames.val, preset)
    empty_test = pd.DataFrame()
    normalize_dense_features(train, val, empty_test, preset.dense_features)
    sparse_maps = build_sparse_maps([train], preset.sparse_features)
    vocab_sizes = sparse_vocab_sizes(sparse_maps, preset.sparse_features)
    x_val = build_model_input(val, sparse_maps, preset, config.data.max_history_len)
    y_val = val["label"].astype("float32").to_numpy()
    prepared_data = SimpleNamespace(
        train_frame=train.iloc[0:0].copy(),
        val_frame=val,
        test_frame=empty_test,
        val_hit_mask=merged_frames.val_hit_mask,
        val_hit_frame=merged_frames.val_hit_frame,
        x_val=x_val,
        y_val=y_val,
        sparse_vocab_sizes=vocab_sizes,
        feature_preset=preset,
    )
    del train, train_frame, val_frame, loaded_frames, merged_frames, sparse_maps
    clear_eval_cache()
    prepared_shape_df = pd.DataFrame(
        [
            {"name": "train_frame_vocab_source", "shape": "released after fitting vocab/scalers"},
            {"name": "val_frame", "shape": None if prepared_data.val_frame is None else prepared_data.val_frame.shape},
            {"name": "val_hit_frame", "shape": None if prepared_data.val_hit_frame is None else prepared_data.val_hit_frame.shape},
            {"name": "test_frame", "shape": prepared_data.test_frame.shape},
            {"name": "x_train_rows", "shape": len(prepared_data.y_train)},
            {"name": "x_val_rows", "shape": None if prepared_data.y_val is None else len(prepared_data.y_val)},
        ]
    )
    prepare_text = f"prepare_elapsed_sec={time.time() - started_at:.1f}\n"
    print(prepare_text, flush=True)
    write_cell(
        3,
        [table_output("loaded shapes", loaded_shape_df), table_output("prepared shapes", prepared_shape_df), stream_output(prepare_text)],
        2,
    )
    write_cell(5, [stream_output("full-val helper functions loaded\n")], 3)

    if prepared_data.x_val is None or prepared_data.y_val is None or prepared_data.val_frame is None:
        raise ValueError("prepared_data does not contain validation data")

    all_summary_frames: list[pd.DataFrame] = []
    all_artifact_rows: list[dict[str, str]] = []
    run_outputs: list[dict[str, Any]] = []

    for scenario_name in scenarios:
        write_status(f"evaluating_{scenario_name}")
        scenario = DEFAULT_SCENARIOS[scenario_name]
        state_path = config.paths.output_path / f"{scenario_name}_best_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(state_path)

        header = "\n".join(
            [
                f"\n=== full val eval: {scenario_name} ===",
                f"protocol={scenario.loss_name} + {scenario.head_type} + {scenario.score_mode}",
                f"state_path={state_path}",
            ]
        )
        print(header, flush=True)
        model = build_model(scenario, prepared_data, config.model, config.data.article_svd_dim).to(device)
        print(f"params={count_parameters(model):,}", flush=True)
        model.load_state_dict(torch.load(state_path, map_location="cpu"))

        scenario_started_at = time.time()
        raw_output = raw_predict_outputs_with_progress(model, prepared_data.x_val, config.train.batch_size, device)
        evaluator = RankingEvaluator(topk=config.train.topk)
        eval_result = evaluator.evaluate_raw(
            raw_output,
            prepared_data.y_val,
            prepared_data.val_frame,
            head_type=scenario.head_type,
            score_mode=scenario.score_mode,
            hit_mask=prepared_data.val_hit_mask,
        )

        row: dict[str, Any] = {
            "scenario": scenario_name,
            "protocol": f"{scenario.loss_name} + {scenario.head_type} + {scenario.score_mode}",
            "eval_scope": "full_validation",
            "val_user_sample_size": None,
            "full_mrr": eval_result.get("full_mrr"),
            "full_ndcg": eval_result.get("full_ndcg"),
            "full_hit_rate": eval_result.get("full_hit_rate"),
            "full_query_count": eval_result.get("full_query_count"),
            "full_pos_query_count": eval_result.get("full_pos_query_count"),
            "hit_mrr": eval_result.get("hit_mrr"),
            "hit_ndcg": eval_result.get("hit_ndcg"),
            "hit_hit_rate": eval_result.get("hit_hit_rate"),
            "hit_query_count": eval_result.get("hit_query_count"),
            "hit_pos_query_count": eval_result.get("hit_pos_query_count"),
            "elapsed_sec": time.time() - scenario_started_at,
        }

        sampled_summary_path = config.paths.output_path / f"{scenario_name}_summary.csv"
        if sampled_summary_path.exists():
            sampled = pd.read_csv(sampled_summary_path).tail(1).iloc[0].to_dict()
            for key in ("best_epoch", "best_metric", "full_mrr", "hit_mrr"):
                if key in sampled:
                    row[f"sampled_{key}"] = sampled[key]

        summary_df = pd.DataFrame([row])
        summary_path = config.paths.output_path / f"{scenario_name}_full_val_summary.csv"
        pred_path = config.paths.output_path / f"{scenario_name}_full_val_predictions.csv"
        hit_pred_path = config.paths.output_path / f"{scenario_name}_full_val_hit_predictions.csv"
        summary_df.to_csv(summary_path, index=False)
        eval_result["val_pred_df"].to_csv(pred_path, index=False)
        artifacts = {"summary": summary_path, "predictions": pred_path}
        if eval_result.get("hit_pred_df") is not None:
            eval_result["hit_pred_df"].to_csv(hit_pred_path, index=False)
            artifacts["hit_predictions"] = hit_pred_path

        artifact_df = pd.DataFrame({"artifact": list(artifacts), "path": [str(path) for path in artifacts.values()]})
        all_summary_frames.append(summary_df)
        for artifact, path in artifacts.items():
            all_artifact_rows.append({"scenario": scenario_name, "artifact": artifact, "path": str(path)})
        run_outputs.extend([table_output(f"{scenario_name} summary", summary_df), table_output(f"{scenario_name} artifacts", artifact_df)])

        del model, raw_output, eval_result
        clear_eval_cache()
        write_cell(7, run_outputs, 4)

    full_val_summary = pd.concat(all_summary_frames, ignore_index=True)
    full_val_artifacts = pd.DataFrame(all_artifact_rows)
    combined_summary_path = config.paths.output_path / "rankmixerclean_full_val_summary.csv"
    combined_artifacts_path = config.paths.output_path / "rankmixerclean_full_val_artifacts.csv"
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
    metric_df = full_val_summary[[col for col in metric_cols if col in full_val_summary.columns]]
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
