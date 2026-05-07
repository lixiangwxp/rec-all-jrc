from __future__ import annotations

import contextlib
import html
import io
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "rankmixerclean.ipynb"

SETUP_CELLS = (2, 4, 6, 8, 11, 13, 16, 27, 31, 35, 37, 39)
EXPERIMENT_CELLS = {
    "old_ge": 41,
    "query_softmax_ce": 43,
    "listwise_bpr_bce": 45,
}


def load_notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK_PATH.read_text())


def save_notebook(nb: dict[str, Any]) -> None:
    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


def source_of(nb: dict[str, Any], cell_index: int) -> str:
    return "".join(nb["cells"][cell_index].get("source", []))


def exec_cell(nb: dict[str, Any], cell_index: int, namespace: dict[str, Any]) -> None:
    print(f"\n>>> cell {cell_index}", flush=True)
    exec(compile(source_of(nb, cell_index), f"rankmixerclean.ipynb:cell-{cell_index}", "exec"), namespace)


def exec_build_model_only(nb: dict[str, Any], namespace: dict[str, Any]) -> None:
    source = source_of(nb, 29).split("\n\nmodel = build_model", 1)[0].rstrip() + "\n"
    print("\n>>> cell 29 build_model definition", flush=True)
    exec(compile(source, "rankmixerclean.ipynb:cell-29-build-model", "exec"), namespace)


def display_for_script(obj: Any) -> None:
    if hasattr(obj, "to_string"):
        print(obj.to_string(index=False), flush=True)
    else:
        print(obj, flush=True)


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def stream_output(text: str) -> dict[str, Any]:
    return {"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}


def error_output(error: BaseException) -> dict[str, Any]:
    return {
        "output_type": "error",
        "ename": error.__class__.__name__,
        "evalue": str(error),
        "traceback": traceback.format_exception(type(error), error, error.__traceback__),
    }


def table_output(title: str, frame: Any) -> dict[str, Any]:
    if frame is None:
        text = f"{title}\nNone"
        body = f"<h4>{html.escape(title)}</h4><pre>None</pre>"
    else:
        text = f"{title}\n{frame.to_string(index=False)}"
        body = f"<h4>{html.escape(title)}</h4>{frame.to_html(index=False)}"
    return {
        "output_type": "display_data",
        "metadata": {},
        "data": {"text/plain": text, "text/html": body},
    }


def write_result_cell(
    scenario_name: str,
    cell_index: int,
    execution_count: int,
    stdout_text: str,
    bundle: dict[str, Any] | None,
    error: BaseException | None,
) -> None:
    outputs = [stream_output(stdout_text)]
    if bundle is not None:
        outputs.append(table_output("summary", bundle["summary"]))
        outputs.append(table_output("history", bundle["history"]))
        pd = sys.modules["pandas"]
        artifacts = pd.DataFrame(
            [{"artifact": key, "path": str(path)} for key, path in bundle["artifacts"].items()]
        )
        outputs.append(table_output("artifacts", artifacts))
    if error is not None:
        outputs.append(error_output(error))

    nb = load_notebook()
    nb["cells"][cell_index]["execution_count"] = execution_count
    nb["cells"][cell_index]["outputs"] = outputs
    save_notebook(nb)
    print(f">>> wrote {scenario_name} results to cell {cell_index}", flush=True)


def selected_experiments() -> dict[str, int]:
    selected = os.getenv("RANKMIXER_SCENARIOS")
    if not selected:
        return EXPERIMENT_CELLS
    names = [name.strip() for name in selected.split(",") if name.strip()]
    unknown = [name for name in names if name not in EXPERIMENT_CELLS]
    if unknown:
        raise ValueError(f"Unknown scenario(s): {unknown}")
    return {name: EXPERIMENT_CELLS[name] for name in names}


def main() -> int:
    os.chdir(PROJECT_ROOT)
    nb = load_notebook()
    module = types.ModuleType("__rankmixerclean_runner__")
    sys.modules[module.__name__] = module
    namespace = module.__dict__

    for cell_index in SETUP_CELLS[:8]:
        exec_cell(nb, cell_index, namespace)
        if cell_index == 2:
            namespace["display"] = display_for_script

    exec_build_model_only(nb, namespace)

    for cell_index in SETUP_CELLS[8:]:
        exec_cell(nb, cell_index, namespace)

    failed = False
    for execution_count, (scenario_name, cell_index) in enumerate(selected_experiments().items(), start=1):
        buffer = io.StringIO()
        bundle = None
        error = None
        with (
            contextlib.redirect_stdout(Tee(sys.stdout, buffer)),
            contextlib.redirect_stderr(Tee(sys.stderr, buffer)),
        ):
            print(f"started_at={time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            try:
                bundle = namespace["run_experiment"](scenario_name)
                print(f"finished_at={time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            except BaseException as exc:  # noqa: BLE001 - keep going and write error into the notebook.
                error = exc
                failed = True
                traceback.print_exc()

        write_result_cell(scenario_name, cell_index, execution_count, buffer.getvalue(), bundle, error)

    return int(failed)


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        status_path = os.getenv("RANKMIXER_STATUS_PATH")
        if status_path:
            Path(status_path).write_text(
                f"exit_code={exit_code}\nfinished_at={time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            )
    raise SystemExit(exit_code)
