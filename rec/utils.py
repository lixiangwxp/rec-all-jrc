from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from rec.config import OUTPUT_PATH, SAVE_PATH


def ensure_project_paths() -> None:
    for path in (SAVE_PATH, OUTPUT_PATH):
        path.mkdir(parents=True, exist_ok=True)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def save_pickle(value: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as file:
        pickle.dump(value, file)


def display(value: Any) -> None:
    print(value)
