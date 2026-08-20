"""設定檔載入。所有路徑都相對於 repo 根目錄解析。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


class Config(dict):
    """dict 加上點記法存取，方便 cfg.data.raw_dir 這樣寫。"""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value


def load(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open(encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def resolve(rel: str | Path) -> Path:
    """把設定檔裡的相對路徑轉成絕對路徑，並確保父目錄存在。"""
    out = ROOT / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def resolve_dir(rel: str | Path) -> Path:
    out = ROOT / rel
    out.mkdir(parents=True, exist_ok=True)
    return out
