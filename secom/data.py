"""SECOM 資料下載、解析與時序切分。

資料格式（UCI 原始檔，非 Kaggle 的合併版）：
  secom.data        1567 列 × 590 欄，空白分隔的浮點數，缺值寫作 "NaN"
  secom_labels.data 1567 列 × 2 欄，`label "DD/MM/YYYY HH:MM:SS"`
                    label: -1 = pass，1 = fail

正樣本（fail）是稀有類別：104 / 1567 = 6.64%，不平衡比約 1:14。
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg_mod

TS_FORMAT = "%d/%m/%Y %H:%M:%S"

# 嚴格比對 f000..f589。不能用 startswith("f") —— 那會把標籤欄 `fail` 也算成特徵，
# 等於把答案餵給模型。這是本專題第一個被抓到的洩漏漏洞，留著當提醒。
FEATURE_RE = re.compile(r"^f\d{3}$")


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if FEATURE_RE.match(c)]


def download(cfg) -> tuple[Path, Path]:
    """把兩個原始檔抓到 data/raw。已存在就跳過。"""
    raw = cfg_mod.resolve_dir(cfg.data.raw_dir)
    targets = [
        (cfg.data.feature_url, raw / "secom.data"),
        (cfg.data.label_url, raw / "secom_labels.data"),
    ]
    for url, dest in targets:
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  已存在，跳過  {dest.name}")
            continue
        print(f"  下載中        {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"  完成          {dest.name}  ({dest.stat().st_size:,} bytes)")
    return targets[0][1], targets[1][1]


def load_raw(cfg) -> pd.DataFrame:
    """讀取兩個原始檔，合併成一張表並依時間排序。

    回傳欄位：ts, fail, f000 ... f589
    """
    raw = cfg_mod.resolve_dir(cfg.data.raw_dir)

    X = pd.read_csv(raw / "secom.data", sep=" ", header=None, na_values=["NaN"])
    X.columns = [f"f{i:03d}" for i in range(X.shape[1])]

    y = pd.read_csv(
        raw / "secom_labels.data",
        sep=" ",
        header=None,
        names=["label", "ts"],
        quotechar='"',
    )

    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(y["ts"], format=TS_FORMAT),
            # -1 pass / 1 fail  ->  0 pass / 1 fail（fail 是我們要抓的正樣本）
            "fail": (y["label"] == 1).astype("int8"),
        }
    )
    df = pd.concat([df, X], axis=1)

    # 原始檔本來就是時間遞增的，但不要相信，明確排一次
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    return df


@dataclass
class Split:
    """一次時序切分的結果。X 保留 NaN —— LightGBM 原生支援。"""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    ts_train: pd.Series
    ts_val: pd.Series
    ts_test: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in ("train", "val", "test"):
            y = getattr(self, f"y_{name}")
            ts = getattr(self, f"ts_{name}")
            rows.append(
                {
                    "split": name,
                    "n": len(y),
                    "n_fail": int(y.sum()),
                    "fail_rate": y.mean(),
                    "from": ts.min(),
                    "to": ts.max(),
                }
            )
        return pd.DataFrame(rows)


def time_split(df: pd.DataFrame, cfg) -> Split:
    """依時間順序切成 train / val / test。

    為什麼不用隨機切分：製程會隨時間漂移（機台保養、換料、換配方）。
    隨機切分會讓同一天的批次同時出現在 train 和 test，模型等於偷看未來，
    離線分數會虛高，上線後直接崩掉。這是 SECOM 公開 notebook 最常見的錯誤。
    """
    n = len(df)
    n_train = int(n * cfg.split.train_frac)
    n_val = int(n * cfg.split.val_frac)

    feats = feature_cols(df)
    idx = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n),
    }

    parts = {}
    for name, sl in idx.items():
        chunk = df.iloc[sl]
        parts[f"X_{name}"] = chunk[feats].reset_index(drop=True)
        parts[f"y_{name}"] = chunk["fail"].reset_index(drop=True)
        parts[f"ts_{name}"] = chunk["ts"].reset_index(drop=True)

    return Split(**parts)


def profile(df: pd.DataFrame) -> dict:
    """給 README 和 EDA 用的資料側寫。數字都是真的，不要手打。"""
    feats = feature_cols(df)
    X = df[feats]
    n_fail = int(df["fail"].sum())
    nunique = X.nunique(dropna=True)
    return {
        "n_rows": len(df),
        "n_features": len(feats),
        "n_fail": n_fail,
        "n_pass": len(df) - n_fail,
        "fail_rate": n_fail / len(df),
        "imbalance_ratio": (len(df) - n_fail) / n_fail,
        "n_missing_cells": int(X.isna().sum().sum()),
        "missing_frac": float(X.isna().sum().sum() / X.size),
        "n_constant_cols": int((nunique <= 1).sum()),
        "n_cols_over_half_missing": int((X.isna().mean() > 0.5).sum()),
        "ts_from": str(df["ts"].min()),
        "ts_to": str(df["ts"].max()),
        "span_days": int((df["ts"].max() - df["ts"].min()).days),
    }


def build(cfg) -> tuple[pd.DataFrame, dict]:
    """完整流程：下載 -> 解析 -> 存 parquet -> 回傳 (df, profile)。"""
    download(cfg)
    df = load_raw(cfg)
    prof = profile(df)

    out = cfg_mod.resolve_dir(cfg.data.processed_dir) / "secom.parquet"
    df.to_parquet(out, index=False)
    print(f"  已寫出        {out.relative_to(cfg_mod.ROOT)}")
    return df, prof


def load_processed(cfg) -> pd.DataFrame:
    path = cfg_mod.resolve_dir(cfg.data.processed_dir) / "secom.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}。請先執行： python scripts/01_build_data.py"
        )
    return pd.read_parquet(path)
