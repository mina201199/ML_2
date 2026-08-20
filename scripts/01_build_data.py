"""步驟 1：下載 SECOM，解析成一張表，存 parquet，印出資料側寫。

    python scripts/01_build_data.py

資料從 UCI 直接抓，不需要 Kaggle API token。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from secom import config as cfg_mod
from secom import data as data_mod
from secom.console import enable_utf8


def main() -> None:
    enable_utf8()
    cfg = cfg_mod.load()

    print("\n[1/3] 下載原始檔")
    print("[2/3] 解析與合併")
    df, prof = data_mod.build(cfg)

    print("\n[3/3] 資料側寫")
    labels = {
        "n_rows": "批次數",
        "n_features": "感測器特徵數",
        "n_fail": "fail 批次",
        "n_pass": "pass 批次",
        "fail_rate": "fail 比例",
        "imbalance_ratio": "不平衡比 (1:N)",
        "n_missing_cells": "缺值格數",
        "missing_frac": "缺值佔比",
        "n_constant_cols": "零變異欄位",
        "n_cols_over_half_missing": "缺值過半的欄位",
        "ts_from": "起始時間",
        "ts_to": "結束時間",
        "span_days": "涵蓋天數",
    }
    for key, label in labels.items():
        v = prof[key]
        shown = f"{v:.2%}" if key in {"fail_rate", "missing_frac"} else (
            f"{v:.1f}" if key == "imbalance_ratio" else f"{v:,}" if isinstance(v, int) else v
        )
        print(f"  {label:<16} {shown}")

    out = cfg_mod.resolve("reports/metrics/data_profile.json")
    out.write_text(json.dumps(prof, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  側寫已存 {out.relative_to(cfg_mod.ROOT)}")

    print("\n  提醒：fail 比例只有 6.6%，所以 accuracy 這個指標在本專題全程禁用。")
    print("  下一步： python scripts/02_train.py\n")


if __name__ == "__main__":
    main()
