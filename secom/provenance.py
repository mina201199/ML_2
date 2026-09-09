"""實驗產物的出處紀錄：寫出指標 JSON，以及「這些數字是在什麼環境跑出來的」。

從 reporting.py 拆出來的理由：**記錄跑了什麼**和**渲染報告文字**是兩件事。
前者處理序列化、套件版本、鎖版檔；後者處理中文散文與 markdown 表格。
它們沒有共用的概念，只是碰巧都寫檔案。

reporting.py 只讀這裡寫出的 JSON，不反向依賴。
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime

import numpy as np

from . import config


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, datetime | date):
        return str(value)
    return value


def write_json(path, value):
    config.resolve(path).write_text(
        json.dumps(_clean(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_environment() -> dict:
    """記錄實際跑出這些數字的環境，並產生對應的鎖版檔。

    以前 reports/metrics/environment.json 是手打的 —— 一個手打的環境紀錄會與實際
    環境漂移，那就失去了它存在的意義。這裡改成從已安裝套件的中介資料讀出來，
    並且相依清單本身也是從 pyproject.toml 宣告的那一份讀的，不另外維護第二份名單。

    requirements.txt 是相容版本下限（給人看的意圖），
    requirements-lock.txt 是這次驗證環境的實際版本（給重現用的事實）。
    注意這只鎖直接相依，不是完整的傳遞相依鎖，檔頭有寫明。
    """
    import platform
    import sys
    from importlib.metadata import PackageNotFoundError, requires, version

    names = []
    for spec in requires("secom") or []:
        if "; extra ==" in spec:            # 只鎖執行時相依，不鎖 dev extras
            continue
        names.append(re.split(r"[<>=!~;\[\s]", spec, maxsplit=1)[0])

    packages = {}
    for name in names:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    payload = {
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "packages": packages,
    }
    write_json("reports/metrics/environment.json", payload)

    lock = [
        "# 這次驗證環境的實際版本，由 scripts/01_build_data.py 產生，不要手改。",
        "# 只鎖直接相依（pyproject.toml 宣告的那一份），不是完整的傳遞相依鎖。",
        f"# python {payload['python']} on {payload['platform']}",
    ]
    lock += [f"{n}=={v}" for n, v in packages.items() if v]
    config.resolve("requirements-lock.txt").write_text(
        "\n".join(lock) + "\n", encoding="utf-8"
    )
    return payload
