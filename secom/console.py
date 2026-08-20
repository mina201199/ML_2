"""Windows 主控台在 cp950 下會把 UTF-8 的中文印成亂碼。每個腳本開頭呼叫一次。"""

from __future__ import annotations

import sys


def enable_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
