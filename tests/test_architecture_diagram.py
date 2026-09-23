"""README 架構圖裡點名的檔案，必須真的存在。

為什麼需要這個檔案：架構圖是兩份 README 裡最大的一塊手寫內容，而它沒有
任何東西看著。腳本改名、頁面搬家、模組刪掉，圖都會安靜地繼續說謊 ——
不會有錯誤訊息，CI 也不會紅。

這個 repo 已經有兩道同樣性質的防線：`test_project_urls.py` 釘住網址、
`test_documented_counts.py` 釘住測試數量，理由都是「手維護的東西散在
多個檔案就一定會過期」。圖是同一類東西，所以用同一套作法守它：
先確定「圖存在」，再確定「圖講的是真的」。

刻意不檢查的東西：

  - **不要求每支腳本都出現在圖上。** `03_decide.py` 是故意不畫的
    （單次切分的附錄，不是結論來源）。只檢查「畫了的必須存在」這個方向，
    反過來檢查會逼著把附錄也畫進去，那會讓圖誤導。
  - **不檢查資料與指標檔名**（`secom.data`、`secom.parquet`）。那些是
    下載或生成的產物，不進版控，在乾淨 clone 上本來就不存在 ——
    檢查它們會讓這支測試在還沒跑過管線的環境變紅。
"""

import re

import pytest

from secom import config

# 兩份 README 都有圖，而且必須講同一條管線
DIAGRAM_FILES = ("README.md", "README.zh-TW.md")

MERMAID_RE = re.compile(r"```mermaid\n(.*?)```", re.S)

# 圖上的腳本一律寫成不含副檔名的字幹（01_build_data），對應 scripts/<字幹>.py
SCRIPT_RE = re.compile(r"\b(\d{2}_[a-z0-9_]+)")

# 只認 .py 與 .html —— 見模組 docstring 對資料檔的說明
CODE_FILE_RE = re.compile(r"\b([A-Za-z0-9_./-]+\.(?:py|html))\b")


def _diagram(rel: str) -> str:
    text = config.resolve(rel).read_text(encoding="utf-8")
    blocks = MERMAID_RE.findall(text)
    assert len(blocks) == 1, (
        f"{rel} 有 {len(blocks)} 個 mermaid 區塊，預期剛好一個。"
        "多一個代表有第二張圖需要一併納入這道防線，少一個代表圖被刪了。"
    )
    return blocks[0]


@pytest.fixture(scope="module")
def diagrams() -> dict[str, str]:
    return {rel: _diagram(rel) for rel in DIAGRAM_FILES}


def test_both_readmes_contain_exactly_one_architecture_diagram(diagrams):
    """圖必須存在。

    這一條看起來多餘，但它跟 `test_documented_counts.py` 裡「必須存在」
    那條的理由一樣：如果有人把圖刪掉，下面兩條就沒有東西可檢查、
    測試仍然全綠。那樣的防線比沒有防線更危險。
    """
    assert set(diagrams) == set(DIAGRAM_FILES)
    for rel, block in diagrams.items():
        assert "flowchart" in block, f"{rel} 的 mermaid 區塊不是流程圖"
        assert SCRIPT_RE.search(block), f"{rel} 的圖沒有點名任何腳本，不再是架構圖"


def test_every_file_the_diagram_names_exists(diagrams):
    """圖上點名的腳本與頁面，在 repo 裡必須找得到。"""
    missing = []
    for rel, block in diagrams.items():
        for stem in sorted(set(SCRIPT_RE.findall(block))):
            if not config.resolve(f"scripts/{stem}.py").exists():
                missing.append(f"{rel}: scripts/{stem}.py")

        for name in sorted(set(CODE_FILE_RE.findall(block))):
            if "/" in name:
                found = config.resolve(name).exists()
            else:
                # 圖上為了寬度只寫檔名（streamlit_app.py），找得到同名檔案就算數
                found = any(config.resolve(".").rglob(name))
            if not found:
                missing.append(f"{rel}: {name}")

    assert not missing, "架構圖指向不存在的檔案：\n  " + "\n  ".join(missing)


def test_both_language_diagrams_name_the_same_scripts(diagrams):
    """中英文兩張圖必須講同一條管線。

    兩張圖是分開手寫的，所以只改一邊是最可能發生的錯誤。節點文字本來就
    不同（一邊中文一邊英文），但被點名的腳本集合沒有理由不一樣。
    """
    named = {rel: set(SCRIPT_RE.findall(block)) for rel, block in diagrams.items()}
    en, zh = named["README.md"], named["README.zh-TW.md"]
    assert en == zh, (
        f"兩張圖點名的腳本不一致：只在英文版 {sorted(en - zh)}，"
        f"只在中文版 {sorted(zh - en)}"
    )
