"""文件裡寫的測試數量，必須等於實際的測試數量。

為什麼需要這個檔案：這些數字已經過期過一次。`ci.yml` 的註解停在 28、
README 的檔案導覽停在 41，而當時實際上有 54 個 —— 而這個 repo 的 CI 賣點
正好是「有測試被跳過就讓 CI 紅掉，否則綠色徽章什麼都不保證」。
拿這件事當賣點，卻讓同一句話裡的數字自己爛掉，是最難看的一種矛盾。

手維護的數字散在五個地方就一定會再過期一次，所以改成讓測試看著它。
作法跟 `test_project_urls.py` 一樣：先確定「只有一種版本」，再確定那個版本是對的。
"""

import ast
import pathlib
import re

import pytest

from secom import config

TESTS_DIR = pathlib.Path(__file__).parent

# 會寫到測試數量的文件。少列一個，這道防線就漏一個。
DOC_FILES = ("README.md", "README.zh-TW.md", ".github/workflows/ci.yml")

# 「54 個測試」與「54 passed」兩種寫法
TOTAL_PATTERNS = (r"(\d+) 個測試", r"(\d+) passed")

# 「13 個需要資料」與「那 13 個正好是全部的洩漏防護測試」
DATA_DEPENDENT_PATTERNS = (r"(\d+) 個需要", r"(\d+) 個正好")


def _test_files():
    return sorted(TESTS_DIR.glob("test_*.py"))


def _parsed():
    return [(p, ast.parse(p.read_text(encoding="utf-8"))) for p in _test_files()]


def _test_functions(tree):
    return [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]


def _found(patterns) -> dict[str, list[int]]:
    """在每份文件裡把 patterns 抓到的數字收集起來。"""
    out = {}
    for rel in DOC_FILES:
        text = config.resolve(rel).read_text(encoding="utf-8")
        numbers = [int(m) for pat in patterns for m in re.findall(pat, text)]
        if numbers:
            out[rel] = numbers
    return out


@pytest.fixture(scope="module")
def collected() -> int:
    return sum(len(_test_functions(tree)) for _, tree in _parsed())


def test_static_counting_is_still_a_valid_way_to_count_this_suite(collected):
    """先驗證計數方式本身：沒有 parametrize、沒有 class 形式的測試。

    `def test_` 的數量等於 pytest 收集到的數量，前提是沒有參數化（一個函式
    展開成多個測試）也沒有測試類別（函式不在模組頂層）。這兩件事任一出現，
    下面那些斷言就會安靜地少算 —— 所以先把前提釘住。

    真的需要 parametrize 的那天，這個測試會紅，那時改成讀 pytest 的收集結果。
    """
    for path, tree in _parsed():
        marks = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and n.attr == "parametrize"]
        assert not marks, (
            f"{path.name} 用了 parametrize —— 靜態計數會少算，"
            "請改用 pytest 的收集結果來數"
        )
        classes = [n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name.startswith("Test")]
        assert not classes, f"{path.name} 有測試類別 —— 靜態計數只看模組頂層函式"
        assert not [n for n in tree.body
                    if isinstance(n, ast.AsyncFunctionDef) and n.name.startswith("test_")]

    assert collected > 0, "一個測試都沒數到，計數器壞了"


def test_documented_test_total_is_stated_and_is_correct(collected):
    """文件裡的總數必須存在、只有一種版本、而且等於實際數量。

    「必須存在」這一條是刻意的：如果有人把那句話改寫掉，數字就不再被檢查，
    而測試仍然是綠的。那樣的防線比沒有防線更危險。
    """
    found = _found(TOTAL_PATTERNS)
    missing = [rel for rel in DOC_FILES if rel not in found]
    assert not missing, f"這些文件沒有寫出測試總數，防線漏掉了：{missing}"

    stated = {n for numbers in found.values() for n in numbers}
    assert stated == {collected}, (
        f"文件寫的測試數 {sorted(stated)} 與實際的 {collected} 不符。"
        f"逐檔：{found}"
    )


def test_documented_data_dependent_count_is_correct():
    """「13 個測試需要資料」也要對。這句話是 CI 那道 skip 關卡的理由。

    需要 data/processed/secom.parquet 的測試，就是吃 `df` fixture 的那些 ——
    那個 fixture 在資料不存在時 skip。CI 一定要下載 UCI 原始檔的唯一理由
    就是這批測試，所以這個數字錯了，那段註解的說服力就沒了。
    """
    needs_data = [f.name for _, tree in _parsed() for f in _test_functions(tree)
                  if any(a.arg == "df" for a in f.args.args)]

    found = _found(DATA_DEPENDENT_PATTERNS)
    assert found, "沒有任何文件寫出「需要資料的測試數」"

    stated = {n for numbers in found.values() for n in numbers}
    assert stated == {len(needs_data)}, (
        f"文件寫的「需要資料的測試數」{sorted(stated)} 與實際的 "
        f"{len(needs_data)} 不符。逐檔：{found}"
    )


def test_every_leakage_test_is_in_the_data_dependent_set():
    """順手釘住那句話的另一半：需要資料的那些，確實就是洩漏防護測試。

    文件說「那 13 個正好是全部的洩漏防護測試」。這條檢查它們都住在
    test_no_leakage.py 裡 —— 若某天洩漏防護測試搬家或新增在別處，
    那句「正好」就不再成立。
    """
    homes = {path.name for path, tree in _parsed() for f in _test_functions(tree)
             if any(a.arg == "df" for a in f.args.args)}
    assert homes == {"test_no_leakage.py"}, (
        f"需要資料的測試散在 {sorted(homes)}，文件那句「正好是全部的洩漏防護測試」"
        "需要改寫"
    )
