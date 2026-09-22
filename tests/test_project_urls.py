"""repo 與 Pages 的網址在整個專案裡必須只有一個版本。

這些網址原本是散在四個檔案裡的手寫字串。repo 改過一次名之後就出現典型症狀：
README 改了、docs/index.html 忘了改，於是作品集首頁的連結指向舊位址 ——
而且不會有任何錯誤訊息。

現在唯一真相來源是 pyproject.toml 的 [project.urls]，docs 的 footer 由
scripts/06_build_pages.py 從那裡生成。這組測試檢查沒有任何檔案跟它脫節。

刻意檢查「內部一致」而不是「符合 git remote」：後者會讓任何 fork 的測試變紅，
而 fork 的人沒有做錯任何事。
"""

import re
import tomllib

import pytest

from secom import config

# 會提到專案網址的檔案
FILES = ["README.md", "README.zh-TW.md", "docs/index.html"]

# 只比對 owner/repo 兩段，忽略後面的路徑
SLUG_RE = re.compile(r"github\.com/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+)")
PAGES_RE = re.compile(r"https://([A-Za-z0-9-]+)\.github\.io/([A-Za-z0-9._-]+)/?")


def _read(rel):
    path = config.resolve(rel)
    if not path.exists():
        pytest.skip(f"尚未產生 {rel}")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def urls():
    with config.resolve("pyproject.toml").open("rb") as fh:
        data = tomllib.load(fh)
    assert "urls" in data["project"], (
        "pyproject.toml 缺少 [project.urls] —— 那是專案網址的唯一真相來源"
    )
    return data["project"]["urls"]


def test_pyproject_declares_repository_and_homepage(urls):
    for key in ("Repository", "Homepage"):
        assert key in urls, f"[project.urls] 缺少 {key}"
    assert SLUG_RE.search(urls["Repository"]), "Repository 不是 github.com 網址"
    assert PAGES_RE.match(urls["Homepage"]), "Homepage 不是 github.io 網址"


def test_repository_slug_is_the_same_everywhere(urls):
    """所有檔案提到的 github.com slug 必須只有一種，且等於 pyproject 宣告的那個。"""
    expected = SLUG_RE.search(urls["Repository"]).groups()

    found = {}
    for rel in FILES:
        slugs = set(SLUG_RE.findall(_read(rel)))
        if slugs:
            found[rel] = slugs

    assert found, "沒有任何檔案提到 repo 網址，連結大概被刪掉了"
    for rel, slugs in found.items():
        assert slugs == {expected}, (
            f"{rel} 的 repo slug 與 pyproject 不一致：{sorted(slugs)}，"
            f"應為 {expected[0]}/{expected[1]}"
        )


def test_pages_url_is_the_same_everywhere(urls):
    """Pages 網址（github.io）同樣不能有第二個版本。"""
    m = PAGES_RE.match(urls["Homepage"])
    expected = m.groups()

    for rel in FILES:
        pages = set(PAGES_RE.findall(_read(rel)))
        if pages:
            assert pages == {expected}, (
                f"{rel} 的 Pages 網址與 pyproject 不一致：{sorted(pages)}"
            )


def test_pages_url_owner_matches_the_repository_owner(urls):
    """github.io 的子網域必須是 repo 的 owner，否則 Pages 網址根本不會存在。"""
    repo_owner, repo_name = SLUG_RE.search(urls["Repository"]).groups()
    pages_owner, pages_path = PAGES_RE.match(urls["Homepage"]).groups()
    assert pages_owner.lower() == repo_owner.lower(), (
        f"Pages 子網域 {pages_owner} 與 repo owner {repo_owner} 不符"
    )
    assert pages_path == repo_name, (
        f"Pages 路徑 {pages_path} 與 repo 名稱 {repo_name} 不符"
    )


def test_docs_footer_links_are_generated_not_handwritten():
    """docs/index.html 的 footer 必須是生成區塊，而不是手寫的死連結。"""
    html = _read("docs/index.html")
    assert "<!-- links:start" in html and "<!-- links:end -->" in html, (
        "footer 的生成標記不見了；連結會變回手寫，改名時又會脫節"
    )
    block = html[html.index("<!-- links:start"): html.index("<!-- links:end -->")]
    assert "executive_summary.md" in block, "footer 少了報告連結"


def test_readme_links_to_the_pages_demo(urls):
    """兩份 README 的最上方都要有可點的 demo 連結 —— 這是整個部署的目的。"""
    for rel in ("README.md", "README.zh-TW.md"):
        head = "\n".join(_read(rel).split("\n")[:8])
        assert urls["Homepage"] in head, f"{rel} 前八行沒有 Pages demo 連結"
        assert "badge.svg" in head, f"{rel} 前八行沒有 CI 徽章"
