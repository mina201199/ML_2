"""步驟 6：把實驗產出灌進 docs/index.html 的資料區塊，供 GitHub Pages 服務。

    python scripts/06_build_pages.py

為什麼要有這一支：docs/index.html 是一頁自包含的靜態試算頁，任何人點連結就能用，
不需要 clone、不需要跑伺服器。但它的數字必須跟 reports/metrics/ 完全一致 ——
這個 repo 的規則是報告裡不准出現手打的數字，那條規則同樣適用於這一頁。

所以頁面的 HTML/CSS/JS 是手寫並進版控的普通檔案，只有 `<script id="payload">`
那一個區塊由這支腳本覆寫。實驗重跑之後再跑這一支，頁面就同步。

頁面裡的決策邏輯是 secom/decision.py 的 JavaScript 移植。移植的正確性由
tests/test_pages_payload.py 驗證：它把 Python 的答案寫成對照表，
瀏覽器端的實作必須逐項相符。
"""

from __future__ import annotations

import json
import tomllib

from secom import config, decision
from secom.console import enable_utf8

PAGE = "docs/index.html"

ACTION_LABELS = {
    "inspect_none": "永遠不檢",
    "inspect_all": "永遠全檢",
    "random_at_capacity": "按配額隨機加驗",
}

# payload 區塊的邊界。用固定字串而不是正則貪婪比對，避免誤吃到頁面其他 script。
OPEN = '<script id="payload" type="application/json">'
CLOSE = "</script>"

# footer 連結區塊的邊界
LINKS_OPEN = "<!-- links:start"
LINKS_CLOSE = "<!-- links:end -->"


def project_urls() -> dict:
    """從 pyproject.toml 讀出 repo 與 Pages 網址。

    這是唯一真相來源。直接讀檔而不是讀已安裝套件的中介資料 ——
    後者在改完 pyproject 但沒重新安裝時會是舊的，那正是要避免的失效模式。
    """
    with config.resolve("pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["urls"]


def build_payload() -> dict:
    cfg = config.load()
    holdout = json.loads(
        config.resolve("reports/metrics/scored_holdout.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        config.resolve("reports/metrics/backtest.json").read_text(encoding="utf-8")
    )

    # 保守策略的放大倍數只取決於驗證窗的 (n_fail, n)，是常數。
    # 內嵌它，前端就不必實作 Clopper-Pearson 的 beta 反函數。
    n = holdout["windows"]["val"]["n"]
    n_fail = holdout["windows"]["val"]["n_fail"]
    p_hat = n_fail / n
    p_up = decision.conservative_base_rate(n_fail, n, 0.95)

    dominance = report["dominance"]
    beaten = set()
    for dom in dominance.values():
        beaten |= set(dom["configurations_beating_model_free"])

    results = [
        {
            "key": key,
            "cost_per_lot": r["summary"]["cost_per_lot"],
            "lift_at_20_median": r["summary"].get("lift_at_20_median"),
            "roc_auc_median": r["summary"].get("roc_auc_median"),
            "total_missed": r["summary"]["total_missed"],
            "beats": key in beaten,
        }
        for key, r in report["results"].items()
    ]

    display = report["display_key"]
    u = report["uncertainty"][display]

    return {
        "generated_by": "scripts/06_build_pages.py",
        "cost_assumptions": report["cost_assumptions"],
        "val": holdout["val"],
        "test": holdout["test"],
        "windows": holdout["windows"],
        "escape_inflation": p_up / p_hat,
        "action_labels": ACTION_LABELS,
        "dominance": dominance,
        "reference_policies": {
            regime: {
                "costs": ref["costs"],
                "feasible": ref["feasible"],
                "best_action": ref["best_action"],
            }
            for regime, ref in report["reference_policies"].items()
        },
        "results": results,
        "uncertainty": {
            "n_lots": u["power"]["n_pos"] + u["power"]["n_neg"],
            "n_pos": u["power"]["n_pos"],
            "min_detectable_auc": u["power"]["min_detectable_auc"],
            "relative_saving_ci": u["bootstrap"]["relative_saving_ci"],
            "permutation_p": u["permutation_vs_random"]["p_value_one_sided"],
        },
        "permutation_p": u["permutation_vs_random"]["p_value_one_sided"],
        "display_key": display,
        "display_lift_median": report["results"][display]["summary"].get(
            "lift_at_20_median"
        ),
        "currency": cfg.cost.currency,
    }


def write_page(payload: dict) -> None:
    path = config.resolve(PAGE)
    html = path.read_text(encoding="utf-8")
    start = html.index(OPEN) + len(OPEN)
    end = html.index(CLOSE, start)
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # </script> 不會出現在 JSON 裡，但 JSON 字串中的 "<" 仍先轉義，避免任何提早收尾
    blob = blob.replace("<", "\\u003c")
    path.write_text(html[:start] + blob + html[end:], encoding="utf-8")


def write_links(urls: dict) -> None:
    """把 footer 的連結從 pyproject 的網址生成。

    刻意在建置時把 href 寫進 HTML，而不是用 JavaScript 設定 ——
    連結在沒有 JS 的情況下也要能用（爬蟲、預覽、關掉 JS 的讀者）。
    """
    path = config.resolve(PAGE)
    html = path.read_text(encoding="utf-8")
    start = html.index(LINKS_OPEN)
    end = html.index(LINKS_CLOSE, start) + len(LINKS_CLOSE)

    repo = urls["Repository"].rstrip("/")
    report = f"{repo}/blob/main/reports/executive_summary.md"
    lines = [
        f"{LINKS_OPEN} 由 scripts/06_build_pages.py 從 pyproject.toml 生成，不要手改 -->",
        '  <p style="margin:0">',
        f'    <a href="{repo}">原始碼與報告</a> ·',
        f'    <a href="{report}">實驗報告</a> ·',
        '    <a href="https://archive.ics.uci.edu/dataset/179/secom">資料來源 UCI SECOM</a>',
        "  </p>",
        f"  {LINKS_CLOSE}",
    ]
    path.write_text(
        html[:start] + "\n".join(lines) + html[end:], encoding="utf-8"
    )


def main() -> None:
    enable_utf8()
    urls = project_urls()
    payload = build_payload()
    payload["repo_url"] = urls["Repository"].rstrip("/")
    payload["pages_url"] = urls["Homepage"]
    write_page(payload)
    write_links(urls)

    path = config.resolve(PAGE)
    size = path.stat().st_size
    print(f"\n  已寫入 {PAGE}（{size / 1024:.0f} KB）")
    print(f"  驗證窗 {payload['windows']['val']['n']} 筆／"
          f"{payload['windows']['val']['n_fail']} fail　"
          f"測試窗 {payload['windows']['test']['n']} 筆／"
          f"{payload['windows']['test']['n_fail']} fail")
    print(f"  內嵌的 escape_inflation = {payload['escape_inflation']:.10f}")
    print(f"  回測配置 {len(payload['results'])} 個，"
          f"跨過零模型門檻 {sum(r['beats'] for r in payload['results'])} 個")
    print(f"  repo  {urls['Repository']}")
    print(f"  pages {urls['Homepage']}")
    print("\n  網址的唯一真相來源是 pyproject.toml 的 [project.urls]；"
          "repo 改名時改那兩行再跑這一支即可。\n")


if __name__ == "__main__":
    main()
