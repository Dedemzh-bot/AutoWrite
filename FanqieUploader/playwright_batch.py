"""Playwright 批量上传脚本 — 自动保存草稿模式。

用法：
1. python -m FanqieUploader.playwright_batch

自动处理所有可上传小说，每篇：新建短故事 → 填标题/正文/封面/AI/Tag → 存草稿 → 下一本
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

# 修复 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from .catalog import Catalog
from .executor import launch_chrome_with_login, FanqieExecutor, PROJECT_ROOT

REPORT_PATH = Path(__file__).resolve().parent / "data" / "playwright-batch-latest.json"


def main() -> None:
    print("=" * 60)
    print("Playwright 番茄批量存草稿")
    print("=" * 60)

    catalog = Catalog(PROJECT_ROOT)
    items = [i for i in catalog.scan() if i.valid]
    if not items:
        print("❌ 没有可上传的小说")
        sys.exit(1)

    print(f"共 {len(items)} 篇可上传小说")
    for i, item in enumerate(items):
        print(f"  [{i+1}] {item.title} ({item.body_chars}字) —— {item.core_tag}")

    print()
    print("启动 Chrome...")
    try:
        pw, context = launch_chrome_with_login()
    except Exception as e:
        print(f"❌ 启动 Chrome 失败：{e}")
        sys.exit(1)
    print("✅ Chrome 已启动")

    # 验证登录
    page = context.new_page()
    page.goto("https://fanqienovel.com/main/writer/short-manage", wait_until="networkidle", timeout=30000)
    time.sleep(3)
    if "login" in page.url.lower():
        print("❌ 番茄未登录，请在弹出的 Chrome 窗口中登录")
        print("等待登录完成...")
        for _ in range(120):
            time.sleep(3)
            if "login" not in page.url.lower():
                print(f"检测到登录完成：{page.url}")
                break
        else:
            print("登录超时（6分钟）")
            context.close()
            pw.stop()
            sys.exit(1)
    print(f"已登录：{page.url}")
    page.close()

    # 批量执行：复用同一个管理页，每篇只关 popup 编辑器页
    results = []
    total = len(items)
    management_page = context.new_page()
    for idx, item in enumerate(items):
        print()
        print("=" * 40)
        print(f"[{idx+1}/{total}] {item.title}")
        print("=" * 40)

        task = {
            "title": item.title,
            "body": catalog.body(item.item_id),
            "core_tag": item.core_tag,
            "tag_groups": item.tag_groups,
        }

        executor = FanqieExecutor(management_page, "batch")
        started = time.time()
        result = executor.execute(task)
        elapsed = time.time() - started

        results.append({
            "title": item.title,
            "status": result.get("status"),
            "elapsed": round(elapsed, 1),
            "url": result.get("url", ""),
            "error": result.get("error", ""),
        })

        status = result.get("status")
        if status == "completed":
            print(f"  ✅ 成功（{elapsed:.1f}s）")
        else:
            print(f"  ❌ 失败（{elapsed:.1f}s）：{result.get('error','')}")
        for line in result.get("logs", []):
            print(f"    {line}")

        # 关闭 popup 编辑器页（executor 内部把 self.page 切到了它）
        # management_page 保持打开，下一篇继续用它点击"新建短故事"
        try:
            if executor.page and not executor.page.is_closed() and executor.page != management_page:
                executor.page.close()
        except Exception:
            pass

    # 汇总
    success = sum(1 for r in results if r["status"] == "completed")
    print()
    print("=" * 60)
    print(f"批量完成：{success}/{total} 成功")
    for r in results:
        icon = "✅" if r["status"] == "completed" else "❌"
        print(f"  {icon} {r['title']} ({r['elapsed']}s)")

    # 保存报告
    report = {
        "total": total,
        "success": success,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"报告：{REPORT_PATH}")

    context.close()
    pw.stop()


if __name__ == "__main__":
    main()
