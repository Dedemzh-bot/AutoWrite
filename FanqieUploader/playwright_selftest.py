"""Playwright 自测脚本 — 连接真实番茄网站，自动执行校准流程。

用法：
1. 确保所有 Chrome 窗口已关闭（脚本会复制登录态到临时目录启动 Chrome）
2. 运行：python -m FanqieUploader.playwright_selftest

会自动选第一篇可上传小说，执行校准流程（填标题/正文/封面/AI/Tag），停在存草稿前。
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
from .executor import launch_chrome_with_login, PROJECT_ROOT, FanqieExecutor

REPORT_PATH = Path(__file__).resolve().parent / "data" / "playwright-selftest-latest.json"


def main() -> None:
    print("=" * 60)
    print("Playwright 番茄自测")
    print("=" * 60)

    # 1. 扫描小说
    catalog = Catalog(PROJECT_ROOT)
    items = [i for i in catalog.scan() if i.valid]
    if not items:
        print("❌ 没有可上传的小说（Novel/ 目录下需要有效 TXT + 大纲 JSON）")
        sys.exit(1)

    item = items[0]
    print(f"选用第一篇小说：{item.title}")
    print(f"  正文：{item.body_chars} 字")
    print(f"  主分类：{item.core_tag}")
    print(f"  标签：{item.all_tags}")
    print()

    # 2. 启动 Chrome（带用户登录态）
    print("启动 Chrome（使用用户登录态）...")
    try:
        pw, context = launch_chrome_with_login()
    except Exception as e:
        print(f"❌ 启动 Chrome 失败：{e}")
        print("请确保所有 Chrome 窗口已关闭（避免 User Data 锁定）")
        sys.exit(1)

    print("✅ Chrome 已启动")
    page = context.new_page()

    # 3. 检查番茄登录状态
    print("检查番茄登录状态...")
    page.goto("https://fanqienovel.com/main/writer/short-manage", wait_until="networkidle", timeout=30000)
    import time as _time
    _time.sleep(3)
    if "login" in page.url.lower():
        print("番茄未登录，请在弹出的 Chrome 窗口中手动登录")
        print("（登录态会保存在临时目录，以后自测自动带上）")
        print("等待登录完成...")
        # 轮询 URL 变化，登录后会跳转到非 login 页
        for _ in range(120):  # 最多等 6 分钟
            _time.sleep(3)
            try:
                cur = page.url
            except Exception:
                break
            if "login" not in cur.lower():
                print(f"检测到登录完成：{cur}")
                break
        else:
            print("登录超时（6分钟），退出")
            context.close()
            pw.stop()
            sys.exit(1)
        # 导航回管理页
        page.goto("https://fanqienovel.com/main/writer/short-manage", wait_until="networkidle", timeout=30000)
        _time.sleep(2)
        if "login" in page.url.lower():
            print("登录后仍跳转到 login 页，登录可能未成功")
            context.close()
            pw.stop()
            sys.exit(1)
    print(f"番茄已登录：{page.url}")

    # 4. 执行校准流程
    task = {
        "title": item.title,
        "body": catalog.body(item.item_id),
        "core_tag": item.core_tag,
        "tag_groups": item.tag_groups,
    }

    started = time.time()
    executor = FanqieExecutor(page, "calibration")
    result = executor.execute(task)
    result["logs"] = executor.logs
    elapsed = time.time() - started

    # 5. 关闭 Chrome
    context.close()
    pw.stop()

    # 4. 输出结果
    print()
    print("=" * 60)
    status = result.get("status", "unknown")
    if status == "awaiting_calibration":
        print(f"✅ 校准成功！耗时 {elapsed:.1f}s")
        print(f"   页面停在校准状态，请人工核对：{result.get('url', '')}")
    else:
        print(f"❌ 校准失败：{result.get('error', '未知错误')}")
        print(f"   耗时 {elapsed:.1f}s")

    print()
    print("执行日志：")
    for line in result.get("logs", []):
        print(f"  {line}")

    # 5. 保存报告
    report = {
        "novel": item.title,
        "status": status,
        "elapsed": round(elapsed, 1),
        "url": result.get("url", ""),
        "error": result.get("error", ""),
        "logs": result.get("logs", []),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存：{REPORT_PATH}")

    sys.exit(0 if status == "awaiting_calibration" else 1)


if __name__ == "__main__":
    main()
