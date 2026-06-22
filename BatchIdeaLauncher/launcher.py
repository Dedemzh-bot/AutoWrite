from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from .core import (
        AutoWriteCLI,
        BatchRunner,
        DEFAULT_RUNS_DIR,
        LAUNCHER_ROOT,
        LauncherError,
        OpenAISelector,
        find_batch,
        format_status,
        initialize_batch,
        load_batch_config,
        read_json,
    )
except ImportError:
    from core import (
        AutoWriteCLI,
        BatchRunner,
        DEFAULT_RUNS_DIR,
        LAUNCHER_ROOT,
        LauncherError,
        OpenAISelector,
        find_batch,
        format_status,
        initialize_batch,
        load_batch_config,
        read_json,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="独立 AI 批量小说启动器（不启动 Web 服务）"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="批次运行目录",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog", help="刷新能力表")
    catalog.add_argument(
        "--output",
        type=Path,
        default=LAUNCHER_ROOT / "catalog" / "capabilities.json",
    )

    run = subparsers.add_parser("run", help="创建并运行一个批次")
    run.add_argument("--ideas", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--batch-id")

    status = subparsers.add_parser("status", help="查看批次状态")
    status.add_argument("--batch-id", required=True)

    retry = subparsers.add_parser("retry", help="继续或重试一个批次")
    retry.add_argument("--batch-id", required=True)
    retry.add_argument(
        "--failed-only",
        action="store_true",
        help="只重试 failed；否则也恢复 pending/selecting/running",
    )
    return parser


def _autowrite() -> AutoWriteCLI:
    return AutoWriteCLI(
        entry=Path(os.environ["AUTOWRITE_ENTRY"])
        if os.environ.get("AUTOWRITE_ENTRY")
        else None,
        python_command=os.environ.get("AUTOWRITE_PYTHON"),
    )


def command_catalog(args) -> int:
    capabilities = _autowrite().export_capabilities(args.output)
    print(f"能力表已刷新：{args.output.resolve()}")
    print(
        f"写手 {len(capabilities.get('writer_styles', []))} 种，"
        f"套路 {len(capabilities.get('story_patterns', []))} 种，"
        f"素材大类 {len(capabilities.get('material_library', {}).get('groups', {}))} 类"
    )
    return 0


def command_run(args) -> int:
    config = load_batch_config(args.config.resolve())
    selector = OpenAISelector(config["selector"])
    autowrite = _autowrite()
    batch_dir = initialize_batch(
        args.ideas.resolve(),
        config,
        autowrite,
        runs_dir=args.runs_dir.resolve(),
        batch_id=args.batch_id,
    )
    print(f"批次已创建：{batch_dir.name}")
    summary = BatchRunner(
        batch_dir, autowrite, selector=selector
    ).process()
    print(format_status(read_json(batch_dir / "batch.json")))
    print(f"汇总报告：{batch_dir / 'summary.csv'}")
    return 1 if summary["counts"].get("failed", 0) else 0


def command_status(args) -> int:
    batch_dir = find_batch(args.batch_id, args.runs_dir.resolve())
    print(format_status(read_json(batch_dir / "batch.json")))
    return 0


def command_retry(args) -> int:
    batch_dir = find_batch(args.batch_id, args.runs_dir.resolve())
    runner = BatchRunner(batch_dir, _autowrite())
    statuses = {"failed"} if args.failed_only else {
        "pending",
        "failed",
        "selecting",
        "running",
    }
    summary = runner.process(statuses=statuses, reuse_selection=True)
    print(format_status(read_json(batch_dir / "batch.json")))
    print(f"汇总报告：{batch_dir / 'summary.csv'}")
    return 1 if summary["counts"].get("failed", 0) else 0


def main() -> int:
    load_dotenv(LAUNCHER_ROOT / ".env", override=False)
    args = build_parser().parse_args()
    try:
        if args.command == "catalog":
            return command_catalog(args)
        if args.command == "run":
            return command_run(args)
        if args.command == "status":
            return command_status(args)
        if args.command == "retry":
            return command_retry(args)
        raise LauncherError(f"未知命令：{args.command}")
    except LauncherError as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
