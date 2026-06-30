import argparse
import datetime
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = PROJECT_ROOT / "BatchIdeaLauncher" / "runs"
DEFAULT_NOVEL_DIR = PROJECT_ROOT / "Novel"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "TestResults"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def discover_failed_partials(
    runs_dir: Path = DEFAULT_RUNS_DIR,
    novel_dir: Path = DEFAULT_NOVEL_DIR,
) -> list[dict]:
    runs_root = runs_dir.resolve()
    novels_root = novel_dir.resolve()
    if not _inside(runs_root, PROJECT_ROOT) or not _inside(novels_root, PROJECT_ROOT):
        raise ValueError("扫描目录必须位于项目工作区内")
    entries = []
    for result_file in sorted(runs_root.glob("*/*/result.json")):
        try:
            result = json.loads(result_file.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("status") != "failed":
            continue
        run_id = str(result.get("run_id", "")).strip()
        if not run_id:
            continue
        for source in sorted(novels_root.glob(f"*_{run_id}.txt")):
            source = source.resolve()
            job_dir = result_file.resolve().parent
            target = (job_dir / "work" / run_id / "partial_novel.txt").resolve()
            if not _inside(source, novels_root) or not _inside(target, job_dir):
                raise ValueError(f"迁移路径越界：{source} -> {target}")
            entries.append({
                "run_id": run_id,
                "result_file": str(result_file.resolve()),
                "source": str(source),
                "target": str(target),
                "size": source.stat().st_size,
                "sha256": _sha256(source),
                "action": "pending",
            })
    return entries


def migrate(entries: list[dict], apply: bool) -> list[dict]:
    migrated = []
    for raw in entries:
        entry = dict(raw)
        source = Path(entry["source"]).resolve()
        target = Path(entry["target"]).resolve()
        if not source.is_file():
            entry["action"] = "source_missing"
            migrated.append(entry)
            continue
        if target.exists():
            entry["action"] = (
                "already_migrated"
                if target.is_file() and _sha256(target) == entry["sha256"]
                else "target_exists_conflict"
            )
            migrated.append(entry)
            continue
        if not apply:
            entry["action"] = "would_move"
            migrated.append(entry)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        actual_hash = _sha256(target)
        if actual_hash != entry["sha256"] or target.stat().st_size != entry["size"]:
            if not source.exists() and target.exists():
                target.replace(source)
            raise RuntimeError(f"迁移后校验失败，已尝试回滚：{source}")
        entry["action"] = "moved"
        entry["verified_sha256"] = actual_hash
        migrated.append(entry)
    return migrated


def remaining_failed_run_ids(
    runs_dir: Path = DEFAULT_RUNS_DIR,
    novel_dir: Path = DEFAULT_NOVEL_DIR,
) -> list[str]:
    remaining = discover_failed_partials(runs_dir, novel_dir)
    return sorted({entry["run_id"] for entry in remaining})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将失败批次的小说半成品从 Novel/ 迁移到各 job 中转区"
    )
    parser.add_argument("--apply", action="store_true", help="执行迁移；默认只预览")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--novel-dir", type=Path, default=DEFAULT_NOVEL_DIR)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    entries = discover_failed_partials(args.runs_dir, args.novel_dir)
    migrated = migrate(entries, args.apply)
    remaining = (
        remaining_failed_run_ids(args.runs_dir, args.novel_dir)
        if args.apply
        else sorted({entry["run_id"] for entry in entries})
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    manifest = (
        args.manifest.resolve()
        if args.manifest
        else (DEFAULT_REPORT_DIR / f"failed_novel_migration_{timestamp}.json").resolve()
    )
    payload = {
        "schema_version": 1,
        "mode": "apply" if args.apply else "dry-run",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "entry_count": len(migrated),
        "total_size": sum(int(entry["size"]) for entry in migrated),
        "entries": migrated,
        "remaining_failed_run_ids_in_novel": remaining,
    }
    _write_json(manifest, payload)
    print(f"mode={payload['mode']}")
    print(f"entries={payload['entry_count']}")
    print(f"total_size={payload['total_size']}")
    print(f"remaining_failed_run_ids={len(remaining)}")
    print(f"manifest={manifest}")
    return 0 if not args.apply or not remaining else 1


if __name__ == "__main__":
    raise SystemExit(main())
