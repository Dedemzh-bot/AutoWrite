from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

from LibraryV2 import (
    default_material_config,
    legacy_pattern_config,
    normalize_material_config,
    sample_materials,
    validate_material_config,
    validate_pattern_config,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTLINE_DIR = ROOT / "Outline"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _needs_material_v2_upgrade(data: dict) -> bool:
    if data.get("schema_version") != 2:
        return True
    config = data.get("material_config")
    if not isinstance(config, dict):
        return True
    if not isinstance(config.get("group_counts"), dict):
        return True
    if not isinstance(config.get("locked_item_keys"), list):
        return True
    return any(
        not isinstance(item, dict) or not item.get("selection_key")
        for item in config.get("items", [])
    )


def convert_outline(data: dict, seed: int) -> dict:
    if data.get("schema_version") == 2:
        migrated = dict(data)
        raw_material = (
            data.get("material_config")
            if isinstance(data.get("material_config"), dict)
            else {}
        )
        material_config = normalize_material_config(raw_material)
        legacy_locked_slots = {
            str(value) for value in raw_material.get("locked_slots", [])
        }
        if legacy_locked_slots:
            material_config["locked_item_keys"] = [
                item["selection_key"]
                for item in material_config["items"]
                if item.get("slot") in legacy_locked_slots
            ]
        pattern_config = data.get("pattern_config", {})
        material_issues = validate_material_config(
            material_config, pattern_config
        )
        if material_issues:
            raise ValueError("素材字段升级失败：" + "；".join(material_issues))
        migrated["material_config"] = material_config
        migrated["material_config_upgraded_at"] = dt.datetime.now().isoformat(
            timespec="seconds"
        )
        return migrated
    pattern_config = legacy_pattern_config(
        data.get("story_pattern"),
        data.get("custom_pattern"),
        data.get("pattern_manifest"),
    )
    pattern_config["structure_plan"] = (
        data.get("pattern_plan")
        if isinstance(data.get("pattern_plan"), dict)
        else {}
    )
    pattern_issues = validate_pattern_config(pattern_config)
    if pattern_issues:
        raise ValueError("套路迁移失败：" + "；".join(pattern_issues))

    raw_material = default_material_config()
    legacy_keywords = data.get("keywords", [])
    if legacy_keywords:
        raw_material["legacy_import"] = list(map(str, legacy_keywords))
    material_config = sample_materials(raw_material, pattern_config, seed=seed)
    material_issues = validate_material_config(material_config, pattern_config)
    if material_issues:
        raise ValueError("素材迁移失败：" + "；".join(material_issues))

    migrated = {
        key: value
        for key, value in data.items()
        if key not in {
            "keywords",
            "story_pattern",
            "custom_pattern",
            "pattern_manifest",
            "pattern_plan",
        }
    }
    migrated.update({
        "schema_version": 2,
        "material_config": normalize_material_config(material_config),
        "pattern_config": pattern_config,
        "migrated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "migration_source_schema": data.get("schema_version", 1),
    })
    return migrated


def migrate_directory(directory: Path, apply: bool) -> dict:
    directory = directory.resolve()
    files = sorted(directory.glob("*.json")) if directory.exists() else []
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = directory.parent / "OutlineBackup" / timestamp
    report = {
        "schema_version": 2,
        "mode": "apply" if apply else "preview",
        "outline_directory": str(directory),
        "backup_directory": str(backup_dir) if apply else "",
        "converted": [],
        "skipped": [],
        "failed": [],
    }
    for index, path in enumerate(files, start=1):
        try:
            data = _read(path)
            if not _needs_material_v2_upgrade(data):
                report["skipped"].append({
                    "file": path.name,
                    "reason": "已经是最新素材字段格式",
                })
                continue
            converted = convert_outline(data, seed=index)
            if apply:
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup_dir / path.name)
                _write_atomic(path, converted)
            report["converted"].append({
                "file": path.name,
                "title": converted.get("title", path.stem),
                "chapters": len(converted.get("chapter_outlines", {})),
                "primary_pattern": converted["pattern_config"]["primary"],
                "material_count": len(converted["material_config"]["items"]),
            })
        except Exception as error:
            report["failed"].append({
                "file": path.name,
                "error": f"{type(error).__name__}: {error}",
            })
    report_path = directory.parent / (
        f"outline_migration_{'apply' if apply else 'preview'}_{timestamp}.json"
    )
    _write_atomic(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将旧版 AutoWrite 大纲一次性迁移到 schema v2"
    )
    parser.add_argument(
        "--outline-dir",
        type=Path,
        default=DEFAULT_OUTLINE_DIR,
        help="大纲目录，默认项目下 Outline",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="实际备份并写入；不提供时只生成预览报告",
    )
    args = parser.parse_args()
    report = migrate_directory(args.outline_dir, args.apply)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
