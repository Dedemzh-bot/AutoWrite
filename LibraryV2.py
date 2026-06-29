from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MATERIAL_LIBRARY_PATH = ROOT / "material_library.json"
PATTERN_LIBRARY_PATH = ROOT / "pattern_library.json"
NOVEL_TAG_LIBRARY_PATH = ROOT / "novel_tag_library.json"
MATERIAL_SCHEMA_VERSION = 2
PATTERN_SCHEMA_VERSION = 2
NOVEL_TAG_SCHEMA_VERSION = 1
NOVEL_TAG_CATEGORIES = ("情节", "角色", "情绪", "背景")
DEFAULT_MATERIAL_COUNT = 4
MIN_MATERIAL_COUNT = 0
MAX_MATERIAL_COUNT = 8
MATERIAL_GROUP_ORDER = [
    "world_stage",
    "protagonist",
    "supporting_role",
    "cheat_device",
    "plot_event",
    "core_conflict",
    "career_resource",
    "atmosphere",
]
MATERIAL_GROUP_LIMITS = {
    "world_stage": 1,
    "protagonist": 1,
    "supporting_role": 2,
    "cheat_device": 2,
    "plot_event": 2,
    "core_conflict": 2,
    "career_resource": 2,
    "atmosphere": 2,
}
DEFAULT_GROUP_COUNTS = {
    "world_stage": 1,
    "protagonist": 1,
    "supporting_role": 0,
    "cheat_device": 1,
    "plot_event": 0,
    "core_conflict": 1,
    "career_resource": 0,
    "atmosphere": 0,
}


class LibraryValidationError(RuntimeError):
    pass


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as error:
        raise LibraryValidationError(f"找不到内容库：{path}") from error
    except json.JSONDecodeError as error:
        raise LibraryValidationError(f"内容库 JSON 格式错误：{path}: {error}") from error


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _normalized_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", str(value or "")).lower()


@lru_cache(maxsize=1)
def load_material_library() -> dict:
    library = _read_json(MATERIAL_LIBRARY_PATH)
    issues = validate_material_library(library)
    if issues:
        raise LibraryValidationError("素材库校验失败：" + "；".join(issues[:10]))
    return library


@lru_cache(maxsize=1)
def load_pattern_library() -> dict:
    library = _read_json(PATTERN_LIBRARY_PATH)
    issues = validate_pattern_library(library)
    if issues:
        raise LibraryValidationError("套路库校验失败：" + "；".join(issues[:10]))
    return library


def load_novel_tag_library() -> dict:
    """Load on every call so manual JSON edits apply to the next outline."""
    library = _read_json(NOVEL_TAG_LIBRARY_PATH)
    issues = validate_novel_tag_library(library)
    if issues:
        raise LibraryValidationError("小说Tag库校验失败：" + "；".join(issues[:10]))
    return {
        "schema_version": NOVEL_TAG_SCHEMA_VERSION,
        "core_tags": [item.strip() for item in library["core_tags"]],
        "secondary_tags": {
            category: [item.strip() for item in library["secondary_tags"][category]]
            for category in NOVEL_TAG_CATEGORIES
        },
    }


def validate_novel_tag_library(library: dict) -> list[str]:
    issues: list[str] = []
    if not isinstance(library, dict):
        return ["词库根节点必须是对象"]
    if library.get("schema_version") != NOVEL_TAG_SCHEMA_VERSION:
        issues.append(f"schema_version 必须为 {NOVEL_TAG_SCHEMA_VERSION}")

    core_tags = library.get("core_tags")
    if not isinstance(core_tags, list):
        issues.append("core_tags 必须是字符串数组")
        core_tags = []

    secondary_tags = library.get("secondary_tags")
    if not isinstance(secondary_tags, dict):
        issues.append("secondary_tags 必须是对象")
        secondary_tags = {}
    else:
        missing = [key for key in NOVEL_TAG_CATEGORIES if key not in secondary_tags]
        extra = [key for key in secondary_tags if key not in NOVEL_TAG_CATEGORIES]
        if missing:
            issues.append("缺少辅助Tag分类：" + "、".join(missing))
        if extra:
            issues.append("存在未知辅助Tag分类：" + "、".join(extra))

    seen: dict[str, str] = {}

    def validate_group(group_name: str, values: Any) -> int:
        if not isinstance(values, list):
            issues.append(f"{group_name}必须是字符串数组")
            return 0
        valid_count = 0
        for index, value in enumerate(values, start=1):
            if not isinstance(value, str):
                issues.append(f"{group_name}第{index}项必须是字符串")
                continue
            tag = value.strip()
            if not tag:
                issues.append(f"{group_name}第{index}项不能为空")
                continue
            valid_count += 1
            if tag in seen:
                issues.append(f"Tag重复：{tag}（{seen[tag]}、{group_name}）")
            else:
                seen[tag] = group_name
        return valid_count

    core_count = validate_group("核心Tag", core_tags)
    secondary_count = sum(
        validate_group(category, secondary_tags.get(category))
        for category in NOVEL_TAG_CATEGORIES
    )
    if core_count < 1:
        issues.append("核心Tag至少需要1个可用词")
    if secondary_count < 5:
        issues.append("辅助Tag总数至少需要5个可用词")
    return list(dict.fromkeys(issues))


def format_novel_tag_library(library: dict | None = None) -> str:
    library = library or load_novel_tag_library()
    lines = ["核心Tag（必须且只能选1个）：" + "、".join(library["core_tags"])]
    lines.append("辅助Tag（四类合计选5-7个，各分类数量不限）：")
    for category in NOVEL_TAG_CATEGORIES:
        lines.append(f"- {category}：" + "、".join(library["secondary_tags"][category]))
    return "\n".join(lines)


def normalize_novel_tag_selection(selection: Any) -> dict:
    selection = selection if isinstance(selection, dict) else {}
    normalized = {"core": str(selection.get("core", "")).strip()}
    for category in NOVEL_TAG_CATEGORIES:
        values = selection.get(category, [])
        if not isinstance(values, list):
            values = []
        normalized[category] = [str(item).strip() for item in values]
    return normalized


def validate_novel_tag_selection(
    selection: Any, library: dict | None = None
) -> list[str]:
    issues: list[str] = []
    if not isinstance(selection, dict):
        return ["novel_tags 必须是对象"]
    library = library or load_novel_tag_library()
    normalized = normalize_novel_tag_selection(selection)

    expected_keys = {"core", *NOVEL_TAG_CATEGORIES}
    missing = [key for key in expected_keys if key not in selection]
    extra = [key for key in selection if key not in expected_keys]
    if missing:
        issues.append("novel_tags 缺少字段：" + "、".join(sorted(missing)))
    if extra:
        issues.append("novel_tags 存在未知字段：" + "、".join(extra))

    core = normalized["core"]
    if core not in library["core_tags"]:
        issues.append(f"核心Tag不在词库中：{core or '空'}")

    selected: list[str] = []
    for category in NOVEL_TAG_CATEGORIES:
        raw_values = selection.get(category)
        if not isinstance(raw_values, list):
            issues.append(f"{category}Tag必须是数组")
            continue
        allowed = set(library["secondary_tags"][category])
        for tag in normalized[category]:
            if not tag:
                issues.append(f"{category}Tag不能为空")
            elif tag not in allowed:
                issues.append(f"{category}Tag不在该分类词库中：{tag}")
            selected.append(tag)

    if not 5 <= len(selected) <= 7:
        issues.append(f"辅助Tag必须为5-7个，当前为{len(selected)}个")
    duplicates = sorted({tag for tag in selected if selected.count(tag) > 1 and tag})
    if duplicates:
        issues.append("辅助Tag不能重复：" + "、".join(duplicates))
    return list(dict.fromkeys(issues))


def validate_material_library(library: dict) -> list[str]:
    issues: list[str] = []
    if library.get("schema_version") != MATERIAL_SCHEMA_VERSION:
        issues.append(f"schema_version 必须为 {MATERIAL_SCHEMA_VERSION}")
    groups = library.get("groups")
    entries = library.get("entries")
    if not isinstance(groups, dict) or len(groups) < 8:
        issues.append("素材库至少需要8个大类")
        groups = {}
    if not isinstance(entries, list) or len(entries) < 500:
        issues.append("素材库至少需要500条素材")
        entries = []
    known_subcategories = {
        child.get("id")
        for group in groups.values()
        if isinstance(group, dict)
        for child in group.get("subcategories", [])
        if isinstance(child, dict)
    }
    ids: set[str] = set()
    texts: set[str] = set()
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            issues.append(f"第{index + 1}条素材不是对象")
            continue
        item_id = str(item.get("id", "")).strip()
        text = str(item.get("text", "")).strip()
        if not item_id:
            issues.append(f"第{index + 1}条素材缺少ID")
        elif item_id in ids:
            issues.append(f"素材ID重复：{item_id}")
        ids.add(item_id)
        normalized = _normalized_text(text)
        if not normalized:
            issues.append(f"素材内容为空：{item_id}")
        elif normalized in texts:
            issues.append(f"素材文本重复：{text}")
        texts.add(normalized)
        category = item.get("category")
        if category not in groups:
            issues.append(f"素材{item_id}引用未知大类：{category}")
        if item.get("subcategory") not in known_subcategories:
            issues.append(f"素材{item_id}引用未知子类：{item.get('subcategory')}")
        if not item.get("slot"):
            issues.append(f"素材{item_id}缺少槽位")
    return list(dict.fromkeys(issues))


def validate_pattern_library(library: dict) -> list[str]:
    issues: list[str] = []
    if library.get("schema_version") != PATTERN_SCHEMA_VERSION:
        issues.append(f"schema_version 必须为 {PATTERN_SCHEMA_VERSION}")
    patterns = library.get("patterns")
    if not isinstance(patterns, dict):
        return issues + ["套路库 patterns 必须是对象"]
    normal_count = sum(
        key not in {"none", "custom"} and not item.get("strong")
        for key, item in patterns.items()
        if isinstance(item, dict)
    )
    strong_count = sum(
        bool(item.get("strong"))
        for item in patterns.values()
        if isinstance(item, dict)
    )
    if normal_count < 60:
        issues.append("常规套路至少需要60个")
    if strong_count != 13:
        issues.append("强套路必须为13个")
    required_base = {"id", "name", "category", "strong", "architect", "writer", "auditor"}
    required_strong = {
        "compatible_styles",
        "ending_options",
        "protagonist_pool",
        "counterpart_pool",
        "foil_pool",
        "background_pool",
        "conflict_modules",
        "beats",
        "writing_techniques",
        "audit_rules",
        "forbidden",
    }
    for key, item in patterns.items():
        if not isinstance(item, dict):
            issues.append(f"套路{key}不是对象")
            continue
        missing = sorted(required_base - set(item))
        if missing:
            issues.append(f"套路{key}缺少字段：{','.join(missing)}")
        if item.get("id") != key:
            issues.append(f"套路键与ID不一致：{key}")
        if item.get("strong"):
            strong_missing = sorted(required_strong - set(item))
            if strong_missing:
                issues.append(f"强套路{key}缺少字段：{','.join(strong_missing)}")
            if len(item.get("conflict_modules", [])) < 4:
                issues.append(f"强套路{key}冲突模块不足")
            if len(item.get("beats", [])) < 5:
                issues.append(f"强套路{key}节拍不足")
        for conflict in item.get("hard_conflicts", []):
            if conflict not in patterns:
                issues.append(f"套路{key}引用未知冲突套路：{conflict}")
    return list(dict.fromkeys(issues))


def material_entries_by_id() -> dict[str, dict]:
    return {
        item["id"]: item
        for item in load_material_library()["entries"]
    }


def pattern_map() -> dict[str, dict]:
    return load_pattern_library()["patterns"]


def default_material_config() -> dict:
    return {
        "schema_version": MATERIAL_SCHEMA_VERSION,
        "filters": {
            "categories": [],
            "subcategories": [],
            "tags": [],
        },
        "group_counts": dict(DEFAULT_GROUP_COUNTS),
        "count": DEFAULT_MATERIAL_COUNT,
        "items": [],
        "locked_item_keys": [],
        "auto_selected_subcategories": [],
        "legacy_import": [],
    }


def default_pattern_config() -> dict:
    return {
        "schema_version": PATTERN_SCHEMA_VERSION,
        "primary": "none",
        "secondary": [],
        "custom_instruction": "",
        "manifest": {},
        "structure_plan": {},
    }


def normalize_material_config(config: dict | None) -> dict:
    raw = config if isinstance(config, dict) else {}
    filters = raw.get("filters") if isinstance(raw.get("filters"), dict) else {}
    items_by_id = material_entries_by_id()
    has_group_counts = isinstance(raw.get("group_counts"), dict)
    raw_counts = raw.get("group_counts") if has_group_counts else {}
    if not has_group_counts and isinstance(raw.get("items"), list):
        derived_counts = {group_id: 0 for group_id in MATERIAL_GROUP_ORDER}
        for value in raw["items"]:
            item_id = value.get("id") if isinstance(value, dict) else value
            item = items_by_id.get(item_id, {})
            category = item.get("category")
            if category in derived_counts:
                derived_counts[category] += 1
        derived_total = sum(derived_counts.values())
        if MIN_MATERIAL_COUNT <= derived_total <= MAX_MATERIAL_COUNT:
            raw_counts = derived_counts
            has_group_counts = True
    group_counts = {}
    for group_id in MATERIAL_GROUP_ORDER:
        try:
            value = int(raw_counts.get(group_id, DEFAULT_GROUP_COUNTS[group_id]))
        except (TypeError, ValueError):
            value = DEFAULT_GROUP_COUNTS[group_id]
        group_counts[group_id] = max(0, value)
    total_count = sum(group_counts.values())
    if not has_group_counts:
        group_counts = dict(DEFAULT_GROUP_COUNTS)
        total_count = DEFAULT_MATERIAL_COUNT
    if total_count > MAX_MATERIAL_COUNT:
        raise LibraryValidationError(
            f"素材总数不能超过{MAX_MATERIAL_COUNT}项"
        )
    items = []
    category_ordinals: dict[str, int] = {}
    for value in raw.get("items", []):
        item_id = value.get("id") if isinstance(value, dict) else value
        if item_id in items_by_id and item_id not in {item["id"] for item in items}:
            item = dict(items_by_id[item_id])
            category = item.get("category", "")
            category_ordinals[category] = category_ordinals.get(category, 0) + 1
            provided_key = (
                str(value.get("selection_key", "")).strip()
                if isinstance(value, dict)
                else ""
            )
            item["selection_key"] = (
                provided_key
                or f"{category}:{category_ordinals[category]}"
            )
            items.append(item)
    return {
        "schema_version": MATERIAL_SCHEMA_VERSION,
        "filters": {
            "categories": _as_string_list(filters.get("categories")),
            "subcategories": _as_string_list(filters.get("subcategories")),
            "tags": _as_string_list(filters.get("tags")),
        },
        "group_counts": group_counts,
        "count": total_count,
        "items": items,
        "locked_item_keys": _as_string_list(raw.get("locked_item_keys")),
        "auto_selected_subcategories": _as_string_list(
            raw.get("auto_selected_subcategories")
        ),
        "legacy_import": _as_string_list(raw.get("legacy_import")),
        "excluded_ids": _as_string_list(raw.get("excluded_ids")),
    }


def normalize_pattern_config(config: dict | None) -> dict:
    raw = config if isinstance(config, dict) else {}
    return {
        "schema_version": PATTERN_SCHEMA_VERSION,
        "primary": str(raw.get("primary") or "none").strip(),
        "secondary": _as_string_list(raw.get("secondary"))[:2],
        "custom_instruction": str(raw.get("custom_instruction") or "").strip(),
        "manifest": raw.get("manifest") if isinstance(raw.get("manifest"), dict) else {},
        "structure_plan": (
            raw.get("structure_plan")
            if isinstance(raw.get("structure_plan"), dict)
            else {}
        ),
    }


def pattern_pair_conflict_reason(primary_id: str, secondary_id: str) -> str:
    patterns = pattern_map()
    primary = patterns.get(primary_id)
    secondary = patterns.get(secondary_id)
    if not primary or not secondary:
        return "套路不存在"
    if secondary.get("strong"):
        return "辅助套路不能使用强套路"
    if secondary_id in {"none", "custom"}:
        return "辅助套路不能选择无套路或自定义套路"
    if secondary_id == primary_id:
        return "主套路与辅助套路不能重复"
    if (
        secondary_id in primary.get("hard_conflicts", [])
        or primary_id in secondary.get("hard_conflicts", [])
    ):
        return f"{primary.get('name')}与{secondary.get('name')}存在硬冲突"
    primary_tags = set(primary.get("tags", []))
    secondary_tags = set(secondary.get("tags", []))
    if primary_tags.intersection(secondary.get("forbidden_pattern_tags", [])):
        return f"{secondary.get('name')}禁止主套路标签：{','.join(primary_tags)}"
    if secondary_tags.intersection(primary.get("forbidden_pattern_tags", [])):
        return f"{primary.get('name')}禁止辅助套路标签：{','.join(secondary_tags)}"
    return ""


def validate_pattern_config(config: dict | None, writer_style: str = "") -> list[str]:
    normalized = normalize_pattern_config(config)
    patterns = pattern_map()
    issues = []
    primary_id = normalized["primary"]
    primary = patterns.get(primary_id)
    if not primary:
        return [f"主套路不存在：{primary_id}"]
    if primary_id == "custom" and not normalized["custom_instruction"]:
        issues.append("自定义主套路必须填写具体要求")
    if len(normalized["secondary"]) > 2:
        issues.append("辅助套路最多两个")
    if len(set(normalized["secondary"])) != len(normalized["secondary"]):
        issues.append("辅助套路不能重复")
    for secondary_id in normalized["secondary"]:
        reason = pattern_pair_conflict_reason(primary_id, secondary_id)
        if reason:
            issues.append(reason)
    compatible_styles = primary.get("compatible_styles", [])
    if writer_style and compatible_styles and writer_style not in compatible_styles:
        issues.append(
            f"{primary.get('name')}仅支持写手风格：{', '.join(compatible_styles)}"
        )
    return list(dict.fromkeys(issues))


def material_pattern_conflict_reason(
    item: dict,
    pattern_config: dict | None,
) -> str:
    normalized = normalize_pattern_config(pattern_config)
    patterns = pattern_map()
    for index, pattern_id in enumerate(
        [normalized["primary"], *normalized["secondary"]]
    ):
        pattern = patterns.get(pattern_id, {})
        category = item.get("category")
        tags = set(item.get("tags", [])) | set(item.get("drivers", []))
        if pattern_id in item.get("forbidden_pattern_ids", []):
            return f"素材{item.get('id')}明确禁止套路：{pattern.get('name', pattern_id)}"
        if category in pattern.get("forbidden_material_categories", []):
            role = "主套路" if index == 0 else "辅助套路"
            return f"{role}{pattern.get('name')}禁止素材大类：{category}"
        blocked_tags = tags.intersection(pattern.get("forbidden_material_tags", []))
        if blocked_tags:
            role = "主套路" if index == 0 else "辅助套路"
            return (
                f"{role}{pattern.get('name')}禁止素材标签："
                + "、".join(sorted(blocked_tags))
            )
    return ""


def material_pair_conflict_reason(left: dict, right: dict) -> str:
    left_id = left.get("id", "")
    right_id = right.get("id", "")
    if right_id in left.get("hard_conflicts", []):
        return f"{left_id}与{right_id}存在素材硬冲突"
    if left_id in right.get("hard_conflicts", []):
        return f"{left_id}与{right_id}存在素材硬冲突"
    left_tags = set(left.get("tags", []))
    right_tags = set(right.get("tags", []))
    left_blocked = set(left.get("incompatible_tags", []))
    right_blocked = set(right.get("incompatible_tags", []))
    blocked = left_blocked.intersection(right_tags) | right_blocked.intersection(
        left_tags
    )
    if blocked:
        return "素材标签硬冲突：" + "、".join(sorted(blocked))
    return ""


def validate_material_config(
    material_config: dict | None,
    pattern_config: dict | None,
    require_items: bool = True,
) -> list[str]:
    normalized = normalize_material_config(material_config)
    issues: list[str] = []
    library = load_material_library()
    groups = library["groups"]
    subcategories = {
        child["id"]
        for group in groups.values()
        for child in group.get("subcategories", [])
    }
    unknown_categories = [
        value
        for value in normalized["filters"]["categories"]
        if value not in groups
    ]
    unknown_subcategories = [
        value
        for value in normalized["filters"]["subcategories"]
        if value not in subcategories
    ]
    if unknown_categories:
        issues.append("未知素材大类：" + "、".join(unknown_categories))
    if unknown_subcategories:
        issues.append("未知素材子类：" + "、".join(unknown_subcategories))
    total_count = sum(normalized["group_counts"].values())
    if not MIN_MATERIAL_COUNT <= total_count <= MAX_MATERIAL_COUNT:
        issues.append(
            f"素材总数必须在{MIN_MATERIAL_COUNT}到{MAX_MATERIAL_COUNT}之间"
        )
    for group_id, value in normalized["group_counts"].items():
        if not 0 <= value <= MATERIAL_GROUP_LIMITS[group_id]:
            issues.append(
                f"{groups.get(group_id, {}).get('name', group_id)}最多选择"
                f"{MATERIAL_GROUP_LIMITS[group_id]}项"
            )
    if require_items and len(normalized["items"]) != total_count:
        issues.append(
            f"已确认素材数量必须为{total_count}，"
            f"当前为{len(normalized['items'])}"
        )
    item_ids: set[str] = set()
    item_texts: set[str] = set()
    selection_keys: set[str] = set()
    actual_group_counts = {group_id: 0 for group_id in MATERIAL_GROUP_ORDER}
    for index, item in enumerate(normalized["items"]):
        reason = material_pattern_conflict_reason(item, pattern_config)
        if reason:
            issues.append(f"素材“{item.get('text')}”冲突：{reason}")
        item_id = item.get("id", "")
        normalized_text = _normalized_text(item.get("text", ""))
        selection_key = item.get("selection_key", "")
        category = item.get("category", "")
        if item_id in item_ids:
            issues.append(f"素材ID重复：{item_id}")
        if normalized_text in item_texts:
            issues.append(f"素材文本重复：{item.get('text', '')}")
        if not selection_key:
            issues.append(f"素材{item_id}缺少selection_key")
        elif selection_key in selection_keys:
            issues.append(f"素材selection_key重复：{selection_key}")
        item_ids.add(item_id)
        item_texts.add(normalized_text)
        selection_keys.add(selection_key)
        if category in actual_group_counts:
            actual_group_counts[category] += 1
        for other in normalized["items"][index + 1:]:
            pair_reason = material_pair_conflict_reason(item, other)
            if pair_reason:
                issues.append(
                    f"素材“{item.get('text')}”与“{other.get('text')}”冲突："
                    f"{pair_reason}"
                )
    if require_items:
        for group_id, expected in normalized["group_counts"].items():
            if actual_group_counts[group_id] != expected:
                issues.append(
                    f"{groups[group_id].get('name', group_id)}应有{expected}项，"
                    f"当前为{actual_group_counts[group_id]}项"
                )
    unknown_locked = set(normalized["locked_item_keys"]) - selection_keys
    if require_items and unknown_locked:
        issues.append(
            "锁定项不存在：" + "、".join(sorted(unknown_locked))
        )
    return list(dict.fromkeys(issues))


def _matches_filters(item: dict, filters: dict) -> bool:
    categories = set(filters.get("categories", []))
    subcategories = set(filters.get("subcategories", []))
    tags = set(filters.get("tags", []))
    if categories and item.get("category") not in categories:
        return False
    if subcategories and item.get("subcategory") not in subcategories:
        return False
    if tags and not tags.intersection(item.get("tags", [])):
        return False
    return True


def sample_materials(
    material_config: dict | None,
    pattern_config: dict | None,
    seed: int | None = None,
    randomize_types: bool = False,
) -> dict:
    config = normalize_material_config(material_config)
    rng = random.Random(seed)
    library = load_material_library()
    if randomize_types:
        explicit = set(config["filters"]["subcategories"]) - set(
            config["auto_selected_subcategories"]
        )
        config["filters"]["subcategories"] = sorted(explicit)
        config["auto_selected_subcategories"] = []
    desired_keys = [
        f"{group_id}:{index}"
        for group_id in MATERIAL_GROUP_ORDER
        for index in range(1, config["group_counts"][group_id] + 1)
    ]
    locked_keys = set(config["locked_item_keys"])
    existing = {
        item.get("selection_key"): item
        for item in config["items"]
        if item.get("selection_key") in locked_keys
        and item.get("selection_key") in desired_keys
    }
    selected = list(existing.values())
    used_ids = {item["id"] for item in selected}
    excluded_ids = set(config.get("excluded_ids", []))
    excluded_ids.update(
        item["id"]
        for item in config["items"]
        if item.get("selection_key") not in existing
    )
    used_texts = {_normalized_text(item["text"]) for item in selected}
    selected_subcategories = set(config["filters"]["subcategories"])
    for selection_key in desired_keys:
        if selection_key in existing:
            continue
        group_id = selection_key.split(":", 1)[0]
        group_subcategories = {
            child["id"]
            for child in library["groups"][group_id].get("subcategories", [])
        }
        preferred_subcategories = (
            selected_subcategories.intersection(group_subcategories)
        )
        candidates = [
            item for item in library["entries"]
            if item.get("category") == group_id
            and (
                not preferred_subcategories
                or item.get("subcategory") in preferred_subcategories
            )
            and _matches_filters(
                item,
                {
                    "categories": [],
                    "subcategories": [],
                    "tags": config["filters"]["tags"],
                },
            )
            and not material_pattern_conflict_reason(item, pattern_config)
            and item["id"] not in excluded_ids
            and item["id"] not in used_ids
            and _normalized_text(item["text"]) not in used_texts
            and not any(
                material_pair_conflict_reason(item, selected_item)
                for selected_item in selected
            )
        ]
        if not candidates:
            raise LibraryValidationError(
                f"筛选和冲突规则下无法为“{group_id}”抽取素材，请扩大素材范围"
            )
        choice = dict(rng.choice(candidates))
        choice["selection_key"] = selection_key
        selected.append(choice)
        used_ids.add(choice["id"])
        used_texts.add(_normalized_text(choice["text"]))
    selected_by_key = {item["selection_key"]: item for item in selected}
    config["items"] = [
        selected_by_key[key] for key in desired_keys if key in selected_by_key
    ]
    explicit_subcategories = set(config["filters"]["subcategories"]) - set(
        config["auto_selected_subcategories"]
    )
    actual_subcategories = {
        item["subcategory"] for item in config["items"]
    }
    config["auto_selected_subcategories"] = sorted(
        actual_subcategories - explicit_subcategories
    )
    config["filters"]["subcategories"] = sorted(
        explicit_subcategories | actual_subcategories
    )
    config["locked_item_keys"] = [
        key for key in config["locked_item_keys"] if key in desired_keys
    ]
    config["excluded_ids"] = []
    issues = validate_material_config(config, pattern_config)
    if issues:
        raise LibraryValidationError("素材抽取结果无效：" + "；".join(issues))
    return config


def resample_material_item(
    material_config: dict | None,
    pattern_config: dict | None,
    selection_key: str,
    seed: int | None = None,
    change_type: bool = False,
) -> dict:
    config = normalize_material_config(material_config)
    target = next(
        (
            item for item in config["items"]
            if item.get("selection_key") == selection_key
        ),
        None,
    )
    if target is None:
        raise LibraryValidationError(f"找不到待重抽素材：{selection_key}")
    rng = random.Random(seed)
    selected = [
        item for item in config["items"]
        if item.get("selection_key") != selection_key
    ]
    used_ids = {item["id"] for item in selected}
    used_texts = {_normalized_text(item["text"]) for item in selected}
    candidates = [
        item for item in load_material_library()["entries"]
        if item.get("category") == target.get("category")
        and item.get("id") != target.get("id")
        and (
            item.get("subcategory") != target.get("subcategory")
            if change_type
            else item.get("subcategory") == target.get("subcategory")
        )
        and item["id"] not in used_ids
        and _normalized_text(item["text"]) not in used_texts
        and not material_pattern_conflict_reason(item, pattern_config)
        and not any(
            material_pair_conflict_reason(item, selected_item)
            for selected_item in selected
        )
    ]
    if not candidates:
        action = "更换子类" if change_type else "保留子类重抽"
        raise LibraryValidationError(f"{action}失败：当前筛选下没有可用素材")
    replacement = dict(rng.choice(candidates))
    replacement["selection_key"] = selection_key
    config["items"] = [
        replacement if item.get("selection_key") == selection_key else item
        for item in config["items"]
    ]
    config["locked_item_keys"] = [
        key for key in config["locked_item_keys"] if key != selection_key
    ]
    explicit_subcategories = set(config["filters"]["subcategories"]) - set(
        config["auto_selected_subcategories"]
    )
    actual_subcategories = {item["subcategory"] for item in config["items"]}
    config["auto_selected_subcategories"] = sorted(
        actual_subcategories - explicit_subcategories
    )
    config["filters"]["subcategories"] = sorted(
        explicit_subcategories | actual_subcategories
    )
    issues = validate_material_config(config, pattern_config)
    if issues:
        raise LibraryValidationError("素材重抽结果无效：" + "；".join(issues))
    return config


def format_selected_materials(material_config: dict | None) -> str:
    config = normalize_material_config(material_config)
    if not config["items"]:
        return "无确认素材"
    group_names = {
        key: value.get("name", key)
        for key, value in load_material_library()["groups"].items()
    }
    lines = [
        f"- [{item.get('selection_key')}] {group_names.get(item.get('category'), item.get('category'))}"
        f"/{item.get('subcategory')}：{item.get('text')}（ID: {item.get('id')}）"
        for item in config["items"]
    ]
    repeated_groups = {
        group_id
        for group_id, count in config["group_counts"].items()
        if count > 1
    }
    if repeated_groups:
        names = "、".join(group_names[group_id] for group_id in repeated_groups)
        lines.append(
            f"- [同类融合要求] {names}存在多项素材。必须明确主次、协同机制、"
            "各自边界与代价；多个冲突或事件必须形成先后因果，禁止简单并列堆砌。"
        )
    if config.get("legacy_import"):
        lines.append(
            "- [旧版原文保留] " + "；".join(config["legacy_import"])
        )
    return "\n".join(lines)


def resolve_pattern_bundle(pattern_config: dict | None) -> dict:
    config = normalize_pattern_config(pattern_config)
    patterns = pattern_map()
    primary = dict(patterns.get(config["primary"], patterns["none"]))
    if config["primary"] == "custom":
        instruction = config["custom_instruction"]
        for field in ("architect", "writer", "auditor"):
            primary[field] = f"{primary.get(field, '')}\n用户自定义主套路：{instruction}"
    secondaries = [
        dict(patterns[item])
        for item in config["secondary"]
        if item in patterns
    ]
    secondary_architect = "\n".join(
        f"- {item['name']}：{item.get('architect', '')}"
        for item in secondaries
    ) or "无"
    secondary_writer = "\n".join(
        f"- {item['name']}：{item.get('writer', '')}"
        for item in secondaries
    ) or "无"
    secondary_auditor = "\n".join(
        f"- {item['name']}：{item.get('auditor', '')}"
        for item in secondaries
    ) or "无"
    return {
        "config": config,
        "primary": primary,
        "secondary": secondaries,
        "architect": (
            f"【主套路（硬约束）】{primary.get('name')}\n{primary.get('architect', '')}\n"
            f"【辅助套路（软约束）】\n{secondary_architect}\n"
            "辅助套路只能补充局部桥段，不得覆盖主套路节拍、人物弧线和结局。"
        ),
        "writer": (
            f"【主套路（必须完成）】{primary.get('name')}\n{primary.get('writer', '')}\n"
            f"【辅助套路（可选桥段）】\n{secondary_writer}"
        ),
        "auditor": (
            f"【主套路（可退稿）】{primary.get('name')}\n{primary.get('auditor', '')}\n"
            f"【辅助套路（只警告）】\n{secondary_auditor}\n"
            "辅助套路未体现不得加入硬问题或触发退稿。"
        ),
    }


def material_library_metadata(pattern_config: dict | None = None) -> dict:
    library = load_material_library()
    patterns = normalize_pattern_config(pattern_config)
    metadata = {
        "schema_version": MATERIAL_SCHEMA_VERSION,
        "count_range": [MIN_MATERIAL_COUNT, MAX_MATERIAL_COUNT],
        "default_count": DEFAULT_MATERIAL_COUNT,
        "default_group_counts": dict(DEFAULT_GROUP_COUNTS),
        "group_limits": dict(MATERIAL_GROUP_LIMITS),
        "slot_order": library.get("slot_order", []),
        "groups": {},
    }
    for group_id, group in library["groups"].items():
        group_items = [
            item for item in library["entries"] if item.get("category") == group_id
        ]
        group_blocked = bool(group_items) and all(
            material_pattern_conflict_reason(item, patterns)
            for item in group_items
        )
        metadata["groups"][group_id] = {
            "name": group.get("name", group_id),
            "default_slot": group.get("default_slot", ""),
            "count": len(group_items),
            "blocked": group_blocked,
            "max_count": MATERIAL_GROUP_LIMITS[group_id],
            "default_count": DEFAULT_GROUP_COUNTS[group_id],
            "subcategories": [],
        }
        for child in group.get("subcategories", []):
            child_items = [
                item
                for item in group_items
                if item.get("subcategory") == child.get("id")
            ]
            conflict_reasons = [
                material_pattern_conflict_reason(item, patterns)
                for item in child_items
            ]
            blocked_reasons = {
                reason for reason in conflict_reasons if reason
            }
            metadata["groups"][group_id]["subcategories"].append({
                **child,
                "count": len(child_items),
                "blocked": bool(child_items) and all(conflict_reasons),
                "conflict_reason": next(iter(blocked_reasons), ""),
            })
    return metadata


def pattern_library_metadata() -> dict:
    patterns = pattern_map()
    return {
        key: {
            "id": key,
            "name": item.get("name", key),
            "category": item.get("category", ""),
            "strong": bool(item.get("strong")),
            "tags": list(item.get("tags", [])),
            "compatible_styles": list(item.get("compatible_styles", [])),
            "ending_options": dict(item.get("ending_options", {})),
            "hard_conflicts": list(item.get("hard_conflicts", [])),
            "forbidden_pattern_tags": list(
                item.get("forbidden_pattern_tags", [])
            ),
            "forbidden_material_categories": list(
                item.get("forbidden_material_categories", [])
            ),
            "forbidden_material_tags": list(
                item.get("forbidden_material_tags", [])
            ),
        }
        for key, item in patterns.items()
    }


def legacy_material_config(keywords: list[str] | None) -> dict:
    config = default_material_config()
    keywords = _as_string_list(keywords)
    if not keywords:
        return sample_materials(config, default_pattern_config(), seed=1)
    entries = load_material_library()["entries"]
    selected = []
    for keyword in keywords:
        normalized = _normalized_text(keyword)
        match = next(
            (
                item
                for item in entries
                if normalized
                and (
                    normalized in _normalized_text(item["text"])
                    or _normalized_text(item["text"]) in normalized
                )
                and item.get("id") not in {value.get("id") for value in selected}
            ),
            None,
        )
        if match:
            selected.append(dict(match))
    if selected:
        group_counts = {group_id: 0 for group_id in MATERIAL_GROUP_ORDER}
        keyed_items = []
        for item in selected[:MAX_MATERIAL_COUNT]:
            group_id = item["category"]
            if group_counts[group_id] >= MATERIAL_GROUP_LIMITS[group_id]:
                continue
            group_counts[group_id] += 1
            item["selection_key"] = f"{group_id}:{group_counts[group_id]}"
            keyed_items.append(item)
        if sum(group_counts.values()) >= MIN_MATERIAL_COUNT:
            config["group_counts"] = group_counts
            config["count"] = sum(group_counts.values())
            config["items"] = keyed_items
            config["locked_item_keys"] = [
                item["selection_key"] for item in keyed_items
            ]
    return sample_materials(config, default_pattern_config(), seed=1)


def legacy_pattern_config(
    story_pattern: str | None,
    custom_pattern: str | None,
    manifest: dict | None,
) -> dict:
    patterns = pattern_map()
    primary = str(story_pattern or "none").strip()
    if primary not in patterns:
        primary = "none"
    return {
        "schema_version": PATTERN_SCHEMA_VERSION,
        "primary": primary,
        "secondary": [],
        "custom_instruction": str(custom_pattern or "").strip(),
        "manifest": manifest if isinstance(manifest, dict) else {},
        "structure_plan": {},
    }
