from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TAG_GROUPS = ("情节", "角色", "情绪", "背景")
MIN_BODY_CHARS = 6000
MAX_TAGS = 8


def body_char_count(text: str) -> int:
    return len(re.sub(r"\s+", "", str(text or "")))


@dataclass(frozen=True)
class CatalogItem:
    item_id: str
    title: str
    novel_file: str
    outline_file: str
    run_id: str
    body_chars: int
    core_tag: str
    tag_groups: dict[str, list[str]]
    all_tags: list[str]
    valid: bool
    errors: list[str]
    matched_by: str
    modified_at: float

    def as_dict(self, uploaded: bool = False, draft_url: str = "") -> dict[str, Any]:
        return {
            "id": self.item_id,
            "title": self.title,
            "novel_file": self.novel_file,
            "outline_file": self.outline_file,
            "run_id": self.run_id,
            "body_chars": self.body_chars,
            "core_tag": self.core_tag,
            "tag_groups": self.tag_groups,
            "all_tags": self.all_tags,
            "tag_count": len(self.all_tags),
            "valid": self.valid,
            "errors": self.errors,
            "matched_by": self.matched_by,
            "uploaded": uploaded,
            "draft_url": draft_url,
            "modified_at": self.modified_at,
        }


@dataclass(frozen=True)
class _OutlineMetadata:
    path: Path
    payload: dict[str, Any] | None
    error: str
    run_id: str


def _read_outline(path: Path) -> _OutlineMetadata:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("顶层必须是JSON对象")
        return _OutlineMetadata(
            path=path,
            payload=payload,
            error="",
            run_id=str(payload.get("run_id") or "").strip(),
        )
    except Exception as error:
        return _OutlineMetadata(path=path, payload=None, error=f"大纲JSON无效：{error}", run_id="")


def _normalize_tags(raw: Any) -> tuple[str, dict[str, list[str]], list[str], list[str]]:
    errors: list[str] = []
    if not isinstance(raw, dict):
        return "", {group: [] for group in TAG_GROUPS}, [], ["缺少novel_tags"]

    core = str(raw.get("core") or "").strip()
    if not core:
        errors.append("缺少主分类core")

    groups: dict[str, list[str]] = {}
    all_tags = [core] if core else []
    for group in TAG_GROUPS:
        values = raw.get(group, [])
        if not isinstance(values, list):
            errors.append(f"{group}必须是数组")
            groups[group] = []
            continue
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        groups[group] = cleaned
        all_tags.extend(cleaned)

    if len(all_tags) > MAX_TAGS:
        errors.append(f"Tag总数{len(all_tags)}超过网站上限{MAX_TAGS}")
    duplicates = sorted({tag for tag in all_tags if all_tags.count(tag) > 1})
    if duplicates:
        errors.append("Tag重复：" + "、".join(duplicates))
    return core, groups, all_tags, errors


class Catalog:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.novel_dir = self.project_root / "Novel"
        self.outline_dir = self.project_root / "Outline"
        self._items: dict[str, CatalogItem] = {}

    def scan(self) -> list[CatalogItem]:
        self.novel_dir.mkdir(parents=True, exist_ok=True)
        self.outline_dir.mkdir(parents=True, exist_ok=True)
        outlines = [_read_outline(path) for path in self.outline_dir.glob("*.json")]
        by_stem = {item.path.stem: item for item in outlines}
        items: list[CatalogItem] = []

        for novel_path in self.novel_dir.glob("*.txt"):
            item_id = novel_path.stem
            matched_by = "exact"
            metadata = by_stem.get(item_id)
            if metadata is None:
                candidates = [
                    item
                    for item in outlines
                    if item.run_id
                    and (item_id.endswith(item.run_id) or f"_{item.run_id}" in item_id)
                ]
                if len(candidates) == 1:
                    metadata = candidates[0]
                    matched_by = "run_id"
                elif len(candidates) > 1:
                    metadata = None
                    matched_by = "ambiguous"
                else:
                    matched_by = "missing"

            errors: list[str] = []
            try:
                body = novel_path.read_text(encoding="utf-8")
            except Exception as error:
                body = ""
                errors.append(f"正文读取失败：{error}")
            chars = body_char_count(body)
            if chars < MIN_BODY_CHARS:
                errors.append(f"正文仅{chars}字，少于{MIN_BODY_CHARS}字")

            payload: dict[str, Any] = {}
            outline_file = ""
            if metadata is None:
                errors.append("找不到唯一匹配的大纲")
            else:
                outline_file = metadata.path.name
                if metadata.error:
                    errors.append(metadata.error)
                payload = metadata.payload or {}

            title = str(payload.get("title") or "").strip()
            if not title:
                errors.append("大纲缺少title")
            core, groups, all_tags, tag_errors = _normalize_tags(payload.get("novel_tags"))
            errors.extend(tag_errors)
            items.append(
                CatalogItem(
                    item_id=item_id,
                    title=title,
                    novel_file=novel_path.name,
                    outline_file=outline_file,
                    run_id=str(payload.get("run_id") or "").strip(),
                    body_chars=chars,
                    core_tag=core,
                    tag_groups=groups,
                    all_tags=all_tags,
                    valid=not errors,
                    errors=errors,
                    matched_by=matched_by,
                    modified_at=novel_path.stat().st_mtime,
                )
            )

        items.sort(key=lambda item: (-item.modified_at, item.item_id))
        self._items = {item.item_id: item for item in items}
        return items

    def get(self, item_id: str) -> CatalogItem:
        if not self._items:
            self.scan()
        try:
            return self._items[item_id]
        except KeyError as error:
            raise KeyError(f"未知小说：{item_id}") from error

    def body(self, item_id: str) -> str:
        item = self.get(item_id)
        path = (self.novel_dir / item.novel_file).resolve()
        if path.parent != self.novel_dir.resolve():
            raise ValueError("正文路径越界")
        return path.read_text(encoding="utf-8")
