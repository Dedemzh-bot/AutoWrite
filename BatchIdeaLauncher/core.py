from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable


LAUNCHER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = LAUNCHER_ROOT.parent
DEFAULT_RUNS_DIR = LAUNCHER_ROOT / "runs"

DEFAULT_CONFIG = {
    "length": {
        "preferred_chapters": 8,
        "min_chapters": 5,
        "max_chapters": 20,
        "preferred_words_per_chapter": 1500,
        "min_words_per_chapter": 1000,
        "max_words_per_chapter": 2500,
    },
    "job_timeout_seconds": 14400,
    "selector": {
        "model": "",
        "temperature": 0.2,
        "max_retries": 2,
        "request_timeout_seconds": 180,
    },
}

LENGTH_FIELDS = (
    "preferred_chapters",
    "min_chapters",
    "max_chapters",
    "preferred_words_per_chapter",
    "min_words_per_chapter",
    "max_words_per_chapter",
)


class LauncherError(RuntimeError):
    pass


class SelectionError(LauncherError):
    pass


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise LauncherError(f"{field} 必须是整数")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise LauncherError(f"{field} 必须是整数") from error


def validate_length_constraints(length: dict) -> dict:
    normalized = {}
    for field in LENGTH_FIELDS:
        normalized[field] = _as_int(length.get(field), field)
        if normalized[field] <= 0:
            raise LauncherError(f"{field} 必须大于 0")

    if not (
        normalized["min_chapters"]
        <= normalized["preferred_chapters"]
        <= normalized["max_chapters"]
    ):
        raise LauncherError(
            "章节偏好必须满足 min_chapters <= preferred_chapters <= max_chapters"
        )
    if not (
        normalized["min_words_per_chapter"]
        <= normalized["preferred_words_per_chapter"]
        <= normalized["max_words_per_chapter"]
    ):
        raise LauncherError(
            "字数偏好必须满足 min_words_per_chapter <= "
            "preferred_words_per_chapter <= max_words_per_chapter"
        )
    return normalized


def load_batch_config(path: Path) -> dict:
    override = read_json(path) if path else {}
    config = deep_merge(DEFAULT_CONFIG, override)
    config["length"] = validate_length_constraints(config["length"])
    config["job_timeout_seconds"] = _as_int(
        config.get("job_timeout_seconds"), "job_timeout_seconds"
    )
    if config["job_timeout_seconds"] <= 0:
        raise LauncherError("job_timeout_seconds 必须大于 0")

    selector = config["selector"]
    try:
        selector["temperature"] = float(selector.get("temperature", 0.2))
    except (TypeError, ValueError) as error:
        raise LauncherError("selector.temperature 必须是数字") from error
    selector["max_retries"] = _as_int(
        selector.get("max_retries", 2), "selector.max_retries"
    )
    selector["request_timeout_seconds"] = _as_int(
        selector.get("request_timeout_seconds", 180),
        "selector.request_timeout_seconds",
    )
    if selector["max_retries"] < 0:
        raise LauncherError("selector.max_retries 不能小于 0")
    if selector["request_timeout_seconds"] <= 0:
        raise LauncherError("selector.request_timeout_seconds 必须大于 0")
    return config


def safe_job_id(value: str, fallback: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "-", str(value or "").strip())
    safe = safe.strip(".-_")
    return safe[:80] or fallback


def _clean_override_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def normalize_idea_item(raw: Any, index: int) -> dict:
    if isinstance(raw, str):
        raw = {"idea": raw}
    if not isinstance(raw, dict):
        raise LauncherError(f"第 {index} 条点子必须是字符串或对象")
    idea = str(raw.get("idea", "")).strip()
    if not idea:
        raise LauncherError(f"第 {index} 条点子的 idea 不能为空")
    job_id = safe_job_id(raw.get("job_id", ""), f"idea-{index:03d}")
    overrides = {}
    nested_length = raw.get("length", {})
    if nested_length and not isinstance(nested_length, dict):
        raise LauncherError(f"{job_id} 的 length 必须是对象")
    for field in LENGTH_FIELDS:
        value = _clean_override_value(raw.get(field))
        if value is None and isinstance(nested_length, dict):
            value = _clean_override_value(nested_length.get(field))
        if value is not None:
            overrides[field] = value
    return {
        "job_id": job_id,
        "idea": idea,
        "length_overrides": overrides,
    }


def _read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    raise LauncherError(f"无法识别文件编码: {path}")


def load_ideas(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    raw_items: list[Any] = []
    if suffix == ".txt":
        raw_items = [
            line.strip()
            for line in _read_text_with_fallback(path).splitlines()
            if line.strip()
        ]
    elif suffix == ".csv":
        text = _read_text_with_fallback(path)
        reader = csv.DictReader(text.splitlines())
        if not reader.fieldnames or "idea" not in reader.fieldnames:
            raise LauncherError("CSV 必须包含 idea 列")
        raw_items = list(reader)
    elif suffix in {".jsonl", ".ndjson"}:
        for line_number, line in enumerate(
            _read_text_with_fallback(path).splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                raw_items.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise LauncherError(
                    f"JSONL 第 {line_number} 行不是有效 JSON：{error}"
                ) from error
    else:
        raise LauncherError("点子清单仅支持 TXT、CSV 或 JSONL")

    if not raw_items:
        raise LauncherError("点子清单为空")
    items = [
        normalize_idea_item(raw, index)
        for index, raw in enumerate(raw_items, start=1)
    ]
    ids = [item["job_id"] for item in items]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise LauncherError("job_id 重复：" + ", ".join(duplicates))
    return items


def resolve_length_constraints(config: dict, item: dict) -> dict:
    length = dict(config["length"])
    length.update(item.get("length_overrides", {}))
    return validate_length_constraints(length)


def _command_prefix(value: str) -> list[str]:
    value = str(value or "").strip()
    if not value:
        return [sys.executable]
    expanded = Path(os.path.expandvars(value)).expanduser()
    if expanded.exists():
        return [str(expanded.resolve())]
    return shlex.split(value, posix=os.name != "nt")


def _decode_timeout_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class AutoWriteCLI:
    def __init__(
        self,
        entry: Path | None = None,
        python_command: str | None = None,
        environment: dict[str, str] | None = None,
    ):
        entry_from_environment = os.environ.get(
            "AUTOWRITE_ENTRY", ""
        ).strip()
        configured_entry = Path(entry) if entry else Path(
            entry_from_environment or PROJECT_ROOT / "TheGraph.py"
        )
        self.entry = configured_entry.expanduser().resolve()
        if not self.entry.is_file():
            raise LauncherError(f"找不到 AutoWrite CLI：{self.entry}")
        configured_python = python_command or os.environ.get(
            "AUTOWRITE_PYTHON", sys.executable
        )
        self.python_prefix = _command_prefix(configured_python)
        self.environment = dict(os.environ)
        if environment:
            self.environment.update(environment)

    def export_capabilities(self, destination: Path) -> dict:
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            *self.python_prefix,
            str(self.entry),
            "--describe-capabilities",
            str(destination),
        ]
        completed = subprocess.run(
            command,
            cwd=str(self.entry.parent),
            env=self.environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            raise LauncherError(
                "能力表导出失败："
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        if not destination.exists():
            raise LauncherError("AutoWrite CLI 未生成能力表")
        return read_json(destination)

    def validate_job(self, job_dir: Path) -> dict:
        job_path = (job_dir / "job.json").resolve()
        result_path = (job_dir / "preflight.json").resolve()
        stdout_path = job_dir / "preflight.stdout.log"
        stderr_path = job_dir / "preflight.stderr.log"
        command = [
            *self.python_prefix,
            str(self.entry),
            "--validate-job-file",
            str(job_path),
            "--result-file",
            str(result_path),
        ]
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=self.environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        result = (
            read_json(result_path)
            if result_path.exists()
            else {
                "schema_version": 2,
                "status": "failed",
                "error": "AutoWrite CLI did not create preflight.json",
            }
        )
        result["process_returncode"] = completed.returncode
        if completed.returncode != 0 and result.get("status") == "validated":
            result["status"] = "failed"
            result["error"] = (
                f"AutoWrite CLI preflight returned {completed.returncode}"
            )
            atomic_write_json(result_path, result)
        return result

    def run_job(self, job_dir: Path, timeout_seconds: int) -> dict:
        job_path = (job_dir / "job.json").resolve()
        result_path = (job_dir / "result.json").resolve()
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        command = [
            *self.python_prefix,
            str(self.entry),
            "--job-file",
            str(job_path),
            "--result-file",
            str(result_path),
            "--auto-approve",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                env=self.environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
            stdout_path.write_text(completed.stdout or "", encoding="utf-8")
            stderr_path.write_text(completed.stderr or "", encoding="utf-8")
            result = (
                read_json(result_path)
                if result_path.exists()
                else {
                    "schema_version": 1,
                    "status": "failed",
                    "error": "AutoWrite CLI 未生成 result.json",
                }
            )
            result["process_returncode"] = completed.returncode
            if completed.returncode != 0 and result.get("status") == "succeeded":
                result["status"] = "failed"
                result["error"] = (
                    f"AutoWrite CLI 返回非零退出码 {completed.returncode}"
                )
                atomic_write_json(result_path, result)
            return result
        except subprocess.TimeoutExpired as error:
            stdout_path.write_text(
                _decode_timeout_output(error.stdout), encoding="utf-8"
            )
            stderr_path.write_text(
                _decode_timeout_output(error.stderr), encoding="utf-8"
            )
            result = {
                "schema_version": 1,
                "status": "failed",
                "error_type": "TimeoutExpired",
                "error": f"单篇任务超过 {timeout_seconds} 秒",
                "finished_at": now_iso(),
            }
            atomic_write_json(result_path, result)
            return result


def _extract_json_object(text: str) -> dict:
    content = str(text or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.I)
        content = re.sub(r"\s*```$", "", content)
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    if start >= 0:
        decoder = json.JSONDecoder()
        try:
            parsed, _ = decoder.raw_decode(content[start:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise SelectionError("选型 Agent 没有返回有效 JSON 对象")


def _compact_capabilities(capabilities: dict) -> dict:
    return {
        "writer_styles": capabilities.get("writer_styles", []),
        "story_patterns": capabilities.get("story_patterns", []),
        "material_library": capabilities.get("material_library", {}),
    }


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(
        str(item).strip() for item in value if str(item).strip()
    ))


def _material_constraints(capabilities: dict) -> dict:
    metadata = capabilities.get("material_library", {})
    groups = metadata.get("groups", {})
    legacy_limits = {
        "world_stage": 1,
        "protagonist": 1,
        "supporting_role": 2,
        "cheat_device": 2,
        "plot_event": 2,
        "core_conflict": 2,
        "career_resource": 2,
        "atmosphere": 2,
    }
    limits = dict(metadata.get("group_limits") or {})
    if not limits:
        limits = {
            group_id: int(group.get("max_count", 0))
            for group_id, group in groups.items()
            if int(group.get("max_count", 0)) > 0
        }
    if not limits:
        limits = legacy_limits
    defaults = dict(metadata.get("default_group_counts") or {})
    for group_id in limits:
        defaults.setdefault(group_id, 0)
    count_range = metadata.get("count_range", [2, 8])
    if not isinstance(count_range, list) or len(count_range) != 2:
        count_range = [2, 8]
    return {
        "schema_version": int(metadata.get("schema_version", 2)),
        "limits": {key: int(value) for key, value in limits.items()},
        "defaults": {key: int(value) for key, value in defaults.items()},
        "min_count": int(count_range[0]),
        "max_count": int(count_range[1]),
    }


def _pattern_constraints(capabilities: dict) -> dict:
    metadata = capabilities.get("pattern_library", {})
    return {
        "schema_version": int(metadata.get("schema_version", 2)),
        "max_secondary": int(metadata.get("max_secondary", 2)),
    }


class OpenAISelector:
    def __init__(self, selector_config: dict):
        self.api_key = os.environ.get("LAUNCHER_API_KEY", "").strip()
        self.base_url = os.environ.get(
            "LAUNCHER_BASE_URL", "https://api.openai.com/v1"
        ).strip()
        self.model = (
            str(selector_config.get("model", "")).strip()
            or os.environ.get("LAUNCHER_MODEL", "").strip()
        )
        self.temperature = float(selector_config.get("temperature", 0.2))
        self.max_retries = int(selector_config.get("max_retries", 2))
        self.timeout = int(
            selector_config.get("request_timeout_seconds", 180)
        )
        if not self.api_key:
            raise LauncherError("缺少 LAUNCHER_API_KEY")
        if not self.base_url:
            raise LauncherError("缺少 LAUNCHER_BASE_URL")
        if not self.model:
            raise LauncherError(
                "缺少 LAUNCHER_MODEL，或 batch_config.json 中的 selector.model"
            )

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    def _request(self, messages: list[dict]) -> str:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                content = payload["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        str(item.get("text", ""))
                        for item in content
                        if isinstance(item, dict)
                    )
                return str(content)
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                KeyError,
                IndexError,
                json.JSONDecodeError,
            ) as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 4))
        raise SelectionError(f"选型模型请求失败：{last_error}")

    def choose(
        self,
        idea: str,
        constraints: dict,
        capabilities: dict,
        repair: dict | None = None,
    ) -> dict:
        material_rules = _material_constraints(capabilities)
        pattern_rules = _pattern_constraints(capabilities)
        group_schema = {
            group_id: f"0 to {limit}"
            for group_id, limit in material_rules["limits"].items()
        }
        system = (
            "你是小说生产任务的配置 Agent。你只能从提供的能力表中选择，"
            "并且必须遵守主辅套路、兼容写手、结局和素材冲突规则。"
            "章节数和每章字数必须落在允许范围内，偏好值只是参考。"
            f"选择1个主套路、最多{pattern_rules['max_secondary']}个非强辅助套路。"
            "素材筛选可选择能力表中的大类、子类与标签；每类数量服从"
            "material_library.group_limits。"
            f"素材总数必须为{material_rules['min_count']}到"
            f"{material_rules['max_count']}。"
            "如果主套路选择 custom，必须给出可执行的 custom_instruction；"
            "强套路必须在 manifest.ending 中选择 ending_options 的一个键。"
            "只输出一个 JSON 对象，不要 Markdown。"
        )
        user_payload = {
            "idea": idea,
            "length_constraints": constraints,
            "capabilities": _compact_capabilities(capabilities),
            "output_schema": {
                "target_chapters": "整数",
                "words_per_chapter": "整数",
                "writer_style": "写手风格键",
                "material_config": {
                    "schema_version": material_rules["schema_version"],
                    "filters": {
                        "categories": ["素材大类键"],
                        "subcategories": ["素材子类键"],
                        "tags": [],
                    },
                    "group_counts": group_schema,
                    "items": [],
                    "locked_item_keys": [],
                    "auto_selected_subcategories": [],
                },
                "pattern_config": {
                    "schema_version": pattern_rules["schema_version"],
                    "primary": "主套路ID",
                    "secondary": ["常规套路ID，数量服从 max_secondary"],
                    "custom_instruction": "仅custom时填写",
                    "manifest": {"ending": "仅强套路填写结局键"},
                    "structure_plan": {},
                },
                "rationale": "一句简短判断依据",
            },
        }
        if repair:
            user_payload["repair"] = repair
        content = self._request([
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    user_payload, ensure_ascii=False, separators=(",", ":")
                ),
            },
        ])
        return _extract_json_object(content)


def validate_selection(
    selection: dict,
    constraints: dict,
    capabilities: dict,
) -> tuple[dict, list[str]]:
    issues = []
    if not isinstance(selection, dict):
        return {}, ["选型结果必须是 JSON 对象"]

    normalized = {
        "target_chapters": 0,
        "words_per_chapter": 0,
        "writer_style": str(selection.get("writer_style", "")).strip(),
        "material_config": selection.get("material_config", {}),
        "pattern_config": selection.get("pattern_config", {}),
        "rationale": str(selection.get("rationale", "")).strip(),
    }
    try:
        normalized["target_chapters"] = _as_int(
            selection.get("target_chapters"), "target_chapters"
        )
    except LauncherError as error:
        issues.append(str(error))
    try:
        normalized["words_per_chapter"] = _as_int(
            selection.get("words_per_chapter"), "words_per_chapter"
        )
    except LauncherError as error:
        issues.append(str(error))

    if not (
        constraints["min_chapters"]
        <= normalized["target_chapters"]
        <= constraints["max_chapters"]
    ):
        issues.append(
            f"target_chapters 必须在 {constraints['min_chapters']}"
            f"-{constraints['max_chapters']} 之间"
        )
    if not (
        constraints["min_words_per_chapter"]
        <= normalized["words_per_chapter"]
        <= constraints["max_words_per_chapter"]
    ):
        issues.append(
            f"words_per_chapter 必须在 "
            f"{constraints['min_words_per_chapter']}"
            f"-{constraints['max_words_per_chapter']} 之间"
        )

    style_keys = {
        item.get("key") for item in capabilities.get("writer_styles", [])
    }
    if normalized["writer_style"] not in style_keys:
        issues.append(f"未知写手风格：{normalized['writer_style']}")

    pattern_map = {
        item.get("id") or item.get("key"): item
        for item in capabilities.get("story_patterns", [])
    }
    pattern_rules = _pattern_constraints(capabilities)
    raw_pattern = (
        normalized["pattern_config"]
        if isinstance(normalized["pattern_config"], dict)
        else {}
    )
    primary_id = str(raw_pattern.get("primary") or "none").strip()
    secondary = raw_pattern.get("secondary", [])
    if not isinstance(secondary, list):
        issues.append("pattern_config.secondary 必须是数组")
        secondary = []
    secondary = list(dict.fromkeys(
        str(item).strip() for item in secondary if str(item).strip()
    ))
    custom_instruction = str(
        raw_pattern.get("custom_instruction") or ""
    ).strip()
    manifest = (
        raw_pattern.get("manifest")
        if isinstance(raw_pattern.get("manifest"), dict)
        else {}
    )
    normalized["pattern_config"] = {
        "schema_version": pattern_rules["schema_version"],
        "primary": primary_id,
        "secondary": secondary,
        "custom_instruction": custom_instruction,
        "manifest": manifest,
        "structure_plan": {},
    }
    primary = pattern_map.get(primary_id)
    if not primary:
        issues.append(f"未知主套路：{primary_id}")
    if len(secondary) > pattern_rules["max_secondary"]:
        issues.append(
            f"辅助套路最多选择{pattern_rules['max_secondary']}个"
        )
    for secondary_id in secondary:
        item = pattern_map.get(secondary_id)
        if not item:
            issues.append(f"未知辅助套路：{secondary_id}")
            continue
        if item.get("strong"):
            issues.append(f"辅助套路不能选择强套路：{secondary_id}")
        if secondary_id in {"none", "custom", primary_id}:
            issues.append(f"无效辅助套路：{secondary_id}")
        if primary and (
            secondary_id in primary.get("hard_conflicts", [])
            or primary_id in item.get("hard_conflicts", [])
        ):
            issues.append(f"主辅套路硬冲突：{primary_id} / {secondary_id}")
        if primary:
            primary_tags = set(primary.get("tags", []))
            secondary_tags = set(item.get("tags", []))
            if primary_tags.intersection(
                item.get("forbidden_pattern_tags", [])
            ) or secondary_tags.intersection(
                primary.get("forbidden_pattern_tags", [])
            ):
                issues.append(
                    f"主辅套路标签冲突：{primary_id} / {secondary_id}"
                )
    if primary_id == "custom" and not custom_instruction:
        issues.append("custom 主套路必须提供 custom_instruction")
    if primary:
        compatible = primary.get("compatible_styles", [])
        if compatible and normalized["writer_style"] not in compatible:
            issues.append(
                f"强套路 {primary_id} 不兼容写手 "
                f"{normalized['writer_style']}"
            )
        endings = primary.get("ending_options", {})
        if endings and manifest.get("ending") not in endings:
            issues.append(f"强套路结局必须是：{', '.join(endings)}")

    raw_material = (
        normalized["material_config"]
        if isinstance(normalized["material_config"], dict)
        else {}
    )
    filters = (
        raw_material.get("filters")
        if isinstance(raw_material.get("filters"), dict)
        else {}
    )
    categories = _as_string_list(filters.get("categories"))
    subcategories = _as_string_list(filters.get("subcategories"))
    tags = _as_string_list(filters.get("tags"))
    material_rules = _material_constraints(capabilities)
    limits = material_rules["limits"]
    defaults = material_rules["defaults"]
    raw_counts = (
        raw_material.get("group_counts")
        if isinstance(raw_material.get("group_counts"), dict)
        else {}
    )
    group_counts = {}
    for group_id, limit in limits.items():
        try:
            value = _as_int(
                raw_counts.get(group_id, defaults[group_id]),
                f"material_config.group_counts.{group_id}",
            )
        except LauncherError as error:
            issues.append(str(error))
            value = defaults[group_id]
        if not 0 <= value <= limit:
            issues.append(f"素材大类 {group_id} 数量必须在0到{limit}之间")
        group_counts[group_id] = value
    material_count = sum(group_counts.values())
    if not (
        material_rules["min_count"]
        <= material_count
        <= material_rules["max_count"]
    ):
        issues.append("素材总数不符合本体能力表范围")

    material_meta = capabilities.get("material_library", {})
    group_map = material_meta.get("groups", {})
    known_subcategories = {
        child.get("id"): (group_id, child)
        for group_id, group in group_map.items()
        for child in group.get("subcategories", [])
    }
    for category in categories:
        if category not in group_map:
            issues.append(f"未知素材大类：{category}")
    for subcategory in subcategories:
        if subcategory not in known_subcategories:
            issues.append(f"未知素材子类：{subcategory}")

    active_patterns = [
        pattern_map.get(item, {})
        for item in [primary_id, *secondary]
    ]
    for category in categories:
        for pattern in active_patterns:
            if category in pattern.get("forbidden_material_categories", []):
                issues.append(
                    f"素材大类 {category} 与套路 {pattern.get('id')} 冲突"
                )
    for subcategory in subcategories:
        if subcategory not in known_subcategories:
            continue
        group_id, child = known_subcategories[subcategory]
        child_tags = set(child.get("tags", []))
        for pattern in active_patterns:
            if group_id in pattern.get("forbidden_material_categories", []):
                issues.append(
                    f"素材子类 {subcategory} 与套路 {pattern.get('id')} 冲突"
                )
            if child_tags.intersection(
                pattern.get("forbidden_material_tags", [])
            ):
                issues.append(
                    f"素材子类 {subcategory} 的标签与套路 "
                    f"{pattern.get('id')} 冲突"
                )

    normalized["material_config"] = {
        "schema_version": material_rules["schema_version"],
        "filters": {
            "categories": categories,
            "subcategories": subcategories,
            "tags": tags,
        },
        "group_counts": group_counts,
        "count": material_count,
        "items": [],
        "locked_item_keys": [],
        "auto_selected_subcategories": [],
    }
    return normalized, list(dict.fromkeys(issues))


def stable_seed(batch_id: str, job_id: str, namespace: str) -> int:
    digest = hashlib.sha256(
        f"{batch_id}:{job_id}:{namespace}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") or 1


def make_job_payload(
    batch_id: str,
    item: dict,
    selection: dict,
) -> dict:
    return {
        "schema_version": 2,
        "job_id": item["job_id"],
        "idea": item["idea"],
        "target_chapters": selection["target_chapters"],
        "words_per_chapter": selection["words_per_chapter"],
        "writer_style": selection["writer_style"],
        "material_config": selection["material_config"],
        "pattern_config": selection["pattern_config"],
        "pattern_seed": stable_seed(batch_id, item["job_id"], "pattern"),
        "material_seed": stable_seed(batch_id, item["job_id"], "material"),
    }


def create_batch_id() -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"batch-{timestamp}"


def _empty_job_record(item: dict, constraints: dict) -> dict:
    return {
        **item,
        "constraints": constraints,
        "status": "pending",
        "attempts": 0,
        "error": "",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "selection": {},
        "outputs": {},
    }


def create_batch_manifest(
    batch_id: str,
    source_path: Path,
    config: dict,
    items: list[dict],
) -> dict:
    return {
        "schema_version": 2,
        "batch_id": batch_id,
        "source_ideas": str(source_path.resolve()),
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "config": config,
        "jobs": [
            _empty_job_record(item, resolve_length_constraints(config, item))
            for item in items
        ],
    }


def _status_counts(jobs: list[dict]) -> dict:
    counts = {
        "pending": 0,
        "selecting": 0,
        "running": 0,
        "succeeded": 0,
        "failed": 0,
    }
    for job in jobs:
        status = job.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_batch_reports(batch_dir: Path, manifest: dict) -> dict:
    counts = _status_counts(manifest["jobs"])
    summary = {
        "schema_version": 2,
        "batch_id": manifest["batch_id"],
        "updated_at": now_iso(),
        "counts": counts,
        "jobs": [
            {
                "job_id": job["job_id"],
                "status": job["status"],
                "attempts": job.get("attempts", 0),
                "target_chapters": job.get("selection", {}).get(
                    "target_chapters", ""
                ),
                "words_per_chapter": job.get("selection", {}).get(
                    "words_per_chapter", ""
                ),
                "writer_style": job.get("selection", {}).get(
                    "writer_style", ""
                ),
                "primary_pattern": job.get("selection", {}).get(
                    "pattern_config", {}
                ).get("primary", ""),
                "novel_file": job.get("outputs", {}).get("novel_file", ""),
                "outline_file": job.get("outputs", {}).get(
                    "outline_file", ""
                ),
                "error": job.get("error", ""),
            }
            for job in manifest["jobs"]
        ],
    }
    atomic_write_json(batch_dir / "summary.json", summary)
    with (batch_dir / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        fields = [
            "job_id",
            "status",
            "attempts",
            "target_chapters",
            "words_per_chapter",
            "writer_style",
            "primary_pattern",
            "novel_file",
            "outline_file",
            "error",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["jobs"])
    return summary


SelectorCallable = Callable[[str, dict, dict, dict | None], dict]


class BatchRunner:
    def __init__(
        self,
        batch_dir: Path,
        autowrite: AutoWriteCLI,
        selector: OpenAISelector | None = None,
    ):
        self.batch_dir = batch_dir.resolve()
        self.manifest_path = self.batch_dir / "batch.json"
        self.autowrite = autowrite
        self.selector = selector
        if not self.manifest_path.exists():
            raise LauncherError(f"找不到批次：{self.manifest_path}")
        self.manifest = read_json(self.manifest_path)

    def save(self) -> None:
        self.manifest["updated_at"] = now_iso()
        atomic_write_json(self.manifest_path, self.manifest)
        write_batch_reports(self.batch_dir, self.manifest)

    def refresh_capabilities(self) -> dict:
        capabilities_path = self.batch_dir / "capabilities.json"
        capabilities = self.autowrite.export_capabilities(capabilities_path)
        if capabilities.get("schema_version") != 2:
            raise LauncherError("BatchIdeaLauncher requires AutoWrite schema v2")
        if not capabilities.get("cli_contract", {}).get(
            "supports_job_preflight"
        ):
            raise LauncherError(
                "AutoWrite CLI does not support --validate-job-file"
            )
        self.capabilities = capabilities
        return capabilities

    def _set_status(
        self,
        record: dict,
        status: str,
        error: str = "",
    ) -> None:
        record["status"] = status
        record["error"] = error
        record["updated_at"] = now_iso()
        status_payload = {
            "job_id": record["job_id"],
            "status": status,
            "attempts": record.get("attempts", 0),
            "error": error,
            "updated_at": record["updated_at"],
        }
        job_dir = self.batch_dir / record["job_id"]
        atomic_write_json(job_dir / "status.json", status_payload)
        self.save()

    def _selector_instance(self) -> OpenAISelector:
        if self.selector is None:
            self.selector = OpenAISelector(self.manifest["config"]["selector"])
        return self.selector

    def _select(self, record: dict) -> dict:
        selector = self._selector_instance()
        raw = selector.choose(
            record["idea"],
            record["constraints"],
            self.capabilities,
        )
        normalized, issues = validate_selection(
            raw, record["constraints"], self.capabilities
        )
        if issues:
            repaired = selector.choose(
                record["idea"],
                record["constraints"],
                self.capabilities,
                repair={
                    "invalid_output": raw,
                    "validation_errors": issues,
                    "instruction": "修复以上错误并重新输出完整 JSON。",
                },
            )
            normalized, repaired_issues = validate_selection(
                repaired, record["constraints"], self.capabilities
            )
            if repaired_issues:
                raise SelectionError("；".join(repaired_issues))
        normalized["selected_at"] = now_iso()
        return normalized

    def _reuse_selection(self, record: dict, job_dir: Path) -> dict | None:
        selection_path = job_dir / "selection.json"
        if not selection_path.exists():
            return None
        raw = read_json(selection_path)
        normalized, issues = validate_selection(
            raw, record["constraints"], self.capabilities
        )
        if issues:
            return None
        normalized["selected_at"] = raw.get("selected_at", now_iso())
        return normalized

    def process(
        self,
        statuses: set[str] | None = None,
        reuse_selection: bool = False,
    ) -> dict:
        self.refresh_capabilities()
        statuses = statuses or {"pending", "failed", "selecting", "running"}
        total = len(self.manifest["jobs"])
        for idx, record in enumerate(self.manifest["jobs"], start=1):
            if record.get("status") == "succeeded":
                continue
            if record.get("status", "pending") not in statuses:
                continue
            job_id = record["job_id"]
            idea = record.get("idea", "")
            idea_summary = idea[:60] + "..." if len(idea) > 60 else idea
            print(f"\n{'='*60}")
            print(f"[{now_iso()}] [{idx}/{total}] 正在处理: {job_id}")
            print(f"  灵感: {idea_summary}")
            print(f"{'='*60}")
            job_dir = self.batch_dir / record["job_id"]
            job_dir.mkdir(parents=True, exist_ok=True)
            for name in ("stdout.log", "stderr.log"):
                path = job_dir / name
                if not path.exists():
                    path.write_text("", encoding="utf-8")
            try:
                selection = (
                    self._reuse_selection(record, job_dir)
                    if reuse_selection
                    else None
                )
                if selection is None:
                    print(f"  [{now_iso()}] 🔍 正在AI选型配置...")
                    self._set_status(record, "selecting")
                    selection = self._select(record)
                    print(f"  [{now_iso()}] ✅ 选型完成 — 风格: {selection.get('writer_style','?')}  套路: {selection.get('pattern_config',{}).get('primary','?')}  {selection.get('target_chapters','?')}章×{selection.get('words_per_chapter','?')}字")
                    atomic_write_json(job_dir / "selection.json", selection)
                record["selection"] = selection
                job_payload = make_job_payload(
                    self.manifest["batch_id"],
                    record,
                    selection,
                )
                atomic_write_json(job_dir / "job.json", job_payload)
                print(f"  [{now_iso()}] 🔬 预检中...")
                preflight = self.autowrite.validate_job(job_dir)
                if preflight.get("status") != "validated":
                    print(f"  [{now_iso()}] ⚠️ 预检失败，重新选型...")
                    selector = self._selector_instance()
                    repaired = selector.choose(
                        record["idea"],
                        record["constraints"],
                        self.capabilities,
                        repair={
                            "invalid_output": selection,
                            "validation_errors": [
                                preflight.get(
                                    "error", "AutoWrite CLI 预检失败"
                                )
                            ],
                            "instruction": (
                                "本体使用最新内容库校验并试抽素材失败；"
                                "请重新选择完整配置。"
                            ),
                        },
                    )
                    selection, repair_issues = validate_selection(
                        repaired, record["constraints"], self.capabilities
                    )
                    if repair_issues:
                        raise SelectionError("；".join(repair_issues))
                    selection["selected_at"] = now_iso()
                    record["selection"] = selection
                    atomic_write_json(job_dir / "selection.json", selection)
                    job_payload = make_job_payload(
                        self.manifest["batch_id"], record, selection
                    )
                    atomic_write_json(job_dir / "job.json", job_payload)
                    preflight = self.autowrite.validate_job(job_dir)
                if preflight.get("status") != "validated":
                    raise SelectionError(
                        preflight.get("error", "AutoWrite CLI preflight failed")
                    )
                print(f"  [{now_iso()}] ✅ 预检通过")
                record["attempts"] = int(record.get("attempts", 0)) + 1
                self._set_status(record, "running")
                target_ch = selection.get("target_chapters", "?")
                target_w = selection.get("words_per_chapter", "?")
                print(f"  [{now_iso()}] ▶ 开始写作... ({target_ch}章×{target_w}字，请耐心等待)")
                result = self.autowrite.run_job(
                    job_dir,
                    self.manifest["config"]["job_timeout_seconds"],
                )
                if result.get("status") != "succeeded":
                    raise LauncherError(
                        result.get("error", "AutoWrite CLI 运行失败")
                    )
                record["outputs"] = {
                    "novel_file": result.get("novel_file", ""),
                    "outline_file": result.get("outline_file", ""),
                    "run_id": result.get("run_id", ""),
                    "material_config": result.get("material_config", {}),
                    "pattern_config": result.get("pattern_config", {}),
                }
                self._set_status(record, "succeeded")
                print(f"  [{now_iso()}] 🎉 完成！{record['outputs'].get('saved_chapter', '?')}章已生成")
            except Exception as error:
                print(f"  [{now_iso()}] ❌ 失败: {error}")
                result_path = job_dir / "result.json"
                if (
                    record.get("status") != "running"
                    or not result_path.exists()
                ):
                    atomic_write_json(result_path, {
                        "schema_version": 2,
                        "status": "failed",
                        "stage": record.get("status", "unknown"),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "finished_at": now_iso(),
                    })
                self._set_status(record, "failed", str(error))
        return write_batch_reports(self.batch_dir, self.manifest)


def initialize_batch(
    source_path: Path,
    config: dict,
    autowrite: AutoWriteCLI,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    batch_id: str | None = None,
) -> Path:
    items = load_ideas(source_path)
    batch_id = safe_job_id(batch_id or create_batch_id(), create_batch_id())
    batch_dir = (runs_dir / batch_id).resolve()
    if batch_dir.exists():
        raise LauncherError(f"批次目录已存在：{batch_dir}")
    batch_dir.mkdir(parents=True)
    manifest = create_batch_manifest(batch_id, source_path, config, items)
    atomic_write_json(batch_dir / "batch.json", manifest)
    for record in manifest["jobs"]:
        job_dir = batch_dir / record["job_id"]
        job_dir.mkdir()
        (job_dir / "stdout.log").write_text("", encoding="utf-8")
        (job_dir / "stderr.log").write_text("", encoding="utf-8")
        atomic_write_json(job_dir / "status.json", {
            "job_id": record["job_id"],
            "status": "pending",
            "attempts": 0,
            "error": "",
            "updated_at": now_iso(),
        })
    autowrite.export_capabilities(batch_dir / "capabilities.json")
    write_batch_reports(batch_dir, manifest)
    return batch_dir


def find_batch(batch_id: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    batch_dir = (runs_dir / safe_job_id(batch_id, batch_id)).resolve()
    if not (batch_dir / "batch.json").exists():
        raise LauncherError(f"找不到批次：{batch_id}")
    return batch_dir


def format_status(manifest: dict) -> str:
    lines = [
        f"批次：{manifest['batch_id']}",
        f"更新时间：{manifest.get('updated_at', '-')}",
        "",
        f"{'JOB ID':<24} {'状态':<12} {'次数':<6} 错误",
        "-" * 90,
    ]
    for job in manifest["jobs"]:
        error = str(job.get("error", "")).replace("\n", " ")[:45]
        lines.append(
            f"{job['job_id']:<24} {job.get('status', 'pending'):<12} "
            f"{job.get('attempts', 0):<6} {error}"
        )
    counts = _status_counts(manifest["jobs"])
    lines.extend([
        "",
        "汇总：" + "，".join(f"{key}={value}" for key, value in counts.items()),
    ])
    return "\n".join(lines)
