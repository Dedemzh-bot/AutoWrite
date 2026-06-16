import datetime
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from State import NovelState 

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("AutoWrite")

if not os.getenv("OPENAI_API_KEY"):
    logger.error("❌ 环境变量 OPENAI_API_KEY 未设置！请在 .env 文件中配置你的 API Key。")
    sys.exit(1)

def _build_output_path(title: str) -> str:
    safe = "".join(c for c in title if c not in r'\/:*?"<>|')
    safe = safe.strip() or "小说输出"
    os.makedirs("Novel", exist_ok=True)
    return os.path.join("Novel", f"{safe}.txt")

def list_outline_files() -> list[dict]:
    files = []
    os.makedirs("Outline", exist_ok=True)
    for name in sorted(os.listdir("Outline"), reverse=True):
        if name.endswith(".json"):
            try:
                with open(os.path.join("Outline", name), "r", encoding="utf-8") as f:
                    data = json.load(f)
                files.append({
                    "file": name,
                    "title": data.get("title", name),
                    "chapters": len(data.get("chapter_outlines", {})),
                    "created_at": data.get("created_at", "")
                })
            except Exception:
                pass
    return files

def load_outline_json(file_name: str) -> dict:
    path = os.path.join("Outline", file_name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

WASH_TITLE_SYSTEM = """你是一位资深编辑，为洗文（同人大纲二次创作）生成新书名。
规则：
1. 新书名必须与原标题明显不同，不能只差一两个字
2. 新书名需体现选定的写手风格
3. 字数8-20字，简洁有力有网感
4. 直接输出新书名，不要任何前缀后缀"""

def generate_wash_title(original_title: str, style: str) -> str:
    style_names = {"hot_blood": "热血爽文", "literary": "文艺细腻", "cold": "冷峻纪实", "humor": "轻松搞笑", "18xx": "18XX", "default": "默认风格"}
    style_cn = style_names.get(style, "默认风格")
    prompt = ChatPromptTemplate.from_messages([
        ("system", WASH_TITLE_SYSTEM),
        ("user", f"原书名《{original_title}》，写手风格【{style_cn}】。请生成洗文新书名。")
    ])
    result = invoke_with_retry(prompt | llm_architect, {}, "洗文书名")
    title = result.content.strip() if result and result.content else f"{original_title}·重制版"
    for ch in r'\/:*?"<>|':
        title = title.replace(ch, "")
    return title.strip() or f"{original_title}·重制版"

DEFAULT_CHAPTERS = int(os.getenv("DEFAULT_CHAPTERS", "10"))
DEFAULT_WORDS_PER_CHAPTER = int(os.getenv("DEFAULT_WORDS_PER_CHAPTER", "1500"))
MIN_OUTLINE_CHARS = 200
MIN_CHAPTER_RATIO = 0.85
STYLE_PASS_SCORE = 7
MAX_REVIEW_ATTEMPTS = 2
MODEL_TIMEOUT_SECONDS = float(os.getenv("MODEL_TIMEOUT_SECONDS", "180"))
MODEL_MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "5"))
APP_INVOKE_ATTEMPTS = int(os.getenv("APP_INVOKE_ATTEMPTS", "3"))
STRONG_PATTERN_KEY = "female_angst_awakening"

CHAPTER_FORMAT_PROMPT = """正文输出必须严格使用以下唯一格式：
第X章 章节名字

正文内容

其中 X 使用当前章节数字，章节名字最多十个字。
正文（不含章节标题）应控制在 {min_words_per_chapter}-{words_per_chapter} 字，建议写到约 {preferred_words_per_chapter} 字，绝不能超过上限。
章节标题必须独占第一行。不要使用 Markdown 标题、括号标题、卷名、序号标题、等号分割线或其他章节格式。
除这一行章节标题和正文外，不要输出任何说明。"""

ARCHITECT_JSON_PROMPT = """必须仅输出一个有效 JSON 对象，不要输出 Markdown 或说明文字。
字段必须完整：novel_title 为字符串；world_bible 为字符串；chapter_outlines 为对象，键是纯数字章节号、值是章节细纲；estimated_words 为整数。
chapter_outlines 中每一章细纲去除空白后必须不少于 200 字。每章细纲必须明确写出本章开场状态、核心冲突、关键行动、人物关系变化、重要信息或伏笔、结尾结果与下一章钩子，禁止用空话凑字数。"""

AUDITOR_JSON_PROMPT = """必须仅输出一个有效 JSON 对象，不要输出 Markdown 或说明文字。
字段必须完整：审核状态为“通过”或“不通过”；发现的问题为逻辑硬伤字符串数组；警告为软性问题字符串数组；套路执行状态为“通过”或“不通过”；套路问题为未完成的强制套路任务字符串数组；修改建议为字符串。
发现的问题或套路问题任一非空时输出“不通过”；只有警告时必须输出“通过”。"""

EDITOR_JSON_PROMPT = """必须仅输出一个有效 JSON 对象，不要输出 Markdown 或说明文字。
字段必须完整：文风评分为 1 到 10 的整数；AI痕迹问题为字符串数组；改进建议为字符串。"""

_CHAPTER_HEADING_RE = re.compile(
    r"^\s*[【\[\(（《]?\s*第\s*[0-9一二三四五六七八九十百千万零〇两]+\s*章"
    r"(?:\s*[:：\-—]\s*|\s+)?(.*?)\s*[】\]\)）》]?\s*$"
)
_LABELED_TITLE_RE = re.compile(r"^\s*(?:章节标题|章节名|标题)\s*[:：]\s*(.+?)\s*$")
_MARKDOWN_TITLE_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
_SEPARATOR_RE = re.compile(r"^\s*[=\-_*~—]{3,}\s*$")


def _clean_chapter_title(title: str) -> str:
    title = re.sub(r"^[\s#=【\[\(（《]+|[\s#=】\]\)）》]+$", "", title)
    title = re.sub(r"^(?:第\s*[0-9一二三四五六七八九十百千万零〇两]+\s*章)\s*", "", title)
    title = re.sub(r"^(?:章节标题|章节名|标题)\s*[:：]\s*", "", title)
    title = " ".join(title.split()).strip("：:，,。.!！?？-— ")
    return title[:10] or "正文"


def _truncate_body(body: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""

    non_whitespace_count = 0
    cutoff = len(body)
    for index, char in enumerate(body):
        if not char.isspace():
            non_whitespace_count += 1
        if non_whitespace_count > max_chars:
            cutoff = index
            break

    if cutoff == len(body):
        return body.rstrip()

    candidate = body[:cutoff].rstrip()
    sentence_end = max(candidate.rfind(mark) for mark in "。！？.!?")
    if sentence_end >= 0:
        complete_sentence = candidate[: sentence_end + 1].rstrip()
        complete_chars = len(re.sub(r"\s+", "", complete_sentence))
        if complete_chars >= int(max_chars * MIN_CHAPTER_RATIO):
            return complete_sentence
    return candidate


def normalize_chapter_output(
    content: str, chapter_num: int, max_body_chars: int | None = None
) -> str:
    """Enforce one plain-text chapter heading and remove decorative separators."""
    title = ""
    body_lines = []

    for line in content.strip().splitlines():
        stripped = line.strip()
        if _SEPARATOR_RE.fullmatch(stripped):
            continue

        heading_match = _CHAPTER_HEADING_RE.fullmatch(stripped)
        labeled_match = _LABELED_TITLE_RE.fullmatch(stripped)
        markdown_match = _MARKDOWN_TITLE_RE.fullmatch(stripped)
        title_match = heading_match or labeled_match or markdown_match
        if title_match:
            if not title:
                title = _clean_chapter_title(title_match.group(1))
            continue

        body_lines.append(line.rstrip())

    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    title = title or "正文"
    body = "\n".join(body_lines)
    if max_body_chars is not None:
        body = _truncate_body(body, max_body_chars)
    return f"第{chapter_num}章 {title}\n\n{body}".rstrip()


def chapter_body_char_count(content: str) -> int:
    parts = content.split("\n", 1)
    body = parts[1] if len(parts) > 1 else ""
    return len(re.sub(r"\s+", "", body))


def should_retry_short_draft(state: NovelState) -> bool:
    words_per = state.get("words_per_chapter", DEFAULT_WORDS_PER_CHAPTER)
    min_words = int(words_per * MIN_CHAPTER_RATIO)
    return (
        chapter_body_char_count(state.get("current_draft", "")) < min_words
        and state.get("iteration_count", 1) < MAX_REVIEW_ATTEMPTS
    )


def chapter_quality_warnings(state: NovelState) -> list[str]:
    warnings = []
    current_chapter = state.get("current_chapter", 1)
    words_per = state.get("words_per_chapter", DEFAULT_WORDS_PER_CHAPTER)
    min_words = int(words_per * MIN_CHAPTER_RATIO)
    body_chars = chapter_body_char_count(state.get("current_draft", ""))
    audit = state.get("audit_report", {})
    editor = state.get("editor_report", {})

    if body_chars < min_words:
        warnings.append(f"第{current_chapter}章正文仅{body_chars}字，低于建议下限{min_words}字")
    if audit.get("发现的问题"):
        warnings.append(f"第{current_chapter}章达到审核上限后逻辑审计仍未通过")
    for warning in audit.get("警告", []):
        warnings.append(f"第{current_chapter}章逻辑警告：{warning}")
    for issue in audit.get("套路问题", []):
        warnings.append(f"第{current_chapter}章达到审核上限后套路任务仍未完成：{issue}")
    if editor.get("文风评分", STYLE_PASS_SCORE) < STYLE_PASS_SCORE:
        warnings.append(
            f"第{current_chapter}章达到审核上限后文风评分仍为"
            f"{editor.get('文风评分')}/10"
        )
    return warnings


def normalize_chapter_outlines(outlines: dict, target_chapters: int) -> dict[str, str]:
    target_chapters = max(1, int(target_chapters))
    outlines = outlines if isinstance(outlines, dict) else {}
    numeric_outlines = {
        str(int(key)): value
        for key, value in outlines.items()
        if str(key).isdigit() and int(key) > 0
    }
    normalized = {}
    for chapter_num in range(1, target_chapters + 1):
        key = str(chapter_num)
        fallback = (
            "收束主线冲突与人物命运，完成本书结局。"
            if chapter_num == target_chapters
            else f"承接上一章继续推进主线，并为第{chapter_num + 1}章制造明确悬念。"
        )
        normalized[key] = numeric_outlines.get(
            key,
            fallback,
        )
    return normalized


def outline_char_count(outline: str) -> int:
    return len(re.sub(r"\s+", "", str(outline or "")))


def outline_validation_issues(outlines: dict, target_chapters: int) -> list[str]:
    outlines = outlines if isinstance(outlines, dict) else {}
    target_chapters = max(1, int(target_chapters))
    issues = []
    expected_keys = [str(index) for index in range(1, target_chapters + 1)]
    actual_keys = sorted(
        (str(int(key)) for key in outlines if str(key).isdigit() and int(key) > 0),
        key=int,
    )
    if actual_keys != expected_keys:
        issues.append(f"章节编号应为{expected_keys}，实际为{actual_keys}")
    for key in expected_keys:
        length = outline_char_count(outlines.get(key, ""))
        if length < MIN_OUTLINE_CHARS:
            issues.append(f"第{key}章细纲仅{length}字，少于{MIN_OUTLINE_CHARS}字")
    return issues

# ==========================================
# 0. 辅助函数：词库操作
# ==========================================
def load_keywords() -> dict:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "keywords.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("⚠️ 词库文件不存在或格式错误，跳过随机关键词功能: %s", path)
        return {}


def load_story_patterns() -> dict:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "story_patterns.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("⚠️ 套路配置不存在或格式错误，回退为无套路: %s", path)
        return {
            "none": {
                "name": "无套路",
                "architect": "不强制套用固定套路。",
                "writer": "自然推进本章。",
                "auditor": "仅检查故事自身逻辑。",
            }
        }


def _format_rule_list(title: str, values: list) -> str:
    if not values:
        return ""
    lines = [
        item.get("requirement", str(item)) if isinstance(item, dict) else str(item)
        for item in values
    ]
    return f"\n【{title}】\n" + "\n".join(f"- {line}" for line in lines)


def resolve_story_pattern(state: NovelState) -> dict:
    patterns = load_story_patterns()
    pattern_key = state.get("story_pattern", "none")
    pattern = patterns.get(pattern_key, patterns.get("none", {})).copy()
    if pattern_key == "custom":
        custom = str(state.get("custom_pattern", "")).strip()
        if custom:
            pattern["name"] = f"自定义：{custom}"
            pattern["architect"] = f"{pattern.get('architect', '')}\n用户要求：{custom}"
            pattern["writer"] = f"{pattern.get('writer', '')}\n用户要求：{custom}"
            pattern["auditor"] = f"{pattern.get('auditor', '')}\n用户要求：{custom}"
    if pattern.get("strong"):
        pattern["architect"] = (
            f"{pattern.get('architect', '')}"
            f"{_format_rule_list('全书比例节拍', pattern.get('beats', []))}"
            f"{_format_rule_list('禁止事项', pattern.get('forbidden', []))}"
        )
        pattern["writer"] = (
            f"{pattern.get('writer', '')}"
            f"{_format_rule_list('强制写作技巧', pattern.get('writing_techniques', []))}"
            f"{_format_rule_list('禁止事项', pattern.get('forbidden', []))}"
        )
        pattern["auditor"] = (
            f"{pattern.get('auditor', '')}"
            f"{_format_rule_list('套路审核规则', pattern.get('audit_rules', []))}"
            f"{_format_rule_list('禁止事项', pattern.get('forbidden', []))}"
        )
    pattern["key"] = pattern_key if pattern_key in patterns else "none"
    return pattern


def is_strong_pattern(pattern_key: str) -> bool:
    return bool(load_story_patterns().get(pattern_key, {}).get("strong"))


def compatible_styles_for_pattern(pattern_key: str) -> list[str]:
    pattern = load_story_patterns().get(pattern_key, {})
    return list(pattern.get("compatible_styles", []))


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def material_rules_for_pattern(pattern_key: str) -> dict:
    pattern = load_story_patterns().get(pattern_key, {})
    return {
        "world_policy": pattern.get("world_policy", "allow_all"),
        "forbidden_drivers": list(pattern.get("forbidden_material_drivers", [])),
        "background_only_drivers": list(pattern.get("background_only_drivers", [])),
        "note": pattern.get("material_note", ""),
    }


def keyword_category_metadata() -> dict:
    metadata = {}
    for key, value in load_keywords().items():
        material_type = value.get("material_type", "world_stage")
        metadata[key] = {
            "description": value.get("description", ""),
            "material_type": material_type,
            "material_type_label": value.get("material_type_label", {
                "world_stage": "世界观舞台",
                "relationship_material": "关系素材",
                "plot_material": "剧情素材",
                "core_driver": "主驱动力",
                "audience_driver": "频道驱动力",
            }.get(material_type, "素材")),
            "driver": value.get("driver", ""),
            "drivers": _as_list(value.get("drivers") or value.get("driver")),
            "usage": value.get("usage", ""),
        }
    return metadata


def material_category_conflict_reason(pattern_key: str, category: str) -> str:
    category_meta = keyword_category_metadata().get(category)
    if not category_meta:
        return ""
    if category_meta.get("material_type") == "world_stage":
        return ""

    rules = material_rules_for_pattern(pattern_key)
    forbidden = set(rules.get("forbidden_drivers", []))
    drivers = set(category_meta.get("drivers", []))
    if not forbidden.intersection(drivers):
        return ""

    pattern_name = load_story_patterns().get(pattern_key, {}).get("name", pattern_key)
    driver_label = category_meta.get("material_type_label", "素材")
    return (
        f"{pattern_name}不能让“{category}”这类{driver_label}抢主线；"
        "世界观舞台仍可自由套用。"
    )


def validate_material_categories_for_pattern(pattern_key: str, categories: list[str]) -> list[str]:
    issues = []
    for category in categories or []:
        reason = material_category_conflict_reason(pattern_key, category)
        if reason:
            issues.append(reason)
    return issues


def filter_material_categories_for_pattern(pattern_key: str, categories: list[str]) -> list[str]:
    return [
        category
        for category in (categories or [])
        if not material_category_conflict_reason(pattern_key, category)
    ]


def roll_pattern_manifest(
    pattern_key: str,
    seed: int | None = None,
    ending: str = "no_reunion",
) -> dict:
    pattern = load_story_patterns().get(pattern_key, {})
    if not pattern.get("strong"):
        return {}

    seed = int(seed) if seed is not None else random.SystemRandom().randint(1, 2_147_483_647)
    rng = random.Random(seed)
    modules = list(pattern.get("conflict_modules", []))
    reproductive = [item for item in modules if item.get("category") == "reproductive"]
    regular = [item for item in modules if item.get("category") != "reproductive"]
    count = rng.choice([2, 3])
    selected = rng.sample(regular, min(count, len(regular)))
    if reproductive and count >= 3 and rng.random() < 0.55:
        selected[-1] = rng.choice(reproductive)
    rng.shuffle(selected)

    endings = pattern.get("ending_options", {})
    if ending not in endings:
        ending = "no_reunion"
    return {
        "pattern_key": pattern_key,
        "seed": seed,
        "heroine": rng.choice(pattern.get("heroine_pool", ["善良隐忍、最终觉醒的女主"])),
        "hero": rng.choice(pattern.get("hero_pool", ["偏信女配、后期追悔的男主"])),
        "rival": rng.choice(pattern.get("rival_pool", ["擅长示弱的女配"])),
        "background": rng.choice(pattern.get("background_pool", ["都市情感"])),
        "conflicts": selected,
        "ending": ending,
        "ending_description": endings.get(ending, ""),
        "compatible_styles": list(pattern.get("compatible_styles", [])),
        "beat_preview": list(pattern.get("beats", [])),
    }


def validate_pattern_manifest(manifest: dict) -> list[str]:
    if not manifest:
        return ["缺少强套路契约"]
    issues = []
    conflicts = manifest.get("conflicts", [])
    if manifest.get("pattern_key") != STRONG_PATTERN_KEY:
        issues.append("套路契约标识不匹配")
    if not isinstance(conflicts, list):
        return issues + ["虐点模块必须为列表"]
    if not 2 <= len(conflicts) <= 3:
        issues.append("虐点模块必须为2至3个")
    conflict_ids = [
        item.get("id")
        for item in conflicts
        if isinstance(item, dict) and item.get("id")
    ]
    if len(conflict_ids) != len(conflicts):
        issues.append("每个虐点模块必须包含有效标识")
    elif len(set(conflict_ids)) != len(conflict_ids):
        issues.append("虐点模块不能重复")
    reproductive_count = sum(
        item.get("category") == "reproductive"
        for item in conflicts
        if isinstance(item, dict)
    )
    if reproductive_count > 1:
        issues.append("生育伤害模块最多只能选择一个")
    for field in ("heroine", "hero", "rival", "background", "ending", "seed"):
        if manifest.get(field) in (None, ""):
            issues.append(f"套路契约缺少字段：{field}")
    try:
        int(manifest.get("seed"))
    except (TypeError, ValueError):
        issues.append("套路随机种子必须为整数")
    if manifest.get("ending") not in {"no_reunion", "costly_reunion"}:
        issues.append("结局方向必须为不复合或高代价复合")
    return issues


def build_pattern_plan(manifest: dict, chapters: int, words_per_chapter: int) -> dict[str, dict]:
    chapters = max(1, int(chapters))
    words_per_chapter = max(1, int(words_per_chapter))
    conflicts = manifest.get("conflicts", [])
    total_words = chapters * words_per_chapter
    paywall_target_word = max(1, int(total_words * 0.475))
    paywall_chapter = min(chapters, (paywall_target_word - 1) // words_per_chapter + 1)
    plan = {}

    for chapter in range(1, chapters + 1):
        midpoint = (chapter - 0.5) / chapters
        if chapters == 1:
            phase = "开篇钩子、最重伤害反转与独立结局"
            heroine_state = "从受伤隐忍走向心死与独立清醒"
            hero_state = "偏信造成伤害后才察觉真相，并承担无法抹平的代价"
            required_event = (
                "前300字内爆发核心伤害；随后完成最严重伤害与反转，"
                "最终兑现女主独立和确认的结局方向"
            )
        elif chapter == 1 and chapter == paywall_chapter:
            phase = "开篇爆钩子与最重伤害反转卡点"
            heroine_state = "受伤隐忍后迅速心死，停止期待"
            hero_state = "偏信造成严重伤害后，第一次察觉事情不对"
            required_event = (
                "前300字内爆发核心伤害，并在本章完成全书最严重伤害与反转，"
                "让男主第一次动摇"
            )
        elif chapter == 1:
            phase = "开篇爆钩子"
            heroine_state = "受伤、隐忍，但开始察觉关系失衡"
            hero_state = "偏信女配，尚未意识到自己在伤害女主"
            required_event = "前300字内让女主遭受伤害、羞辱或被抛弃，立即爆发核心冲突"
        elif chapter == paywall_chapter:
            phase = "最重伤害与反转卡点"
            heroine_state = "由痛苦转为心死，停止期待"
            hero_state = "首次察觉事情不对，但尚未掌握完整真相"
            required_event = "发生全书最严重伤害，并用反转证据或异常让男主第一次动摇"
        elif midpoint < 0.25:
            phase = "虐点叠加"
            heroine_state = "继续隐忍，同时逐渐看清偏爱与不公"
            hero_state = "继续偏袒女配，把女主反应误判为矫情或嫉妒"
            required_event = "升级一个已选虐点，并让女配在女主受伤时获得关注或利益"
        elif midpoint < 0.50:
            phase = "伤害升级与心死"
            heroine_state = "痛苦逐步耗尽，准备停止解释"
            hero_state = "仍被信息差蒙蔽，做出会后悔的选择"
            required_event = "让伤害产生不可轻易撤销的后果，推进女主心死"
        elif chapter < chapters and midpoint < 0.80:
            phase = "真相揭露与追悔"
            heroine_state = "平静坚定，拒绝解释、补偿和回头"
            hero_state = "逐步看见真相，追悔并付出实际代价"
            required_event = "揭露一层真相，让男主的补偿失败并承担新的损失"
        else:
            phase = "独立离开与结局"
            heroine_state = "独立清醒，主动选择自己的新生活"
            hero_state = "承担长期后果，无法用道歉抹平伤害"
            if manifest.get("ending") == "costly_reunion":
                required_event = (
                    "先完成女主独立与边界建立；男主付出长期且不可逆的代价后，"
                    "由女主自主决定是否重新开始"
                )
            else:
                required_event = "完成女主独立离开与更好生活，男主追悔但无法挽回，明确不复合"

        module = conflicts[(chapter - 1) % len(conflicts)] if conflicts else {}
        if module.get("category") == "reproductive" and midpoint < 0.25:
            conflict_stage = "只铺垫风险、伤病或信息差，禁止提前发生实际生育伤害"
        elif chapter < paywall_chapter:
            conflict_stage = "建立前因并升级过程，不得无因果突然发生"
        elif chapter == paywall_chapter:
            conflict_stage = "让既有前因造成不可轻易撤销的后果"
        elif midpoint < 0.80:
            conflict_stage = "通过真相揭露呈现该虐点的后果与责任"
        else:
            conflict_stage = "兑现长期后果，禁止重复制造同一种伤害"
        if chapter == chapters:
            relationship_change = (
                "女主保持独立边界，在男主付出高代价后自主决定是否重新开始"
                if manifest.get("ending") == "costly_reunion"
                else "女主彻底离开且不复合，男主承担长期后果"
            )
        elif chapter == paywall_chapter:
            relationship_change = "女主彻底心死，男主第一次动摇但仍未掌握完整真相"
        else:
            relationship_change = "本章结束时关系必须发生可观察的变化"
        plan[str(chapter)] = {
            "phase": phase,
            "progress_range": f"{int((chapter - 1) / chapters * 100)}%-{int(chapter / chapters * 100)}%",
            "word_range": f"{(chapter - 1) * words_per_chapter + 1}-{chapter * words_per_chapter}",
            "heroine_state": heroine_state,
            "hero_awareness": hero_state,
            "required_event": required_event,
            "conflict_module": module.get("name", "按人物因果推进"),
            "conflict_stage": conflict_stage,
            "relationship_change": relationship_change,
            "ending_hook": "用新的伤害后果、真相证据或离开行动形成钩子",
            "is_paywall_turn": chapter == paywall_chapter,
            "paywall_target_word": paywall_target_word if chapter == paywall_chapter else None,
            "total_words": total_words,
            "is_final": chapter == chapters,
        }
    return plan


def format_pattern_manifest(manifest: dict) -> str:
    if not manifest:
        return "无结构化套路契约"
    conflicts = "；".join(
        (
            f"{item.get('name', '')}（{item.get('description', '')}）"
            if isinstance(item, dict)
            else str(item)
        )
        for item in manifest.get("conflicts", [])
    )
    return (
        f"背景：{manifest.get('background', '')}\n"
        f"女主：{manifest.get('heroine', '')}\n"
        f"男主：{manifest.get('hero', '')}\n"
        f"女配：{manifest.get('rival', '')}\n"
        f"选定虐点：{conflicts}\n"
        f"结局：{manifest.get('ending_description', '')}\n"
        f"随机种子：{manifest.get('seed', '')}"
    )


def format_pattern_chapter_task(task: dict) -> str:
    if not task:
        return "无结构化章节套路任务"
    return "\n".join(
        f"{label}：{task.get(key, '')}"
        for label, key in (
            ("套路阶段", "phase"),
            ("全书进度", "progress_range"),
            ("女主状态", "heroine_state"),
            ("男主认知", "hero_awareness"),
            ("本章必须事件", "required_event"),
            ("本章虐点模块", "conflict_module"),
            ("虐点执行阶段", "conflict_stage"),
            ("关系变化", "relationship_change"),
            ("结尾钩子", "ending_hook"),
        )
    )


def attach_pattern_plan_to_outlines(
    outlines: dict[str, str], pattern_plan: dict[str, dict]
) -> dict[str, str]:
    if not pattern_plan:
        return outlines
    return {
        key: f"【结构化套路任务】\n{format_pattern_chapter_task(pattern_plan.get(key, {}))}\n\n【剧情细纲】\n{outline}"
        for key, outline in outlines.items()
    }


def strip_pattern_plan_from_outlines(outlines: dict[str, str]) -> dict[str, str]:
    marker = "\n\n【剧情细纲】\n"
    cleaned = {}
    for key, outline in (outlines or {}).items():
        text = str(outline)
        if text.startswith("【结构化套路任务】") and marker in text:
            text = text.split(marker, 1)[1]
        cleaned[str(key)] = text
    return cleaned


def strong_pattern_validation_issues(
    manifest: dict, pattern_plan: dict, outlines: dict, target_chapters: int
) -> list[str]:
    issues = validate_pattern_manifest(manifest)
    expected = {str(index) for index in range(1, max(1, int(target_chapters)) + 1)}
    if set(pattern_plan) != expected:
        issues.append("逐章节拍计划与目标章节数不一致")
    paywall = [
        int(key)
        for key, task in pattern_plan.items()
        if isinstance(task, dict) and task.get("is_paywall_turn")
    ]
    if len(paywall) != 1:
        issues.append("必须且只能有一个45%-50%的最重伤害反转卡点")
    elif not pattern_plan[str(paywall[0])].get("paywall_target_word"):
        issues.append("最重伤害反转卡点缺少全书字数目标")
    else:
        paywall_task = pattern_plan[str(paywall[0])]
        ratio = paywall_task["paywall_target_word"] / max(1, paywall_task.get("total_words", 0))
        if not 0.45 <= ratio <= 0.50:
            issues.append("最重伤害反转卡点必须位于全书累计字数45%-50%")
    if manifest.get("ending") == "no_reunion":
        final_task = pattern_plan.get(str(max(1, int(target_chapters))), {})
        if "独立" not in final_task.get("phase", ""):
            issues.append("默认不复合结局必须以女主独立离开收束")
    for key in expected:
        outline = str(outlines.get(key, ""))
        if "【结构化套路任务】" not in outline:
            issues.append(f"第{key}章缺少结构化套路任务")
    return issues


def strong_pattern_outline_content_warnings(
    manifest: dict, pattern_plan: dict, outlines: dict, target_chapters: int
) -> list[str]:
    """Best-effort lexical diagnostics; never use these warnings to reject an outline."""
    chapters = max(1, int(target_chapters))
    outlines = {str(key): str(value) for key, value in (outlines or {}).items()}
    issues = []

    first = outlines.get("1", "")
    if "前300字" not in first or not any(word in first for word in ("伤害", "羞辱", "抛弃")):
        issues.append("第1章细纲必须明确前300字内发生伤害、羞辱或抛弃")

    paywall_keys = [
        key for key, task in pattern_plan.items()
        if isinstance(task, dict) and task.get("is_paywall_turn")
    ]
    if paywall_keys:
        paywall_outline = outlines.get(paywall_keys[0], "")
        if not any(word in paywall_outline for word in ("最严重", "最重伤害", "致命", "不可逆")):
            issues.append(f"第{paywall_keys[0]}章细纲缺少最严重或不可逆伤害")
        if not any(word in paywall_outline for word in ("反转", "证据")):
            issues.append(f"第{paywall_keys[0]}章细纲缺少最重伤害后的反转证据")
        if not any(
            word in paywall_outline
            for word in ("男主第一次", "男主首次", "第一次动摇", "首次察觉")
        ):
            issues.append(f"第{paywall_keys[0]}章细纲缺少男主首次察觉异常的反转")
        if not any(word in paywall_outline for word in ("心死", "停止期待")):
            issues.append(f"第{paywall_keys[0]}章细纲缺少女主由痛苦转为心死")

    truth_outlines = "".join(
        outlines.get(key, "")
        for key, task in pattern_plan.items()
        if isinstance(task, dict) and task.get("phase") == "真相揭露与追悔"
    )
    if truth_outlines and "真相" not in truth_outlines:
        issues.append("50%-80%阶段细纲缺少真相揭露")
    if truth_outlines and not any(word in truth_outlines for word in ("追悔", "补偿", "代价")):
        issues.append("50%-80%阶段细纲缺少男主追悔、补偿或实际代价")

    final = outlines.get(str(chapters), "")
    if not any(word in first for word in ("隐忍", "察觉", "失望")):
        issues.append("开篇阶段细纲缺少女主隐忍或开始察觉异常")
    if not any(word in final for word in ("平静", "坚定", "独立")):
        issues.append("结局细纲缺少女主平静坚定或独立状态")
    if manifest.get("ending") == "costly_reunion":
        if "代价" not in final or not any(word in final for word in ("复合", "重新开始")):
            issues.append("高代价复合结局必须写明男主代价与女主自主决定")
    elif "独立" not in final or not any(word in final for word in ("离开", "不复合", "无法挽回")):
        issues.append("默认结局必须明确女主独立离开且不复合")

    all_outlines = "".join(outlines.values())
    for conflict in manifest.get("conflicts", []):
        name = conflict.get("name", "") if isinstance(conflict, dict) else str(conflict)
        if name and name not in all_outlines:
            issues.append(f"细纲未安排已确认虐点模块：{name}")
    return issues


# Backward-compatible alias for callers that used the original helper name.
strong_pattern_outline_content_issues = strong_pattern_outline_content_warnings

def pick_keywords(categories: list[str], count: int = 2, story_pattern: str = "none") -> list[str]:
    keyword_db = load_keywords()
    pool = []
    for cat in filter_material_categories_for_pattern(story_pattern, categories):
        if cat in keyword_db and keyword_db[cat].get("keywords"):
            pool.extend(keyword_db[cat]["keywords"])
    if not pool:
        return []
    return random.sample(pool, min(count, len(pool)))

# ==========================================
# 0. 辅助函数：读取本地 Prompt 文件 (绝对路径版)
# ==========================================
def load_prompt(file_name: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "Role", file_name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error("❌ 严重错误：找不到提示词文件！请检查路径：%s", path)
        raise
# ==========================================
# 1. 模型插座配置
# ==========================================
def _create_llm(temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=temperature,
        timeout=MODEL_TIMEOUT_SECONDS,
        max_retries=MODEL_MAX_RETRIES,
    )


llm_architect = _create_llm(0.7)
llm_writer = _create_llm(0.8)
llm_editor = _create_llm(0.5)
llm_auditor_raw = _create_llm(0)
llm_summarizer = _create_llm(0.3)

# ==========================================
# 2. 强制 JSON 结构化定义 (键名保持中文)
# ==========================================
class ArchitectOutput(BaseModel):
    novel_title: str = Field(description="小说的书名，8-20字，简洁有力有网感")
    world_bible: str = Field(description="不少于500字的世界观、力量体系、主角人设详细设定。")
    chapter_outlines: dict[str, str] = Field(description="章节号(纯数字)映射到不少于200字的详细细纲文本。键必须为纯数字如'1', '2'。")
    estimated_words: int = Field(description="预估总字数")

class AuditReport(BaseModel):
    审核状态: str = Field(description="严格输出 '通过' 或 '不通过'。")
    发现的问题: list[str] = Field(description="具体的逻辑硬伤或偏离大纲的问题点。无问题则为空列表。")
    警告: list[str] = Field(default_factory=list, description="不触发退稿的软性逻辑问题。")
    套路执行状态: str = Field(default="通过", description="严格输出 '通过' 或 '不通过'。")
    套路问题: list[str] = Field(default_factory=list, description="当前章节未完成的强制套路任务。")
    修改建议: str = Field(description="具体的修改指导。若通过则填'无'。")

class EditorReport(BaseModel):
    文风评分: int = Field(description="给出1-10的评分，7分及格。")
    AI痕迹问题: list[str] = Field(default_factory=list, description="可定位的具体AI写作痕迹。")
    改进建议: str = Field(description="关于遣词造句、剧情节奏的润色建议。")

llm_architect_structured = llm_architect.with_structured_output(ArchitectOutput, method="json_mode")
llm_auditor_structured = llm_auditor_raw.with_structured_output(AuditReport, method="json_mode")
llm_editor_structured = llm_editor.with_structured_output(EditorReport, method="json_mode")


def _normalize_issue_list(value) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item or "").strip()]


def normalize_audit_report(report: dict) -> dict:
    normalized = dict(report or {})
    normalized["发现的问题"] = _normalize_issue_list(normalized.get("发现的问题", []))
    normalized["警告"] = _normalize_issue_list(normalized.get("警告", []))
    normalized["套路问题"] = _normalize_issue_list(normalized.get("套路问题", []))
    normalized["套路执行状态"] = "不通过" if normalized["套路问题"] else "通过"
    normalized["审核状态"] = (
        "不通过"
        if normalized["发现的问题"] or normalized["套路问题"]
        else "通过"
    )
    normalized["修改建议"] = str(normalized.get("修改建议", "无"))
    return normalized


def normalize_editor_report(report: dict) -> dict:
    normalized = dict(report or {})
    normalized["AI痕迹问题"] = _normalize_issue_list(normalized.get("AI痕迹问题", []))
    score = max(1, min(10, int(normalized.get("文风评分", 8))))
    if not normalized["AI痕迹问题"] and score < STYLE_PASS_SCORE:
        score = STYLE_PASS_SCORE
    normalized["文风评分"] = score
    normalized["改进建议"] = str(normalized.get("改进建议", "无"))
    return normalized

def invoke_with_retry(chain, inputs, node_name: str, max_attempts: int = APP_INVOKE_ATTEMPTS):
    last_error = None
    for attempt in range(max_attempts):
        try:
            result = chain.invoke(inputs)
            if result is not None:
                return result
        except KeyError as error:
            missing = str(error).strip("'")
            raise RuntimeError(
                f"{node_name}提示词模板变量缺失：{missing}。"
                "请检查对应 Role 提示词中的花括号是否需要转义。"
            ) from error
        except Exception as e:
            last_error = e
            logger.warning(
                "   ⚠️ [%s] 模型调用尝试 %d/%d 失败: %s",
                node_name,
                attempt + 1,
                max_attempts,
                e,
            )
        if attempt + 1 < max_attempts:
            delay = min(2 ** attempt, 8)
            logger.info("   ⏳ [%s] %d秒后自动重试...", node_name, delay)
            time.sleep(delay)
    detail = f"{type(last_error).__name__}: {last_error}" if last_error else "模型返回空结果"
    raise RuntimeError(
        f"{node_name}模型调用失败，已自动重试{max_attempts}次。"
        f"最后错误：{detail}"
    ) from last_error


def _safe_invoke(chain, inputs, node_name: str, max_retries: int = APP_INVOKE_ATTEMPTS):
    try:
        return invoke_with_retry(chain, inputs, node_name, max_retries)
    except RuntimeError as error:
        logger.error("   ❌ %s", error)
    return None

# ==========================================
# 3. 核心节点
# ==========================================

def architect_node(state: NovelState):
    logger.info("🧠 架构师正在深度推演世界观与大纲...")
    if state.get("world_bible"):
        return {}

    pattern = resolve_story_pattern(state)
    strong_pattern = bool(pattern.get("strong"))
    manifest = state.get("pattern_manifest", {}) if strong_pattern else {}
    if strong_pattern and validate_pattern_manifest(manifest):
        manifest = roll_pattern_manifest(pattern.get("key", STRONG_PATTERN_KEY))

    keywords = state.get("keywords", [])
    keywords_str = "、".join(keywords) if keywords else "无"

    chapters = state.get("target_chapters", DEFAULT_CHAPTERS)
    words_per = state.get("words_per_chapter", DEFAULT_WORDS_PER_CHAPTER)
    pattern_plan = (
        build_pattern_plan(manifest, chapters, words_per)
        if strong_pattern
        else state.get("pattern_plan", {})
    )
    strong_requirement = ""
    if strong_pattern:
        strong_requirement = (
            "\n【已确认强套路契约】\n"
            f"{format_pattern_manifest(manifest)}\n"
            "【逐章节拍计划】\n"
            f"{json.dumps(pattern_plan, ensure_ascii=False)}\n"
            "每章细纲必须严格服从对应章节任务，尤其不得提前让男主得知真相，"
            "不得让后期女主重新反复解释或无代价回头。"
            "请在细纲中清楚描述开篇伤害、最重伤害与反转、女主觉醒过程、"
            "确认的结局方向和已选虐点如何建立前因与后果；允许使用符合剧情的自然措辞。"
            "若本次随机素材非空，素材只能作为世界观舞台、人物关系或局部冲突补充，"
            "不得替代强套路的核心驱动力和逐章节拍计划。"
        )
    chapter_req = (
        f"规划严格且仅有{chapters}章的详细细纲，每章正文目标{words_per}字。"
        f"每章细纲去除空白后必须不少于{MIN_OUTLINE_CHARS}字，"
        "必须包含开场状态、核心冲突、关键行动、人物关系变化、伏笔、"
        "结尾结果和下一章钩子。"
        f"\n【创作套路：{pattern.get('name', '无套路')}】\n{pattern.get('architect', '')}"
        f"{strong_requirement}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("architect_system.md")),
        ("system", ARCHITECT_JSON_PROMPT),
        ("user", load_prompt("architect_user.md"))
        ])
    
    architect_chain = prompt | llm_architect_structured
    result = None
    outline_issues = []
    outline_warnings = []
    active_chapter_req = chapter_req
    for outline_attempt in range(APP_INVOKE_ATTEMPTS):
        result = invoke_with_retry(architect_chain, {
            "user_idea": state.get("user_idea"),
            "keywords": keywords_str,
            "chapter_requirement": active_chapter_req
        }, "架构师")
        chapter_outlines = normalize_chapter_outlines(result.chapter_outlines, chapters)
        outline_issues = outline_validation_issues(chapter_outlines, chapters)
        if strong_pattern:
            outline_warnings = strong_pattern_outline_content_warnings(
                manifest, pattern_plan, chapter_outlines, chapters
            )
            chapter_outlines = attach_pattern_plan_to_outlines(chapter_outlines, pattern_plan)
            outline_issues.extend(
                strong_pattern_validation_issues(
                    manifest, pattern_plan, chapter_outlines, chapters
                )
            )
        if not outline_issues:
            if outline_warnings:
                logger.warning(
                    "   ⚠️ 强套路细纲存在措辞层面的提醒，但结构化任务已补齐，不阻断生成: %s",
                    "；".join(outline_warnings[:5]),
                )
            break
        logger.warning(
            "   ⚠️ 架构师大纲验收未通过 (%d/%d): %s",
            outline_attempt + 1,
            APP_INVOKE_ATTEMPTS,
            "；".join(outline_issues[:5]),
        )
        active_chapter_req = (
            f"{chapter_req}\n【上一稿确定性验收失败，下一稿必须逐项修复】\n"
            + "\n".join(f"- {issue}" for issue in outline_issues)
        )
    if result is None or outline_issues:
        raise RuntimeError(
            f"架构师连续{APP_INVOKE_ATTEMPTS}次未能生成合格大纲："
            f"{'；'.join(outline_issues[:5])}"
        )
    chapter_outlines = normalize_chapter_outlines(result.chapter_outlines, chapters)
    if strong_pattern:
        chapter_outlines = attach_pattern_plan_to_outlines(chapter_outlines, pattern_plan)

    # 保存大纲 JSON 到 Outline/ 目录
    try:
        os.makedirs("Outline", exist_ok=True)
        safe_title = "".join(c for c in result.novel_title if c not in r'\/:*?"<>|')
        save_path = os.path.join("Outline", f"{safe_title}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "title": result.novel_title,
                "world_bible": result.world_bible,
                "chapter_outlines": chapter_outlines,
                "story_pattern": state.get("story_pattern", "none"),
                "custom_pattern": state.get("custom_pattern", ""),
                "pattern_manifest": manifest,
                "pattern_plan": pattern_plan,
                "estimated_words": chapters * words_per,
                "created_at": datetime.datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        logger.info("📁 大纲已保存 → %s", save_path)
    except Exception as e:
        logger.warning("⚠️ 大纲保存失败: %s", e)

    return {
        "novel_title": result.novel_title,
        "world_bible": result.world_bible,
        "chapter_outlines": chapter_outlines,
        "pattern_manifest": manifest,
        "pattern_plan": pattern_plan,
        "current_chapter": 1
    }
def writer_node(state: NovelState):
    chapter_num = state.get("current_chapter", 1)
    iteration = state.get("iteration_count", 0) + 1
    logger.info("✍️ 写手正在奋笔疾书 第 %d 章 (第 %d 稿)...", chapter_num, iteration)
    
    world_bible = state.get("world_bible", "")
    outlines = state.get("chapter_outlines", {})
    current_outline = outlines.get(str(chapter_num), "自由发挥。")
    next_outline = outlines.get(str(chapter_num + 1), "这是最后一章，收束已建立的冲突与伏笔。")
    words_per = state.get("words_per_chapter", DEFAULT_WORDS_PER_CHAPTER)
    min_words = int(words_per * MIN_CHAPTER_RATIO)
    preferred_words = int(words_per * 0.93)
    style = state.get("writer_style", "default")
    pattern = resolve_story_pattern(state)
    pattern_task = state.get("pattern_plan", {}).get(str(chapter_num), {})
    continuity_state = state.get("continuity_state") or state.get("story_summary", "故事刚刚开始。")
    
    style_map = {
        "hot_blood": "writer_system_hot_blood.md",
        "literary": "writer_system_literary.md",
        "cold": "writer_system_cold.md",
        "humor": "writer_system_humor.md",
        "18xx": "writer_system_18xx.md",
    }
    system_file = style_map.get(style, "writer_system.md")
    
    audit_report = state.get("audit_report", {})
    editor_report = state.get("editor_report", {})
    feedback = ""
    previous_draft_chars = chapter_body_char_count(state.get("current_draft", ""))
    if iteration > 1 and previous_draft_chars < min_words:
        feedback += (
            f"\n\n【篇幅补写令】：上一稿正文仅{previous_draft_chars}字，"
            f"低于最低要求{min_words}字。请补足必要场景、动作、对话和情节推进，"
            f"将正文写到{min_words}-{words_per}字。"
        )
    if audit_report.get("审核状态") == "不通过":
        feedback += (
            f"\n\n【审计退稿修改令】：逻辑硬伤：{audit_report.get('发现的问题')}；"
            f"套路执行问题：{audit_report.get('套路问题', [])}。"
            f"请只针对这些明确问题重写本章：{audit_report.get('修改建议')}"
        )
    if editor_report.get("文风评分", 10) < STYLE_PASS_SCORE:
        feedback += (
            f"\n\n【责编退稿润色令】：文风不达标(当前评分{editor_report.get('文风评分')}/10)。"
            f"只修改以下明确问题：{editor_report.get('AI痕迹问题', [])}。"
            f"改进建议：{editor_report.get('改进建议')}"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("writer_common_rules.md")),
        ("system", load_prompt(system_file)),
        ("system", CHAPTER_FORMAT_PROMPT),
        ("user", load_prompt("writer_user.md"))
    ])

    result = invoke_with_retry(prompt | llm_writer, {
        "world_bible": world_bible,
        "continuity_state": continuity_state,
        "outline": current_outline,
        "next_outline": next_outline,
        "pattern_writer": pattern.get("writer", ""),
        "pattern_manifest": format_pattern_manifest(state.get("pattern_manifest", {})),
        "pattern_chapter_task": format_pattern_chapter_task(pattern_task),
        "feedback": feedback,
        "chapter_num": chapter_num,
        "words_per_chapter": words_per,
        "min_words_per_chapter": min_words,
        "preferred_words_per_chapter": preferred_words,
    }, f"写手第{chapter_num}章第{iteration}稿")

    content = result.content if result and result.content else "[写手产出为空，请重试]"
    content = normalize_chapter_output(content, chapter_num, words_per)
    
    return {
        "current_draft": content,
        "iteration_count": iteration
    }

def auditor_node(state: NovelState):
    logger.info("🕵️ 审计员正在进行地毯式排查...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("auditor_system.md")),
        ("system", AUDITOR_JSON_PROMPT),
        ("user", load_prompt("auditor_user.md"))
    ])

    result = _safe_invoke(prompt | llm_auditor_structured, _audit_inputs(state), "auditor")
    
    if result is None:
        logger.warning("🕵️ 审计失败，默认放行")
        return {"audit_report": {"审核状态": "通过", "发现的问题": [], "警告": ["审计模型调用失败"], "套路执行状态": "通过", "套路问题": [], "修改建议": "无"}}
    
    logger.info("🕵️ 审计结果: %s", result.审核状态)
    return {"audit_report": normalize_audit_report(result.model_dump())}

def editor_node(state: NovelState):
    logger.info("👓 责编正在审视文笔与爽点...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("editor_system.md")),
        ("system", EDITOR_JSON_PROMPT),
        ("user", load_prompt("editor_user.md"))
    ])
    
    result = _safe_invoke(prompt | llm_editor_structured, _editor_inputs(state), "editor")
    
    if result is None:
        logger.warning("👓 责编评估失败，默认放行(评分8)")
        result = EditorReport(文风评分=8, AI痕迹问题=[], 改进建议="无")
    
    logger.info("👓 责编评分: %d/10", result.文风评分)
    
    return {"editor_report": normalize_editor_report(result.model_dump())}

def _audit_inputs(state: NovelState) -> dict:
    chapter = state.get("current_chapter", 1)
    outlines = state.get("chapter_outlines", {})
    pattern = resolve_story_pattern(state)
    return {
        "world_bible": state.get("world_bible", ""),
        "continuity_state": state.get("continuity_state") or state.get("story_summary", "故事刚刚开始。"),
        "pattern_auditor": pattern.get("auditor", ""),
        "pattern_manifest": format_pattern_manifest(state.get("pattern_manifest", {})),
        "pattern_chapter_task": format_pattern_chapter_task(
            state.get("pattern_plan", {}).get(str(chapter), {})
        ),
        "outline": outlines.get(str(chapter), ""),
        "next_outline": outlines.get(str(chapter + 1), "这是最后一章。"),
        "draft": state.get("current_draft", ""),
    }


def _editor_inputs(state: NovelState) -> dict:
    pattern = resolve_story_pattern(state)
    style_names = {
        "hot_blood": "热血爽文",
        "literary": "文艺细腻",
        "cold": "冷峻纪实",
        "humor": "轻松搞笑",
        "18xx": "18XX",
        "default": "默认风格",
    }
    return {
        "writer_style": style_names.get(state.get("writer_style", "default"), "默认风格"),
        "story_pattern": pattern.get("name", "无套路"),
        "draft": state.get("current_draft", ""),
    }


def _auditor_internal(state: NovelState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("auditor_system.md")),
        ("system", AUDITOR_JSON_PROMPT),
        ("user", load_prompt("auditor_user.md"))
    ])
    result = _safe_invoke(prompt | llm_auditor_structured, _audit_inputs(state), "auditor")
    if result is None:
        return {"audit_report": {"审核状态": "通过", "发现的问题": [], "警告": ["审计模型调用失败"], "套路执行状态": "通过", "套路问题": [], "修改建议": "无"}}
    return {"audit_report": normalize_audit_report(result.model_dump())}

def _editor_internal(state: NovelState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("editor_system.md")),
        ("system", EDITOR_JSON_PROMPT),
        ("user", load_prompt("editor_user.md"))
    ])
    result = _safe_invoke(prompt | llm_editor_structured, _editor_inputs(state), "editor")
    if result is None:
        return {"editor_report": EditorReport(文风评分=8, AI痕迹问题=[], 改进建议="无").model_dump()}
    return {"editor_report": normalize_editor_report(result.model_dump())}

def reviewer_node(state: NovelState):
    logger.info("🔍 审稿员正在进行逻辑+文风并行双检...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_audit = pool.submit(_auditor_internal, state)
        fut_editor = pool.submit(_editor_internal, state)
        audit_result = fut_audit.result()
        editor_result = fut_editor.result()
    audit_report = audit_result.get("audit_report", {})
    editor_report = editor_result.get("editor_report", {})
    logger.info("🔍 审计: %s | 文风评分: %d/10", audit_report.get("审核状态"), editor_report.get("文风评分", 0))
    return {**audit_result, **editor_result}

def summarizer_node(state: NovelState):
    logger.info("📝 书记员正在提炼记忆档案...")
    
    current_chap_num = state.get("current_chapter", 1)
    latest_chapter = state.get("current_draft", "")
    file_path = _build_output_path(state.get("novel_title", "小说输出"))
    has_existing_content = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    
    with open(file_path, "a", encoding="utf-8") as f:
        if has_existing_content:
            f.write("\n\n")
        f.write(latest_chapter)
    logger.info("💾 第 %d 章已安全入库 → %s", current_chap_num, file_path)

    warnings = chapter_quality_warnings(state)
    for warning in warnings:
        logger.warning("⚠️ %s", warning)

    outlines = state.get("chapter_outlines", {})
    is_last_chapter = current_chap_num >= len(outlines)
    story_summary = state.get("story_summary", "")
    continuity_state = state.get("continuity_state") or story_summary
    if not is_last_chapter:
        prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("summarizer_system.md")),
            ("user", load_prompt("summarizer_user.md"))
        ])

        try:
            result = invoke_with_retry(prompt | llm_summarizer, {
                "old_summary": continuity_state,
                "new_chapter": latest_chapter
            }, f"第{current_chap_num}章剧情摘要")
            continuity_state = result.content
            story_summary = continuity_state
        except RuntimeError as error:
            logger.warning("⚠️ %s；保留旧摘要并继续下一章", error)
            warnings.append(f"第{current_chap_num}章剧情摘要更新失败，已保留旧摘要")
    else:
        logger.info("⏭️ 最后一章已保存，跳过无后续用途的剧情摘要")
    
    return {
        "story_summary": story_summary,
        "continuity_state": continuity_state,
        "current_chapter": current_chap_num + 1,
        "current_draft": latest_chapter,
        "iteration_count": 0,
        "saved_chapter": current_chap_num,
        "audit_report": {},
        "editor_report": {},
        "chapter_warnings": warnings,
        "summary_skipped": is_last_chapter,
    }
