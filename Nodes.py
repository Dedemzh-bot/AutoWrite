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

def _safe_file_stem(text: str, fallback: str) -> str:
    safe = "".join(c for c in str(text or "") if c not in r'\/:*?"<>|')
    return safe.strip() or fallback


def _build_output_path(title: str, run_id: str = "") -> str:
    safe = _safe_file_stem(title, "小说输出")
    if run_id:
        safe = f"{safe}_{_safe_file_stem(run_id, 'run')}"
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
MAX_REVIEW_ATTEMPTS = 4
MAX_FINALE_REVIEW_ATTEMPTS = 6
MAX_CONTINUITY_REPAIR_ATTEMPTS = 2
LEDGER_CONTEXT_CHARS = 1500
NORMAL_RECOMMENDED_MIN_RATIO = 0.85
NORMAL_RECOMMENDED_MAX_RATIO = 1.15
NORMAL_HARD_MIN_RATIO = 0.70
NORMAL_HARD_MAX_RATIO = 1.35
FINALE_RECOMMENDED_MIN_RATIO = 0.80
FINALE_RECOMMENDED_MAX_RATIO = 1.40
FINALE_HARD_MIN_RATIO = 0.60
FINALE_HARD_MAX_RATIO = 1.60
MODEL_TIMEOUT_SECONDS = float(os.getenv("MODEL_TIMEOUT_SECONDS", "180"))
MODEL_MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "5"))
APP_INVOKE_ATTEMPTS = int(os.getenv("APP_INVOKE_ATTEMPTS", "3"))
STRONG_PATTERN_KEY = "female_angst_awakening"

CHAPTER_FORMAT_PROMPT = """正文输出必须严格使用以下唯一格式：
第X章 章节名字

正文内容

其中 X 使用当前章节数字，章节名字最多十个字。
正文（不含章节标题）应遵循本次任务给出的篇幅规则。字数是软目标，不得为了卡上限删掉句子、场景、关键事件或结局。
章节标题必须独占第一行。不要使用 Markdown 标题、括号标题、卷名、序号标题、等号分割线或其他章节格式。
除这一行章节标题和正文外，不要输出任何说明。"""

ARCHITECT_JSON_PROMPT = """必须仅输出一个有效 JSON 对象，不要输出 Markdown 或说明文字。
字段必须完整：novel_title 为字符串；world_bible 为字符串；chapter_outlines 为对象，键是纯数字章节号、值是章节细纲；estimated_words 为整数。
chapter_outlines 中每一章细纲去除空白后必须不少于 200 字。每章细纲必须明确写出本章开场状态、核心冲突、关键行动、人物关系变化、重要信息或伏笔、结尾结果与下一章钩子，禁止用空话凑字数。"""

AUDITOR_JSON_PROMPT = """必须仅输出一个有效 JSON 对象，不要输出 Markdown 或说明文字。
字段必须完整：审核状态为“通过”或“不通过”；发现的问题为逻辑硬伤或大纲偏离字符串数组；警告为软性问题字符串数组；套路执行状态为“通过”或“不通过”；套路问题为未完成的强制套路任务字符串数组；修改建议为字符串；大纲完成度、连续性评分、衔接评分为0到100整数；已完成事件、未完成事件、阻断问题、结局问题为字符串数组；结局完整性为布尔值。
发现的问题或套路问题任一非空时输出“不通过”；只有警告时必须输出“通过”。"""

SCENE_PLAN_SYSTEM_PROMPT = """你是小说执行导演。请先把本章契约转成可执行场景计划，再交给写手。
必须仅输出有效 JSON：scenes 为按顺序排列的场景字符串数组；coverage 为“契约事件→对应场景”的对象；ending_strategy 为本章如何到达规定结束状态的字符串。
每个必须事件都要在 coverage 中出现。不得新增改变主线方向的事件。最终章必须把终局事件分配到具体场景并留出明确收束场景。"""

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


def normalize_chapter_output(
    content: str, chapter_num: int, max_body_chars: int | None = None
) -> str:
    """Enforce plain-text chapter formatting without truncating story content."""
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
    return f"第{chapter_num}章 {title}\n\n{body}".rstrip()


def chapter_body_char_count(content: str) -> int:
    parts = content.split("\n", 1)
    body = parts[1] if len(parts) > 1 else ""
    return len(re.sub(r"\s+", "", body))


def should_retry_short_draft(state: NovelState) -> bool:
    body_chars = chapter_body_char_count(state.get("current_draft", ""))
    return (
        body_chars < 20
        and state.get("iteration_count", 1) < 2
    )


def is_final_chapter(state: NovelState) -> bool:
    outlines = state.get("chapter_outlines", {})
    return bool(outlines) and state.get("current_chapter", 1) >= len(outlines)


def chapter_length_limits(words_per: int, final_chapter: bool = False) -> dict:
    words_per = max(1, int(words_per or DEFAULT_WORDS_PER_CHAPTER))
    if final_chapter:
        recommended_min_ratio = FINALE_RECOMMENDED_MIN_RATIO
        recommended_max_ratio = FINALE_RECOMMENDED_MAX_RATIO
        hard_min_ratio = FINALE_HARD_MIN_RATIO
        hard_max_ratio = FINALE_HARD_MAX_RATIO
    else:
        recommended_min_ratio = NORMAL_RECOMMENDED_MIN_RATIO
        recommended_max_ratio = NORMAL_RECOMMENDED_MAX_RATIO
        hard_min_ratio = NORMAL_HARD_MIN_RATIO
        hard_max_ratio = NORMAL_HARD_MAX_RATIO
    return {
        "target": words_per,
        "recommended_min": int(words_per * recommended_min_ratio),
        "recommended_max": int(words_per * recommended_max_ratio),
        "hard_min": int(words_per * hard_min_ratio),
        "hard_max": int(words_per * hard_max_ratio),
    }


def chapter_length_guidance(words_per: int, final_chapter: bool = False) -> str:
    limits = chapter_length_limits(words_per, final_chapter)
    if final_chapter:
        return (
            f"最终章目标约{limits['target']}字；建议{limits['recommended_min']}-"
            f"{limits['recommended_max']}字。若终局事件全部兑现，可接受"
            f"{limits['hard_min']}-{limits['hard_max']}字；不得超过硬上限"
            f"{limits['hard_max']}字，也不得为了压字数删掉结局。"
        )
    return (
        f"本章目标约{limits['target']}字；建议{limits['recommended_min']}-"
        f"{limits['recommended_max']}字，可接受范围{limits['hard_min']}-"
        f"{limits['hard_max']}字。不得截断完整句子或关键事件。"
    )


def chapter_length_assessment(state: NovelState, draft: str | None = None) -> dict:
    final_chapter = is_final_chapter(state)
    limits = chapter_length_limits(
        state.get("words_per_chapter", DEFAULT_WORDS_PER_CHAPTER),
        final_chapter,
    )
    body_chars = chapter_body_char_count(
        state.get("current_draft", "") if draft is None else draft
    )
    blocking = []
    warnings = []
    if body_chars < limits["hard_min"]:
        blocking.append(
            f"正文仅{body_chars}字，低于篇幅硬下限{limits['hard_min']}字，"
            "大概率未完成必要事件"
        )
    elif body_chars < limits["recommended_min"]:
        warnings.append(
            f"正文{body_chars}字，低于建议下限{limits['recommended_min']}字；"
            "只有契约事件完整时才可接受"
        )
    if body_chars > limits["hard_max"]:
        blocking.append(
            f"正文{body_chars}字，超过篇幅硬上限{limits['hard_max']}字，"
            "需要压缩重复内容但不得机械截断"
        )
    elif body_chars > limits["recommended_max"]:
        warnings.append(
            f"正文{body_chars}字，高于建议上限{limits['recommended_max']}字；"
            "只有剧情紧凑且契约事件必要时才可接受"
        )

    target = limits["target"]
    deviation = abs(body_chars - target) / max(1, target)
    score = max(0, round(100 - deviation * 100))
    if limits["recommended_min"] <= body_chars <= limits["recommended_max"]:
        score = max(score, 90)
    return {
        "body_chars": body_chars,
        "limits": limits,
        "blocking": blocking,
        "warnings": warnings,
        "score": score,
    }


def chapter_quality_warnings(state: NovelState) -> list[str]:
    warnings = []
    current_chapter = state.get("current_chapter", 1)
    audit = state.get("audit_report", {})
    editor = state.get("editor_report", {})

    length_assessment = chapter_length_assessment(state)
    for issue in length_assessment["blocking"] + length_assessment["warnings"]:
        warnings.append(f"第{current_chapter}章篇幅提示：{issue}")
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


_OUTLINE_SECTION_RE = re.compile(
    r"(?:^|[\n。；])\s*"
    r"(开场状态|开场|核心冲突|关键行动|人物关系变化|关系变化|重要信息或伏笔|伏笔|"
    r"结尾结果与下一章钩子|结尾结果|下一章钩子)\s*[:：]",
    re.MULTILINE,
)


def _outline_sections(outline: str) -> dict[str, str]:
    text = str(outline or "").strip()
    if "【剧情细纲】" in text:
        text = text.split("【剧情细纲】", 1)[1].strip()
    matches = list(_OUTLINE_SECTION_RE.finditer(text))
    sections = {}
    aliases = {
        "开场": "开场状态",
        "关系变化": "人物关系变化",
    }
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        key = aliases.get(match.group(1), match.group(1))
        sections[key] = text[start:end].strip(" \n。；")
    return sections


def _split_required_events(text: str) -> list[str]:
    parts = re.split(
        r"[；。\n]+|(?:随后|然后|接着|最终|结果|于是)[，,]?",
        str(text or ""),
    )
    events = []
    for part in parts:
        event = re.sub(
            r"^(?:开场|核心冲突|关键行动|人物关系变化|关系变化|结尾结果|下一章钩子)\s*[:：]\s*",
            "",
            part,
        ).strip(" ，,；;。")
        if len(event) >= 6 and event not in events:
            events.append(event[:180])
    return events[:8]


def build_chapter_contracts(outlines: dict) -> dict[str, dict]:
    outlines = outlines if isinstance(outlines, dict) else {}
    contracts = {}
    total = len(outlines)
    for key, outline in outlines.items():
        sections = _outline_sections(outline)
        ending_state = sections.get("结尾结果", "")
        handoff = sections.get("下一章钩子", "")
        combined_ending = sections.get("结尾结果与下一章钩子", "")
        if combined_ending and not ending_state:
            ending_state = combined_ending
        required_events = _split_required_events(sections.get("关键行动", ""))
        if not required_events:
            required_events = _split_required_events(sections.get("核心冲突", ""))
        if ending_state:
            ending_event = f"本章结束时：{ending_state}"[:180]
            if ending_event not in required_events:
                required_events.append(ending_event)
        relationship_change = sections.get("人物关系变化", "")
        if relationship_change:
            relationship_event = f"关系变化：{relationship_change}"[:180]
            if relationship_event not in required_events:
                required_events.append(relationship_event)
        if not required_events:
            story_text = str(outline).split("【剧情细纲】", 1)[-1]
            required_events = _split_required_events(story_text)
        contracts[str(key)] = {
            "chapter": int(key) if str(key).isdigit() else key,
            "is_final": str(key).isdigit() and int(key) >= total,
            "opening_state": sections.get("开场状态", ""),
            "core_conflict": sections.get("核心冲突", ""),
            "required_events": required_events,
            "relationship_change": relationship_change,
            "facts_and_foreshadowing": (
                sections.get("重要信息或伏笔", "") or sections.get("伏笔", "")
            ),
            "ending_state": ending_state,
            "next_handoff": "" if str(key).isdigit() and int(key) >= total else handoff,
            "source_outline": str(outline),
        }
    return contracts


def build_finale_contract(
    chapter_contracts: dict, pattern_manifest: dict | None = None
) -> dict:
    if not chapter_contracts:
        return {}
    final_key = max(chapter_contracts, key=lambda key: int(key) if str(key).isdigit() else 0)
    contract = dict(chapter_contracts[final_key])
    return {
        "chapter": contract.get("chapter"),
        "required_finale_events": contract.get("required_events", []),
        "required_resolution": contract.get("ending_state", ""),
        "relationship_resolution": contract.get("relationship_change", ""),
        "selected_ending": (pattern_manifest or {}).get("ending", ""),
        "must_resolve_main_conflict": True,
        "must_signal_story_end": True,
        "must_not_create_new_main_arc": True,
    }


def format_contract(contract: dict) -> str:
    if not contract:
        return "未提供结构化契约，以本章细纲为准。"
    return json.dumps(contract, ensure_ascii=False, indent=2)


IMMUTABLE_FACT_PRIORITIES = {
    "death": 100,
    "reproductive": 100,
    "accident": 95,
    "legal_financial": 90,
    "evidence": 90,
    "irreversible_relationship": 85,
    "identity": 80,
    "history": 70,
}


class LedgerMergeError(RuntimeError):
    pass


def empty_story_ledger() -> dict:
    return {
        "version": 1,
        "immutable_facts": [],
        "current_states": {},
        "foreshadowing": [],
        "last_chapter_ending": "",
        "next_handoff": "",
        "last_updated_chapter": 0,
    }


def normalize_story_ledger(value) -> dict:
    raw = value if isinstance(value, dict) else {}
    ledger = empty_story_ledger()
    ledger["version"] = max(1, int(raw.get("version", 1)))
    ledger["immutable_facts"] = [
        dict(item)
        for item in raw.get("immutable_facts", [])
        if isinstance(item, dict) and item.get("statement")
    ]
    states = raw.get("current_states", {})
    if isinstance(states, list):
        states = {
            str(item.get("state_key", "")): dict(item)
            for item in states
            if isinstance(item, dict) and item.get("state_key")
        }
    ledger["current_states"] = {
        str(key): dict(item)
        for key, item in (states.items() if isinstance(states, dict) else [])
        if isinstance(item, dict)
    }
    ledger["foreshadowing"] = [
        dict(item)
        for item in raw.get("foreshadowing", [])
        if isinstance(item, dict) and item.get("description")
    ]
    ledger["last_chapter_ending"] = str(raw.get("last_chapter_ending", ""))
    ledger["next_handoff"] = str(raw.get("next_handoff", ""))
    ledger["last_updated_chapter"] = max(
        0, int(raw.get("last_updated_chapter", 0))
    )
    return ledger


def _normalized_fact_text(value: str) -> str:
    text = str(value or "").lower()
    return "".join(
        char
        for char in text
        if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    )


def _ledger_relevance_terms(*values) -> set[str]:
    text = " ".join(
        json.dumps(value, ensure_ascii=False)
        if isinstance(value, (dict, list))
        else str(value or "")
        for value in values
    )
    return {
        token
        for token in re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z0-9_]{3,}", text)
        if token
    }


def _item_relevance(item: dict, terms: set[str]) -> int:
    haystack = " ".join(
        str(item.get(key, ""))
        for key in (
            "subject",
            "statement",
            "value",
            "description",
            "source_evidence",
            "state_key",
        )
    )
    keywords = item.get("keywords", [])
    if isinstance(keywords, list):
        haystack += " " + " ".join(map(str, keywords))
    return sum(1 for term in terms if term in haystack)


def _fit_ledger_lines(lines: list[str], budget: int) -> str:
    selected = []
    used = 0
    for line in lines:
        remaining = budget - used
        if remaining <= 0:
            break
        clean = str(line).strip()
        if not clean:
            continue
        if len(clean) > remaining:
            clean = clean[:remaining]
        selected.append(clean)
        used += len(clean)
    return "\n".join(selected)


def render_story_ledger(
    ledger_value,
    current_contract: dict | None = None,
    next_contract: dict | None = None,
    draft: str = "",
    max_chars: int = LEDGER_CONTEXT_CHARS,
) -> str:
    ledger = normalize_story_ledger(ledger_value)
    if not ledger["immutable_facts"] and not ledger["current_states"]:
        return "暂无已入库历史事实。"

    terms = _ledger_relevance_terms(current_contract or {}, next_contract or {}, draft)
    immutable = sorted(
        ledger["immutable_facts"],
        key=lambda item: (
            _item_relevance(item, terms),
            IMMUTABLE_FACT_PRIORITIES.get(item.get("category", "history"), 50),
            int(item.get("chapter", 0)),
        ),
        reverse=True,
    )
    states = sorted(
        ledger["current_states"].values(),
        key=lambda item: (
            _item_relevance(item, terms),
            int(item.get("chapter", 0)),
        ),
        reverse=True,
    )
    foreshadowing = sorted(
        (
            item
            for item in ledger["foreshadowing"]
            if item.get("status", "open") == "open"
        ),
        key=lambda item: (
            _item_relevance(item, terms),
            int(item.get("chapter", 0)),
        ),
        reverse=True,
    )

    content_budget = max(200, max_chars - 100)
    immutable_budget = int(content_budget * 0.50)
    state_budget = int(content_budget * 0.25)
    foreshadow_budget = int(content_budget * 0.15)
    handoff_budget = (
        content_budget - immutable_budget - state_budget - foreshadow_budget
    )

    immutable_text = _fit_ledger_lines(
        [
            f"- [{item.get('id', 'F-?')}] {item.get('statement', '')}"
            for item in immutable
        ],
        immutable_budget,
    )
    state_text = _fit_ledger_lines(
        [
            f"- [{item.get('state_key', key)}] {item.get('subject', '')}"
            f"{item.get('category', '')}：{item.get('value', '')}"
            for key, item in (
                (item.get("state_key", ""), item) for item in states
            )
        ],
        state_budget,
    )
    foreshadow_text = _fit_ledger_lines(
        [
            f"- [{item.get('id', 'P-?')}] {item.get('description', '')}"
            for item in foreshadowing
        ],
        foreshadow_budget,
    )
    handoff_text = _fit_ledger_lines(
        [
            f"- 上一章结尾：{ledger.get('last_chapter_ending', '')}",
            f"- 当前交接目标：{ledger.get('next_handoff', '')}",
        ],
        handoff_budget,
    )
    rendered = (
        "【不可变历史事实】\n"
        f"{immutable_text or '- 无'}\n"
        "【当前状态】\n"
        f"{state_text or '- 无'}\n"
        "【未解决伏笔】\n"
        f"{foreshadow_text or '- 无'}\n"
        "【章节交接】\n"
        f"{handoff_text or '- 无'}"
    )
    return rendered[:max_chars]


def format_story_ledger(state: NovelState) -> str:
    chapter = state.get("current_chapter", 1)
    contracts = state.get("chapter_contracts", {})
    return render_story_ledger(
        state.get("story_ledger", {}),
        contracts.get(str(chapter), {}),
        contracts.get(str(chapter + 1), {}),
        state.get("current_draft", ""),
    )

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


def _manifest_labels(pattern: dict) -> dict:
    labels = dict(pattern.get("manifest_labels", {}))
    return {
        "protagonist": labels.get("protagonist", "女主"),
        "counterpart": labels.get("counterpart", "男主"),
        "foil": labels.get("foil", "女配"),
        "conflict": labels.get("conflict", "虐点"),
        "ending": labels.get("ending", "结局"),
    }


def _choice_from_pool(pattern: dict, rng: random.Random, key: str, legacy_key: str, fallback: str) -> str:
    pool = pattern.get(key) or pattern.get(legacy_key) or [fallback]
    return rng.choice(pool)


def _pattern_from_manifest(manifest: dict) -> dict:
    return load_story_patterns().get((manifest or {}).get("pattern_key", ""), {})


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
    selected_pool = regular if regular else modules
    selected = rng.sample(selected_pool, min(count, len(selected_pool)))
    if reproductive and count >= 3 and rng.random() < 0.55:
        selected[-1] = rng.choice(reproductive)
    rng.shuffle(selected)

    endings = pattern.get("ending_options", {})
    if ending not in endings:
        ending = next(iter(endings), "default")
    labels = _manifest_labels(pattern)
    protagonist = _choice_from_pool(pattern, rng, "protagonist_pool", "heroine_pool", "被强套路推到选择边界的主角")
    counterpart = _choice_from_pool(pattern, rng, "counterpart_pool", "hero_pool", "制造核心压力的关键关系方")
    foil = _choice_from_pool(pattern, rng, "foil_pool", "rival_pool", "推动误判和冲突升级的对照人物")
    return {
        "pattern_key": pattern_key,
        "pattern_name": pattern.get("name", pattern_key),
        "seed": seed,
        "labels": labels,
        "protagonist": protagonist,
        "counterpart": counterpart,
        "foil": foil,
        "heroine": protagonist,
        "hero": counterpart,
        "rival": foil,
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
    pattern_key = manifest.get("pattern_key")
    pattern = load_story_patterns().get(pattern_key, {})
    labels = _manifest_labels(pattern)
    conflict_label = labels["conflict"]
    conflicts = manifest.get("conflicts", [])
    if not pattern.get("strong"):
        issues.append("套路契约标识不匹配")
    if not isinstance(conflicts, list):
        return issues + [f"{conflict_label}模块必须为列表"]
    if not 2 <= len(conflicts) <= 3:
        issues.append(f"{conflict_label}模块必须为2至3个")
    conflict_ids = [
        item.get("id")
        for item in conflicts
        if isinstance(item, dict) and item.get("id")
    ]
    if len(conflict_ids) != len(conflicts):
        issues.append(f"每个{conflict_label}模块必须包含有效标识")
    elif len(set(conflict_ids)) != len(conflict_ids):
        issues.append(f"{conflict_label}模块不能重复")
    valid_ids = {
        item.get("id")
        for item in pattern.get("conflict_modules", [])
        if isinstance(item, dict) and item.get("id")
    }
    unknown_ids = [item for item in conflict_ids if valid_ids and item not in valid_ids]
    if unknown_ids:
        issues.append(f"{conflict_label}模块不属于当前套路：{', '.join(unknown_ids)}")
    reproductive_count = sum(
        item.get("category") == "reproductive"
        for item in conflicts
        if isinstance(item, dict)
    )
    if reproductive_count > 1:
        issues.append("生育伤害模块最多只能选择一个")
    for field, legacy in (
        ("protagonist", "heroine"),
        ("counterpart", "hero"),
        ("foil", "rival"),
    ):
        if manifest.get(field) in (None, "") and manifest.get(legacy) in (None, ""):
            issues.append(f"套路契约缺少字段：{field}")
    for field in ("background", "ending", "seed"):
        if manifest.get(field) in (None, ""):
            issues.append(f"套路契约缺少字段：{field}")
    try:
        int(manifest.get("seed"))
    except (TypeError, ValueError):
        issues.append("套路随机种子必须为整数")
    ending_options = pattern.get("ending_options", {})
    if ending_options and manifest.get("ending") not in ending_options:
        issues.append(f"结局方向必须为：{', '.join(ending_options)}")
    return issues


def build_pattern_plan(manifest: dict, chapters: int, words_per_chapter: int) -> dict[str, dict]:
    chapters = max(1, int(chapters))
    words_per_chapter = max(1, int(words_per_chapter))
    pattern = _pattern_from_manifest(manifest)
    labels = dict(manifest.get("labels") or _manifest_labels(pattern))
    profile = pattern.get("plan_profile", {})
    conflicts = manifest.get("conflicts", [])
    total_words = chapters * words_per_chapter
    paywall_target_word = max(1, int(total_words * 0.475))
    paywall_chapter = min(chapters, (paywall_target_word - 1) // words_per_chapter + 1)
    beat_map = {
        item.get("id"): item.get("requirement", "")
        for item in pattern.get("beats", [])
        if isinstance(item, dict)
    }

    def beat(name: str, default: str) -> str:
        return beat_map.get(name) or profile.get(f"{name}_event") or default

    ending_events = profile.get("ending_events", {})
    ending_descriptions = pattern.get("ending_options", {})
    is_female_angst = manifest.get("pattern_key") == STRONG_PATTERN_KEY
    plan = {}

    for chapter in range(1, chapters + 1):
        midpoint = (chapter - 0.5) / chapters
        if chapters == 1:
            phase = profile.get("one_chapter_phase", "开篇钩子、核心反转与结局兑现")
            protagonist_state = profile.get(
                "one_chapter_protagonist",
                "从核心压力中完成关键认知变化，并做出最终选择",
            )
            counterpart_state = profile.get(
                "one_chapter_counterpart",
                "制造压力的一方暴露真实代价或被迫承担后果",
            )
            required_event = beat(
                "one_chapter",
                "前300字内爆发核心冲突；随后完成强套路卡点、反转和结局方向兑现",
            )
            if is_female_angst:
                phase = "开篇钩子、最重伤害反转与独立结局"
                protagonist_state = "从受伤隐忍走向心死与独立清醒"
                counterpart_state = "偏信造成伤害后才察觉真相，并承担无法抹平的代价"
                required_event = (
                    "前300字内爆发核心伤害；随后完成最严重伤害与反转，"
                    "最终兑现女主独立和确认的结局方向"
                )
        elif chapter == 1 and chapter == paywall_chapter:
            phase = profile.get("hook_turn_phase", "开篇强钩子与核心反转卡点")
            protagonist_state = profile.get("hook_turn_protagonist", "遭遇核心压力后迅速进入不可回头的选择")
            counterpart_state = profile.get("hook_turn_counterpart", "关键对立力量首次露出破绽或代价")
            required_event = beat(
                "hook_turn",
                "前300字内爆发核心冲突，并在本章完成全书第一次强反转卡点",
            )
            if is_female_angst:
                phase = "开篇爆钩子与最重伤害反转卡点"
                protagonist_state = "受伤隐忍后迅速心死，停止期待"
                counterpart_state = "偏信造成严重伤害后，第一次察觉事情不对"
                required_event = (
                    "前300字内爆发核心伤害，并在本章完成全书最严重伤害与反转，"
                    "让男主第一次动摇"
                )
        elif chapter == 1:
            phase = profile.get("hook_phase", "开篇强钩子")
            protagonist_state = profile.get("hook_protagonist", "被迫进入核心困境，并看见第一条风险线索")
            counterpart_state = profile.get("hook_counterpart", "关键压力尚未完全暴露，但已造成直接后果")
            required_event = beat("hook", "前300字内爆发核心冲突，让强套路主线立刻成立")
            if is_female_angst:
                phase = "开篇爆钩子"
                protagonist_state = "受伤、隐忍，但开始察觉关系失衡"
                counterpart_state = "偏信女配，尚未意识到自己在伤害女主"
                required_event = "前300字内让女主遭受伤害、羞辱或被抛弃，立即爆发核心冲突"
        elif chapter == paywall_chapter:
            phase = profile.get("turn_phase", "45%-50%核心反转卡点")
            protagonist_state = profile.get("turn_protagonist", "在最强压力下完成关键认知转折")
            counterpart_state = profile.get("turn_counterpart", "对立力量或关系方第一次显露不可逆代价")
            required_event = beat("paywall_turn", "发生全书核心反转，并抛出必须追读的证据、规则或局势变化")
            if is_female_angst:
                phase = "最重伤害与反转卡点"
                protagonist_state = "由痛苦转为心死，停止期待"
                counterpart_state = "首次察觉事情不对，但尚未掌握完整真相"
                required_event = "发生全书最严重伤害，并用反转证据或异常让男主第一次动摇"
        elif midpoint < 0.25:
            phase = profile.get("accumulation_phase", "压力叠加")
            protagonist_state = profile.get("accumulation_protagonist", "继续承压，同时积累线索、能力或边界感")
            counterpart_state = profile.get("accumulation_counterpart", "持续扩大误判、规则压迫或局势优势")
            required_event = beat("accumulation", "升级一个已选强套路模块，让主线压力变得更具体")
            if is_female_angst:
                phase = "虐点叠加"
                protagonist_state = "继续隐忍，同时逐渐看清偏爱与不公"
                counterpart_state = "继续偏袒女配，把女主反应误判为矫情或嫉妒"
                required_event = "升级一个已选虐点，并让女配在女主受伤时获得关注或利益"
        elif midpoint < 0.50:
            phase = profile.get("escalation_phase", "压力升级与转折预埋")
            protagonist_state = profile.get("escalation_protagonist", "从被动承压转向主动判断，准备改变策略")
            counterpart_state = profile.get("escalation_counterpart", "继续误判局势，并做出会引发反噬的选择")
            required_event = beat("escalation", "让强套路模块产生不可轻易撤销的后果，逼近核心卡点")
            if is_female_angst:
                phase = "伤害升级与心死"
                protagonist_state = "痛苦逐步耗尽，准备停止解释"
                counterpart_state = "仍被信息差蒙蔽，做出会后悔的选择"
                required_event = "让伤害产生不可轻易撤销的后果，推进女主心死"
        elif chapter < chapters and midpoint < 0.80:
            phase = profile.get("truth_phase", "真相揭露与反击代价")
            protagonist_state = profile.get("truth_protagonist", "掌握更多真相，开始主动重排局势")
            counterpart_state = profile.get("truth_counterpart", "逐步付出代价，但仍无法完全修复前因")
            required_event = beat("truth_regret", "揭露一层关键真相，让既有选择产生新的代价")
            if is_female_angst:
                phase = "真相揭露与追悔"
                protagonist_state = "平静坚定，拒绝解释、补偿和回头"
                counterpart_state = "逐步看见真相，追悔并付出实际代价"
                required_event = "揭露一层真相，让男主的补偿失败并承担新的损失"
        else:
            phase = profile.get("ending_phase", "终局兑现")
            protagonist_state = profile.get("ending_protagonist", "主动掌控结局，并兑现新的身份、秩序或边界")
            counterpart_state = profile.get("ending_counterpart", "承担长期后果，旧秩序无法轻易复原")
            required_event = ending_events.get(
                manifest.get("ending"),
                f"兑现结局方向：{ending_descriptions.get(manifest.get('ending'), manifest.get('ending_description', '完成强套路结局'))}",
            )
            if is_female_angst:
                phase = "独立离开与结局"
                protagonist_state = "独立清醒，主动选择自己的新生活"
                counterpart_state = "承担长期后果，无法用道歉抹平伤害"
                if manifest.get("ending") == "costly_reunion":
                    required_event = (
                        "先完成女主独立与边界建立；男主付出长期且不可逆的代价后，"
                        "由女主自主决定是否重新开始"
                    )
                else:
                    required_event = "完成女主独立离开与更好生活，男主追悔但无法挽回，明确不复合"

        module = conflicts[(chapter - 1) % len(conflicts)] if conflicts else {}
        conflict_label = labels.get("conflict", "强套路")
        if module.get("category") == "reproductive" and midpoint < 0.25:
            conflict_stage = "只铺垫风险、伤病或信息差，禁止提前发生实际生育伤害"
        elif chapter < paywall_chapter:
            conflict_stage = profile.get("before_turn_conflict_stage", "建立前因并升级过程，不得无因果突然发生")
        elif chapter == paywall_chapter:
            conflict_stage = profile.get("turn_conflict_stage", "让既有前因造成核心反转或不可轻易撤销的后果")
        elif midpoint < 0.80:
            conflict_stage = profile.get("after_turn_conflict_stage", f"通过真相揭露呈现该{conflict_label}模块的后果与责任")
        else:
            conflict_stage = profile.get("ending_conflict_stage", f"兑现长期后果，禁止重复制造同一种{conflict_label}")
        if chapter == chapters:
            relationship_change = ending_descriptions.get(manifest.get("ending"), manifest.get("ending_description", "完成强套路结局"))
            if is_female_angst:
                relationship_change = (
                    "女主保持独立边界，在男主付出高代价后自主决定是否重新开始"
                    if manifest.get("ending") == "costly_reunion"
                    else "女主彻底离开且不复合，男主承担长期后果"
                )
        elif chapter == paywall_chapter:
            relationship_change = profile.get("turn_relationship_change", "核心关系、规则或局势发生不可逆变化")
            if is_female_angst:
                relationship_change = "女主彻底心死，男主第一次动摇但仍未掌握完整真相"
        else:
            relationship_change = profile.get("relationship_change", "本章结束时关系、规则或局势必须发生可观察的变化")
        plan[str(chapter)] = {
            "phase": phase,
            "labels": labels,
            "progress_range": f"{int((chapter - 1) / chapters * 100)}%-{int(chapter / chapters * 100)}%",
            "word_range": f"{(chapter - 1) * words_per_chapter + 1}-{chapter * words_per_chapter}",
            "protagonist_state": protagonist_state,
            "counterpart_state": counterpart_state,
            "heroine_state": protagonist_state,
            "hero_awareness": counterpart_state,
            "required_event": required_event,
            "conflict_module": module.get("name", "按人物因果推进"),
            "conflict_stage": conflict_stage,
            "relationship_change": relationship_change,
            "ending_hook": profile.get("ending_hook", "用新的后果、证据、规则或局势变化形成下一章钩子"),
            "is_paywall_turn": chapter == paywall_chapter,
            "paywall_target_word": paywall_target_word if chapter == paywall_chapter else None,
            "total_words": total_words,
            "is_final": chapter == chapters,
        }
    return plan


def format_pattern_manifest(manifest: dict) -> str:
    if not manifest:
        return "无结构化套路契约"
    labels = dict(manifest.get("labels") or _manifest_labels(_pattern_from_manifest(manifest)))
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
        f"{labels.get('protagonist', '主角')}：{manifest.get('protagonist') or manifest.get('heroine', '')}\n"
        f"{labels.get('counterpart', '关系方')}：{manifest.get('counterpart') or manifest.get('hero', '')}\n"
        f"{labels.get('foil', '对照方')}：{manifest.get('foil') or manifest.get('rival', '')}\n"
        f"选定{labels.get('conflict', '模块')}：{conflicts}\n"
        f"{labels.get('ending', '结局')}：{manifest.get('ending_description', '')}\n"
        f"随机种子：{manifest.get('seed', '')}"
    )


def format_pattern_chapter_task(task: dict) -> str:
    if not task:
        return "无结构化章节套路任务"
    labels = task.get("labels", {})
    return "\n".join(
        f"{label}：{task.get(key, '')}"
        for label, key in (
            ("套路阶段", "phase"),
            ("全书进度", "progress_range"),
            (f"{labels.get('protagonist', '主角')}状态", "protagonist_state"),
            (f"{labels.get('counterpart', '关系方')}状态", "counterpart_state"),
            ("本章必须事件", "required_event"),
            (f"本章{labels.get('conflict', '强套路')}模块", "conflict_module"),
            (f"{labels.get('conflict', '强套路')}执行阶段", "conflict_stage"),
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
    pattern = _pattern_from_manifest(manifest)
    expected = {str(index) for index in range(1, max(1, int(target_chapters)) + 1)}
    if set(pattern_plan) != expected:
        issues.append("逐章节拍计划与目标章节数不一致")
    paywall = [
        int(key)
        for key, task in pattern_plan.items()
        if isinstance(task, dict) and task.get("is_paywall_turn")
    ]
    if len(paywall) != 1:
        issues.append("必须且只能有一个45%-50%的核心反转卡点")
    elif not pattern_plan[str(paywall[0])].get("paywall_target_word"):
        issues.append("核心反转卡点缺少全书字数目标")
    else:
        paywall_task = pattern_plan[str(paywall[0])]
        ratio = paywall_task["paywall_target_word"] / max(1, paywall_task.get("total_words", 0))
        if not 0.45 <= ratio <= 0.50:
            issues.append("核心反转卡点必须位于全书累计字数45%-50%")
    if manifest.get("pattern_key") == STRONG_PATTERN_KEY and manifest.get("ending") == "no_reunion":
        final_task = pattern_plan.get(str(max(1, int(target_chapters))), {})
        if "独立" not in final_task.get("phase", ""):
            issues.append("默认不复合结局必须以女主独立离开收束")
    ending_options = pattern.get("ending_options", {})
    if ending_options and manifest.get("pattern_key") != STRONG_PATTERN_KEY:
        final_task = pattern_plan.get(str(max(1, int(target_chapters))), {})
        ending_description = ending_options.get(manifest.get("ending"), "")
        if ending_description and ending_description not in final_task.get("relationship_change", ""):
            issues.append("最终章必须兑现已确认的结局方向")
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
    if manifest.get("pattern_key") != STRONG_PATTERN_KEY:
        all_outlines = "".join(outlines.values())
        labels = dict(manifest.get("labels") or _manifest_labels(_pattern_from_manifest(manifest)))
        for conflict in manifest.get("conflicts", []):
            name = conflict.get("name", "") if isinstance(conflict, dict) else str(conflict)
            if name and name not in all_outlines:
                issues.append(f"细纲未安排已确认{labels.get('conflict', '强套路')}模块：{name}")
        return issues

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
llm_summarizer = _create_llm(0)

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
    大纲完成度: int = Field(default=100, description="本章契约事件兑现比例，0到100。")
    连续性评分: int = Field(default=100, description="与连续性台账一致程度，0到100。")
    衔接评分: int = Field(default=100, description="结束状态与下一章交接要求匹配程度，0到100。")
    已完成事件: list[str] = Field(default_factory=list, description="正文已经兑现的契约事件。")
    未完成事件: list[str] = Field(default_factory=list, description="正文尚未兑现的契约事件。")
    阻断问题: list[str] = Field(default_factory=list, description="会污染后续剧情的致命问题。")
    结局完整性: bool = Field(default=True, description="最终章是否形成完整结局；非最终章填true。")
    结局问题: list[str] = Field(default_factory=list, description="最终章未收束的终局问题。")

class EditorReport(BaseModel):
    文风评分: int = Field(description="给出1-10的评分，7分及格。")
    AI痕迹问题: list[str] = Field(default_factory=list, description="可定位的具体AI写作痕迹。")
    改进建议: str = Field(description="关于遣词造句、剧情节奏的润色建议。")


class ScenePlan(BaseModel):
    scenes: list[str] = Field(description="按顺序排列的场景执行计划。")
    coverage: dict[str, str] = Field(description="每个契约事件由哪个场景完成。")
    ending_strategy: str = Field(description="如何到达契约规定的章节结束状态。")


class LedgerFact(BaseModel):
    id: str = ""
    fact_key: str
    chapter: int = 0
    category: str
    subject: str
    statement: str
    source_evidence: str
    keywords: list[str] = Field(default_factory=list)


class LedgerState(BaseModel):
    state_key: str
    chapter: int = 0
    category: str
    subject: str
    value: str
    source_evidence: str


class LedgerThread(BaseModel):
    id: str = ""
    thread_key: str
    chapter: int = 0
    description: str
    status: str = "open"
    source_evidence: str = ""
    resolved_chapter: int = 0


class StoryLedger(BaseModel):
    version: int = 1
    immutable_facts: list[LedgerFact] = Field(default_factory=list)
    current_states: dict[str, LedgerState] = Field(default_factory=dict)
    foreshadowing: list[LedgerThread] = Field(default_factory=list)
    last_chapter_ending: str = ""
    next_handoff: str = ""
    last_updated_chapter: int = 0


class NewLedgerFact(BaseModel):
    fact_key: str = Field(description="稳定事实键；同一历史事实后续必须复用相同键。")
    category: str = Field(description="death/reproductive/accident/legal_financial/evidence/irreversible_relationship/identity/history")
    subject: str
    statement: str
    source_evidence: str
    keywords: list[str] = Field(default_factory=list)


class LedgerStateUpdate(BaseModel):
    state_key: str = Field(description="稳定状态键，如 location:沈念、injury:陆廷烨左腿。")
    category: str = Field(description="location/injury/possession/relationship/knowledge/role")
    subject: str
    value: str
    source_evidence: str


class NewLedgerThread(BaseModel):
    thread_key: str
    description: str
    source_evidence: str


class LedgerDelta(BaseModel):
    new_immutable_facts: list[NewLedgerFact] = Field(default_factory=list)
    state_updates: list[LedgerStateUpdate] = Field(default_factory=list)
    new_foreshadowing: list[NewLedgerThread] = Field(default_factory=list)
    resolved_foreshadowing_ids: list[str] = Field(default_factory=list)
    chapter_ending: str = ""
    next_handoff: str = ""


class ContinuityConflict(BaseModel):
    fact_id: str
    established_fact: str
    draft_claim: str
    draft_evidence: str
    repair_instruction: str


class ContinuityReport(BaseModel):
    status: str = Field(description="严格输出 pass 或 fail。")
    conflicts: list[ContinuityConflict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ContinuityReview(BaseModel):
    # DeepSeek 对复杂嵌套 JSON schema 的遵循不稳定，因此接收宽松的扁平结构，
    # 再由程序确定性转换为内部 LedgerDelta / ContinuityReport。
    new_immutable_facts: list[dict] = Field(default_factory=list)
    state_updates: list[dict] = Field(default_factory=list)
    new_foreshadowing: list[dict] = Field(default_factory=list)
    resolved_foreshadowing_ids: list[str] = Field(default_factory=list)
    chapter_ending: str | dict = ""
    next_handoff: str | dict = ""
    conflicts: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    status: str = "pass"
    ledger_delta: dict = Field(default_factory=dict)
    continuity_report: dict = Field(default_factory=dict)


llm_architect_structured = llm_architect.with_structured_output(ArchitectOutput, method="json_mode")
llm_auditor_structured = llm_auditor_raw.with_structured_output(AuditReport, method="json_mode")
llm_editor_structured = llm_editor.with_structured_output(EditorReport, method="json_mode")
llm_scene_planner_structured = llm_writer.with_structured_output(ScenePlan, method="json_mode")
llm_continuity_structured = llm_summarizer.with_structured_output(
    ContinuityReview, method="json_mode"
)


def _normalize_issue_list(value) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item or "").strip()]


def normalize_audit_report(report: dict) -> dict:
    normalized = dict(report or {})
    for field in (
        "发现的问题",
        "警告",
        "套路问题",
        "已完成事件",
        "未完成事件",
        "阻断问题",
        "结局问题",
    ):
        normalized[field] = _normalize_issue_list(normalized.get(field, []))
    normalized["套路执行状态"] = "不通过" if normalized["套路问题"] else "通过"
    normalized["审核状态"] = (
        "不通过"
        if (
            normalized["发现的问题"]
            or normalized["套路问题"]
            or normalized["未完成事件"]
            or normalized["阻断问题"]
            or normalized["结局问题"]
        )
        else "通过"
    )
    normalized["修改建议"] = str(normalized.get("修改建议", "无"))
    for field in ("大纲完成度", "连续性评分", "衔接评分"):
        normalized[field] = max(0, min(100, int(normalized.get(field, 100))))
    normalized["结局完整性"] = bool(normalized.get("结局完整性", True))
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


def _compact_json_text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if value in (None, {}, []):
        return ""
    return json.dumps(value, ensure_ascii=False, separators=("，", "："))


def _infer_fact_category(fact_key: str, statement: str) -> str:
    text = f"{fact_key} {statement}".lower()
    if any(word in text for word in ("死亡", "死去", "复活", "death")):
        return "death"
    if any(word in text for word in ("怀孕", "流产", "孩子", "生育", "reproductive")):
        return "reproductive"
    if any(word in text for word in ("事故", "摔倒", "车祸", "失踪", "accident")):
        return "accident"
    if any(word in text for word in ("资金", "转账", "财务", "信托", "犯罪", "法律", "legal", "financial")):
        return "legal_financial"
    if any(word in text for word in ("证据", "录音", "监控", "文件", "evidence")):
        return "evidence"
    if any(word in text for word in ("分手", "离婚", "不复合", "决裂", "relationship")):
        return "irreversible_relationship"
    if any(word in text for word in ("身份", "身世", "重生", "identity")):
        return "identity"
    return "history"


def _normalize_fact_payload(raw: dict) -> dict:
    fact_key = str(
        raw.get("fact_key") or raw.get("key") or raw.get("id") or ""
    ).strip()
    statement = str(
        raw.get("statement") or raw.get("description") or raw.get("fact") or ""
    ).strip()
    source_evidence = _compact_json_text(
        raw.get("source_evidence") or raw.get("source") or raw.get("evidence")
    )
    subject = str(raw.get("subject") or raw.get("entity") or fact_key).strip()
    category = str(raw.get("category") or "").strip()
    if not category:
        category = _infer_fact_category(fact_key, statement)
    return {
        "fact_key": fact_key,
        "category": category,
        "subject": subject,
        "statement": statement,
        "source_evidence": source_evidence,
        "keywords": _normalize_issue_list(raw.get("keywords", [])),
    }


def _expand_state_payload(raw: dict) -> list[dict]:
    if raw.get("state_key") and raw.get("value") not in (None, ""):
        return [{
            "state_key": str(raw.get("state_key")).strip(),
            "category": str(raw.get("category", "")).strip(),
            "subject": str(raw.get("subject") or raw.get("entity") or "").strip(),
            "value": _compact_json_text(raw.get("value")),
            "source_evidence": _compact_json_text(
                raw.get("source_evidence") or raw.get("source")
            ),
        }]

    entity = str(raw.get("entity") or raw.get("subject") or "未知角色").strip()
    source = _compact_json_text(
        raw.get("source_evidence") or raw.get("source") or "当前章节正文"
    )
    field_categories = {
        "location": "location",
        "physical_state": "injury",
        "injury": "injury",
        "possessions": "possession",
        "relationships": "relationship",
        "awareness": "knowledge",
        "knowledge": "knowledge",
        "role": "role",
        "status": "status",
        "attitude": "relationship",
        "speech": "knowledge",
    }
    updates = []
    for field, value in raw.items():
        if field in {"entity", "subject", "source", "source_evidence"}:
            continue
        if value in (None, "", [], {}):
            continue
        category = field_categories.get(field, "status")
        if field == "relationships" and isinstance(value, dict):
            for target, relationship in value.items():
                updates.append({
                    "state_key": f"relationship:{entity}-{target}",
                    "category": "relationship",
                    "subject": f"{entity}-{target}",
                    "value": _compact_json_text(relationship),
                    "source_evidence": source,
                })
            continue
        updates.append({
            "state_key": f"{category}:{entity}:{field}",
            "category": category,
            "subject": entity,
            "value": _compact_json_text(value),
            "source_evidence": source,
        })
    return updates


def _normalize_thread_payload(raw: dict) -> dict:
    thread_key = str(
        raw.get("thread_key") or raw.get("id") or raw.get("key") or ""
    ).strip()
    return {
        "thread_key": thread_key,
        "description": str(
            raw.get("description") or raw.get("thread") or ""
        ).strip(),
        "source_evidence": _compact_json_text(
            raw.get("source_evidence") or raw.get("source")
        ),
    }


def normalize_ledger_delta(delta: dict | None) -> dict:
    raw = dict(delta or {})
    facts = [
        _normalize_fact_payload(item)
        for item in raw.get("new_immutable_facts", [])
        if isinstance(item, dict)
    ]
    facts = [
        item for item in facts if item["fact_key"] and item["statement"]
    ]
    states = []
    for item in raw.get("state_updates", []):
        if isinstance(item, dict):
            states.extend(_expand_state_payload(item))
    threads = [
        _normalize_thread_payload(item)
        for item in raw.get("new_foreshadowing", [])
        if isinstance(item, dict)
    ]
    threads = [
        item for item in threads if item["thread_key"] and item["description"]
    ]
    resolved = raw.get("resolved_foreshadowing_ids", [])
    return {
        "new_immutable_facts": facts,
        "state_updates": states,
        "new_foreshadowing": threads,
        "resolved_foreshadowing_ids": (
            list(map(str, resolved)) if isinstance(resolved, list) else []
        ),
        "chapter_ending": _compact_json_text(raw.get("chapter_ending")),
        "next_handoff": _compact_json_text(raw.get("next_handoff")),
    }


def normalize_continuity_report(report: dict | None) -> dict:
    normalized = dict(report or {})
    conflicts = [
        dict(item)
        for item in normalized.get("conflicts", [])
        if isinstance(item, dict)
    ]
    normalized["conflicts"] = conflicts
    normalized["warnings"] = _normalize_issue_list(normalized.get("warnings", []))
    normalized["status"] = "fail" if conflicts else "pass"
    return normalized


def continuity_conflict_messages(report: dict | None) -> list[str]:
    normalized = normalize_continuity_report(report)
    messages = []
    for conflict in normalized["conflicts"]:
        fact_id = conflict.get("fact_id", "未知事实")
        established = conflict.get("established_fact", "")
        draft_claim = conflict.get("draft_claim", "")
        messages.append(
            f"{fact_id}规定“{established}”，当前稿写成“{draft_claim}”"
        )
    return messages


def has_continuity_conflict(report: dict | None) -> bool:
    normalized = normalize_continuity_report(report)
    return normalized["status"] != "pass" or bool(normalized["conflicts"])


def merge_story_ledger(
    ledger_value,
    delta_value,
    chapter_num: int,
    continuity_report: dict | None = None,
) -> dict:
    report = normalize_continuity_report(continuity_report)
    if report["status"] != "pass" or report["conflicts"]:
        raise LedgerMergeError(
            "连续性审核未通过：" + "；".join(continuity_conflict_messages(report))
        )

    ledger = normalize_story_ledger(ledger_value)
    delta = normalize_ledger_delta(delta_value)
    facts = list(ledger["immutable_facts"])
    by_key = {
        str(item.get("fact_key", "")).strip(): item
        for item in facts
        if str(item.get("fact_key", "")).strip()
    }
    by_statement = {
        _normalized_fact_text(item.get("statement", "")): item
        for item in facts
        if _normalized_fact_text(item.get("statement", ""))
    }

    next_fact_index = 1 + max(
        (
            int(match.group(1))
            for item in facts
            if (match := re.search(r"-(\d+)$", str(item.get("id", ""))))
            and int(item.get("chapter", 0)) == int(chapter_num)
        ),
        default=0,
    )
    for raw_fact in delta["new_immutable_facts"]:
        if not isinstance(raw_fact, dict):
            continue
        fact = dict(raw_fact)
        fact_key = str(fact.get("fact_key", "")).strip()
        statement = str(fact.get("statement", "")).strip()
        subject = str(fact.get("subject", "")).strip()
        category = str(fact.get("category", "history")).strip() or "history"
        if not fact_key or not statement:
            continue
        existing = by_key.get(fact_key)
        if existing:
            if _normalized_fact_text(existing.get("statement", "")) == _normalized_fact_text(statement):
                continue
            raise LedgerMergeError(
                f"{existing.get('id', fact_key)}不可变事实冲突："
                f"已入库“{existing.get('statement', '')}”，新增“{statement}”"
            )
        normalized_statement = _normalized_fact_text(statement)
        if normalized_statement in by_statement:
            continue
        fact.update({
            "id": f"F-C{chapter_num}-{next_fact_index:02d}",
            "fact_key": fact_key,
            "chapter": int(chapter_num),
            "category": category,
            "subject": subject,
            "statement": statement,
            "source_evidence": str(fact.get("source_evidence", "")).strip(),
            "keywords": _normalize_issue_list(fact.get("keywords", [])),
        })
        facts.append(fact)
        by_key[fact_key] = fact
        by_statement[normalized_statement] = fact
        next_fact_index += 1

    states = dict(ledger["current_states"])
    for raw_state in delta["state_updates"]:
        if not isinstance(raw_state, dict):
            continue
        state_key = str(raw_state.get("state_key", "")).strip()
        value = str(raw_state.get("value", "")).strip()
        if not state_key or not value:
            continue
        states[state_key] = {
            "state_key": state_key,
            "chapter": int(chapter_num),
            "category": str(raw_state.get("category", "")).strip(),
            "subject": str(raw_state.get("subject", "")).strip(),
            "value": value,
            "source_evidence": str(raw_state.get("source_evidence", "")).strip(),
        }

    threads = list(ledger["foreshadowing"])
    by_thread_key = {
        str(item.get("thread_key", "")).strip(): item
        for item in threads
        if str(item.get("thread_key", "")).strip()
    }
    next_thread_index = 1 + max(
        (
            int(match.group(1))
            for item in threads
            if (match := re.search(r"-(\d+)$", str(item.get("id", ""))))
            and int(item.get("chapter", 0)) == int(chapter_num)
        ),
        default=0,
    )
    for raw_thread in delta["new_foreshadowing"]:
        if not isinstance(raw_thread, dict):
            continue
        thread_key = str(raw_thread.get("thread_key", "")).strip()
        description = str(raw_thread.get("description", "")).strip()
        if not thread_key or not description or thread_key in by_thread_key:
            continue
        thread = {
            "id": f"P-C{chapter_num}-{next_thread_index:02d}",
            "thread_key": thread_key,
            "chapter": int(chapter_num),
            "description": description,
            "status": "open",
            "source_evidence": str(raw_thread.get("source_evidence", "")).strip(),
            "resolved_chapter": 0,
        }
        threads.append(thread)
        by_thread_key[thread_key] = thread
        next_thread_index += 1

    resolved_ids = set(map(str, delta["resolved_foreshadowing_ids"]))
    for thread in threads:
        if str(thread.get("id", "")) in resolved_ids:
            thread["status"] = "resolved"
            thread["resolved_chapter"] = int(chapter_num)

    return {
        "version": max(1, int(ledger.get("version", 1))),
        "immutable_facts": facts,
        "current_states": states,
        "foreshadowing": threads,
        "last_chapter_ending": delta["chapter_ending"],
        "next_handoff": delta["next_handoff"],
        "last_updated_chapter": int(chapter_num),
    }


def apply_deterministic_quality_checks(report: dict, state: NovelState) -> dict:
    normalized = normalize_audit_report(report)
    assessment = chapter_length_assessment(state)
    normalized["警告"] = _normalize_issue_list(
        normalized.get("警告", []) + assessment["warnings"]
    )
    normalized["阻断问题"] = _normalize_issue_list(
        normalized.get("阻断问题", []) + assessment["blocking"]
    )
    if assessment["blocking"]:
        normalized["审核状态"] = "不通过"
    if is_final_chapter(state) and not normalized.get("结局完整性", False):
        normalized["审核状态"] = "不通过"
        if not normalized["结局问题"]:
            normalized["结局问题"] = ["最终章未形成明确、完整的故事收束"]
    return normalize_audit_report(normalized)


def draft_quality_score(
    state: NovelState,
    audit_report: dict,
    editor_report: dict,
    draft: str,
    continuity_report: dict | None = None,
) -> float:
    length_score = chapter_length_assessment(state, draft)["score"]
    score = (
        audit_report.get("大纲完成度", 0) * 0.50
        + audit_report.get("连续性评分", 0) * 0.20
        + audit_report.get("衔接评分", 0) * 0.15
        + editor_report.get("文风评分", 1) * 10 * 0.10
        + length_score * 0.05
    )
    score -= len(audit_report.get("阻断问题", [])) * 25
    score -= len(
        normalize_continuity_report(continuity_report).get("conflicts", [])
    ) * 40
    if is_final_chapter(state) and not audit_report.get("结局完整性", False):
        score -= 30
    return round(max(0, score), 2)


def build_draft_candidate(
    state: NovelState,
    audit_report: dict,
    editor_report: dict,
    ledger_delta: dict,
    continuity_report: dict,
) -> dict:
    draft = state.get("current_draft", "")
    return {
        "chapter": state.get("current_chapter", 1),
        "iteration": state.get("iteration_count", 1),
        "draft": draft,
        "audit_report": audit_report,
        "editor_report": editor_report,
        "ledger_delta": normalize_ledger_delta(ledger_delta),
        "continuity_report": normalize_continuity_report(continuity_report),
        "score": draft_quality_score(
            state,
            audit_report,
            editor_report,
            draft,
            continuity_report,
        ),
        "body_chars": chapter_body_char_count(draft),
    }


def select_best_draft(
    state: NovelState,
    require_complete_finale: bool = False,
    require_continuity_clean: bool = True,
) -> dict:
    chapter = state.get("current_chapter", 1)
    candidates = [
        candidate
        for candidate in state.get("draft_candidates", [])
        if candidate.get("chapter") == chapter and candidate.get("draft")
    ]
    if require_continuity_clean:
        candidates = [
            candidate
            for candidate in candidates
            if not has_continuity_conflict(candidate.get("continuity_report", {}))
        ]
    if require_complete_finale:
        candidates = [
            candidate
            for candidate in candidates
            if (
                candidate.get("audit_report", {}).get("审核状态") == "通过"
                and candidate.get("audit_report", {}).get("结局完整性", False)
                and not candidate.get("audit_report", {}).get("阻断问题", [])
            )
        ]
    return max(candidates, key=lambda item: item.get("score", 0), default={})


def current_draft_is_acceptable(state: NovelState) -> bool:
    audit = state.get("audit_report", {})
    editor = state.get("editor_report", {})
    if audit.get("审核状态") != "通过":
        return False
    if audit.get("阻断问题") or editor.get("文风评分", 0) < STYLE_PASS_SCORE:
        return False
    if has_continuity_conflict(state.get("continuity_report", {})):
        return False
    return not is_final_chapter(state) or audit.get("结局完整性", True)


def route_after_review_decision(state: NovelState) -> str:
    if current_draft_is_acceptable(state):
        return "summarizer"

    iteration = state.get("iteration_count", 1)
    if not is_final_chapter(state):
        if iteration < MAX_REVIEW_ATTEMPTS:
            return "writer"
        if select_best_draft(state, require_continuity_clean=True):
            return "summarizer"
        max_attempts = MAX_REVIEW_ATTEMPTS + MAX_CONTINUITY_REPAIR_ATTEMPTS
        if iteration < max_attempts:
            return "writer"
        raise RuntimeError(
            f"第{state.get('current_chapter', 1)}章连续{max_attempts}稿均存在"
            "不可变事实冲突，已停止入库。"
        )

    if iteration < MAX_FINALE_REVIEW_ATTEMPTS:
        return "writer"
    if select_best_draft(state, require_complete_finale=True):
        return "summarizer"
    raise RuntimeError(
        f"最终章连续{MAX_FINALE_REVIEW_ATTEMPTS}稿仍未形成完整结局，"
        "已停止入库，所有候选稿均保留在运行状态中。"
    )


def invoke_with_retry(chain, inputs, node_name: str, max_attempts: int = APP_INVOKE_ATTEMPTS):
    last_error = None
    attempts_made = 0
    for attempt in range(max_attempts):
        attempts_made = attempt + 1
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
            error_text = str(e)
            logger.warning(
                "   ⚠️ [%s] 模型调用尝试 %d/%d 失败: %s",
                node_name,
                attempt + 1,
                max_attempts,
                e,
            )
            if (
                "Prompt must contain the word 'json'" in error_text
                or "response_format" in error_text
                and "invalid_request_error" in error_text
            ):
                break
        if attempt + 1 < max_attempts:
            delay = min(2 ** attempt, 8)
            logger.info("   ⏳ [%s] %d秒后自动重试...", node_name, delay)
            time.sleep(delay)
    detail = f"{type(last_error).__name__}: {last_error}" if last_error else "模型返回空结果"
    raise RuntimeError(
        f"{node_name}模型调用失败，已尝试{attempts_made}次。"
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
        labels = dict(manifest.get("labels") or _manifest_labels(pattern))
        strong_requirement = (
            "\n【已确认强套路契约】\n"
            f"{format_pattern_manifest(manifest)}\n"
            "【逐章节拍计划】\n"
            f"{json.dumps(pattern_plan, ensure_ascii=False)}\n"
            "每章细纲必须严格服从对应章节任务，不得跳过已确认的核心反转卡点，"
            f"不得让{labels.get('conflict', '强套路')}模块无因果突然发生或被随机素材替代。"
            f"请在细纲中清楚描述开篇钩子、45%-50%核心反转、{labels.get('protagonist', '主角')}状态变化、"
            f"确认的{labels.get('ending', '结局')}方向和已选{labels.get('conflict', '强套路')}模块如何建立前因与后果；"
            "允许使用符合剧情的自然措辞。"
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
    chapter_contracts = build_chapter_contracts(chapter_outlines)
    finale_contract = build_finale_contract(chapter_contracts, manifest)

    # 保存大纲 JSON 到 Outline/ 目录
    try:
        os.makedirs("Outline", exist_ok=True)
        safe_title = _safe_file_stem(result.novel_title, "未命名大纲")
        if state.get("run_id"):
            safe_title = f"{safe_title}_{_safe_file_stem(state.get('run_id'), 'run')}"
        save_path = os.path.join("Outline", f"{safe_title}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "title": result.novel_title,
                "run_id": state.get("run_id", ""),
                "world_bible": result.world_bible,
                "chapter_outlines": chapter_outlines,
                "chapter_contracts": chapter_contracts,
                "finale_contract": finale_contract,
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
        "chapter_contracts": chapter_contracts,
        "finale_contract": finale_contract,
        "pattern_manifest": manifest,
        "pattern_plan": pattern_plan,
        "story_ledger": {},
        "ledger_delta": {},
        "continuity_report": {},
        "draft_candidates": [],
        "current_chapter": 1
    }


def _build_scene_plan(
    state: NovelState, chapter_contract: dict, finale_contract: dict
) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SCENE_PLAN_SYSTEM_PROMPT),
        ("user", """【连续性台账】
{story_ledger}

【本章契约】
{chapter_contract}

【最终章契约】
{finale_contract}

【篇幅规则】
{length_guidance}

【上一稿审核反馈】
{review_feedback}
"""),
    ])
    audit = state.get("audit_report", {})
    review_feedback = (
        f"未完成事件：{audit.get('未完成事件', [])}；"
        f"阻断问题：{audit.get('阻断问题', [])}；"
        f"修改建议：{audit.get('修改建议', '无')}"
    )
    result = _safe_invoke(
        prompt | llm_scene_planner_structured,
        {
            "story_ledger": format_story_ledger(state),
            "chapter_contract": format_contract(chapter_contract),
            "finale_contract": (
                format_contract(finale_contract)
                if is_final_chapter(state)
                else "非最终章，不适用。"
            ),
            "length_guidance": chapter_length_guidance(
                state.get("words_per_chapter", DEFAULT_WORDS_PER_CHAPTER),
                is_final_chapter(state),
            ),
            "review_feedback": review_feedback,
        },
        f"第{state.get('current_chapter', 1)}章场景计划",
    )
    if result is not None:
        return result.model_dump()

    required_events = chapter_contract.get("required_events", [])
    scenes = [f"场景{index + 1}：完成{event}" for index, event in enumerate(required_events)]
    if chapter_contract.get("ending_state"):
        scenes.append(f"收尾场景：到达{chapter_contract['ending_state']}")
    return {
        "scenes": scenes or ["按本章细纲顺序完成全部事件"],
        "coverage": {event: f"场景{index + 1}" for index, event in enumerate(required_events)},
        "ending_strategy": chapter_contract.get("ending_state", "完成本章规定的结束状态"),
    }


def writer_node(state: NovelState):
    chapter_num = state.get("current_chapter", 1)
    iteration = state.get("iteration_count", 0) + 1
    logger.info("✍️ 写手正在奋笔疾书 第 %d 章 (第 %d 稿)...", chapter_num, iteration)
    
    world_bible = state.get("world_bible", "")
    outlines = state.get("chapter_outlines", {})
    current_outline = outlines.get(str(chapter_num), "自由发挥。")
    final_chapter = is_final_chapter(state)
    next_outline = (
        "无下一章。本章必须完成终局冲突、人物命运与情感落点，不得把收束推迟。"
        if final_chapter
        else outlines.get(str(chapter_num + 1), "承接本章结果继续推进主线。")
    )
    words_per = state.get("words_per_chapter", DEFAULT_WORDS_PER_CHAPTER)
    min_words = int(words_per * MIN_CHAPTER_RATIO)
    preferred_words = int(words_per * 0.93)
    style = state.get("writer_style", "default")
    pattern = resolve_story_pattern(state)
    pattern_task = state.get("pattern_plan", {}).get(str(chapter_num), {})
    continuity_state = format_story_ledger(state)
    contracts = state.get("chapter_contracts") or build_chapter_contracts(outlines)
    chapter_contract = contracts.get(str(chapter_num), {})
    finale_contract = state.get("finale_contract") or build_finale_contract(
        contracts, state.get("pattern_manifest", {})
    )
    length_guidance = chapter_length_guidance(words_per, final_chapter)
    
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
    if iteration == 2:
        feedback += "\n\n【第二稿模式】：根据审核反馈完整重写，优先纠正大纲偏离和连续性问题。"
    elif iteration == 3:
        feedback += (
            "\n\n【定向修复模式】：围绕未完成事件和阻断问题重写相关场景，"
            "其余已合格内容保持稳定。"
        )
    elif iteration >= 4:
        feedback += (
            "\n\n【编辑补丁模式】：以上一稿为底稿，保留已完成事件，"
            "补齐缺失事件、修正错误结束状态，并输出修复后的完整章节，不要输出差异说明。"
        )
    if final_chapter and iteration > MAX_REVIEW_ATTEMPTS:
        feedback += (
            "\n\n【最终章强制收束模式】：只处理尚未完成的终局事件与结局问题，"
            "禁止开启新主线；在篇幅硬护栏内完成冲突解决、人物命运和明确结束信号。"
        )
    elif iteration > MAX_REVIEW_ATTEMPTS:
        feedback += (
            "\n\n【纯连续性修复模式】：前四稿均因历史事实冲突无法入库。"
            "本稿只修正下列冲突，不改动已完成的大纲事件、文风和章节结构。"
        )
    if iteration > 1 and previous_draft_chars < chapter_length_limits(words_per, final_chapter)["recommended_min"]:
        feedback += (
            f"\n\n【篇幅补写令】：上一稿正文仅{previous_draft_chars}字，"
            "请只补足契约要求的必要场景、动作、对话和情节推进，不得灌水。"
        )
    if audit_report.get("审核状态") == "不通过":
        feedback += (
            f"\n\n【审计退稿修改令】：未完成事件：{audit_report.get('未完成事件', [])}；"
            f"阻断问题：{audit_report.get('阻断问题', [])}；"
            f"逻辑或大纲问题：{audit_report.get('发现的问题')}；"
            f"套路执行问题：{audit_report.get('套路问题', [])}。"
            f"请按以下建议修复：{audit_report.get('修改建议')}"
        )
    continuity_report = normalize_continuity_report(
        state.get("continuity_report", {})
    )
    if continuity_report["conflicts"]:
        feedback += "\n\n【连续性硬冲突修复令】：\n" + "\n".join(
            f"- {conflict.get('fact_id', '未知事实')}规定"
            f"“{conflict.get('established_fact', '')}”；当前稿写成"
            f"“{conflict.get('draft_claim', '')}”。"
            f"{conflict.get('repair_instruction', '恢复既定事实。')}"
            for conflict in continuity_report["conflicts"]
        )
    if editor_report.get("文风评分", 10) < STYLE_PASS_SCORE:
        feedback += (
            f"\n\n【责编退稿润色令】：文风不达标(当前评分{editor_report.get('文风评分')}/10)。"
            f"只修改以下明确问题：{editor_report.get('AI痕迹问题', [])}。"
            f"改进建议：{editor_report.get('改进建议')}"
        )

    scene_plan = state.get("scene_plan", {})
    if not scene_plan or iteration in (1, 3):
        plan_state = dict(state)
        plan_state["iteration_count"] = iteration
        scene_plan = _build_scene_plan(plan_state, chapter_contract, finale_contract)

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
        "chapter_contract": format_contract(chapter_contract),
        "finale_contract": (
            format_contract(finale_contract) if final_chapter else "非最终章，不适用。"
        ),
        "scene_plan": json.dumps(scene_plan, ensure_ascii=False, indent=2),
        "chapter_type": "最终章" if final_chapter else "普通章节",
        "length_guidance": length_guidance,
        "previous_draft": state.get("current_draft", "") if iteration > 1 else "无",
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
    content = normalize_chapter_output(content, chapter_num)
    
    return {
        "current_draft": content,
        "iteration_count": iteration,
        "scene_plan": scene_plan,
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
        return {"audit_report": apply_deterministic_quality_checks({
            "审核状态": "通过",
            "发现的问题": [],
            "警告": ["审计模型调用失败"],
            "套路执行状态": "通过",
            "套路问题": [],
            "修改建议": "无",
            "结局完整性": not is_final_chapter(state),
        }, state)}
    
    logger.info("🕵️ 审计结果: %s", result.审核状态)
    return {
        "audit_report": apply_deterministic_quality_checks(
            result.model_dump(), state
        )
    }

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
    is_last = is_final_chapter(state)
    draft = state.get("current_draft", "")
    target_words = state.get("words_per_chapter", DEFAULT_WORDS_PER_CHAPTER)
    assessment = chapter_length_assessment(state)
    draft_len = assessment["body_chars"]
    hint = (
        f"当前正文{draft_len}字。{chapter_length_guidance(target_words, is_last)}"
        f"确定性篇幅问题：{assessment['blocking'] or '无'}；"
        f"篇幅提醒：{assessment['warnings'] or '无'}。"
    )
    contracts = state.get("chapter_contracts") or build_chapter_contracts(outlines)
    chapter_contract = contracts.get(str(chapter), {})
    finale_contract = state.get("finale_contract") or build_finale_contract(
        contracts, state.get("pattern_manifest", {})
    )
    
    return {
        "world_bible": state.get("world_bible", ""),
        "continuity_state": format_story_ledger(state),
        "pattern_auditor": pattern.get("auditor", ""),
        "pattern_manifest": format_pattern_manifest(state.get("pattern_manifest", {})),
        "pattern_chapter_task": format_pattern_chapter_task(
            state.get("pattern_plan", {}).get(str(chapter), {})
        ),
        "outline": outlines.get(str(chapter), ""),
        "chapter_contract": format_contract(chapter_contract),
        "finale_contract": (
            format_contract(finale_contract) if is_last else "非最终章，不适用。"
        ),
        "next_outline": outlines.get(
            str(chapter + 1),
            "无下一章。本章就是最终章，必须在本章完成全部收束。",
        ),
        "target_words": target_words,
        "word_count_hint": hint,
        "chapter_type": "末尾结局章节 — 必须检查完整性收束，大纲中的终局事件必须全部兑现" if is_last else "中间章节",
        "draft": draft,
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


def _continuity_inputs(state: NovelState) -> dict:
    chapter = state.get("current_chapter", 1)
    contracts = state.get("chapter_contracts") or build_chapter_contracts(
        state.get("chapter_outlines", {})
    )
    return {
        "chapter_num": chapter,
        "story_ledger": render_story_ledger(
            state.get("story_ledger", {}),
            contracts.get(str(chapter), {}),
            contracts.get(str(chapter + 1), {}),
            state.get("current_draft", ""),
        ),
        "chapter_contract": format_contract(contracts.get(str(chapter), {})),
        "next_contract": format_contract(contracts.get(str(chapter + 1), {})),
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
        fallback = {
            "审核状态": "通过",
            "发现的问题": [],
            "警告": ["审计模型调用失败"],
            "套路执行状态": "通过",
            "套路问题": [],
            "修改建议": "无",
            "大纲完成度": 70,
            "连续性评分": 70,
            "衔接评分": 70,
            "结局完整性": not is_final_chapter(state),
        }
        return {"audit_report": apply_deterministic_quality_checks(fallback, state)}
    return {
        "audit_report": apply_deterministic_quality_checks(
            result.model_dump(), state
        )
    }

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


def _continuity_internal(state: NovelState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("summarizer_system.md")),
        ("user", load_prompt("summarizer_user.md")),
    ])
    result = invoke_with_retry(
        prompt | llm_continuity_structured,
        _continuity_inputs(state),
        f"第{state.get('current_chapter', 1)}章连续性台账",
    )
    payload = result.model_dump()
    delta_payload = payload.get("ledger_delta") or {
        "new_immutable_facts": payload.get("new_immutable_facts", []),
        "state_updates": payload.get("state_updates", []),
        "new_foreshadowing": payload.get("new_foreshadowing", []),
        "resolved_foreshadowing_ids": payload.get(
            "resolved_foreshadowing_ids", []
        ),
        "chapter_ending": payload.get("chapter_ending", ""),
        "next_handoff": payload.get("next_handoff", ""),
    }
    report_payload = payload.get("continuity_report") or {
        "status": payload.get("status", "pass"),
        "conflicts": payload.get("conflicts", []),
        "warnings": payload.get("warnings", []),
    }
    return {
        "ledger_delta": normalize_ledger_delta(delta_payload),
        "continuity_report": normalize_continuity_report(report_payload),
    }


def reviewer_node(state: NovelState):
    logger.info("🔍 审稿员正在进行逻辑+文风+连续性并行三检...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_audit = pool.submit(_auditor_internal, state)
        fut_editor = pool.submit(_editor_internal, state)
        fut_continuity = pool.submit(_continuity_internal, state)
        audit_result = fut_audit.result()
        editor_result = fut_editor.result()
        continuity_result = fut_continuity.result()
    audit_report = audit_result.get("audit_report", {})
    editor_report = editor_result.get("editor_report", {})
    ledger_delta = continuity_result.get("ledger_delta", {})
    continuity_report = continuity_result.get("continuity_report", {})
    conflict_messages = continuity_conflict_messages(continuity_report)
    if conflict_messages:
        audit_report["阻断问题"] = _normalize_issue_list(
            audit_report.get("阻断问题", []) + conflict_messages
        )
        audit_report["连续性评分"] = 0
        audit_report["审核状态"] = "不通过"
    if continuity_report.get("warnings"):
        audit_report["警告"] = _normalize_issue_list(
            audit_report.get("警告", []) + continuity_report["warnings"]
        )
    audit_report = normalize_audit_report(audit_report)
    audit_result = {"audit_report": audit_report}
    candidate = build_draft_candidate(
        state,
        audit_report,
        editor_report,
        ledger_delta,
        continuity_report,
    )
    candidates = [
        item
        for item in state.get("draft_candidates", [])
        if item.get("chapter") == state.get("current_chapter", 1)
    ]
    candidates.append(candidate)
    logger.info(
        "🔍 审计:%s | 连续性:%s | 文风:%d/10",
        audit_report.get("审核状态"),
        continuity_report.get("status", "pass"),
        editor_report.get("文风评分", 0),
    )
    return {
        **audit_result,
        **editor_result,
        **continuity_result,
        "outline_report": {
            "完成度": audit_report.get("大纲完成度", 0),
            "已完成事件": audit_report.get("已完成事件", []),
            "未完成事件": audit_report.get("未完成事件", []),
        },
        "finale_report": {
            "完整": audit_report.get("结局完整性", True),
            "问题": audit_report.get("结局问题", []),
        },
        "draft_candidates": candidates,
    }

def summarizer_node(state: NovelState):
    logger.info("📝 书记员正在验证并合并结构化剧情台账...")
    
    current_chap_num = state.get("current_chapter", 1)
    final_chapter = is_final_chapter(state)
    selected_candidate = {}
    if not current_draft_is_acceptable(state):
        selected_candidate = select_best_draft(
            state,
            require_complete_finale=final_chapter,
            require_continuity_clean=True,
        )
    if not selected_candidate and not current_draft_is_acceptable(state):
        raise LedgerMergeError(
            f"第{current_chap_num}章没有可安全入库的无连续性冲突候选稿。"
        )
    latest_chapter = selected_candidate.get(
        "draft", state.get("current_draft", "")
    )
    selected_audit = selected_candidate.get(
        "audit_report", state.get("audit_report", {})
    )
    selected_editor = selected_candidate.get(
        "editor_report", state.get("editor_report", {})
    )
    selected_delta = selected_candidate.get(
        "ledger_delta", state.get("ledger_delta", {})
    )
    selected_continuity = selected_candidate.get(
        "continuity_report", state.get("continuity_report", {})
    )
    if selected_candidate:
        logger.info(
            "🏆 第 %d 章采用第 %d 稿，综合评分 %.2f",
            current_chap_num,
            selected_candidate.get("iteration", 0),
            selected_candidate.get("score", 0),
        )

    try:
        story_ledger = merge_story_ledger(
            state.get("story_ledger", {}),
            selected_delta,
            current_chap_num,
            selected_continuity,
        )
    except LedgerMergeError as error:
        raise LedgerMergeError(
            f"第{current_chap_num}章台账合并失败，正文未入库：{error}"
        ) from error

    file_path = _build_output_path(state.get("novel_title", "小说输出"), state.get("run_id", ""))
    has_existing_content = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    
    with open(file_path, "a", encoding="utf-8") as f:
        if has_existing_content:
            f.write("\n\n")
        f.write(latest_chapter)
    logger.info("💾 第 %d 章已安全入库 → %s", current_chap_num, file_path)

    selected_state = dict(state)
    selected_state.update({
        "current_draft": latest_chapter,
        "audit_report": selected_audit,
        "editor_report": selected_editor,
    })
    warnings = chapter_quality_warnings(selected_state)
    for warning in warnings:
        logger.warning("⚠️ %s", warning)

    outlines = state.get("chapter_outlines", {})
    is_last_chapter = current_chap_num >= len(outlines)
    contracts = state.get("chapter_contracts", {})
    continuity_state = render_story_ledger(
        story_ledger,
        contracts.get(str(current_chap_num + 1), {}),
        contracts.get(str(current_chap_num + 2), {}),
    )
    story_summary = continuity_state
    if is_last_chapter:
        logger.info("⏭️ 最后一章台账已合并，无需生成后续上下文")
    
    return {
        "story_summary": story_summary,
        "continuity_state": continuity_state,
        "story_ledger": story_ledger,
        "current_chapter": current_chap_num + 1,
        "current_draft": latest_chapter,
        "iteration_count": 0,
        "scene_plan": {},
        "draft_candidates": [],
        "saved_chapter": current_chap_num,
        "audit_report": {},
        "editor_report": {},
        "ledger_delta": {},
        "continuity_report": {},
        "outline_report": {},
        "finale_report": {},
        "chapter_warnings": warnings,
        "summary_skipped": is_last_chapter,
    }
