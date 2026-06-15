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
字段必须完整：审核状态为“通过”或“不通过”；发现的问题为字符串数组；修改建议为字符串。"""

EDITOR_JSON_PROMPT = """必须仅输出一个有效 JSON 对象，不要输出 Markdown 或说明文字。
字段必须完整：文风评分为 1 到 10 的整数；改进建议为字符串。"""

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
    if audit.get("审核状态") == "不通过":
        warnings.append(f"第{current_chapter}章达到审核上限后逻辑审计仍未通过")
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

def pick_keywords(categories: list[str], count: int = 2) -> list[str]:
    keyword_db = load_keywords()
    pool = []
    for cat in categories:
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
    修改建议: str = Field(description="具体的修改指导。若通过则填'无'。")

class EditorReport(BaseModel):
    文风评分: int = Field(description="给出1-10的评分，7分及格。")
    改进建议: str = Field(description="关于遣词造句、剧情节奏的润色建议。")

llm_architect_structured = llm_architect.with_structured_output(ArchitectOutput, method="json_mode")
llm_auditor_structured = llm_auditor_raw.with_structured_output(AuditReport, method="json_mode")
llm_editor_structured = llm_editor.with_structured_output(EditorReport, method="json_mode")

def invoke_with_retry(chain, inputs, node_name: str, max_attempts: int = APP_INVOKE_ATTEMPTS):
    last_error = None
    for attempt in range(max_attempts):
        try:
            result = chain.invoke(inputs)
            if result is not None:
                return result
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
    raise RuntimeError(
        f"{node_name}连接模型失败，已自动重试{max_attempts}次。"
        "请检查网络、代理或模型服务状态后重试。"
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

    keywords = state.get("keywords", [])
    keywords_str = "、".join(keywords) if keywords else "无"

    chapters = state.get("target_chapters", DEFAULT_CHAPTERS)
    words_per = state.get("words_per_chapter", DEFAULT_WORDS_PER_CHAPTER)
    chapter_req = (
        f"规划严格且仅有{chapters}章的详细细纲，每章正文目标{words_per}字。"
        f"每章细纲去除空白后必须不少于{MIN_OUTLINE_CHARS}字，"
        "必须包含开场状态、核心冲突、关键行动、人物关系变化、伏笔、"
        "结尾结果和下一章钩子。"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("architect_system.md")),
        ("system", ARCHITECT_JSON_PROMPT),
        ("user", load_prompt("architect_user.md"))
        ])
    
    architect_chain = prompt | llm_architect_structured
    result = None
    outline_issues = []
    for outline_attempt in range(APP_INVOKE_ATTEMPTS):
        result = invoke_with_retry(architect_chain, {
            "user_idea": state.get("user_idea"),
            "keywords": keywords_str,
            "chapter_requirement": chapter_req
        }, "架构师")
        outline_issues = outline_validation_issues(result.chapter_outlines, chapters)
        if not outline_issues:
            break
        logger.warning(
            "   ⚠️ 架构师大纲验收未通过 (%d/%d): %s",
            outline_attempt + 1,
            APP_INVOKE_ATTEMPTS,
            "；".join(outline_issues[:5]),
        )
    if result is None or outline_issues:
        raise RuntimeError(
            f"架构师连续{APP_INVOKE_ATTEMPTS}次未能生成合格大纲："
            f"{'；'.join(outline_issues[:5])}"
        )
    chapter_outlines = normalize_chapter_outlines(result.chapter_outlines, chapters)

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
        "current_chapter": 1
    }
def writer_node(state: NovelState):
    chapter_num = state.get("current_chapter", 1)
    iteration = state.get("iteration_count", 0) + 1
    logger.info("✍️ 写手正在奋笔疾书 第 %d 章 (第 %d 稿)...", chapter_num, iteration)
    
    summary = state.get("story_summary", "故事刚刚开始。")
    world_bible = state.get("world_bible", "")
    outlines = state.get("chapter_outlines", {})
    current_outline = outlines.get(str(chapter_num), "自由发挥。")
    words_per = state.get("words_per_chapter", DEFAULT_WORDS_PER_CHAPTER)
    min_words = int(words_per * MIN_CHAPTER_RATIO)
    preferred_words = int(words_per * 0.93)
    style = state.get("writer_style", "default")
    
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
        feedback += f"\n\n【审计退稿修改令】：发现严重问题：{audit_report.get('发现的问题')}。请务必根据以下建议重写本章：{audit_report.get('修改建议')}"
    if editor_report.get("文风评分", 10) < STYLE_PASS_SCORE:
        feedback += f"\n\n【责编退稿润色令】：文风不达标(当前评分{editor_report.get('文风评分')}/10)。改进建议：{editor_report.get('改进建议')}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt(system_file)),
        ("system", CHAPTER_FORMAT_PROMPT),
        ("user", load_prompt("writer_user.md"))
    ])

    result = invoke_with_retry(prompt | llm_writer, {
        "world_bible": world_bible,
        "summary": summary,
        "outline": current_outline,
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

    result = _safe_invoke(prompt | llm_auditor_structured, {
        "outline": state.get("chapter_outlines", {}).get(str(state.get("current_chapter", 1)), ""), 
        "draft": state.get("current_draft")
    }, "auditor")
    
    if result is None:
        logger.warning("🕵️ 审计失败，默认放行")
        return {"audit_report": {"审核状态": "通过", "发现的问题": [], "修改建议": "无"}}
    
    logger.info("🕵️ 审计结果: %s", result.审核状态)
    return {"audit_report": result.model_dump()}

def editor_node(state: NovelState):
    logger.info("👓 责编正在审视文笔与爽点...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("editor_system.md")),
        ("system", EDITOR_JSON_PROMPT),
        ("user", load_prompt("editor_user.md"))
    ])
    
    result = _safe_invoke(prompt | llm_editor_structured, {"draft": state.get("current_draft")}, "editor")
    
    if result is None:
        logger.warning("👓 责编评估失败，默认放行(评分8)")
        result = EditorReport(文风评分=8, 改进建议="无")
    
    logger.info("👓 责编评分: %d/10", result.文风评分)
    
    return {
        "editor_report": result.model_dump(),
    }

def _auditor_internal(state: NovelState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("auditor_system.md")),
        ("system", AUDITOR_JSON_PROMPT),
        ("user", load_prompt("auditor_user.md"))
    ])
    result = _safe_invoke(prompt | llm_auditor_structured, {
        "outline": state.get("chapter_outlines", {}).get(str(state.get("current_chapter", 1)), ""),
        "draft": state.get("current_draft")
    }, "auditor")
    if result is None:
        return {"audit_report": {"审核状态": "通过", "发现的问题": [], "修改建议": "无"}}
    return {"audit_report": result.model_dump()}

def _editor_internal(state: NovelState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("editor_system.md")),
        ("system", EDITOR_JSON_PROMPT),
        ("user", load_prompt("editor_user.md"))
    ])
    result = _safe_invoke(prompt | llm_editor_structured, {"draft": state.get("current_draft")}, "editor")
    if result is None:
        return {"editor_report": EditorReport(文风评分=8, 改进建议="无").model_dump()}
    return {"editor_report": result.model_dump()}

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
    if not is_last_chapter:
        prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("summarizer_system.md")),
            ("user", load_prompt("summarizer_user.md"))
        ])

        try:
            result = invoke_with_retry(prompt | llm_summarizer, {
                "old_summary": story_summary,
                "new_chapter": latest_chapter
            }, f"第{current_chap_num}章剧情摘要")
            story_summary = result.content
        except RuntimeError as error:
            logger.warning("⚠️ %s；保留旧摘要并继续下一章", error)
            warnings.append(f"第{current_chap_num}章剧情摘要更新失败，已保留旧摘要")
    else:
        logger.info("⏭️ 最后一章已保存，跳过无后续用途的剧情摘要")
    
    return {
        "story_summary": story_summary,
        "current_chapter": current_chap_num + 1,
        "current_draft": latest_chapter,
        "iteration_count": 0,
        "saved_chapter": current_chap_num,
        "audit_report": {},
        "editor_report": {},
        "chapter_warnings": warnings,
        "summary_skipped": is_last_chapter,
    }
