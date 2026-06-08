import datetime
import json
import logging
import os
import random
import sys
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
    result = (prompt | llm_architect).invoke({})
    title = result.content.strip() if result and result.content else f"{original_title}·重制版"
    for ch in r'\/:*?"<>|':
        title = title.replace(ch, "")
    return title.strip() or f"{original_title}·重制版"

DEFAULT_CHAPTERS = int(os.getenv("DEFAULT_CHAPTERS", "12"))
DEFAULT_WORDS_PER_CHAPTER = int(os.getenv("DEFAULT_WORDS_PER_CHAPTER", "2500"))

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
llm_architect = ChatOpenAI(model="deepseek-v4-flash", temperature=0.7)
llm_writer = ChatOpenAI(model="deepseek-v4-flash", temperature=0.8)
llm_editor = ChatOpenAI(model="deepseek-v4-flash", temperature=0.5)
llm_auditor_raw = ChatOpenAI(model="deepseek-v4-flash", temperature=0)
llm_summarizer = ChatOpenAI(model="deepseek-v4-flash", temperature=0.3)

# ==========================================
# 2. 强制 JSON 结构化定义 (键名保持中文)
# ==========================================
class ArchitectOutput(BaseModel):
    novel_title: str = Field(description="小说的书名，8-20字，简洁有力有网感")
    world_bible: str = Field(description="不少于500字的世界观、力量体系、主角人设详细设定。")
    chapter_outlines: dict[str, str] = Field(description="章节号(纯数字)映射到细纲文本。键必须为纯数字如'1', '2'。")
    estimated_words: int = Field(description="预估总字数")

class AuditReport(BaseModel):
    审核状态: str = Field(description="严格输出 '通过' 或 '不通过'。")
    发现的问题: list[str] = Field(description="具体的逻辑硬伤或偏离大纲的问题点。无问题则为空列表。")
    修改建议: str = Field(description="具体的修改指导。若通过则填'无'。")

class EditorReport(BaseModel):
    文风评分: int = Field(description="给出1-10的评分，8分及格。")
    改进建议: str = Field(description="关于遣词造句、剧情节奏的润色建议。")

llm_architect_structured = llm_architect.with_structured_output(ArchitectOutput, method="function_calling")
llm_auditor_structured = llm_auditor_raw.with_structured_output(AuditReport, method="function_calling")
llm_editor_structured = llm_editor.with_structured_output(EditorReport, method="function_calling")

def _safe_invoke(chain, inputs, node_name: str, max_retries: int = 2):
    for attempt in range(max_retries):
        try:
            result = chain.invoke(inputs)
            if result is not None:
                return result
        except Exception as e:
            logger.warning("   ⚠️ [%s] 结构化输出尝试 %d/%d 失败: %s", node_name, attempt + 1, max_retries, e)
    logger.error("   ❌ [%s] 结构化输出全部%d次尝试失败，返回 None", node_name, max_retries)
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
    chapter_req = f"规划{chapters}章详细细纲，每章目标{words_per}字"

    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("architect_system.md")),
        ("user", load_prompt("architect_user.md"))
        ])
    
    result = _safe_invoke(prompt | llm_architect_structured, {
        "user_idea": state.get("user_idea"),
        "keywords": keywords_str,
        "chapter_requirement": chapter_req
    }, "architect")
    if result is None:
        raise RuntimeError("架构师结构化输出失败，无法生成大纲")

    # 保存大纲 JSON 到 Outline/ 目录
    try:
        os.makedirs("Outline", exist_ok=True)
        safe_title = "".join(c for c in result.novel_title if c not in r'\/:*?"<>|')
        save_path = os.path.join("Outline", f"{safe_title}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "title": result.novel_title,
                "world_bible": result.world_bible,
                "chapter_outlines": result.chapter_outlines,
                "estimated_words": result.estimated_words,
                "created_at": datetime.datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        logger.info("📁 大纲已保存 → %s", save_path)
    except Exception as e:
        logger.warning("⚠️ 大纲保存失败: %s", e)

    return {
        "novel_title": result.novel_title,
        "world_bible": result.world_bible,
        "chapter_outlines": result.chapter_outlines,
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
    if audit_report.get("审核状态") == "不通过":
        feedback += f"\n\n【审计退稿修改令】：发现严重问题：{audit_report.get('发现的问题')}。请务必根据以下建议重写本章：{audit_report.get('修改建议')}"
    if editor_report.get("文风评分", 10) < 8:
        feedback += f"\n\n【责编退稿润色令】：文风不达标(当前评分{editor_report.get('文风评分')}/10)。改进建议：{editor_report.get('改进建议')}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt(system_file)),
        ("user", load_prompt("writer_user.md"))
    ])

    result = (prompt | llm_writer).invoke({
        "world_bible": world_bible,
        "summary": summary,
        "outline": current_outline,
        "feedback": feedback,
        "chapter_num": chapter_num,
        "words_per_chapter": words_per
    })

    content = result.content if result and result.content else "[写手产出为空，请重试]"
    
    return {
        "current_draft": content,
        "iteration_count": iteration
    }

def auditor_node(state: NovelState):
    logger.info("🕵️ 审计员正在进行地毯式排查...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("auditor_system.md")),
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
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n{'='*20} 第 {current_chap_num} 章 {'='*20}\n\n")
        f.write(latest_chapter)
    logger.info("💾 第 %d 章已安全入库 → %s", current_chap_num, file_path)

    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("summarizer_system.md")),
        ("user", load_prompt("summarizer_user.md"))
    ])
    
    result = (prompt | llm_summarizer).invoke({
        "old_summary": state.get("story_summary", ""),
        "new_chapter": latest_chapter
    })
    
    return {
        "story_summary": result.content,
        "current_chapter": current_chap_num + 1,
        "current_draft": latest_chapter,
        "iteration_count": 0,
        "saved_chapter": current_chap_num
    }