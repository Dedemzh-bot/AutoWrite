import os
import time
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from State import NovelState 

load_dotenv()

# ==========================================
# 0. 辅助函数：读取本地 Prompt 文件 (绝对路径版)
# ==========================================
def load_prompt(file_name: str) -> str:
    # 1. 获取当前文件 (Nodes.py) 所在的绝对路径目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 拼接出准确的 prompts 文件夹路径
    path = os.path.join(current_dir, "Role", file_name)
    
    # 3. 读取文件
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ 严重错误：找不到提示词文件！请检查路径：\n{path}")
        raise
# ==========================================
# 1. 模型插座配置
# ==========================================
llm_architect = ChatOpenAI(model="gemini-2.5-pro", temperature=0.7)
llm_writer = ChatOpenAI(model="gemini-2.5-flash", temperature=0.8)
llm_editor = ChatOpenAI(model="gemini-2.5-flash", temperature=0.5)
llm_auditor_raw = ChatOpenAI(model="gemini-2.5-flash", temperature=0)
llm_summarizer = ChatOpenAI(model="gemini-2.5-flash", temperature=0.3)

# ==========================================
# 2. 强制 JSON 结构化定义 (键名保持中文)
# ==========================================
class ArchitectOutput(BaseModel):
    world_bible: str = Field(description="不少于500字的世界观、力量体系、主角人设详细设定。")
    chapter_outlines: dict[int, str] = Field(description="章节号映射到具体的细纲。需规划前50章，格式如 {1: '第1章剧情', 2: '第2章剧情'}")

class AuditReport(BaseModel):
    审核状态: str = Field(description="严格输出 '通过' 或 '不通过'。")
    发现的问题: list[str] = Field(description="具体的逻辑硬伤或偏离大纲的问题点。无问题则为空列表。")
    修改建议: str = Field(description="具体的修改指导。若通过则填'无'。")

class EditorReport(BaseModel):
    文风评分: int = Field(description="给出1-10的评分，8分及格。")
    改进建议: str = Field(description="关于遣词造句、剧情节奏的润色建议。")

llm_architect_structured = llm_architect.with_structured_output(ArchitectOutput)
llm_auditor_structured = llm_auditor_raw.with_structured_output(AuditReport)
llm_editor_structured = llm_editor.with_structured_output(EditorReport)

# ==========================================
# 3. 核心节点
# ==========================================

def architect_node(state: NovelState):
    print("🧠 架构师正在深度推演世界观与大纲...")
# 只有当世界观为空时才生成
    if state.get("world_bible"):
        return {}

    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("architect_system.md")),
        ("user", load_prompt("architect_user.md"))
        ])
    
    result = (prompt | llm_architect_structured).invoke({"user_idea": state.get("user_idea")})

    return {
        "world_bible": result.world_bible,
        "chapter_outlines": result.chapter_outlines,
        "current_chapter": 1
    }
def writer_node(state: NovelState):
    chapter_num = state.get("current_chapter", 1)
    iteration = state.get("iteration_count", 0) + 1
    print(f"✍️ 写手正在奋笔疾书 第 {chapter_num} 章 (第 {iteration} 稿)...")
    
    summary = state.get("story_summary", "故事刚刚开始。")
    world_bible = state.get("world_bible", "")
    outlines = state.get("chapter_outlines", {})
    current_outline = outlines.get(chapter_num, "自由发挥。")
    
    audit_report = state.get("audit_report", {})
    editor_report = state.get("editor_report", {})
    feedback = ""
    if audit_report.get("审核状态") == "不通过":
        feedback += f"\n\n【审计退稿修改令】：发现严重问题：{audit_report.get('发现的问题')}。请务必根据以下建议重写本章：{audit_report.get('修改建议')}"
    if editor_report.get("文风评分", 10) < 8:
        feedback += f"\n\n【责编退稿润色令】：文风不达标(当前评分{editor_report.get('文风评分')}/10)。改进建议：{editor_report.get('改进建议')}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("writer_system.md")),
        ("user", load_prompt("writer_user.md"))
    ])

    result = (prompt | llm_writer).invoke({
        "world_bible": world_bible,
        "summary": summary,
        "outline": current_outline,
        "feedback": feedback,
        "chapter_num": chapter_num
    })

    return {
        "current_draft": result.content,
        "iteration_count": iteration
    }

def auditor_node(state: NovelState):
    print("⏳ [减速带] 审计员正在喝茶等待 (15秒)...")
    time.sleep(15)  
    print("🕵️ 审计员正在进行地毯式排查...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("auditor_system.md")),
        ("user", load_prompt("auditor_user.md"))
    ])

    result = (prompt | llm_auditor_structured).invoke({
        "outline": state.get("chapter_outlines", {}).get(state.get("current_chapter", 1)), 
        "draft": state.get("current_draft")
    })
    
    print(f"🕵️ 审计结果: {result.审核状态}")
    return {"audit_report": result.model_dump()}

def editor_node(state: NovelState):
    print("👓 责编正在审视文笔与爽点...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("editor_system.md")),
        ("user", load_prompt("editor_user.md"))
    ])
    
    result = (prompt | llm_editor_structured).invoke({"draft": state.get("current_draft")})
    print(f"👓 责编评分: {result.文风评分}/10")
    
    editor_iter = state.get("editor_iteration_count", 0) + 1
    
    return {
        "editor_report": result.model_dump(),
        "editor_iteration_count": editor_iter
    }

def summarizer_node(state: NovelState):
    print("📝 书记员正在提炼记忆档案...")
    
    current_chap_num = state.get("current_chapter", 1)
    latest_chapter = state.get("current_draft", "")
    
    # 自动保存到本地
    file_path = "我的修仙大作.txt"
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n{'='*20} 第 {current_chap_num} 章 {'='*20}\n\n")
        f.write(latest_chapter)
    print(f"💾 第 {current_chap_num} 章已安全入库。")

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
        "iteration_count": 0,
        "editor_iteration_count": 0
    }