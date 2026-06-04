import logging
import os
import sys
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from State import NovelState
from Nodes import (
    architect_node, writer_node, auditor_node, editor_node, summarizer_node,
    load_keywords, pick_keywords,
    SHORT_NOVEL_MAX_WORDS, LONG_NOVEL_DEFAULT_CHAPTERS
)

logger = logging.getLogger("AutoWrite")

# 1. 初始化图
workflow = StateGraph(NovelState)

# 2. 添加所有节点
workflow.add_node("architect", architect_node)
workflow.add_node("writer", writer_node)
workflow.add_node("auditor", auditor_node)
workflow.add_node("editor", editor_node)
workflow.add_node("summarizer", summarizer_node)

# 3. 设置入口点
workflow.set_entry_point("architect")

# 4. 固定连接
workflow.add_edge("architect", "writer")
workflow.add_edge("writer", "auditor")

# 5. 核心路由一：审计完了去哪？
def route_after_audit(state: NovelState):
    report = state.get("audit_report", {})
    if report.get("审核状态") == "不通过" and state.get("iteration_count", 0) < 3:
        return "writer"
    return "editor"
workflow.add_conditional_edges("auditor", route_after_audit, {"writer": "writer", "editor": "editor"})

# 6. 核心路由二：责编完了去哪？
def route_after_editor(state: NovelState):
    report = state.get("editor_report", {})
    outlines = state.get("chapter_outlines", {})
    
    if report.get("文风评分", 0) < 8 and state.get("editor_iteration_count", 0) < 2:
        logger.warning("   ⚠️ 责编退稿：文风评分 %d/10 未达标，发回写手润色 (第%d次)", report.get('文风评分'), state.get('editor_iteration_count', 0))
        return "writer"
    
    if state.get("current_chapter", 1) <= len(outlines):
        return "summarizer"
    return END

workflow.add_conditional_edges("editor", route_after_editor, {
    "writer": "writer",
    "summarizer": "summarizer",
    END: END
})

# 7. 闭环路线
workflow.add_edge("summarizer", "writer")

# ==========================================
# 【核心升级】：挂载存档器与设置断点
# ==========================================
# 初始化一个内存存档器（真实项目中可换成 SQLite 或 Postgres 数据库存档）
memory = MemorySaver()

# 编译图时，把 memory 挂上去，并告诉程序：在 architect（架构师）干完活后，立刻暂停！
app = workflow.compile(
    checkpointer=memory,
    interrupt_after=["architect"] 
)

# ==========================================
# 8. 启动测试执行 (带交互式输入版)
# ==========================================
if __name__ == "__main__":
    print("🚀 小说工业流水线 v4.0 (词库 + 篇幅自适应) 启动...\n")
    
    # ======== 步骤1: 输入灵感 ========
    print("-" * 50)
    my_idea = input("💡 请输入你的小说灵感/点子 (直接回车将使用默认设定)：\n> ")
    if not my_idea.strip():
        my_idea = "一个能在梦里修仙的现代程序员"
    print()
    
    # ======== 步骤2: 词库题材选择 ========
    keywords = []
    keyword_db = load_keywords()
    if keyword_db:
        print("-" * 50)
        print("📚 随机附加词库 — 请选择题材类型 (混抽逗号分隔，回车跳过):")
        cats = list(keyword_db.keys())
        for i, cat in enumerate(cats):
            desc = keyword_db[cat].get("description", "")
            end_char = "\n" if (i + 1) % 4 == 0 else "  "
            print(f"  [{i + 1}] {cat}({desc})", end=end_char)
        if len(cats) % 4 != 0:
            print()
        
        choice = input("\n👉 输入编号: ").strip()
        if choice:
            selected_cats = []
            for part in choice.replace("，", ",").split(","):
                try:
                    idx = int(part.strip()) - 1
                    if 0 <= idx < len(cats):
                        selected_cats.append(cats[idx])
                except ValueError:
                    pass
            
            if selected_cats:
                print(f"   已选: {', '.join(selected_cats)}")
                
                while True:
                    keywords = pick_keywords(selected_cats, 2)
                    if not keywords:
                        print("   ⚠️ 该分类下无词条，跳过。")
                        break
                    print(f"   🎲 命中: [{keywords[0]}] [{keywords[1] if len(keywords) > 1 else '—'}]")
                    confirm = input("   确认(Y) / 重抽(R) / 跳过(N): ").strip().upper()
                    if confirm == 'Y':
                        break
                    elif confirm == 'N':
                        keywords = []
                        break
                    # R → loop again
    print()
    
    # ======== 步骤3: 篇幅选择 ========
    scope = "short"
    max_words = SHORT_NOVEL_MAX_WORDS
    target_chapters = 0
    
    print("-" * 50)
    scope_input = input("📏 篇幅: 短篇-S (默认≤5W字) / 长篇-L (默认50章): ").strip().upper()
    if scope_input == 'L':
        scope = "long"
        ch_input = input(f"   章节数 (默认{LONG_NOVEL_DEFAULT_CHAPTERS}): ").strip()
        try:
            target_chapters = int(ch_input) if ch_input else LONG_NOVEL_DEFAULT_CHAPTERS
        except ValueError:
            target_chapters = LONG_NOVEL_DEFAULT_CHAPTERS
        print(f"   ✅ 长篇模式，规划 {target_chapters} 章")
    else:
        w_input = input(f"   字数上限 (默认{SHORT_NOVEL_MAX_WORDS}字): ").strip()
        try:
            max_words = int(w_input) if w_input else SHORT_NOVEL_MAX_WORDS
        except ValueError:
            max_words = SHORT_NOVEL_MAX_WORDS
        print(f"   ✅ 短篇模式，上限 {max_words} 字")
    print()
    
    config = {"configurable": {"thread_id": "novel_project_001"}}
    
    initial_state = {
        "user_idea": my_idea,
        "keywords": keywords,
        "scope": scope,
        "max_words": max_words,
        "target_chapters": target_chapters,
        "current_chapter": 1,
        "iteration_count": 0,
        "editor_iteration_count": 0
    }
    
    try:
        print("--- 第一阶段：呼叫架构师出大纲 ---")
        for output in app.stream(initial_state, config=config):
            for node_name, node_state in output.items():
                print(f"✅ 节点 [{node_name}] 执行完毕！")
                if node_name == "architect":
                    print("\n【架构师产出的全局大纲如下】：")
                    print(node_state.get('world_bible'))
                    print(node_state.get('chapter_outlines'))

        current_state = app.get_state(config)
        print(f"\n⏸️ 流程已暂停！下一步原本应该前往：{current_state.next}")
        
        user_input = input("\n👉 主编大人，大纲是否满意？输入 'Y' 批准执行，或输入 'N' 退出修改：")
        
        if user_input.strip().upper() == 'Y':
            print("\n--- 第二阶段：大纲已批准，唤醒写手全自动码字 ---")
            for output in app.stream(None, config=config):
                for node_name, node_state in output.items():
                    if node_name == "writer":
                        print(f"✍️ 写手产出 第 {node_state.get('current_chapter')} 章 (第 {node_state.get('iteration_count')} 稿)...")
                    elif node_name == "auditor":
                        report = node_state.get('audit_report', {})
                        print(f"   -> 审计状态: {report.get('审核状态')}")
                    elif node_name == "summarizer":
                        print(f"   -> 🗂️ 记忆已更新，准备进入下一章。")
        else:
            print("🛑 流程已终止。你可以调整提示词后重新运行。")
    except Exception as e:
        logger.error("❌ 流程异常终止：%s", e)
        logger.info("💡 提示：请检查 API Key 是否有效、网络连接是否正常。")
        sys.exit(1)