import logging
import os
import sys
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from State import NovelState
from Nodes import (
    architect_node, writer_node, reviewer_node, summarizer_node,
    load_keywords, pick_keywords, load_story_patterns,
    DEFAULT_CHAPTERS, DEFAULT_WORDS_PER_CHAPTER,
    MAX_REVIEW_ATTEMPTS, STYLE_PASS_SCORE, should_retry_short_draft,
    is_strong_pattern, compatible_styles_for_pattern, roll_pattern_manifest,
    format_pattern_manifest, filter_material_categories_for_pattern,
    validate_material_categories_for_pattern,
)

logger = logging.getLogger("AutoWrite")

# 1. 初始化图
workflow = StateGraph(NovelState)

# 2. 添加节点 (审查合并为一个并行节点)
workflow.add_node("architect", architect_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("summarizer", summarizer_node)

# 3. 设置入口点
workflow.set_entry_point("architect")

# 4. 固定连接
workflow.add_edge("architect", "writer")

def route_after_writer(state: NovelState):
    if should_retry_short_draft(state):
        logger.warning("   ⚠️ 正文低于最低字数 → 发回写手补写后再审核")
        return "writer"
    return "reviewer"

workflow.add_conditional_edges("writer", route_after_writer, {
    "writer": "writer",
    "reviewer": "reviewer",
})

# 5. 统一路由：审查完去哪？
def route_after_review(state: NovelState):
    audit = state.get("audit_report", {})
    editor = state.get("editor_report", {})
    outlines = state.get("chapter_outlines", {})
    
    need_retry = (
        audit.get("审核状态") == "不通过" or
        editor.get("文风评分", 10) < STYLE_PASS_SCORE
    )
    if need_retry and state.get("iteration_count", 1) < MAX_REVIEW_ATTEMPTS:
        logger.warning("   ⚠️ 审稿退稿 审计:%s 评分:%d/10 → 发回写手重写",
                      audit.get("审核状态"), editor.get("文风评分", 0))
        return "writer"
    
    if state.get("current_chapter", 1) <= len(outlines):
        return "summarizer"
    return END

workflow.add_conditional_edges("reviewer", route_after_review, {
    "writer": "writer",
    "summarizer": "summarizer",
    END: END,
})

# 6. 摘要保存后，仅在仍有下一章时继续写作
def route_after_summary(state: NovelState):
    outlines = state.get("chapter_outlines", {})
    if state.get("current_chapter", 1) <= len(outlines):
        return "writer"
    return END

workflow.add_conditional_edges("summarizer", route_after_summary, {
    "writer": "writer",
    END: END
})

# ==========================================
# 挂载存档器与设置断点
# ==========================================
memory = MemorySaver()

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
    selected_cats = []
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
    print("-" * 50)
    ch_input = input(f"📏 章节数 (默认{DEFAULT_CHAPTERS}章): ").strip()
    try:
        target_chapters = int(ch_input) if ch_input else DEFAULT_CHAPTERS
    except ValueError:
        target_chapters = DEFAULT_CHAPTERS
    
    w_input = input(f"   每章字数 (默认{DEFAULT_WORDS_PER_CHAPTER}字): ").strip()
    try:
        words_per_chapter = int(w_input) if w_input else DEFAULT_WORDS_PER_CHAPTER
    except ValueError:
        words_per_chapter = DEFAULT_WORDS_PER_CHAPTER
    print(f"   ✅ {target_chapters}章 × {words_per_chapter}字 = 约{target_chapters * words_per_chapter}字")
    print()
    
    # ======== 步骤4: 写手风格 ========
    print("-" * 50)
    print("✍️ 写手风格: [1] 默认  [2] 热血爽文  [3] 文艺细腻  [4] 冷峻纪实  [5] 轻松搞笑")
    style_input = input("   选择风格 (默认1): ").strip()
    style_map_cli = {"2": "hot_blood", "3": "literary", "4": "cold", "5": "humor"}
    writer_style = style_map_cli.get(style_input, "default")
    style_names = {"default": "默认", "hot_blood": "热血爽文", "literary": "文艺细腻", "cold": "冷峻纪实", "humor": "轻松搞笑", "18xx": "18XX"}
    print(f"   ✅ 写手风格: {style_names[writer_style]}")
    print()

    # ======== 步骤5: 创作套路 ========
    patterns = load_story_patterns()
    pattern_keys = [key for key in patterns if key != "custom"]
    print("-" * 50)
    print("🎭 创作套路: " + "  ".join(
        f"[{index + 1}] {patterns[key].get('name', key)}"
        for index, key in enumerate(pattern_keys)
    ))
    pattern_input = input("   选择套路 (默认1=无套路，输入C自定义): ").strip()
    custom_pattern = ""
    if pattern_input.upper() == "C":
        story_pattern = "custom"
        custom_pattern = input("   自定义套路要求: ").strip()
    else:
        try:
            story_pattern = pattern_keys[int(pattern_input) - 1] if pattern_input else "none"
        except (ValueError, IndexError):
            story_pattern = "none"
    print(f"   ✅ 创作套路: {patterns.get(story_pattern, patterns['none']).get('name', '无套路')}")
    pattern_manifest = {}
    pattern_plan = {}
    material_issues = validate_material_categories_for_pattern(story_pattern, selected_cats)
    if material_issues:
        print("   ⚠️ 已移除与当前套路冲突的随机素材：")
        for issue in material_issues:
            print(f"      - {issue}")
        selected_cats = filter_material_categories_for_pattern(story_pattern, selected_cats)
        keywords = pick_keywords(selected_cats, 2, story_pattern) if selected_cats else []
    if is_strong_pattern(story_pattern):
        compatible_styles = compatible_styles_for_pattern(story_pattern)
        if writer_style not in compatible_styles:
            writer_style = "default"
            print("   ⚠️ 当前风格与强套路冲突，已切换为默认风格")
        ending_choice = input("   结局方向: [1] 不复合(默认)  [2] 高代价后由女主决定: ").strip()
        ending = "costly_reunion" if ending_choice == "2" else "no_reunion"
        pattern_manifest = roll_pattern_manifest(story_pattern, ending=ending)
        print("   🎲 已生成强套路契约：")
        print(format_pattern_manifest(pattern_manifest))
    print()
    
    config = {"configurable": {"thread_id": "novel_project_001"}}
    
    initial_state = {
        "user_idea": my_idea,
        "keywords": keywords,
        "target_chapters": target_chapters,
        "words_per_chapter": words_per_chapter,
        "writer_style": writer_style,
        "story_pattern": story_pattern,
        "custom_pattern": custom_pattern,
        "pattern_manifest": pattern_manifest,
        "pattern_plan": pattern_plan,
        "continuity_state": "",
        "current_chapter": 1,
        "iteration_count": 0
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
                    elif node_name == "reviewer":
                        audit = node_state.get('audit_report', {})
                        editor = node_state.get('editor_report', {})
                        print(
                            f"   -> 审稿 逻辑:{'不通过' if audit.get('发现的问题') else '通过'} "
                            f"套路:{audit.get('套路执行状态', '通过')} "
                            f"评分:{editor.get('文风评分')}/10"
                        )
                    elif node_name == "summarizer":
                        print(f"   -> 🗂️ 记忆已更新，准备进入下一章。")
        else:
            print("🛑 流程已终止。你可以调整提示词后重新运行。")
    except Exception as e:
        logger.error("❌ 流程异常终止：%s", e)
        logger.info("💡 提示：请检查 API Key 是否有效、网络连接是否正常。")
        sys.exit(1)
