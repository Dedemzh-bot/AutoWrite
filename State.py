from typing import TypedDict

class NovelState(TypedDict):
    # --- 基础输入与设定 ---
    user_idea: str
    world_bible: str
    chapter_outlines: dict
    keywords: list[str]
    
    # --- 篇幅配置 ---
    target_chapters: int      # 目标章节数
    words_per_chapter: int    # 每章目标字数
    writer_style: str         # 写手风格标识
    
    # --- 运行进度与暂存 ---
    current_chapter: int
    current_draft: str
    
    # --- 审核与反馈 ---
    audit_report: dict
    editor_report: dict
    
    # --- 统计与存储 ---
    iteration_count: int
    editor_iteration_count: int
    saved_chapter: int        # 刚刚保存的章节号

    story_summary: str