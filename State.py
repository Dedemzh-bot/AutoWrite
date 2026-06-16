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
    story_pattern: str        # 创作套路标识
    custom_pattern: str       # 用户自定义套路要求
    pattern_manifest: dict    # 本次确认的人物、背景、虐点与结局契约
    pattern_plan: dict        # 按章节映射的套路节拍任务
    
    # --- 运行进度与暂存 ---
    current_chapter: int
    current_draft: str
    
    # --- 审核与反馈 ---
    audit_report: dict
    editor_report: dict
    chapter_warnings: list[str]  # 达到重试上限后仍存在的篇幅或质量警告
    
    # --- 统计与存储 ---
    iteration_count: int      # 当前章节稿件次数，最多2稿
    saved_chapter: int        # 刚刚保存的章节号
    summary_skipped: bool     # 最后一章保存后是否跳过摘要模型
    novel_title: str          # 小说标题
    run_id: str               # 单次网页/命令行运行标识，用于多开时隔离输出
    outline_file: str         # 洗文来源大纲文件名
    wash_original_title: str  # 洗文原始标题

    story_summary: str
    continuity_state: str     # 结构化连续性档案
