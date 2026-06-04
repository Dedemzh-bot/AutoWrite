from typing import TypedDict

class NovelState(TypedDict):
    # --- 基础输入与设定 ---
    user_idea: str
    world_bible: str          # 架构师生成的设定
    chapter_outlines: dict    # 各章细纲字典
    keywords: list[str]       # 随机抽取的创作关键词
    
    # --- 篇幅配置 ---
    scope: str                # "short" 或 "long"
    max_words: int            # 短篇字数上限
    target_chapters: int      # 长篇目标章数
    
    # --- 运行进度与暂存 ---
    current_chapter: int
    current_draft: str
    
    # --- 审核与反馈 ---
    audit_report: dict        # 包含 "审核状态", "发现的问题", "修改建议"
    editor_report: dict       # 包含 "文风评分", "改进建议"
    
    # --- 统计与存储 ---
    iteration_count: int         # 审计打回计数器，最多3次
    editor_iteration_count: int  # 责编打回计数器，最多2次

    story_summary: str        # 记录到上一章为止的剧情提要