from __future__ import annotations


WRITER_STYLES = [
    {
        "key": "default",
        "name": "默认自然",
        "prompt_file": "writer_system.md",
        "editor_focus": "语言自然、信息清楚、动作对白与必要描写平衡。",
    },
    {
        "key": "hot_blood",
        "name": "热血爽文",
        "prompt_file": "writer_system_hot_blood.md",
        "editor_focus": "压迫与反击有因果，爽点改变局势，不靠口号和惊叹号制造热血。",
    },
    {
        "key": "literary",
        "name": "文艺细腻",
        "prompt_file": "writer_system_literary.md",
        "editor_focus": "细节、留白和心理层次准确，避免万能意象、滥用比喻和自我感动。",
    },
    {
        "key": "cold",
        "name": "冷峻纪实",
        "prompt_file": "writer_system_cold.md",
        "editor_focus": "叙述克制、因果扎实、细节可信，不机械堆短句或模仿具体作家。",
    },
    {
        "key": "humor",
        "name": "轻松幽默",
        "prompt_file": "writer_system_humor.md",
        "editor_focus": "笑点来自人物、处境和反差，不靠密集网络梗、强行吐槽或定时插科打诨。",
    },
    {
        "key": "18xx",
        "name": "成人情感",
        "prompt_file": "writer_system_18xx.md",
        "editor_focus": "只描写成年人之间自愿、明确、有边界的亲密关系，服务人物和剧情。",
    },
    {
        "key": "suspense",
        "name": "悬疑压迫",
        "prompt_file": "writer_system_suspense.md",
        "editor_focus": "线索可复盘、压力逐步升级、信息差公平，不用故弄玄虚代替悬疑。",
    },
    {
        "key": "emotional_tension",
        "name": "情感拉扯",
        "prompt_file": "writer_system_emotional_tension.md",
        "editor_focus": "情感变化来自选择、边界和现实代价，不用重复误会和内心独白拖延。",
    },
    {
        "key": "sweet_romcom",
        "name": "甜宠轻喜",
        "prompt_file": "writer_system_sweet_romcom.md",
        "editor_focus": "甜感来自尊重、照顾和共同经历，喜感自然，不靠降智和工业糖精。",
    },
    {
        "key": "ancient_elegant",
        "name": "古言雅致",
        "prompt_file": "writer_system_ancient_elegant.md",
        "editor_focus": "语言符合时代和身份，简净有韵，不堆砌伪古风辞藻与现代口吻。",
    },
    {
        "key": "realist_ensemble",
        "name": "现实群像",
        "prompt_file": "writer_system_realist_ensemble.md",
        "editor_focus": "人物立场各有现实根源，群像相互影响，不用作者旁白统一裁判。",
    },
    {
        "key": "business",
        "name": "商战职场",
        "prompt_file": "writer_system_business.md",
        "editor_focus": "商业行为有流程、数据和利益依据，不用万能合同、降智对手和口号翻盘。",
    },
]

WRITER_STYLE_MAP = {item["key"]: item for item in WRITER_STYLES}
WRITER_STYLE_KEYS = set(WRITER_STYLE_MAP)

AI_STOCK_EXPRESSIONS = [
    "这一刻他明白了",
    "这一刻她明白了",
    "空气仿佛凝固",
    "复杂的情绪涌上心头",
    "复杂情绪涌上心头",
    "嘴角勾起一抹",
    "眼底闪过一丝",
    "心中五味杂陈",
    "时间仿佛静止",
    "命运的齿轮",
    "一切都将不同",
]


def writer_style_options() -> list[dict]:
    return [
        {"key": item["key"], "name": item["name"]}
        for item in WRITER_STYLES
    ]


def writer_style(key: str) -> dict:
    return WRITER_STYLE_MAP.get(key, WRITER_STYLE_MAP["default"])


def writer_style_editor_context(key: str, prompt_text: str) -> str:
    item = writer_style(key)
    banned = "、".join(f"“{value}”" for value in AI_STOCK_EXPRESSIONS)
    return (
        f"风格名称：{item['name']}\n"
        f"编辑关注：{item['editor_focus']}\n"
        f"写手专项规则：\n{prompt_text}\n"
        f"库存表达观察表：{banned}\n"
        "库存表达单次出现只记警告；同一表达或同类模板在一章重复出现，"
        "必须列入可定位的AI痕迹问题。"
    )
