from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

from Nodes import (  # noqa: E402
    MODEL_MAX_RETRIES,
    MODEL_TIMEOUT_SECONDS,
    OPENAI_MODEL,
    invoke_with_retry,
)


REFINE_SYSTEM = """你是一位资深小说编辑，帮作者把粗略的点子精炼为完整的故事设定。

你的工作方式是：
1. 先看作者的点子，找出最需要明确的关键信息（主角身份、核心冲突、世界观基调、金手指类型）
2. 一次只问一个问题，问题要具体，引导作者给出有用信息
3. 累计问 2 轮后，输出精炼后的完整设定

输出格式：
- 如果是提问，直接输出问题，不要前缀标记
- 如果是精炼结果，输出格式为 "【精炼设定】\n<完整的300字设定描述>"
- 精炼设定必须整合所有已获取的信息，用一段通顺的文字呈现"""

BATCH_REFINE_SYSTEM = """你是一位资深小说编辑，帮作者把粗略的点子精炼为完整的故事设定。

这是批量无人值守创作流程，不能向作者追问。你必须根据已有点子合理补全缺失信息，
直接输出可用于小说创作的完整设定。

要求：
1. 保留原始点子的核心冲突、人物关系、情绪方向和题材基调。
2. 补足主角身份、核心矛盾、关键转折、爽点/虐点、结局走向等创作必要信息。
3. 不要添加与原点子相反的设定，不要改变主角立场。
4. 输出 250-450 字的一段完整设定。

输出格式固定为：
【精炼设定】
<完整设定描述>"""


llm_refine = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0.5,
    timeout=MODEL_TIMEOUT_SECONDS,
    max_retries=MODEL_MAX_RETRIES,
    max_tokens=4096,
    extra_body={
        "thinking": {"type": "disabled"},
    },
)


def _strip_refine_prefix(content: str) -> str:
    text = (content or "").strip()
    prefix = "【精炼设定】"
    if text.startswith(prefix):
        text = text[len(prefix):].lstrip("\r\n ")
    return text.strip()


def call_refine(history: list[dict]) -> str:
    messages = [("system", REFINE_SYSTEM)]
    for item in history:
        role = "user" if item["role"] == "user" else "assistant"
        messages.append((role, item["content"]))
    prompt = ChatPromptTemplate.from_messages(messages)
    result = invoke_with_retry(prompt | llm_refine, {}, "灵感精炼")
    return result.content.strip() if result and result.content else ""


def refine_idea_for_batch(idea: str) -> str:
    source = str(idea or "").strip()
    if not source:
        raise ValueError("点子不能为空")
    prompt = ChatPromptTemplate.from_messages([
        ("system", BATCH_REFINE_SYSTEM),
        ("user", f"原始小说点子：\n{source}\n\n请直接输出精炼后的完整设定。"),
    ])
    result = invoke_with_retry(prompt | llm_refine, {}, "批量灵感精炼")
    refined = _strip_refine_prefix(
        result.content.strip() if result and result.content else ""
    )
    if not refined:
        raise RuntimeError("精炼模型返回空内容")
    return refined