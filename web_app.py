import asyncio
import json
import logging
import os
import queue
import socket
import sys
import threading
import time
import uuid
import urllib.request
from collections import defaultdict
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from State import NovelState
from Nodes import (
    architect_node, writer_node, reviewer_node, summarizer_node,
    list_outline_files, load_outline_json, generate_wash_title,
    DEFAULT_CHAPTERS, DEFAULT_WORDS_PER_CHAPTER,
    MAX_REVIEW_ATTEMPTS, STYLE_PASS_SCORE, normalize_chapter_outlines,
    MODEL_MAX_RETRIES, MODEL_TIMEOUT_SECONDS, invoke_with_retry,
    outline_validation_issues, should_retry_short_draft,
    route_after_review_decision, build_chapter_contracts, build_finale_contract,
    is_strong_pattern, roll_pattern_manifest,
    validate_pattern_manifest, build_pattern_plan, attach_pattern_plan_to_outlines,
    strip_pattern_plan_from_outlines,
)
from LibraryV2 import (
    LibraryValidationError,
    default_material_config,
    default_pattern_config,
    material_library_metadata,
    normalize_material_config,
    normalize_pattern_config,
    pattern_library_metadata,
    resample_material_item,
    sample_materials,
    validate_material_config,
    validate_pattern_config,
)
from WriterStyles import writer_style_options

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("AutoWrite")


class RetryCounterHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self._count = 0
        self._lock = threading.Lock()

    def emit(self, record):
        if "Retrying request" in record.getMessage():
            with self._lock:
                self._count += 1

    def reset(self):
        with self._lock:
            self._count = 0

    def value(self):
        with self._lock:
            return self._count


retry_counter = RetryCounterHandler()
logging.getLogger().addHandler(retry_counter)

# ── 构建图（不去除 interrupt，Web 层手动控制暂停） ──
workflow = StateGraph(NovelState)
workflow.add_node("architect", architect_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("summarizer", summarizer_node)
workflow.set_entry_point("architect")
workflow.add_edge("architect", "writer")


def route_after_writer(state: NovelState):
    return "writer" if should_retry_short_draft(state) else "reviewer"


workflow.add_conditional_edges("writer", route_after_writer, {
    "writer": "writer", "reviewer": "reviewer"
})


def route_after_review(state: NovelState):
    return route_after_review_decision(state)


workflow.add_conditional_edges("reviewer", route_after_review, {
    "writer": "writer", "summarizer": "summarizer", END: END
})


def route_after_summary(state: NovelState):
    outlines = state.get("chapter_outlines", {})
    if state.get("current_chapter", 1) <= len(outlines):
        return "writer"
    return END


workflow.add_conditional_edges("summarizer", route_after_summary, {
    "writer": "writer", END: END
})

memory = MemorySaver()
graph_app = workflow.compile(checkpointer=memory)

# ── 灵感精炼 LLM ──
llm_refine = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.5,
    timeout=MODEL_TIMEOUT_SECONDS,
    max_retries=MODEL_MAX_RETRIES,
    max_tokens=4096,
    extra_body={
        "thinking": {"type": "disabled"},
    },
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

app = FastAPI(title="AutoWrite Web")

def _call_refine(history: list[dict]) -> str:
    messages = [("system", REFINE_SYSTEM)]
    for h in history:
        role = "user" if h["role"] == "user" else "assistant"
        messages.append((role, h["content"]))
    prompt = ChatPromptTemplate.from_messages(messages)
    result = invoke_with_retry(prompt | llm_refine, {}, "灵感精炼")
    return result.content.strip() if result and result.content else ""


# ═══════════════════════════════════════════════════
#  HTML 前端 (嵌入式单页)
# ═══════════════════════════════════════════════════
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoWrite 小说工业流水线</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Microsoft YaHei','PingFang SC',sans-serif;background:#0f1117;color:#c9d1d9;height:100vh;overflow:hidden}
#app{display:grid;grid-template-columns:minmax(680px,48vw) minmax(420px,1fr);grid-template-rows:1fr;height:100vh;min-width:1100px}
#panel{background:#161b22;border-right:1px solid #30363d;display:flex;flex-direction:column;overflow-y:auto;padding:20px 24px}
#panel h1{font-size:18px;color:#58a6ff;margin-bottom:12px;text-align:center}
.section{margin-bottom:16px}
.section label{display:block;font-size:13px;color:#8b949e;margin-bottom:6px}
#idea{width:100%;height:96px;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:8px;border-radius:6px;resize:vertical;font-size:13px}
.cat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px;max-height:260px;overflow-y:auto}
.cat-item{display:flex;align-items:center;gap:6px;font-size:12px;padding:4px 6px;background:#0d1117;border-radius:4px;cursor:pointer;border:1px solid #30363d;transition:all .15s}
.cat-item.active{border-color:#58a6ff;background:#1a2332}
.cat-item.disabled{opacity:.45;cursor:not-allowed}
.cat-item input{accent-color:#58a6ff}
.cat-tag{margin-left:auto;color:#8b949e;font-size:10px;white-space:nowrap}
.material-groups{display:grid;gap:10px;max-height:380px;overflow-y:auto;padding-right:3px}
.material-group{border:1px solid #30363d;border-radius:7px;background:#0d1117;overflow:hidden}
.material-group-head{display:flex;align-items:center;gap:8px;padding:7px 9px;border-bottom:1px solid #30363d;background:#161b22}
.material-group-head strong{color:#58a6ff;font-size:12px}
.material-group-head .quota{margin-left:auto;display:flex;align-items:center;gap:5px;font-size:11px;color:#8b949e}
.material-group-head select{background:#0d1117;border:1px solid #30363d;color:#c9d1d9;border-radius:4px;padding:3px 5px}
.material-group-body{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:6px;padding:7px}
.material-card{display:flex;align-items:flex-start;gap:7px;padding:7px;border:1px solid #30363d;border-radius:6px;background:#0d1117;font-size:11px}
.material-card .material-text{flex:1;line-height:1.5}
.material-card-actions{display:flex;gap:4px;flex-wrap:wrap}
.material-card-actions button{width:auto;padding:3px 6px;font-size:10px}
.scope-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.scope-row select,.scope-row input{background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:6px 8px;border-radius:4px;font-size:12px}
.scope-row input{width:80px}
.btn{display:block;width:100%;padding:10px;border:none;border-radius:6px;font-size:14px;cursor:pointer;font-weight:600;transition:all .15s}
.btn-primary{background:#238636;color:#fff}
.btn-primary:hover{background:#2ea043}
.btn-primary:disabled{background:#30363d;color:#8b949e;cursor:not-allowed}
.btn-danger{background:#da3633;color:#fff}
.btn-warning{background:#d29922;color:#000}
.btn-bar{display:flex;gap:8px;margin-top:4px}
.btn-bar .btn{flex:1;padding:8px;font-size:12px}
#right{display:flex;flex-direction:column;overflow:hidden}
#tabs{display:flex;background:#161b22;border-bottom:1px solid #30363d;flex-shrink:0}
.tab{padding:10px 20px;font-size:13px;cursor:pointer;border-bottom:2px solid transparent;color:#8b949e;transition:all .15s}
.tab.active{color:#58a6ff;border-bottom-color:#58a6ff}
#content{flex:1;overflow-y:auto;padding:16px;background:#0d1117}
#content pre{white-space:pre-wrap;font-family:inherit;font-size:13px;line-height:1.7;color:#c9d1d9}
#log-panel{height:180px;overflow-y:auto;background:#0d1117;border-top:1px solid #30363d;padding:8px 16px;flex-shrink:0}
.log-item{font-size:12px;padding:3px 0;border-bottom:1px solid #161b22;font-family:'Consolas','Courier New',monospace}
.log-item.info{color:#8b949e}
.log-item.success{color:#3fb950}
.log-item.warn{color:#d29922}
.log-item.error{color:#f85149}
.agent-bar{display:flex;align-items:center;gap:4px;margin:2px 0;font-size:12px;flex-wrap:wrap}
.agent-dot{width:8px;height:8px;border-radius:50%;background:#30363d;transition:background .3s}
.agent-dot.running{background:#d29922;animation:pulse .8s infinite}
.agent-dot.done{background:#3fb950}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.approval-bar{display:flex;gap:8px;margin:8px 0}
.approval-bar .btn{flex:1;padding:8px;font-size:13px}
.hidden{display:none!important}
.mode-tab{flex:1;text-align:center;padding:8px;font-size:13px;cursor:pointer;border-bottom:2px solid #30363d;color:#8b949e;transition:all .15s}
.mode-tab.active{color:#58a6ff;border-bottom-color:#58a6ff}
.pattern-tabs{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:8px}
.pattern-tab{text-align:center;padding:7px 4px;border:1px solid #30363d;border-radius:6px;background:#0d1117;color:#8b949e;font-size:12px;cursor:pointer;transition:all .15s}
.pattern-tab.active{border-color:#58a6ff;background:#1a2332;color:#58a6ff}
.pattern-tab.disabled{opacity:.45;cursor:not-allowed}
.pattern-select{width:100%;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:6px 8px;border-radius:4px;font-size:12px}
.pattern-value{display:none}
.outline-item{padding:8px 10px;border-bottom:1px solid #161b22;cursor:pointer;transition:all .15s;display:flex;justify-content:space-between;align-items:center}
.outline-item:hover{background:#1a2332}
.outline-item.active{background:#1a2332;border-left:3px solid #58a6ff}
.outline-item .otitle{color:#c9d1d9;font-weight:600}
.outline-item .ometa{color:#8b949e;font-size:10px}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:#0d1117}
::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}
</style>
</head>
<body>
<div id="app">
  <div id="panel">
    <div id="modeBar" style="display:flex;margin-bottom:12px">
      <div class="mode-tab active" onclick="switchMode('create')">🎨 创作</div>
      <div class="mode-tab" onclick="switchMode('wash')">🔄 洗文</div>
    </div>
    <h1>🚀 AutoWrite 小说流水线</h1>
    <div class="agent-bar" id="agentStatus">
      <span class="agent-dot" id="dot-architect"></span>架构师
      <span class="agent-dot" id="dot-writer"></span>写手
      <span class="agent-dot" id="dot-reviewer"></span>审稿
      <span class="agent-dot" id="dot-summarizer"></span>书记
    </div>
    <div id="createPanel">
    <div class="section">
      <label>📝 小说灵感</label>
      <textarea id="idea" placeholder="输入你的创意点子..."></textarea>
      <div class="btn-bar" style="margin-top:4px">
        <button class="btn" style="flex:1;padding:6px;font-size:12px;background:#1f6feb;color:#fff" onclick="startRefine()">🔍 AI 精炼灵感</button>
        <button class="btn" style="flex:1;padding:6px;font-size:12px;background:#30363d;color:#c9d1d9" onclick="skipRefine()">跳过</button>
      </div>
    </div>
    <div class="section hidden" id="refineSection">
      <label>💬 灵感精炼</label>
      <div id="refineChat" style="background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px;max-height:200px;overflow-y:auto;font-size:12px;margin-bottom:6px"></div>
      <div style="display:flex;gap:6px">
        <textarea id="refineAnswer" placeholder="输入你的回答..." style="flex:1;height:48px;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:6px;border-radius:4px;font-size:12px;resize:none"></textarea>
        <button class="btn btn-primary" style="width:50px;padding:6px;font-size:12px" onclick="sendRefineAnswer()">发送</button>
      </div>
    </div>
    <div class="section hidden" id="refineResultSection">
      <label>✨ 精炼后的设定</label>
      <div id="refineResult" style="background:#0d1117;border:1px solid #3fb950;border-radius:6px;padding:8px;font-size:12px;line-height:1.6;color:#c9d1d9;max-height:150px;overflow-y:auto"></div>
      <div class="btn-bar" style="margin-top:4px">
        <button class="btn btn-primary" onclick="confirmRefine()">✅ 使用此设定</button>
        <button class="btn" style="background:#30363d;color:#c9d1d9" onclick="editRefineResult()">✏️ 手动修改</button>
      </div>
    </div>
    <div class="section">
      <label>✍️ 写手风格</label>
      <select id="writerStyle" onchange="onWriterStyleChange('create')" style="width:100%;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:6px 8px;border-radius:4px;font-size:12px">
        <option value="default">默认自然</option>
      </select>
    </div>
    <div class="section">
      <label>🎭 创作套路</label>
      <div class="pattern-tabs">
        <div class="pattern-tab active" data-pattern-mode="none" onclick="switchPatternMode('create','none')">无套路</div>
        <div class="pattern-tab" data-pattern-mode="normal" onclick="switchPatternMode('create','normal')">普通套路</div>
        <div class="pattern-tab" data-pattern-mode="strong" onclick="switchPatternMode('create','strong')">强套路</div>
      </div>
      <input id="primaryPatternSearch" class="pattern-select" placeholder="搜索主套路名称、分类或标签" oninput="syncPatternControls('create')" style="margin-bottom:6px">
      <select id="storyPattern" class="pattern-value" onchange="onPatternChange('create')">
        <option value="none">无套路</option>
        <option value="wife_chasing">追妻火葬场</option>
        <option value="rule_horror">规则怪谈</option>
        <option value="counterattack">逆袭打脸</option>
        <option value="marriage_first">先婚后爱</option>
        <option value="infinite_trials">无限流闯关</option>
        <option value="revenge_rebirth">复仇重生</option>
        <option value="female_angst_awakening">女频虐恋觉醒</option>
        <option value="strong_rule_horror">强规则怪谈</option>
        <option value="strong_historical_power">历史权谋强套路</option>
        <option value="strong_male_power_progression">男频升级打脸强套路</option>
        <option value="male_angst_awakening">虐恋觉醒性转</option>
        <option value="custom">自定义套路</option>
      </select>
      <div id="normalPatternPanel" class="hidden">
        <select id="normalStoryPattern" class="pattern-select" onchange="selectVisiblePattern('create','normal')"></select>
      </div>
      <div id="strongPatternPanel" class="hidden">
        <select id="strongStoryPattern" class="pattern-select" onchange="selectVisiblePattern('create','strong')"></select>
      </div>
      <div id="customPatternWrap" class="hidden" style="margin-top:6px">
        <textarea id="customPattern" placeholder="描述套路节拍、必须出现的桥段与禁忌..." style="width:100%;height:64px;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:6px;border-radius:4px;font-size:12px;resize:vertical"></textarea>
      </div>
      <div style="margin-top:8px">
        <label>辅助套路（最多2个，仅作软约束）</label>
        <input id="secondaryPatternSearch" class="pattern-select" placeholder="搜索辅助套路" oninput="renderSecondaryPatterns('create')">
        <div id="secondaryPatterns" class="cat-grid" style="margin-top:6px;max-height:150px"></div>
      </div>
      <div id="patternManifestSection" class="hidden" style="margin-top:8px;background:#0d1117;border:1px solid #d29922;border-radius:6px;padding:8px">
        <label style="margin-bottom:4px">结局方向</label>
        <select id="patternEnding" onchange="changePatternEnding('create')" style="width:100%;margin-bottom:6px;background:#161b22;border:1px solid #30363d;color:#c9d1d9;padding:5px;border-radius:4px;font-size:11px">
          <option value="no_reunion">女主彻底离开，不复合</option>
          <option value="costly_reunion">男主付出高代价后，由女主决定是否复合</option>
        </select>
        <div id="patternManifestSummary" style="font-size:11px;line-height:1.6;white-space:pre-wrap"></div>
        <textarea id="patternManifestEditor" class="hidden" style="width:100%;height:160px;margin-top:6px;background:#161b22;border:1px solid #30363d;color:#c9d1d9;padding:6px;border-radius:4px;font-size:11px"></textarea>
        <div class="btn-bar" style="margin-top:6px">
          <button class="btn btn-primary" onclick="confirmPatternManifest('create')">确认契约</button>
          <button class="btn btn-warning" onclick="rollPatternManifest('create')">重抽</button>
          <button class="btn" style="background:#30363d;color:#c9d1d9" onclick="editPatternManifest('create')">手动修改</button>
        </div>
      </div>
    </div>
    <div class="section" id="kwSection">
      <label>📚 结构化素材库</label>
      <div id="materialHint" style="font-size:11px;color:#8b949e;margin-bottom:6px;line-height:1.5">按八个大类分别设置0/1/2项配额；世界舞台、主角人设最多1项，其余最多2项。与主辅套路硬冲突的分类会被禁用。</div>
      <div class="scope-row" style="margin-bottom:6px">
        <input id="materialSearch" placeholder="搜索分类" style="width:150px" oninput="renderCategories()">
        总数 <strong id="materialTotal" style="color:#58a6ff">4</strong>
        <button class="btn btn-warning" style="width:auto;padding:6px 12px" onclick="drawMaterials('create',false)">重抽素材</button>
        <button class="btn" style="width:auto;padding:6px 12px;background:#30363d;color:#c9d1d9" onclick="drawMaterials('create',true)">换类型重抽</button>
      </div>
      <div class="material-groups" id="catGrid"></div>
      <div id="materialCards" style="margin-top:8px;display:grid;gap:6px"></div>
    </div>
    <div class="section">
      <label>📏 篇幅设置</label>
      <div class="scope-row">
        章节数 <input id="chapters" type="number" value="8" min="1" max="200" style="width:60px"> 章
        &nbsp;每章 <input id="wordsPerCh" type="number" value="1500" min="500" max="10000" step="100" style="width:70px"> 字
      </div>
      <div style="font-size:11px;color:#8b949e;margin-top:4px" id="estWords">预估: 约 12,000 字</div>
    </div>
    <button class="btn btn-primary" id="btnStart" onclick="startPipeline()">▶ 启动流水线</button>
    <div class="approval-bar hidden" id="approvalBar">
      <button class="btn btn-primary" onclick="sendCmd('approval',true)">✅ 批准大纲，开始写作</button>
      <button class="btn btn-danger" onclick="sendCmd('approval',false)">❌ 拒绝，重新设定</button>
    </div>
    <p id="progressStatus" style="font-size:11px;color:#8b949e;margin-top:8px"></p>
  </div><!-- /createPanel -->
  <div id="washPanel" class="hidden">
    <div class="section">
      <label>✍️ 写手风格</label>
      <select id="washWriterStyle" onchange="onWriterStyleChange('wash')" style="width:100%;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:6px 8px;border-radius:4px;font-size:12px">
        <option value="default">默认自然</option>
      </select>
    </div>
    <div class="section">
      <label>🎭 创作套路</label>
      <div class="pattern-tabs">
        <div class="pattern-tab active" data-pattern-mode="none" onclick="switchPatternMode('wash','none')">无套路</div>
        <div class="pattern-tab" data-pattern-mode="normal" onclick="switchPatternMode('wash','normal')">普通套路</div>
        <div class="pattern-tab" data-pattern-mode="strong" onclick="switchPatternMode('wash','strong')">强套路</div>
      </div>
      <input id="washPrimaryPatternSearch" class="pattern-select" placeholder="搜索主套路名称、分类或标签" oninput="syncPatternControls('wash')" style="margin-bottom:6px">
      <select id="washStoryPattern" class="pattern-value" onchange="onPatternChange('wash')">
        <option value="none">无套路</option>
        <option value="wife_chasing">追妻火葬场</option>
        <option value="rule_horror">规则怪谈</option>
        <option value="counterattack">逆袭打脸</option>
        <option value="marriage_first">先婚后爱</option>
        <option value="infinite_trials">无限流闯关</option>
        <option value="revenge_rebirth">复仇重生</option>
        <option value="female_angst_awakening">女频虐恋觉醒</option>
        <option value="strong_rule_horror">强规则怪谈</option>
        <option value="strong_historical_power">历史权谋强套路</option>
        <option value="strong_male_power_progression">男频升级打脸强套路</option>
        <option value="male_angst_awakening">虐恋觉醒性转</option>
        <option value="custom">自定义套路</option>
      </select>
      <div id="washNormalPatternPanel" class="hidden">
        <select id="washNormalStoryPattern" class="pattern-select" onchange="selectVisiblePattern('wash','normal')"></select>
      </div>
      <div id="washStrongPatternPanel" class="hidden">
        <select id="washStrongStoryPattern" class="pattern-select" onchange="selectVisiblePattern('wash','strong')"></select>
      </div>
      <div id="washCustomPatternWrap" class="hidden" style="margin-top:6px">
        <textarea id="washCustomPattern" placeholder="描述套路节拍、必须出现的桥段与禁忌..." style="width:100%;height:64px;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:6px;border-radius:4px;font-size:12px;resize:vertical"></textarea>
      </div>
      <div style="margin-top:8px">
        <label>辅助套路（最多2个，仅作软约束）</label>
        <input id="washSecondaryPatternSearch" class="pattern-select" placeholder="搜索辅助套路" oninput="renderSecondaryPatterns('wash')">
        <div id="washSecondaryPatterns" class="cat-grid" style="margin-top:6px;max-height:150px"></div>
      </div>
      <div id="washPatternManifestSection" class="hidden" style="margin-top:8px;background:#0d1117;border:1px solid #d29922;border-radius:6px;padding:8px">
        <label style="margin-bottom:4px">结局方向</label>
        <select id="washPatternEnding" onchange="changePatternEnding('wash')" style="width:100%;margin-bottom:6px;background:#161b22;border:1px solid #30363d;color:#c9d1d9;padding:5px;border-radius:4px;font-size:11px">
          <option value="no_reunion">女主彻底离开，不复合</option>
          <option value="costly_reunion">男主付出高代价后，由女主决定是否复合</option>
        </select>
        <div id="washPatternManifestSummary" style="font-size:11px;line-height:1.6;white-space:pre-wrap"></div>
        <textarea id="washPatternManifestEditor" class="hidden" style="width:100%;height:160px;margin-top:6px;background:#161b22;border:1px solid #30363d;color:#c9d1d9;padding:6px;border-radius:4px;font-size:11px"></textarea>
        <div class="btn-bar" style="margin-top:6px">
          <button class="btn btn-primary" onclick="confirmPatternManifest('wash')">确认契约</button>
          <button class="btn btn-warning" onclick="rollPatternManifest('wash')">重抽</button>
          <button class="btn" style="background:#30363d;color:#c9d1d9" onclick="editPatternManifest('wash')">手动修改</button>
        </div>
      </div>
    </div>
    <div class="section">
      <label>📚 洗文素材配置</label>
      <div style="font-size:11px;color:#8b949e;margin-bottom:6px;line-height:1.5">按八个大类分别设置配额；同类多项会作为独立卡片，可分别锁定和重抽。</div>
      <div class="scope-row" style="margin-bottom:6px">
        <input id="washMaterialSearch" placeholder="搜索分类" style="width:150px" oninput="renderCategories('wash')">
        总数 <strong id="washMaterialTotal" style="color:#58a6ff">4</strong>
        <button class="btn btn-warning" style="width:auto;padding:6px 12px" onclick="drawMaterials('wash',false)">重抽素材</button>
        <button class="btn" style="width:auto;padding:6px 12px;background:#30363d;color:#c9d1d9" onclick="drawMaterials('wash',true)">换类型重抽</button>
      </div>
      <div class="material-groups" id="washCatGrid"></div>
      <div id="washMaterialCards" style="margin-top:8px;display:grid;gap:6px"></div>
    </div>
    <div class="section">
      <label>📏 篇幅 (0=保持原大纲)</label>
      <div class="scope-row">
        章节数 <input id="washChapters" type="number" value="0" min="0" max="200" style="width:60px"> 章
        &nbsp;每章 <input id="washWords" type="number" value="1500" min="500" max="10000" step="100" style="width:70px"> 字
      </div>
    </div>
    <div class="section">
      <label>📚 大纲列表</label>
      <div id="outlineList" style="background:#0d1117;border:1px solid #30363d;border-radius:6px;max-height:240px;overflow-y:auto;font-size:12px">
        <div style="color:#8b949e;padding:12px;text-align:center">点击"洗文"标签后自动加载...</div>
      </div>
    </div>
    <button class="btn btn-primary" id="btnWashStart" disabled onclick="startWash()">▶ 确认创作</button>
    <p id="washStatus" style="font-size:11px;color:#8b949e;margin-top:4px"></p>
  </div>
  </div><!-- /panel -->
  <div id="right">
    <div id="tabs">
      <div class="tab active" onclick="switchTab('outline')">📋 大纲</div>
      <div class="tab" onclick="switchTab('novel')">📖 正文</div>
    </div>
    <div id="content">
      <pre id="outlineArea" style="white-space:pre-wrap;font-family:inherit;font-size:13px;line-height:1.7;color:#c9d1d9">等待启动...</pre>
      <pre id="novelArea" class="hidden" style="white-space:pre-wrap;font-family:inherit;font-size:13px;line-height:1.7;color:#c9d1d9"></pre>
    </div>
    <div id="log-panel"><div id="logArea"></div></div>
  </div>
</div>

<script>
let ws=null,token=0,selectedCats=[];
let materialMeta={groups:{}},patternMeta={};
const defaultGroupCounts={world_stage:1,protagonist:1,supporting_role:0,cheat_device:1,plot_event:0,core_conflict:1,career_resource:0,atmosphere:0};
let materialConfig={schema_version:2,filters:{categories:[],subcategories:[],tags:[]},group_counts:{...defaultGroupCounts},count:4,items:[],locked_item_keys:[],auto_selected_subcategories:[]};
let washSelectedCats=[];
let washMaterialConfig={schema_version:2,filters:{categories:[],subcategories:[],tags:[]},group_counts:{...defaultGroupCounts},count:4,items:[],locked_item_keys:[],auto_selected_subcategories:[]};
let createSecondaryPatterns=[],washSecondaryPatterns=[];
let createPatternManifest=null,washPatternManifest=null;
let createPatternConfirmed=false,washPatternConfirmed=false;
const agentMap={architect:'dot-architect',writer:'dot-writer',reviewer:'dot-reviewer',summarizer:'dot-summarizer'};
const agentNames={architect:'架构师',writer:'写手',reviewer:'审稿员',summarizer:'书记员'};

function log(msg,cls='info'){
  let d=document.getElementById('logArea');
  d.innerHTML+=`<div class="log-item ${cls}">${msg}</div>`;
  d.scrollTop=d.scrollHeight;
}

function connect(){
  let proto=location.protocol==='https:'?'wss':'ws';
  ws=new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen=()=>{log('✅ 已连接服务器','success');loadCategories()};
  ws.onmessage=e=>handleMsg(JSON.parse(e.data));
  ws.onclose=()=>{log('⚠️ 连接断开，3秒后重连...','warn');setTimeout(connect,3000)};
  ws.onerror=()=>log('❌ 连接错误','error');
}

function sendMsg(obj){if(ws&&ws.readyState===1)ws.send(JSON.stringify(obj))}
function sendCmd(action,data){
  sendMsg({action,data,token});
  if(action==='approval' && data){
    document.getElementById('approvalBar').classList.add('hidden');
    document.getElementById('progressStatus').textContent='▶ 写作流水线启动中...';
    document.querySelectorAll('#approvalBar .btn').forEach(b=>b.disabled=true);
    log('▶ 大纲已批准，写作流水线启动...','success');
  }
}

function loadCategories(){
  sendMsg({action:'get_material_library'});
  sendMsg({action:'get_patterns'});
  sendMsg({action:'get_writer_styles'});
}

function escapeHtml(text){
  return String(text||'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

function isStrongPatternKey(pattern){
  return !!(patternMeta[pattern]&&patternMeta[pattern].strong);
}

function patternKind(pattern){
  if(!pattern||pattern==='none')return 'none';
  return isStrongPatternKey(pattern)?'strong':'normal';
}

function patternEntries(kind){
  return Object.entries(patternMeta||{}).filter(([key,item])=>{
    if(kind==='normal')return key!=='none'&&!item.strong;
    if(kind==='strong')return !!item.strong;
    return key==='none';
  });
}

function fillPatternSelect(selectId,entries){
  let select=document.getElementById(selectId);
  if(!select)return;
  select.innerHTML=entries.length
    ? entries.map(([key,item])=>`<option value="${key}">[${escapeHtml(item.category||'其他')}] ${escapeHtml(item.name||key)}</option>`).join('')
    : '<option value="">暂无可选套路</option>';
  select.disabled=!entries.length;
}

function endingOptionsFor(pattern){
  let options=(patternMeta[pattern]&&patternMeta[pattern].ending_options)||{};
  if(Object.keys(options).length)return options;
  return {
    no_reunion:'女主彻底离开并拥有更好生活，男主追悔但无法挽回。',
    costly_reunion:'男主付出长期且不可逆的代价后，女主自主决定是否重新开始。'
  };
}

function updateEndingOptions(mode,pattern,selected){
  let ids=patternIds(mode);
  let select=document.getElementById(ids.ending);
  if(!select)return;
  let options=endingOptionsFor(pattern);
  select.innerHTML=Object.entries(options).map(([key,text])=>
    `<option value="${key}">${escapeHtml(text)}</option>`
  ).join('');
  let target=selected&&options[selected]?selected:Object.keys(options)[0];
  select.value=target||'';
}

function syncPatternControls(mode){
  let ids=patternIds(mode);
  let hidden=document.getElementById(ids.select);
  if(!hidden)return;
  let previous=hidden.value||'none';
  let allOptions=Object.entries(patternMeta||{}).map(([key,item])=>
    `<option value="${key}">${escapeHtml(item.name||key)}</option>`
  ).join('');
  hidden.innerHTML=allOptions||'<option value="none">无套路</option>';
  hidden.value=[...hidden.options].some(option=>option.value===previous)?previous:'none';

  let search=(document.getElementById(mode==='wash'?'washPrimaryPatternSearch':'primaryPatternSearch')?.value||'').trim().toLowerCase();
  let matches=entries=>entries.filter(([key,item])=>{
    if(!search)return true;
    return `${item.name||''}${item.category||''}${(item.tags||[]).join('')}${key}`.toLowerCase().includes(search);
  });
  fillPatternSelect(ids.normalSelect,matches(patternEntries('normal')));
  fillPatternSelect(ids.strongSelect,matches(patternEntries('strong')));
  renderPatternControls(mode);
  renderSecondaryPatterns(mode);
}

function renderPatternControls(mode){
  let ids=patternIds(mode);
  let hidden=document.getElementById(ids.select);
  if(!hidden)return;
  let activeKind=patternKind(hidden.value||'none');
  document.querySelectorAll(ids.tabs).forEach(tab=>{
    let kind=tab.dataset.patternMode;
    let disabled=kind!=='none'&&patternEntries(kind).length===0;
    tab.classList.toggle('active',kind===activeKind);
    tab.classList.toggle('disabled',disabled);
  });
  document.getElementById(ids.normalPanel).classList.toggle('hidden',activeKind!=='normal');
  document.getElementById(ids.strongPanel).classList.toggle('hidden',activeKind!=='strong');
  let visibleSelect=document.getElementById(activeKind==='strong'?ids.strongSelect:ids.normalSelect);
  if(visibleSelect&&activeKind!=='none'&&[...visibleSelect.options].some(option=>option.value===hidden.value)){
    visibleSelect.value=hidden.value;
  }
}

function setPatternValue(mode,pattern){
  let ids=patternIds(mode);
  document.getElementById(ids.select).value=pattern||'none';
  onPatternChange(mode);
}

function fillWriterStyles(items){
  for(let id of ['writerStyle','washWriterStyle']){
    let select=document.getElementById(id);
    let previous=select.value||'default';
    select.innerHTML=(items||[]).map(item=>
      `<option value="${escapeHtml(item.key)}">${escapeHtml(item.name)}</option>`
    ).join('');
    if([...select.options].some(option=>option.value===previous))select.value=previous;
  }
}

function switchPatternMode(mode,kind){
  if(kind!=='none'&&patternEntries(kind).length===0)return;
  if(kind==='none'){setPatternValue(mode,'none');return}
  let ids=patternIds(mode);
  let select=document.getElementById(kind==='strong'?ids.strongSelect:ids.normalSelect);
  setPatternValue(mode,select.value||patternEntries(kind)[0]?.[0]||'none');
}

function selectVisiblePattern(mode,kind){
  let ids=patternIds(mode);
  let select=document.getElementById(kind==='strong'?ids.strongSelect:ids.normalSelect);
  setPatternValue(mode,select.value||'none');
}

function selectedSecondary(mode){
  return mode==='wash'?washSecondaryPatterns:createSecondaryPatterns;
}

function patternConfigFor(mode){
  let ids=patternIds(mode);
  let primary=document.getElementById(ids.select)?.value||'none';
  let custom=document.getElementById(mode==='wash'?'washCustomPattern':'customPattern')?.value.trim()||'';
  let manifest=mode==='wash'?washPatternManifest:createPatternManifest;
  return {schema_version:2,primary,secondary:[...selectedSecondary(mode)],custom_instruction:custom,manifest:manifest||{},structure_plan:{}};
}

function patternPairConflict(primary,secondary){
  let p=patternMeta[primary]||{},s=patternMeta[secondary]||{};
  if(secondary===primary)return '不能与主套路重复';
  if(s.strong)return '辅助套路不能使用强套路';
  if((p.hard_conflicts||[]).includes(secondary)||(s.hard_conflicts||[]).includes(primary))return '与主套路存在硬冲突';
  return '';
}

function renderSecondaryPatterns(mode){
  let primary=document.getElementById(patternIds(mode).select)?.value||'none';
  let selected=selectedSecondary(mode);
  let search=(document.getElementById(mode==='wash'?'washSecondaryPatternSearch':'secondaryPatternSearch')?.value||'').trim().toLowerCase();
  let target=document.getElementById(mode==='wash'?'washSecondaryPatterns':'secondaryPatterns');
  if(!target)return;
  target.innerHTML=patternEntries('normal').filter(([key,item])=>{
    if(key==='custom')return false;
    return !search||`${item.name||''}${item.category||''}${(item.tags||[]).join('')}`.toLowerCase().includes(search);
  }).map(([key,item])=>{
    let reason=patternPairConflict(primary,key);
    let checked=selected.includes(key);
    let disabled=!!reason||(!checked&&selected.length>=2);
    return `<label class="cat-item ${disabled?'disabled':''}" title="${escapeHtml(reason)}">
      <input type="checkbox" value="${key}" ${checked?'checked':''} ${disabled?'disabled':''} onchange="toggleSecondaryPattern('${mode}','${key}',this.checked)">
      ${escapeHtml(item.name||key)}<span class="cat-tag">${escapeHtml(item.category||'')}</span>
    </label>`;
  }).join('');
}

function toggleSecondaryPattern(mode,key,checked){
  let selected=selectedSecondary(mode);
  if(checked&&!selected.includes(key)&&selected.length<2)selected.push(key);
  if(!checked){
    let idx=selected.indexOf(key);if(idx>=0)selected.splice(idx,1);
  }
  renderSecondaryPatterns(mode);
  if(mode==='create'){
    materialConfig.items=[];renderMaterialCards();renderCategories();
  }else{
    washMaterialConfig.items=[];renderMaterialCards('wash');renderCategories('wash');
  }
}

function isSubcategoryBlocked(groupId,child,mode='create'){
  let config=patternConfigFor(mode);
  for(let patternId of [config.primary,...config.secondary]){
    let pattern=patternMeta[patternId]||{};
    if((pattern.forbidden_material_categories||[]).includes(groupId))return `${pattern.name}禁止该素材大类`;
    if((child.tags||[]).some(tag=>(pattern.forbidden_material_tags||[]).includes(tag)))return `${pattern.name}禁止标签：${(child.tags||[]).join('、')}`;
  }
  return '';
}

function renderCategories(mode='create'){
  let isWash=mode==='wash';
  let search=(document.getElementById(isWash?'washMaterialSearch':'materialSearch')?.value||'').trim().toLowerCase();
  let selected=isWash?washSelectedCats:selectedCats;
  let config=isWash?washMaterialConfig:materialConfig;
  let html=[];
  for(let [groupId,group] of Object.entries(materialMeta.groups||{})){
    let children=[];
    for(let child of group.subcategories||[]){
      let hay=`${group.name||''}${child.name||''}${(child.tags||[]).join('')}`.toLowerCase();
      if(search&&!hay.includes(search))continue;
      let reason=isSubcategoryBlocked(groupId,child,mode);
      let checked=selected.includes(child.id)&&!reason;
      let hitCount=(config.items||[]).filter(item=>item.subcategory===child.id).length;
      children.push(`<label class="cat-item ${reason?'disabled':''}" title="${escapeHtml(reason)}">
        <input type="checkbox" value="${escapeHtml(child.id)}" data-group="${escapeHtml(groupId)}" ${checked?'checked':''} ${reason?'disabled':''} onchange="onCatChange('${mode}')">
        ${escapeHtml(child.name)}<span style="color:#8b949e;font-size:10px">${escapeHtml(group.name)}</span>
        <span class="cat-tag">${hitCount?`命中${hitCount} · `:''}${child.count||0}条</span>
      </label>`);
    }
    if(search&&!children.length)continue;
    let max=group.max_count??1;
    let current=(config.group_counts||{})[groupId]??0;
    let groupBlocked=(group.subcategories||[]).length>0&&(group.subcategories||[]).every(
      child=>!!isSubcategoryBlocked(groupId,child,mode)
    );
    if(groupBlocked&&current>0){
      config.group_counts[groupId]=0;
      config.items=(config.items||[]).filter(item=>item.category!==groupId);
      let remainingKeys=new Set((config.items||[]).map(item=>item.selection_key));
      config.locked_item_keys=(config.locked_item_keys||[]).filter(key=>remainingKeys.has(key));
      current=0;
    }
    let options=Array.from({length:max+1},(_,index)=>
      `<option value="${index}" ${index===current?'selected':''}>${index}</option>`
    ).join('');
    html.push(`<section class="material-group">
      <div class="material-group-head">
        <strong>${escapeHtml(group.name||groupId)}</strong>
        <span style="color:#8b949e;font-size:10px">${group.count||0}条素材</span>
        <label class="quota">抽取数量
          <select onchange="changeGroupCount('${mode}','${groupId}',this.value)" ${groupBlocked?'disabled':''}>${options}</select>
          / ${max}
        </label>
      </div>
      <div class="material-group-body">${children.join('')||'<span style="color:#8b949e;font-size:11px">无匹配子类</span>'}</div>
    </section>`);
  }
  document.getElementById(isWash?'washCatGrid':'catGrid').innerHTML=html.join('');
  updateMaterialTotal(mode);
}

function updateMaterialTotal(mode='create'){
  let config=mode==='wash'?washMaterialConfig:materialConfig;
  let total=Object.values(config.group_counts||{}).reduce((sum,value)=>sum+(parseInt(value)||0),0);
  config.count=total;
  document.getElementById(mode==='wash'?'washMaterialTotal':'materialTotal').textContent=String(total);
}

function changeGroupCount(mode,groupId,value){
  let config=mode==='wash'?washMaterialConfig:materialConfig;
  config.group_counts={...(config.group_counts||defaultGroupCounts),[groupId]:parseInt(value)||0};
  let desired=new Set(Object.entries(config.group_counts).flatMap(([group,count])=>
    Array.from({length:count},(_,index)=>`${group}:${index+1}`)
  ));
  config.items=(config.items||[]).filter(item=>desired.has(item.selection_key));
  config.locked_item_keys=(config.locked_item_keys||[]).filter(key=>desired.has(key));
  updateMaterialTotal(mode);
  renderMaterialCards(mode);
}

function drawMaterials(mode='create',randomizeTypes=false){
  let isWash=mode==='wash';
  let selected=isWash?washSelectedCats:selectedCats;
  let config=isWash?washMaterialConfig:materialConfig;
  updateMaterialTotal(mode);
  if(config.count<2||config.count>8){log('素材总数必须在2到8之间','warn');return}
  config={...config,filters:{categories:[],subcategories:[...selected],tags:[]}};
  if(isWash)washMaterialConfig=config;else materialConfig=config;
  sendMsg({action:'sample_materials',data:{mode,randomize_types:randomizeTypes,material_config:config,pattern_config:patternConfigFor(mode)}});
}

function rerollMaterial(selectionKey,mode='create',changeType=false){
  let config=mode==='wash'?washMaterialConfig:materialConfig;
  sendMsg({action:'resample_material',data:{mode,selection_key:selectionKey,change_type:changeType,material_config:config,pattern_config:patternConfigFor(mode)}});
}

function toggleMaterialLock(selectionKey,mode='create'){
  let config=mode==='wash'?washMaterialConfig:materialConfig;
  let locks=new Set(config.locked_item_keys||[]);
  if(locks.has(selectionKey))locks.delete(selectionKey);else locks.add(selectionKey);
  config.locked_item_keys=[...locks];
  renderMaterialCards(mode);
}

function renderMaterialCards(mode='create'){
  let isWash=mode==='wash';
  let config=isWash?washMaterialConfig:materialConfig;
  let target=document.getElementById(isWash?'washMaterialCards':'materialCards');
  if(!target)return;
  target.innerHTML=(config.items||[]).map(item=>{
    let group=materialMeta.groups?.[item.category]||{};
    let child=(group.subcategories||[]).find(value=>value.id===item.subcategory)||{};
    let locked=(config.locked_item_keys||[]).includes(item.selection_key);
    return `<div class="material-card">
    <span style="color:#58a6ff;white-space:nowrap">${escapeHtml(group.name||item.category)} ${escapeHtml(item.selection_key.split(':').pop())}</span>
    <div class="material-text"><strong>${escapeHtml(child.name||item.subcategory)}</strong><br>${escapeHtml(item.text)}</div>
    <div class="material-card-actions">
      <button class="btn" style="background:${locked?'#1f6feb':'#30363d'};color:#fff" onclick="toggleMaterialLock('${escapeHtml(item.selection_key)}','${mode}')">${locked?'已锁定':'锁定'}</button>
      <button class="btn" style="background:#30363d;color:#c9d1d9" onclick="rerollMaterial('${escapeHtml(item.selection_key)}','${mode}',false)">重抽</button>
      <button class="btn" style="background:#30363d;color:#c9d1d9" onclick="rerollMaterial('${escapeHtml(item.selection_key)}','${mode}',true)">换类型</button>
    </div>
  </div>`;
  }).join('');
}

function handleMsg(msg){
  switch(msg.type){
    case 'material_library':
      materialMeta=msg.data||{groups:{}};
      renderCategories();
      renderCategories('wash');
      break;
    case 'material_result':
      if(msg.mode==='wash'){
        washMaterialConfig=msg.data||washMaterialConfig;
        washSelectedCats=[...((washMaterialConfig.filters||{}).subcategories||[])];
        updateMaterialTotal('wash');
        renderMaterialCards('wash');
        renderCategories('wash');
      }else{
        materialConfig=msg.data||materialConfig;
        selectedCats=[...((materialConfig.filters||{}).subcategories||[])];
        updateMaterialTotal('create');
        renderMaterialCards('create');
        renderCategories('create');
      }
      log('✅ 素材已生成，可独立锁定、重抽或更换类型','success');
      break;
    case 'patterns':
      patternMeta=msg.data||{};
      syncPatternControls('create');
      syncPatternControls('wash');
      renderCategories();
      renderCategories('wash');
      break;
    case 'writer_styles':
      fillWriterStyles(msg.data||[]);
      break;
    case 'pattern_manifest_result':
      applyPatternManifest(msg.mode||'create',msg.data||{},false);
      break;
    case 'refine_question':
      document.getElementById('refineChat').innerHTML+=`<div style="color:#d29922;margin:4px 0">🤖 AI: ${msg.question}</div>`;
      document.getElementById('refineChat').scrollTop=document.getElementById('refineChat').scrollHeight;
      break;
    case 'refine_done':
      document.getElementById('refineChat').innerHTML+=`<div style="color:#3fb950;margin:4px 0">✨ 精炼完成</div>`;
      document.getElementById('refineChat').scrollTop=document.getElementById('refineChat').scrollHeight;
      refinedIdea=msg.refined;
      document.getElementById('refineResult').textContent=msg.refined;
      document.getElementById('refineResultSection').classList.remove('hidden');
      document.getElementById('refineAnswer').disabled=true;
      document.querySelector('#refineSection .btn-primary').disabled=true;
      break;
    case 'refine_skip':
      skipRefine();
      break;
    case 'outline_list':
      let listHtml='';
      if(!msg.data||!msg.data.length){
        listHtml='<div style="color:#8b949e;padding:12px;text-align:center">暂无可洗文大纲<br><span style="font-size:10px">请先在"创作"模式生成大纲</span></div>';
      }else{
        for(let o of msg.data){
          listHtml+=`<div class="outline-item" data-file="${o.file}" onclick="selectOutline('${o.file}')">
            <span class="otitle">${o.title}</span>
            <span class="ometa">${o.chapters}章 · ${(o.created_at||'').slice(0,10)}</span>
          </div>`;
        }
      }
      document.getElementById('outlineList').innerHTML=listHtml;
      break;
    case 'outline_content':
      selectedOutlineData=msg.data;
      let ocText=`《${msg.data.title||'未命名'}》\n\n${msg.data.world_bible||''}\n\n======== 章节细纲 ========\n\n`;
      for(let[k,v]of Object.entries(msg.data.chapter_outlines||{})){
        ocText+='第'+k+'章: '+v.replace(/\\n/g,'\n')+'\n\n';
      }
      document.getElementById('outlineArea').textContent=ocText;
      document.getElementById('washChapters').value=msg.data.chapter_outlines?Object.keys(msg.data.chapter_outlines).length:0;
      let loadedPattern=msg.data.pattern_config||{primary:'none',secondary:[],custom_instruction:'',manifest:{}};
      document.getElementById('washStoryPattern').value=loadedPattern.primary||'none';
      document.getElementById('washCustomPattern').value=loadedPattern.custom_instruction||'';
      washSecondaryPatterns=[...(loadedPattern.secondary||[])];
      washMaterialConfig=msg.data.material_config||washMaterialConfig;
      washSelectedCats=[...((washMaterialConfig.filters||{}).subcategories||[])];
      updateMaterialTotal('wash');
      renderPatternControls('wash');
      renderSecondaryPatterns('wash');
      renderCategories('wash');
      renderMaterialCards('wash');
      if(loadedPattern.manifest&&Object.keys(loadedPattern.manifest).length){
        applyPatternManifest('wash',loadedPattern.manifest,true);
      }else{
        washPatternManifest=null;washPatternConfirmed=false;
        document.getElementById('washPatternManifestSection').classList.add('hidden');
        setCompatibleStyles('wash',(patternMeta[loadedPattern.primary]||{}).compatible_styles||[]);
      }
      break;
    case 'rewash_title':
      document.getElementById('washStatus').textContent='✨ 新书名: 《'+msg.title+'》';
      break;
    case 'architect_start':
      log('🧠 架构师正在推演大纲...','info');
      setAgentState('architect','running');
      break;
    case 'architect_result':
      setAgentState('architect','done');
      log(`✅ 架构师完成大纲 — 《${msg.data.novel_title||'未命名'}》`,'success');
      let outlineText='';
      if(msg.data.novel_title) outlineText+=`《${msg.data.novel_title}》\n\n`;
      outlineText+=msg.data.world_bible+'\n\n======== 章节细纲 ========\n\n';
      for(let[k,v]of Object.entries(msg.data.chapter_outlines||{})){
        outlineText+='第'+k+'章: '+v.replace(/\\n/g,'\n')+'\n\n';
      }
      document.getElementById('outlineArea').textContent=outlineText;
      document.getElementById('approvalBar').classList.remove('hidden');
      document.getElementById('progressStatus').textContent='⏸️ 请审批大纲';
      document.getElementById('novelArea').textContent='';
      break;
    case 'node_start':
      log(`${['🧠','✍️','🕵️','👓','📝'][['architect','writer','auditor','editor','summarizer'].indexOf(msg.node)]||'▶'} ${agentNames[msg.node]||msg.node} 工作中...`,'info');
      setAgentState(msg.node,'running');
      document.getElementById('progressStatus').textContent=`${agentNames[msg.node]||msg.node} 执行中...`;
      break;
    case 'node_done':
      setAgentState(msg.node,'done');
      if(msg.node==='summarizer'){
        log('💾 '+msg.message,'success');
      }else if(msg.node==='writer'){
        log(msg.message,'info');
      }
      break;
    case 'node_done_review':
      setAgentState('reviewer','done');
      if(msg.data){
        let a=msg.data.audit_report||{};
        let e=msg.data.editor_report||{};
        let logicStatus=(a['发现的问题']||[]).length?'不通过':'通过';
        log(`  审稿: 逻辑${logicStatus} | 套路${a['套路执行状态']||'通过'} | 文风${e['文风评分']}/10`,(a['审核状态']==='通过'&&e['文风评分']>=7)?'success':'warn');
        for(let warning of (a['警告']||[]))log('  逻辑警告: '+warning,'warn');
        for(let issue of (a['套路问题']||[]))log('  套路问题: '+issue,'warn');
        for(let issue of (e['AI痕迹问题']||[]))log('  AI痕迹: '+issue,'warn');
        for(let warning of (e['AI痕迹警告']||[]))log('  AI提醒: '+warning,'info');
      }
      break;
    case 'chapter_saved':
      let na=document.getElementById('novelArea');
      na.textContent+=msg.data+'\n\n';
      na.scrollTop=na.scrollHeight;
      break;
    case 'pipeline_done':
      log('🎉 流水线完成！全部章节已产出','success');
      document.getElementById('progressStatus').textContent='✅ 全部完成';
      document.getElementById('btnStart').disabled=false;
      break;
    case 'timing_report':
      let t=msg.data||{};
      log(`⏱️ 总耗时 ${(t.wall_seconds||0).toFixed(1)}秒，模型流水线 ${(t.pipeline_seconds||0).toFixed(1)}秒，审批等待 ${(t.approval_wait_seconds||0).toFixed(1)}秒，API重试 ${t.api_retry_count||0}次`,'success');
      for(let [name,item] of Object.entries(t.nodes||{})){
        log(`  ${agentNames[name]||name}: ${item.calls}次 / ${item.total_seconds.toFixed(1)}秒 / 平均${item.average_seconds.toFixed(1)}秒`,'info');
      }
      break;
    case 'log':
      log(msg.message,msg.cls||'info');
      break;
    case 'error':
      log('❌ '+msg.message,'error');
      document.getElementById('btnStart').disabled=isStrongPatternKey(document.getElementById('storyPattern').value)&&!createPatternConfirmed;
      document.getElementById('btnWashStart').disabled=!selectedOutlineFile||(isStrongPatternKey(document.getElementById('washStoryPattern').value)&&!washPatternConfirmed);
      document.getElementById('progressStatus').textContent='❌ 错误: '+msg.message;
      break;
  }
}

function setAgentState(name,state){
  let dot=document.getElementById(agentMap[name]);
  if(!dot)return;
  dot.classList.remove('running','done');
  if(state==='running')dot.classList.add('running');
  else if(state==='done')dot.classList.add('done');
}

function onCatChange(mode='create'){
  let isWash=mode==='wash';
  let target=isWash?'#washCatGrid input:checked':'#catGrid input:checked';
  let selected=[...document.querySelectorAll(target)].map(c=>c.value);
  if(isWash){
    washSelectedCats=selected;
    washMaterialConfig={...washMaterialConfig,filters:{categories:[],subcategories:[...selected],tags:[]},items:[],locked_item_keys:[],auto_selected_subcategories:[]};
    renderMaterialCards('wash');
  }else{
    selectedCats=selected;
    materialConfig={...materialConfig,filters:{categories:[],subcategories:[...selected],tags:[]},items:[],locked_item_keys:[],auto_selected_subcategories:[]};
    renderMaterialCards('create');
  }
}

let refineActive=false,refinedIdea='';

function startRefine(){
  let idea=document.getElementById('idea').value.trim();
  if(!idea){log('请先输入灵感','warn');return}
  refineActive=true;refinedIdea='';
  document.getElementById('refineSection').classList.remove('hidden');
  document.getElementById('refineResultSection').classList.add('hidden');
  document.getElementById('refineChat').innerHTML='<div style="color:#8b949e">⏳ AI 正在分析你的点子...</div>';
  document.getElementById('refineAnswer').value='';
  document.getElementById('btnStart').disabled=true;
  document.getElementById('idea').disabled=true;
  log('🔍 启动灵感精炼...','info');
  sendMsg({action:'refine_start',data:{idea}});
}

function skipRefine(){
  refineActive=false;refinedIdea='';
  document.getElementById('refineSection').classList.add('hidden');
  document.getElementById('refineResultSection').classList.add('hidden');
  document.getElementById('btnStart').disabled=false;
  document.getElementById('idea').disabled=false;
}

function sendRefineAnswer(){
  let ans=document.getElementById('refineAnswer').value.trim();
  if(!ans)return;
  let chat=document.getElementById('refineChat');
  chat.innerHTML+=`<div style="color:#58a6ff;margin:4px 0">👤 ${ans}</div>`;
  document.getElementById('refineAnswer').value='';
  chat.scrollTop=chat.scrollHeight;
  sendMsg({action:'refine_answer',data:ans});
}

function confirmRefine(){
  document.getElementById('idea').value=refinedIdea;
  document.getElementById('idea').disabled=false;
  document.getElementById('btnStart').disabled=false;
  document.getElementById('refineSection').classList.add('hidden');
  document.getElementById('refineResultSection').classList.add('hidden');
  refineActive=false;
  log('✅ 已采用精炼设定','success');
}

function editRefineResult(){
  document.getElementById('idea').value=refinedIdea;
  document.getElementById('idea').disabled=false;
  document.getElementById('btnStart').disabled=false;
  document.getElementById('refineSection').classList.add('hidden');
  document.getElementById('refineResultSection').classList.add('hidden');
  refineActive=false;
  log('✏️ 精炼设定已放入编辑框，可手动修改','info');
}

function startPipeline(){
  let idea=document.getElementById('idea').value.trim();
  if(!idea){log('请输入小说灵感','warn');return}
  let targetChapters=parseInt(document.getElementById('chapters').value)||8;
  let wordsPerChapter=parseInt(document.getElementById('wordsPerCh').value)||1500;
  let writerStyle=document.getElementById('writerStyle').value||'default';
  let patternConfig=patternConfigFor('create');
  if(patternConfig.primary==='custom'&&!patternConfig.custom_instruction){log('请填写自定义套路要求','warn');return}
  if(isStrongPatternKey(patternConfig.primary)&&!createPatternConfirmed){log('请先确认强套路契约','warn');return}
  if((materialConfig.items||[]).length!==(materialConfig.count||4)){log('请先抽取并确认完整素材配置','warn');return}
  document.getElementById('btnStart').disabled=true;
  document.getElementById('approvalBar').classList.add('hidden');
  document.getElementById('progressStatus').textContent='提交中...';
  Object.keys(agentMap).forEach(k=>setAgentState(k,'idle'));
  sendMsg({action:'start',data:{idea,material_config:materialConfig,pattern_config:patternConfig,target_chapters:targetChapters,words_per_chapter:wordsPerChapter,writer_style:writerStyle}});
}

// Live word count estimate
document.getElementById('chapters').oninput=updateEstimate;
document.getElementById('wordsPerCh').oninput=updateEstimate;
function updateEstimate(){
  let ch=parseInt(document.getElementById('chapters').value)||0;
  let w=parseInt(document.getElementById('wordsPerCh').value)||0;
  document.getElementById('estWords').textContent='预估: 约 '+ (ch*w).toLocaleString() +' 字';
}

function toggleCustomPattern(selectId,wrapId){
  document.getElementById(wrapId).classList.toggle('hidden',document.getElementById(selectId).value!=='custom');
}

function patternIds(mode){
  return mode==='wash'
    ? {select:'washStoryPattern',normalSelect:'washNormalStoryPattern',strongSelect:'washStrongStoryPattern',tabs:'#washPanel .pattern-tab',normalPanel:'washNormalPatternPanel',strongPanel:'washStrongPatternPanel',style:'washWriterStyle',ending:'washPatternEnding',section:'washPatternManifestSection',summary:'washPatternManifestSummary',editor:'washPatternManifestEditor',button:'btnWashStart'}
    : {select:'storyPattern',normalSelect:'normalStoryPattern',strongSelect:'strongStoryPattern',tabs:'#createPanel .pattern-tab',normalPanel:'normalPatternPanel',strongPanel:'strongPatternPanel',style:'writerStyle',ending:'patternEnding',section:'patternManifestSection',summary:'patternManifestSummary',editor:'patternManifestEditor',button:'btnStart'};
}

function setCompatibleStyles(mode,styles){
  let select=document.getElementById(patternIds(mode).style);
  let selected=select.value||'default';
  let incompatible=styles.length>0&&!styles.includes(selected);
  for(let option of select.options){
    option.disabled=styles.length>0&&!styles.includes(option.value)&&option.value!==selected;
  }
  select.style.borderColor=incompatible?'#f85149':'#30363d';
  select.title=incompatible?`当前风格与所选套路不兼容，可用风格：${styles.join('、')}`:'';
  if(incompatible)log(`⚠️ 当前写手风格与所选套路不兼容，请改选：${styles.join('、')}`,'warn');
  return !incompatible;
}

function onWriterStyleChange(mode){
  let ids=patternIds(mode);
  let pattern=document.getElementById(ids.select).value||'none';
  let manifest=mode==='wash'?washPatternManifest:createPatternManifest;
  let styles=(manifest&&manifest.pattern_key===pattern?manifest.compatible_styles:null)||(patternMeta[pattern]||{}).compatible_styles||[];
  let compatible=setCompatibleStyles(mode,styles);
  let strong=isStrongPatternKey(pattern);
  let confirmed=mode==='wash'?washPatternConfirmed:createPatternConfirmed;
  document.getElementById(ids.button).disabled=!compatible||(strong&&!confirmed)||(mode==='wash'&&!selectedOutlineFile);
}
function onPatternChange(mode){
  let ids=patternIds(mode);
  let pattern=document.getElementById(ids.select).value||'none';
  renderPatternControls(mode);
  toggleCustomPattern(ids.select,mode==='wash'?'washCustomPatternWrap':'customPatternWrap');
  let strong=isStrongPatternKey(pattern);
  let styleCompatible=setCompatibleStyles(mode,(patternMeta[pattern]||{}).compatible_styles||[]);
  document.getElementById(ids.section).classList.toggle('hidden',!strong);
  if(strong)updateEndingOptions(mode,pattern,document.getElementById(ids.ending).value);
  if(mode==='create'){
    document.getElementById('kwSection').classList.remove('hidden');
    materialConfig.items=[];
    materialConfig.locked_item_keys=[];
    renderMaterialCards();
    renderCategories();
  }else{
    washMaterialConfig.items=[];
    washMaterialConfig.locked_item_keys=[];
    renderMaterialCards('wash');
    renderCategories('wash');
  }
  renderSecondaryPatterns(mode);
  if(!strong){
    if(mode==='wash'){washPatternManifest=null;washPatternConfirmed=false}
    else{createPatternManifest=null;createPatternConfirmed=false}
    document.getElementById(ids.button).disabled=!styleCompatible||(mode==='wash'&&!selectedOutlineFile);
    return;
  }
  if(mode==='create')renderCategories();
  rollPatternManifest(mode);
}

function rollPatternManifest(mode){
  let ids=patternIds(mode);
  let pattern=document.getElementById(ids.select).value||'none';
  if(!isStrongPatternKey(pattern))return;
  updateEndingOptions(mode,pattern,document.getElementById(ids.ending).value);
  if(mode==='wash')washPatternConfirmed=false;else createPatternConfirmed=false;
  document.getElementById(ids.button).disabled=true;
  document.getElementById(ids.summary).textContent='正在抽取强套路契约...';
  sendMsg({action:'roll_pattern_manifest',data:{pattern,mode,ending:document.getElementById(ids.ending).value}});
}

function manifestSummary(manifest){
  let conflicts=(manifest.conflicts||[]).map(item=>item.name||item).join('、');
  let beats=(manifest.beat_preview||[]).map(item=>`${item.range} ${item.requirement}`).join('\n');
  let labels=manifest.labels||{};
  return `背景：${manifest.background||''}\n${labels.protagonist||'主角'}：${manifest.protagonist||manifest.heroine||''}\n${labels.counterpart||'关系方'}：${manifest.counterpart||manifest.hero||''}\n${labels.foil||'对照方'}：${manifest.foil||manifest.rival||''}\n${labels.conflict||'模块'}：${conflicts}\n${labels.ending||'结局'}：${manifest.ending_description||''}\n\n节拍预览：\n${beats}`;
}

function applyPatternManifest(mode,manifest,confirmed){
  let ids=patternIds(mode);
  if(mode==='wash'){washPatternManifest=manifest;washPatternConfirmed=confirmed}
  else{createPatternManifest=manifest;createPatternConfirmed=confirmed}
  renderPatternControls(mode);
  document.getElementById(ids.section).classList.remove('hidden');
  updateEndingOptions(mode,manifest.pattern_key||document.getElementById(ids.select).value,manifest.ending);
  if(manifest.ending)document.getElementById(ids.ending).value=manifest.ending;
  document.getElementById(ids.summary).textContent=manifestSummary(manifest)+(confirmed?'\n\n✅ 契约已确认':'\n\n⏸️ 请确认、重抽或手动修改');
  document.getElementById(ids.editor).classList.add('hidden');
  let styleCompatible=setCompatibleStyles(mode,manifest.compatible_styles||[]);
  document.getElementById(ids.button).disabled=!styleCompatible||!confirmed||(mode==='wash'&&!selectedOutlineFile);
}

function changePatternEnding(mode){
  let ids=patternIds(mode);
  let manifest=mode==='wash'?washPatternManifest:createPatternManifest;
  if(!manifest)return;
  let ending=document.getElementById(ids.ending).value||'no_reunion';
  let options=endingOptionsFor(manifest.pattern_key||document.getElementById(ids.select).value);
  manifest={...manifest,ending,ending_description:options[ending]||''};
  applyPatternManifest(mode,manifest,false);
  log('结局方向已修改，请重新确认契约','warn');
}

function confirmPatternManifest(mode){
  let ids=patternIds(mode);
  let editor=document.getElementById(ids.editor);
  let manifest=mode==='wash'?washPatternManifest:createPatternManifest;
  if(!editor.classList.contains('hidden')){
    try{manifest=JSON.parse(editor.value)}
    catch(e){log('套路契约 JSON 格式错误','error');return}
  }
  if(!manifest){log('请先抽取套路契约','warn');return}
  applyPatternManifest(mode,manifest,true);
  log('✅ 强套路契约已确认','success');
}

function editPatternManifest(mode){
  let ids=patternIds(mode);
  let editor=document.getElementById(ids.editor);
  let manifest=mode==='wash'?washPatternManifest:createPatternManifest;
  if(editor.classList.contains('hidden')){
    if(mode==='wash')washPatternConfirmed=false;else createPatternConfirmed=false;
    document.getElementById(ids.button).disabled=true;
    editor.value=JSON.stringify(manifest||{},null,2);
    editor.classList.remove('hidden');
    return;
  }
  try{
    applyPatternManifest(mode,JSON.parse(editor.value),false);
    log('手动修改已载入，请确认契约','warn');
  }catch(e){log('套路契约 JSON 格式错误','error')}
}

function switchTab(tab){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('outlineArea').classList.toggle('hidden',tab!=='outline');
  document.getElementById('novelArea').classList.toggle('hidden',tab!=='novel');
}

// --- 洗文模式 ---
let currentMode='create',selectedOutlineFile='',selectedOutlineData=null;

function switchMode(mode){
  currentMode=mode;
  document.querySelectorAll('.mode-tab').forEach(t=>t.classList.toggle('active',t.textContent.includes(mode==='create'?'创作':'洗文')));
  document.getElementById('createPanel').classList.toggle('hidden',mode!=='create');
  document.getElementById('washPanel').classList.toggle('hidden',mode!=='wash');
  document.getElementById('approvalBar').classList.add('hidden');
  if(mode==='wash'){
    document.getElementById('btnWashStart').disabled=true;
    selectedOutlineFile='';selectedOutlineData=null;
    loadOutlines();
  }else{
    document.getElementById('btnStart').disabled=false;
  }
}

function loadOutlines(){
  document.getElementById('outlineList').innerHTML='<div style="color:#8b949e;padding:12px;text-align:center">加载中...</div>';
  sendMsg({action:'list_outlines'});
}

function selectOutline(file){
  selectedOutlineFile=file;
  document.querySelectorAll('#outlineList .outline-item').forEach(i=>i.classList.remove('active'));
  let items=document.querySelectorAll('#outlineList .outline-item');
  for(let el of items){if(el.dataset.file===file){el.classList.add('active');break}}
  let needsManifest=isStrongPatternKey(document.getElementById('washStoryPattern').value)&&!washPatternConfirmed;
  document.getElementById('btnWashStart').disabled=needsManifest;
  sendMsg({action:'load_outline',data:{file}});
}

function startWash(){
  if(!selectedOutlineFile||!selectedOutlineData)return;
  let ch=parseInt(document.getElementById('washChapters').value)||0;
  let w=parseInt(document.getElementById('washWords').value)||1500;
  let style=document.getElementById('washWriterStyle').value||'default';
  let patternConfig=patternConfigFor('wash');
  if(patternConfig.primary==='custom'&&!patternConfig.custom_instruction){log('请填写自定义套路要求','warn');return}
  if(isStrongPatternKey(patternConfig.primary)&&!washPatternConfirmed){log('请先确认强套路契约','warn');return}
  if((washMaterialConfig.items||[]).length!==(washMaterialConfig.count||4)){log('请先确认完整的洗文素材配置','warn');return}
  document.getElementById('btnWashStart').disabled=true;
  document.getElementById('washStatus').textContent='生成新书名...';
  Object.keys(agentMap).forEach(k=>setAgentState(k,'idle'));
  document.getElementById('novelArea').textContent='';
  document.getElementById('approvalBar').classList.add('hidden');
  sendMsg({action:'start_rewash',data:{
    file:selectedOutlineFile,writer_style:style,pattern_config:patternConfig,material_config:washMaterialConfig,
    target_chapters:ch,words_per_chapter:w
  }});
}

// Reset all agent dots
Object.keys(agentMap).forEach(k=>setAgentState(k,'idle'));
updateEstimate();
connect();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML_PAGE)


# ═══════════════════════════════════════════════════
#  WebSocket 核心逻辑
# ═══════════════════════════════════════════════════
@app.websocket("/ws")
async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    state = None
    token_counter = 0

    async def send(msg: dict):
        try:
            await websocket.send_json(msg)
        except Exception:
            pass

    while True:
        try:
            raw = await websocket.receive_json()
        except WebSocketDisconnect:
            break
        except Exception:
            continue

        action = raw.get("action", "")
        data = raw.get("data", {})

        if action == "get_material_library":
            await send({
                "type": "material_library",
                "data": material_library_metadata(),
            })

        elif action == "get_writer_styles":
            await send({
                "type": "writer_styles",
                "data": writer_style_options(),
            })

        elif action == "get_patterns":
            await send({"type": "patterns", "data": pattern_library_metadata()})

        elif action == "sample_materials":
            try:
                material_config = sample_materials(
                    data.get("material_config"),
                    data.get("pattern_config"),
                    seed=data.get("seed"),
                    randomize_types=bool(data.get("randomize_types")),
                )
                await send({
                    "type": "material_result",
                    "mode": data.get("mode", "create"),
                    "data": material_config,
                })
            except (LibraryValidationError, ValueError) as error:
                await send({"type": "error", "message": str(error)})

        elif action == "resample_material":
            try:
                material_config = resample_material_item(
                    data.get("material_config"),
                    data.get("pattern_config"),
                    str(data.get("selection_key") or ""),
                    seed=data.get("seed"),
                    change_type=bool(data.get("change_type")),
                )
                await send({
                    "type": "material_result",
                    "mode": data.get("mode", "create"),
                    "data": material_config,
                })
            except (LibraryValidationError, ValueError) as error:
                await send({"type": "error", "message": str(error)})

        elif action == "roll_pattern_manifest":
            pattern_key = data.get("pattern", "")
            mode = data.get("mode", "create")
            if not is_strong_pattern(pattern_key):
                await send({"type": "error", "message": "所选套路不需要生成强套路契约"})
                continue
            try:
                manifest = roll_pattern_manifest(
                    pattern_key,
                    seed=data.get("seed"),
                    ending=data.get("ending", "no_reunion"),
                )
            except (TypeError, ValueError):
                await send({"type": "error", "message": "套路随机种子必须为整数"})
                continue
            await send({"type": "pattern_manifest_result", "mode": mode, "data": manifest})

        # ── 灵感精炼：开始 ──
        elif action == "refine_start":
            idea = data.get("idea", "")
            if not idea:
                await send({"type": "error", "message": "请先输入灵感"})
                continue
            # 用临时对话历史（累积上下文）
            refine_history = []
            refine_history.append({"role": "user", "content": f"我的小说灵感是：【{idea}】。请帮我精炼。"})
            # 第一轮提问
            try:
                resp = await asyncio.get_event_loop().run_in_executor(None, _call_refine, refine_history)
            except Exception as e:
                await send({"type": "error", "message": f"精炼LLM调用失败: {e}"})
                continue

            if resp.startswith("【精炼设定】"):
                # 直接出结果了
                await send({"type": "refine_done", "refined": resp.replace("【精炼设定】\n", "").strip()})
            else:
                refine_history.append({"role": "assistant", "content": resp})
                await send({"type": "refine_question", "question": resp, "round": 1})

                # 等待用户回答
                while True:
                    try:
                        ans = await websocket.receive_json()
                    except WebSocketDisconnect:
                        return
                    if ans.get("action") == "refine_answer":
                        answer = ans.get("data", "")
                        refine_history.append({"role": "user", "content": answer})
                        # 让LLM继续
                        try:
                            resp2 = await asyncio.get_event_loop().run_in_executor(None, _call_refine, refine_history)
                        except Exception as e:
                            await send({"type": "error", "message": f"精炼LLM调用失败: {e}"})
                            break
                        if resp2.startswith("【精炼设定】"):
                            await send({"type": "refine_done", "refined": resp2.replace("【精炼设定】\n", "").strip()})
                            break
                        else:
                            refine_history.append({"role": "assistant", "content": resp2})
                            await send({"type": "refine_question", "question": resp2, "round": 2})
                            # 第二轮回答后直接给精炼结果
                            try:
                                ans2 = await websocket.receive_json()
                            except WebSocketDisconnect:
                                return
                            if ans2.get("action") == "refine_answer":
                                refine_history.append({"role": "user", "content": ans2.get("data", "")})
                            refine_history.append({"role": "user", "content": "请根据以上对话输出精炼设定。"})
                            try:
                                resp3 = await asyncio.get_event_loop().run_in_executor(None, _call_refine, refine_history)
                            except Exception as e:
                                await send({"type": "error", "message": f"精炼LLM调用失败: {e}"})
                                break
                            if resp3.startswith("【精炼设定】"):
                                resp3 = resp3.replace("【精炼设定】\n", "").strip()
                            await send({"type": "refine_done", "refined": resp3})
                            break
                    elif ans.get("action") == "refine_skip":
                        await send({"type": "refine_skip"})
                        break
                    else:
                        break

        # ── 灵感精炼：确认使用精炼结果 ──
        elif action == "refine_confirm":
            # 前端会重新发 start 包含精炼后的 idea
            pass
        # ── 启动流水线 ──
        elif action == "start":
            run_id = f"web-{uuid.uuid4().hex[:8]}"
            run_config = {"configurable": {"thread_id": run_id}}
            run_metrics = {
                "run_id": run_id,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "wall_started": time.perf_counter(),
                "pre_pipeline": {},
                "retry_counter_start": retry_counter.value(),
            }
            idea = data.get("idea", "")
            target_chapters = data.get("target_chapters", DEFAULT_CHAPTERS)
            words_per_chapter = data.get("words_per_chapter", DEFAULT_WORDS_PER_CHAPTER)
            writer_style = data.get("writer_style", "default")
            pattern_config = normalize_pattern_config(data.get("pattern_config"))
            material_config = normalize_material_config(data.get("material_config"))
            configuration_issues = validate_pattern_config(
                pattern_config, writer_style
            )
            configuration_issues.extend(
                validate_material_config(material_config, pattern_config)
            )
            if configuration_issues:
                await send({
                    "type": "error",
                    "message": "创作配置无效：" + "；".join(configuration_issues),
                })
                continue
            primary_pattern = pattern_config["primary"]
            if is_strong_pattern(primary_pattern):
                manifest_issues = validate_pattern_manifest(
                    pattern_config.get("manifest", {})
                )
                if manifest_issues:
                    await send({"type": "error", "message": f"强套路契约未确认或无效：{'；'.join(manifest_issues)}"})
                    continue
            run_metrics["configuration"] = {
                "target_chapters": target_chapters,
                "words_per_chapter": words_per_chapter,
                "writer_style": writer_style,
                "pattern_config": pattern_config,
                "material_config": material_config,
            }

            # ── 第一阶段: 运行架构师 ──
            init_state = {
                "user_idea": idea,
                "run_id": run_id,
                "material_config": material_config,
                "pattern_config": pattern_config,
                "target_chapters": target_chapters,
                "words_per_chapter": words_per_chapter,
                "writer_style": writer_style,
                "continuity_state": "",
                "story_ledger": {},
                "ledger_delta": {},
                "continuity_report": {},
                "scene_plan": {},
                "draft_candidates": [],
                "current_chapter": 1,
                "iteration_count": 0,
            }
            await send({"type": "architect_start"})
            architect_started = time.perf_counter()
            try:
                arch_result = await asyncio.get_event_loop().run_in_executor(
                    None, architect_node, init_state
                )
            except Exception as e:
                await send({"type": "error", "message": str(e)})
                continue
            run_metrics["pre_pipeline"]["architect_seconds"] = round(
                time.perf_counter() - architect_started, 3
            )

            if arch_result is None:
                await send({"type": "error", "message": "架构师输出为空"})
                continue

            init_state.update(arch_result)
            await send({"type": "architect_result", "data": {
                "novel_title": init_state.get("novel_title", ""),
                "world_bible": init_state.get("world_bible", ""),
                "chapter_outlines": init_state.get("chapter_outlines", {})
            }})

            # ── 等待用户审批大纲 ──
            approved = False
            approval_started = time.perf_counter()
            while True:
                try:
                    resp = await websocket.receive_json()
                except WebSocketDisconnect:
                    return
                if resp.get("action") == "approval":
                    approved = resp.get("data", False)
                    break
            run_metrics["pre_pipeline"]["approval_wait_seconds"] = round(
                time.perf_counter() - approval_started, 3
            )
            if not approved:
                await send({"type": "error", "message": "用户拒绝大纲"})
                continue

            # ── 第二阶段: 运行完整流水线 ──
            state = init_state
            await _run_pipeline(websocket, send, state, run_config, run_metrics)

        # ── 洗文：列出大纲 ──
        elif action == "list_outlines":
            files = list_outline_files()
            await send({"type": "outline_list", "data": files})

        # ── 洗文：加载大纲内容 ──
        elif action == "load_outline":
            try:
                content = load_outline_json(data.get("file", ""))
                await send({"type": "outline_content", "data": content})
            except Exception as e:
                await send({"type": "error", "message": f"加载大纲失败: {e}"})

        # ── 洗文：启动洗文流水线 ──
        elif action == "start_rewash":
            run_id = f"web-{uuid.uuid4().hex[:8]}"
            run_config = {"configurable": {"thread_id": run_id}}
            run_metrics = {
                "run_id": run_id,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "wall_started": time.perf_counter(),
                "pre_pipeline": {},
                "retry_counter_start": retry_counter.value(),
            }
            file_name = data.get("file", "")
            writer_style = data.get("writer_style", "default")
            target_chapters = data.get("target_chapters", 0)
            words_per_chapter = data.get("words_per_chapter", DEFAULT_WORDS_PER_CHAPTER)

            try:
                outline = load_outline_json(file_name)
            except Exception as e:
                await send({"type": "error", "message": f"加载大纲失败: {e}"})
                continue

            pattern_config = normalize_pattern_config(
                data.get("pattern_config") or outline.get("pattern_config")
            )
            material_config = normalize_material_config(
                data.get("material_config") or outline.get("material_config")
            )
            configuration_issues = validate_pattern_config(
                pattern_config, writer_style
            )
            configuration_issues.extend(
                validate_material_config(material_config, pattern_config)
            )
            if configuration_issues:
                await send({
                    "type": "error",
                    "message": "洗文配置无效：" + "；".join(configuration_issues),
                })
                continue
            primary_pattern = pattern_config["primary"]
            pattern_manifest = pattern_config.get("manifest", {})
            if is_strong_pattern(primary_pattern):
                manifest_issues = validate_pattern_manifest(pattern_manifest)
                if manifest_issues:
                    await send({"type": "error", "message": f"强套路契约未确认或无效：{'；'.join(manifest_issues)}"})
                    continue
            run_metrics["configuration"] = {
                "target_chapters": target_chapters,
                "words_per_chapter": words_per_chapter,
                "writer_style": writer_style,
                "pattern_config": pattern_config,
                "material_config": material_config,
            }

            chapters = target_chapters if target_chapters > 0 else len(outline.get("chapter_outlines", {}))
            run_metrics["configuration"]["target_chapters"] = chapters
            base_outlines = normalize_chapter_outlines(
                strip_pattern_plan_from_outlines(outline.get("chapter_outlines", {})),
                chapters,
            )
            outline_issues = outline_validation_issues(
                base_outlines, chapters
            )
            if outline_issues:
                await send({
                    "type": "error",
                    "message": (
                        "所选大纲不符合写作要求："
                        f"{'；'.join(outline_issues[:5])}。"
                        "每章细纲至少需要200字，请重新生成合格大纲后再启动写作。"
                    ),
                })
                continue

            original_title = outline.get("title", "未命名")
            pattern_plan = (
                build_pattern_plan(pattern_manifest, chapters, words_per_chapter)
                if is_strong_pattern(primary_pattern)
                else {}
            )
            pattern_config["structure_plan"] = pattern_plan
            chapter_outlines = (
                attach_pattern_plan_to_outlines(base_outlines, pattern_plan)
                if pattern_plan
                else base_outlines
            )
            chapter_contracts = build_chapter_contracts(chapter_outlines)
            finale_contract = build_finale_contract(
                chapter_contracts, pattern_manifest
            )

            # 生成洗文新书名
            await send({"type": "log", "message": f"🤖 为《{original_title}》生成洗文新书名...", "cls": "info"})
            title_started = time.perf_counter()
            try:
                new_title = await asyncio.get_event_loop().run_in_executor(
                    None, generate_wash_title, original_title, writer_style
                )
            except Exception as e:
                new_title = f"{original_title}·重制版"
            run_metrics["pre_pipeline"]["title_generation_seconds"] = round(
                time.perf_counter() - title_started, 3
            )
            await send({"type": "rewash_title", "title": new_title})

            init_state = {
                "user_idea": f"洗文:《{original_title}》→《{new_title}》",
                "run_id": run_id,
                "novel_title": new_title,
                "wash_original_title": original_title,
                "outline_file": file_name,
                "world_bible": outline.get("world_bible", ""),
                "chapter_outlines": chapter_outlines,
                "chapter_contracts": chapter_contracts,
                "finale_contract": finale_contract,
                "material_config": material_config,
                "pattern_config": pattern_config,
                "target_chapters": chapters,
                "words_per_chapter": words_per_chapter,
                "writer_style": writer_style,
                "continuity_state": "",
                "story_ledger": {},
                "ledger_delta": {},
                "continuity_report": {},
                "scene_plan": {},
                "draft_candidates": [],
                "current_chapter": 1,
                "iteration_count": 0,
            }
            await send({"type": "log", "message": f"📝 洗文启动: 《{new_title}》 | {chapters}章 × {words_per_chapter}字 | {writer_style}", "cls": "info"})
            state = init_state
            await _run_pipeline(websocket, send, state, run_config, run_metrics)

        elif action == "approval":
            pass  # handled in the inner loop above


def _summarize_web_timings(node_timings: list[dict]) -> dict:
    grouped = defaultdict(list)
    for timing in node_timings:
        grouped[timing["node"]].append(timing["duration_seconds"])
    return {
        node: {
            "calls": len(values),
            "total_seconds": round(sum(values), 3),
            "average_seconds": round(sum(values) / len(values), 3),
        }
        for node, values in grouped.items()
    }


async def _run_pipeline(websocket, send, state, config, run_metrics=None):
    """ 在独立线程中运行 LangGraph pipeline，通过队列同步到 WebSocket """
    run_metrics = run_metrics or {
        "run_id": f"web-{uuid.uuid4().hex[:8]}",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "wall_started": time.perf_counter(),
        "pre_pipeline": {},
        "retry_counter_start": retry_counter.value(),
    }
    q: queue.Queue = queue.Queue()

    def runner():
        node_timings = []
        pipeline_started = time.perf_counter()
        previous_completed = pipeline_started
        current_chapter = state.get("current_chapter", 1)
        last_writer_attempt = 0
        error_message = None
        try:
            for output in graph_app.stream(state, config=config):
                for node_name, node_data in output.items():
                    if node_name == "__interrupt__":
                        continue
                    completed = time.perf_counter()
                    duration = round(completed - previous_completed, 3)
                    previous_completed = completed

                    # 追踪当前章节号
                    if node_name == "writer":
                        ch = current_chapter
                        last_writer_attempt = node_data.get("iteration_count", last_writer_attempt)
                    elif node_name == "summarizer":
                        ch = node_data.get("saved_chapter", node_data.get("current_chapter", "?"))
                    else:
                        ch = current_chapter
                    timing_entry = {
                        "node": node_name,
                        "chapter": ch,
                        "attempt": last_writer_attempt,
                        "duration_seconds": duration,
                    }
                    if node_name == "reviewer":
                        audit = node_data.get("audit_report", {})
                        editor = node_data.get("editor_report", {})
                        timing_entry.update({
                            "audit_status": audit.get("审核状态"),
                            "logic_issues": audit.get("发现的问题", []),
                            "logic_warnings": audit.get("警告", []),
                            "pattern_status": audit.get("套路执行状态"),
                            "pattern_issues": audit.get("套路问题", []),
                            "style_score": editor.get("文风评分"),
                            "ai_trace_issues": editor.get("AI痕迹问题", []),
                        })
                    node_timings.append(timing_entry)

                    if node_name == "reviewer":
                        q.put({"type": "node_done_review", "node": "reviewer", "data": node_data})
                    else:
                        q.put({"type": "node_done", "node": node_name, "data": node_data,
                               "message": f"✍️ 写手产出 第{ch}章 第{node_data.get('iteration_count','?')}稿" if node_name=="writer" else ""})
                    if node_name == "summarizer":
                        chap_num = node_data.get("saved_chapter", node_data.get("current_chapter", "?"))
                        draft = node_data.get("current_draft", "")
                        current_chapter = chap_num + 1
                        q.put({"type": "chapter_saved", "data": draft})
                        for warning in node_data.get("chapter_warnings", []):
                            q.put({"type": "log", "message": f"⚠️ {warning}", "cls": "warn"})
                        if node_data.get("summary_skipped"):
                            q.put({"type": "log", "message": "⏭️ 最后一章已保存，已跳过剧情摘要", "cls": "info"})
        except Exception as e:
            error_message = str(e)
        finally:
            pipeline_finished = time.perf_counter()
            pre_pipeline = run_metrics.get("pre_pipeline", {})
            nodes = _summarize_web_timings(node_timings)
            if "architect_seconds" in pre_pipeline:
                architect_seconds = pre_pipeline["architect_seconds"]
                nodes["architect"] = {
                    "calls": 1,
                    "total_seconds": architect_seconds,
                    "average_seconds": architect_seconds,
                }
            report = {
                "run_id": run_metrics["run_id"],
                "started_at": run_metrics["started_at"],
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "wall_seconds": round(pipeline_finished - run_metrics["wall_started"], 3),
                "pipeline_seconds": round(pipeline_finished - pipeline_started, 3),
                "approval_wait_seconds": pre_pipeline.get("approval_wait_seconds", 0),
                "api_retry_count": max(0, retry_counter.value() - run_metrics.get("retry_counter_start", 0)),
                "pre_pipeline": pre_pipeline,
                "configuration": run_metrics.get("configuration", {}),
                "pattern_plan": (
                    state.get("pattern_config", {}).get("structure_plan", {})
                ),
                "nodes": nodes,
                "node_calls": node_timings,
                "rewrite_reasons": [
                    {
                        "chapter": item.get("chapter"),
                        "attempt": item.get("attempt"),
                        "logic_issues": item.get("logic_issues", []),
                        "pattern_issues": item.get("pattern_issues", []),
                        "ai_trace_issues": item.get("ai_trace_issues", []),
                    }
                    for item in node_timings
                    if item.get("node") == "reviewer"
                    and (
                        item.get("audit_status") == "不通过"
                        or (item.get("style_score") or 10) < STYLE_PASS_SCORE
                    )
                ],
                "error": error_message,
            }
            try:
                os.makedirs("TestResults", exist_ok=True)
                report_path = os.path.join("TestResults", f"{run_metrics['run_id']}.json")
                with open(report_path, "w", encoding="utf-8") as file:
                    json.dump(report, file, ensure_ascii=False, indent=2)
                report["report_path"] = report_path
            except Exception as report_error:
                logger.warning("⚠️ 网页计时报告保存失败: %s", report_error)

            q.put({"type": "timing_report", "data": report})
            if error_message:
                q.put({"type": "error", "message": error_message})
            else:
                q.put({"type": "log", "message": "🎉 全部章节写作完成！", "cls": "success"})
                q.put({"type": "done"})

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()

    while True:
        try:
            msg = q.get(timeout=0.2)
            if msg["type"] == "done":
                await send({"type": "pipeline_done"})
                break
            elif msg["type"] == "error":
                await send({"type": "error", "message": msg["message"]})
                break
            elif msg["type"] == "log":
                await send({"type": "log", "message": msg.get("message", ""), "cls": msg.get("cls", "info")})
            else:
                await send(msg)
        except queue.Empty:
            try:
                # 检查客户端是否还在
                await asyncio.sleep(0.1)
            except Exception:
                break
        except WebSocketDisconnect:
            break


# ═══════════════════════════════════════════════════
def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def _is_existing_autowrite(port: int) -> bool:
    if _port_is_available(port):
        return False
    try:
        direct_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with direct_opener.open(f"http://127.0.0.1:{port}/", timeout=1) as response:
            content = response.read(4096).decode("utf-8", errors="ignore")
        return "AutoWrite" in content
    except Exception:
        return False


def choose_web_port(preferred_port: int, max_attempts: int = 20) -> tuple[int, bool]:
    """Return (port, already_running) without terminating another process."""
    if _is_existing_autowrite(preferred_port):
        return preferred_port, True
    for port in range(preferred_port, preferred_port + max_attempts):
        if _port_is_available(port):
            return port, False
    raise RuntimeError(
        f"端口 {preferred_port}-{preferred_port + max_attempts - 1} 均被占用，"
        "请关闭占用程序或设置 WEB_PORT。"
    )


# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    import webbrowser

    preferred_port = int(os.getenv("WEB_PORT", "8080"))
    port, already_running = choose_web_port(preferred_port)
    url = f"http://127.0.0.1:{port}"
    if already_running:
        print(f"ℹ️ AutoWrite Web 已在运行，直接打开: {url}")
        webbrowser.open(url)
        raise SystemExit(0)
    if port != preferred_port:
        print(f"⚠️ 端口 {preferred_port} 已被其他程序占用，自动改用端口 {port}")
    print(f"🚀 AutoWrite Web 服务启动: {url}")
    threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
