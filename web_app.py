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
    load_keywords, pick_keywords,
    list_outline_files, load_outline_json, generate_wash_title,
    DEFAULT_CHAPTERS, DEFAULT_WORDS_PER_CHAPTER,
    MAX_REVIEW_ATTEMPTS, STYLE_PASS_SCORE, normalize_chapter_outlines,
    MODEL_MAX_RETRIES, MODEL_TIMEOUT_SECONDS, invoke_with_retry,
    outline_validation_issues, should_retry_short_draft,
)

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
    audit = state.get("audit_report", {})
    editor = state.get("editor_report", {})
    outlines = state.get("chapter_outlines", {})
    need_retry = (
        audit.get("审核状态") == "不通过" or
        editor.get("文风评分", 10) < STYLE_PASS_SCORE
    )
    if need_retry and state.get("iteration_count", 1) < MAX_REVIEW_ATTEMPTS:
        return "writer"
    if state.get("current_chapter", 1) <= len(outlines):
        return "summarizer"
    return END


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
pipeline_lock = threading.Lock()
generation_active = threading.Event()

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
#app{display:grid;grid-template-columns:340px 1fr;grid-template-rows:1fr;height:100vh}
#panel{background:#161b22;border-right:1px solid #30363d;display:flex;flex-direction:column;overflow-y:auto;padding:16px}
#panel h1{font-size:18px;color:#58a6ff;margin-bottom:12px;text-align:center}
.section{margin-bottom:16px}
.section label{display:block;font-size:13px;color:#8b949e;margin-bottom:6px}
#idea{width:100%;height:72px;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:8px;border-radius:6px;resize:vertical;font-size:13px}
.cat-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;max-height:180px;overflow-y:auto}
.cat-item{display:flex;align-items:center;gap:6px;font-size:12px;padding:4px 6px;background:#0d1117;border-radius:4px;cursor:pointer;border:1px solid #30363d;transition:all .15s}
.cat-item.active{border-color:#58a6ff;background:#1a2332}
.cat-item input{accent-color:#58a6ff}
.scope-row{display:flex;gap:8px;align-items:center}
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
    <div class="section" id="kwSection">
      <label>📚 随机词库 (可选多选)</label>
      <div class="cat-grid" id="catGrid"></div>
      <div id="kwResult" class="hidden" style="margin-top:6px;font-size:12px;color:#d29922"></div>
      <div class="btn-bar hidden" id="kwBtns">
        <button class="btn btn-warning" onclick="sendCmd('keywords_decision','accept')">确认</button>
        <button class="btn" style="background:#30363d;color:#c9d1d9" onclick="sendCmd('keywords_decision','retry')">重抽</button>
        <button class="btn" style="background:#30363d;color:#c9d1d9" onclick="sendCmd('keywords_decision','skip')">跳过</button>
      </div>
    </div>
    <div class="section">
      <label>📏 篇幅设置</label>
      <div class="scope-row">
        章节数 <input id="chapters" type="number" value="10" min="1" max="200" style="width:60px"> 章
        &nbsp;每章 <input id="wordsPerCh" type="number" value="1500" min="500" max="10000" step="100" style="width:70px"> 字
      </div>
      <div style="font-size:11px;color:#8b949e;margin-top:4px" id="estWords">预估: 约 30,000 字</div>
    </div>
    <div class="section">
      <label>✍️ 写手风格</label>
      <select id="writerStyle" style="width:100%;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:6px 8px;border-radius:4px;font-size:12px">
        <option value="default">默认</option>
        <option value="hot_blood">热血爽文</option>
        <option value="literary">文艺细腻</option>
        <option value="cold">冷峻纪实</option>
        <option value="humor">轻松搞笑</option>
        <option value="18xx">18XX</option>
      </select>
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
      <select id="washWriterStyle" style="width:100%;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:6px 8px;border-radius:4px;font-size:12px">
        <option value="default">默认</option>
        <option value="hot_blood">热血爽文</option>
        <option value="literary">文艺细腻</option>
        <option value="cold">冷峻纪实</option>
        <option value="humor">轻松搞笑</option>
        <option value="18xx">18XX</option>
      </select>
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
  sendMsg({action:'get_categories'});
}

function handleMsg(msg){
  switch(msg.type){
    case 'categories':
      document.getElementById('catGrid').innerHTML=Object.entries(msg.data).map(([k,v],i)=>
        `<label class="cat-item"><input type="checkbox" value="${k}" onchange="onCatChange()">${k}<span style="color:#8b949e;font-size:10px">${v}</span></label>`
      ).join('');
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
      break;
    case 'rewash_title':
      document.getElementById('washStatus').textContent='✨ 新书名: 《'+msg.title+'》';
      break;
    case 'keywords':
      document.getElementById('kwResult').textContent='🎲 命中: ['+msg.data.join('] [')+']';
      document.getElementById('kwResult').classList.remove('hidden');
      document.getElementById('kwBtns').classList.remove('hidden');
      token=msg.token;
      break;
    case 'keywords_skip':
      document.getElementById('kwResult').classList.add('hidden');
      document.getElementById('kwBtns').classList.add('hidden');
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
        log(`  审稿: 审计${a['审核状态']}${a['审核状态']==='不通过'?' ('+((a['发现的问题']||[]).length)+'个问题)':''} | 评分${e['文风评分']}/10`,(a['审核状态']==='通过'&&e['文风评分']>=7)?'success':'warn');
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
      document.getElementById('btnStart').disabled=false;
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

function onCatChange(){
  selectedCats=[...document.querySelectorAll('#catGrid input:checked')].map(c=>c.value);
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
  let targetChapters=parseInt(document.getElementById('chapters').value)||10;
  let wordsPerChapter=parseInt(document.getElementById('wordsPerCh').value)||1500;
  let writerStyle=document.getElementById('writerStyle').value||'default';
  document.getElementById('btnStart').disabled=true;
  document.getElementById('approvalBar').classList.add('hidden');
  document.getElementById('progressStatus').textContent='提交中...';
  Object.keys(agentMap).forEach(k=>setAgentState(k,'idle'));
  sendMsg({action:'start',data:{idea,selected_cats:selectedCats,target_chapters:targetChapters,words_per_chapter:wordsPerChapter,writer_style:writerStyle}});
}

// Live word count estimate
document.getElementById('chapters').oninput=updateEstimate;
document.getElementById('wordsPerCh').oninput=updateEstimate;
function updateEstimate(){
  let ch=parseInt(document.getElementById('chapters').value)||0;
  let w=parseInt(document.getElementById('wordsPerCh').value)||0;
  document.getElementById('estWords').textContent='预估: 约 '+ (ch*w).toLocaleString() +' 字';
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
  document.getElementById('btnWashStart').disabled=false;
  sendMsg({action:'load_outline',data:{file}});
}

function startWash(){
  if(!selectedOutlineFile||!selectedOutlineData)return;
  let ch=parseInt(document.getElementById('washChapters').value)||0;
  let w=parseInt(document.getElementById('washWords').value)||1500;
  let style=document.getElementById('washWriterStyle').value||'default';
  document.getElementById('btnWashStart').disabled=true;
  document.getElementById('washStatus').textContent='生成新书名...';
  Object.keys(agentMap).forEach(k=>setAgentState(k,'idle'));
  document.getElementById('novelArea').textContent='';
  document.getElementById('approvalBar').classList.add('hidden');
  sendMsg({action:'start_rewash',data:{
    file:selectedOutlineFile,writer_style:style,
    target_chapters:ch,words_per_chapter:w
  }});
}

// Reset all agent dots
Object.keys(agentMap).forEach(k=>setAgentState(k,'idle'));
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
    config = {"configurable": {"thread_id": f"web_{id(websocket)}"}}
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

        if action == "get_categories":
            db = load_keywords()
            cats = {k: v.get("description", "") for k, v in db.items()}
            await send({"type": "categories", "data": cats})

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
        if action == "get_categories":
            db = load_keywords()
            cats = {k: v.get("description", "") for k, v in db.items()}
            await send({"type": "categories", "data": cats})

        # ── 启动流水线 ──
        elif action == "start":
            if generation_active.is_set():
                await send({"type": "error", "message": "已有小说生成任务正在运行，请等待其完成"})
                continue
            generation_active.set()
            retry_counter.reset()
            run_metrics = {
                "run_id": f"web-{uuid.uuid4().hex[:8]}",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "wall_started": time.perf_counter(),
                "pre_pipeline": {},
            }
            idea = data.get("idea", "")
            cats = data.get("selected_cats", [])
            target_chapters = data.get("target_chapters", DEFAULT_CHAPTERS)
            words_per_chapter = data.get("words_per_chapter", DEFAULT_WORDS_PER_CHAPTER)
            writer_style = data.get("writer_style", "default")

            # 抽取关键词
            keywords = []
            if cats:
                keywords = pick_keywords(cats, 2)
                token_counter += 1
                await send({"type": "keywords", "data": keywords, "token": token_counter})

                # 等待用户确认关键词
                kw_decided = False
                while not kw_decided:
                    try:
                        resp = await websocket.receive_json()
                    except WebSocketDisconnect:
                        return
                    if resp.get("action") == "keywords_decision":
                        dec = resp.get("data", "accept")
                        if dec == "retry":
                            keywords = pick_keywords(cats, 2)
                            token_counter += 1
                            await send({"type": "keywords", "data": keywords, "token": token_counter})
                        elif dec == "skip":
                            keywords = []
                            kw_decided = True
                            await send({"type": "keywords_skip"})
                        else:
                            kw_decided = True

            # ── 第一阶段: 运行架构师 ──
            init_state = {
                "user_idea": idea,
                "keywords": keywords,
                "target_chapters": target_chapters,
                "words_per_chapter": words_per_chapter,
                "writer_style": writer_style,
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
                generation_active.clear()
                continue
            run_metrics["pre_pipeline"]["architect_seconds"] = round(
                time.perf_counter() - architect_started, 3
            )

            if arch_result is None:
                await send({"type": "error", "message": "架构师输出为空"})
                generation_active.clear()
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
                    generation_active.clear()
                    return
                if resp.get("action") == "approval":
                    approved = resp.get("data", False)
                    break
            run_metrics["pre_pipeline"]["approval_wait_seconds"] = round(
                time.perf_counter() - approval_started, 3
            )
            if not approved:
                await send({"type": "error", "message": "用户拒绝大纲"})
                generation_active.clear()
                continue

            # ── 第二阶段: 运行完整流水线 ──
            state = init_state
            await _run_pipeline(websocket, send, state, config, run_metrics)

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
            if generation_active.is_set():
                await send({"type": "error", "message": "已有小说生成任务正在运行，请等待其完成"})
                continue
            generation_active.set()
            retry_counter.reset()
            run_metrics = {
                "run_id": f"web-{uuid.uuid4().hex[:8]}",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "wall_started": time.perf_counter(),
                "pre_pipeline": {},
            }
            file_name = data.get("file", "")
            writer_style = data.get("writer_style", "default")
            target_chapters = data.get("target_chapters", 0)
            words_per_chapter = data.get("words_per_chapter", DEFAULT_WORDS_PER_CHAPTER)

            try:
                outline = load_outline_json(file_name)
            except Exception as e:
                await send({"type": "error", "message": f"加载大纲失败: {e}"})
                generation_active.clear()
                continue

            chapters = target_chapters if target_chapters > 0 else len(outline.get("chapter_outlines", {}))
            outline_issues = outline_validation_issues(
                outline.get("chapter_outlines", {}), chapters
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
                generation_active.clear()
                continue

            original_title = outline.get("title", "未命名")

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
                "novel_title": new_title,
                "wash_original_title": original_title,
                "outline_file": file_name,
                "world_bible": outline.get("world_bible", ""),
                "chapter_outlines": normalize_chapter_outlines(
                    outline.get("chapter_outlines", {}), chapters
                ),
                "keywords": [],
                "target_chapters": chapters,
                "words_per_chapter": words_per_chapter,
                "writer_style": writer_style,
                "current_chapter": 1,
                "iteration_count": 0,
            }
            await send({"type": "log", "message": f"📝 洗文启动: 《{new_title}》 | {chapters}章 × {words_per_chapter}字 | {writer_style}", "cls": "info"})
            state = init_state
            await _run_pipeline(websocket, send, state, config, run_metrics)

        # ── 关键词决策 (在 start 流程中通过 receive_json 内循环处理) ──
        elif action == "keywords_decision":
            pass  # handled in the inner loop above

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
    if not pipeline_lock.acquire(blocking=False):
        await send({"type": "error", "message": "已有小说生成任务正在运行，请等待其完成"})
        return

    run_metrics = run_metrics or {
        "run_id": f"web-{uuid.uuid4().hex[:8]}",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "wall_started": time.perf_counter(),
        "pre_pipeline": {},
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
                    node_timings.append({
                        "node": node_name,
                        "chapter": ch,
                        "attempt": last_writer_attempt,
                        "duration_seconds": duration,
                    })

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
                "api_retry_count": retry_counter.value(),
                "pre_pipeline": pre_pipeline,
                "nodes": nodes,
                "node_calls": node_timings,
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
            pipeline_lock.release()
            generation_active.clear()

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
