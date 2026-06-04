import asyncio
import json
import logging
import os
import queue
import sys
import threading

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
    architect_node, writer_node, auditor_node, editor_node, summarizer_node,
    load_keywords, pick_keywords,
    DEFAULT_CHAPTERS, DEFAULT_WORDS_PER_CHAPTER
)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("AutoWrite")

# ── 构建图（不去除 interrupt，Web 层手动控制暂停） ──
workflow = StateGraph(NovelState)
workflow.add_node("architect", architect_node)
workflow.add_node("writer", writer_node)
workflow.add_node("auditor", auditor_node)
workflow.add_node("editor", editor_node)
workflow.add_node("summarizer", summarizer_node)
workflow.set_entry_point("architect")
workflow.add_edge("architect", "writer")
workflow.add_edge("writer", "auditor")


def route_after_audit(state: NovelState):
    report = state.get("audit_report", {})
    if report.get("审核状态") == "不通过" and state.get("iteration_count", 0) < 3:
        return "writer"
    return "editor"


workflow.add_conditional_edges("auditor", route_after_audit, {"writer": "writer", "editor": "editor"})


def route_after_editor(state: NovelState):
    report = state.get("editor_report", {})
    outlines = state.get("chapter_outlines", {})
    if report.get("文风评分", 0) < 8 and state.get("editor_iteration_count", 0) < 2:
        return "writer"
    if state.get("current_chapter", 1) <= len(outlines):
        return "summarizer"
    return END


workflow.add_conditional_edges("editor", route_after_editor, {
    "writer": "writer", "summarizer": "summarizer", END: END
})
workflow.add_edge("summarizer", "writer")

memory = MemorySaver()
graph_app = workflow.compile(checkpointer=memory)

# ── 灵感精炼 LLM ──
llm_refine = ChatOpenAI(model="deepseek-chat", temperature=0.5)

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
    result = (prompt | llm_refine).invoke({})
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
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:#0d1117}
::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}
</style>
</head>
<body>
<div id="app">
  <div id="panel">
    <h1>🚀 AutoWrite 小说流水线</h1>
    <div class="agent-bar" id="agentStatus">
      <span class="agent-dot" id="dot-architect"></span>架构师
      <span class="agent-dot" id="dot-writer"></span>写手
      <span class="agent-dot" id="dot-auditor"></span>审计
      <span class="agent-dot" id="dot-editor"></span>责编
      <span class="agent-dot" id="dot-summarizer"></span>书记
    </div>
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
        章节数 <input id="chapters" type="number" value="12" min="1" max="200" style="width:60px"> 章
        &nbsp;每章 <input id="wordsPerCh" type="number" value="2500" min="500" max="10000" step="100" style="width:70px"> 字
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
      </select>
    </div>
    <button class="btn btn-primary" id="btnStart" onclick="startPipeline()">▶ 启动流水线</button>
    <div class="approval-bar hidden" id="approvalBar">
      <button class="btn btn-primary" onclick="sendCmd('approval',true)">✅ 批准大纲，开始写作</button>
      <button class="btn btn-danger" onclick="sendCmd('approval',false)">❌ 拒绝，重新设定</button>
    </div>
    <p id="progressStatus" style="font-size:11px;color:#8b949e;margin-top:8px"></p>
  </div>
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
const agentMap={architect:'dot-architect',writer:'dot-writer',auditor:'dot-auditor',editor:'dot-editor',summarizer:'dot-summarizer'};
const agentNames={architect:'架构师',writer:'写手',auditor:'审计员',editor:'责编',summarizer:'书记员'};

function log(msg,cls='info'){let d=document.getElementById('logArea');d.innerHTML+=`<div class="log-item ${cls}">${msg}</div>`;d.scrollTop=d.scrollHeight}

function connect(){
  let proto=location.protocol==='https:'?'wss':'ws';
  ws=new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen=()=>{log('✅ 已连接服务器','success');loadCategories()};
  ws.onmessage=e=>handleMsg(JSON.parse(e.data));
  ws.onclose=()=>{log('⚠️ 连接断开，3秒后重连...','warn');setTimeout(connect,3000)};
  ws.onerror=()=>log('❌ 连接错误','error');
}

function sendMsg(obj){if(ws&&ws.readyState===1)ws.send(JSON.stringify(obj))}
function sendCmd(action,data){sendMsg({action,data,token})}

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
      log('✅ 架构师完成大纲','success');
      document.getElementById('outlineArea').textContent=msg.data.world_bible+'\n\n======== 章节细纲 ========\n\n'+JSON.stringify(msg.data.chapter_outlines,null,2);
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
      }else if(msg.node==='auditor'&&msg.data){
        let r=msg.data.audit_report||{};
        log(`  审计: ${r['审核状态']}${r['审核状态']==='不通过'?' (发现'+((r['发现的问题']||[]).length)+'个问题)':''}`,r['审核状态']==='不通过'?'warn':'success');
      }else if(msg.node==='editor'&&msg.data){
        let r=msg.data.editor_report||{};
        log(`  责编评分: ${r['文风评分']}/10`,r['文风评分']>=8?'success':'warn');
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
  let targetChapters=parseInt(document.getElementById('chapters').value)||12;
  let wordsPerChapter=parseInt(document.getElementById('wordsPerCh').value)||2500;
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
                "editor_iteration_count": 0,
            }
            await send({"type": "architect_start"})
            try:
                arch_result = await asyncio.get_event_loop().run_in_executor(
                    None, architect_node, init_state
                )
            except Exception as e:
                await send({"type": "error", "message": str(e)})
                continue

            if arch_result is None:
                await send({"type": "error", "message": "架构师输出为空"})
                continue

            init_state.update(arch_result)
            await send({"type": "architect_result", "data": {
                "world_bible": init_state.get("world_bible", ""),
                "chapter_outlines": init_state.get("chapter_outlines", {})
            }})

            # ── 等待用户审批大纲 ──
            approved = False
            while True:
                try:
                    resp = await websocket.receive_json()
                except WebSocketDisconnect:
                    return
                if resp.get("action") == "approval":
                    approved = resp.get("data", False)
                    break
            if not approved:
                await send({"type": "error", "message": "用户拒绝大纲"})
                continue

            # ── 第二阶段: 运行完整流水线 ──
            state = init_state
            await _run_pipeline(websocket, send, state, config)

        # ── 关键词决策 (在 start 流程中通过 receive_json 内循环处理) ──
        elif action == "keywords_decision":
            pass  # handled in the inner loop above

        elif action == "approval":
            pass  # handled in the inner loop above


async def _run_pipeline(websocket, send, state, config):
    """ 在独立线程中运行 LangGraph pipeline，通过队列同步到 WebSocket """
    q: queue.Queue = queue.Queue()

    def runner():
        try:
            for output in graph_app.stream(state, config=config):
                for node_name, node_data in output.items():
                    if node_name == "__interrupt__":
                        continue
                    q.put({"type": "node_done", "node": node_name, "data": node_data,
                           "message": f"✍️ 写手产出 第{node_data.get('current_chapter','?')}章 第{node_data.get('iteration_count','?')}稿" if node_name=="writer" else ""})
                    if node_name == "summarizer":
                        chap_num = node_data.get("saved_chapter", node_data.get("current_chapter", "?"))
                        draft = node_data.get("current_draft", "")
                        q.put({"type": "chapter_saved", "data": f"══════════ 第{chap_num}章 ══════════\n{draft}"})
            q.put({"type": "log", "message": "🎉 全部章节写作完成！", "cls": "success"})
            q.put({"type": "done"})
        except Exception as e:
            q.put({"type": "error", "message": str(e)})

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
if __name__ == "__main__":
    import uvicorn
    import webbrowser
    url = "http://127.0.0.1:8080"
    webbrowser.open(url)
    print(f"🚀 AutoWrite Web 服务启动: {url}")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
