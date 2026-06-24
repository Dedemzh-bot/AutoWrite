import argparse
import datetime
import json
import logging
import os
import re
import sys
import traceback
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

from State import NovelState
from Nodes import (
    architect_node, writer_node, reviewer_node, summarizer_node,
    load_story_patterns,
    DEFAULT_CHAPTERS, DEFAULT_WORDS_PER_CHAPTER,
    MAX_REVIEW_ATTEMPTS, STYLE_PASS_SCORE, should_retry_short_draft,
    route_after_review_decision,
    is_strong_pattern, roll_pattern_manifest, format_pattern_manifest,
)
from LibraryV2 import (
    default_material_config,
    default_pattern_config,
    material_library_metadata,
    normalize_material_config,
    normalize_pattern_config,
    pattern_library_metadata,
    sample_materials,
    validate_material_config,
    validate_pattern_config,
)
from WriterStyles import WRITER_STYLE_KEYS, writer_style_options

logger = logging.getLogger("AutoWrite")

STYLE_OPTIONS = writer_style_options()
STYLE_KEYS = WRITER_STYLE_KEYS

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
    decision = route_after_review_decision(state)
    if decision == "writer":
        audit = state.get("audit_report", {})
        editor = state.get("editor_report", {})
        logger.warning("   ⚠️ 审稿退稿 审计:%s 评分:%d/10 → 发回写手重写",
                      audit.get("审核状态"), editor.get("文风评分", 0))
    return decision

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


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _write_json(path: str | os.PathLike, payload: dict) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def build_capabilities() -> dict:
    pattern_items = list(pattern_library_metadata().values())
    material_library = material_library_metadata()
    pattern_defaults = default_pattern_config()

    return {
        "schema_version": 2,
        "generated_at": _now_iso(),
        "writer_styles": STYLE_OPTIONS,
        "story_patterns": pattern_items,
        "pattern_library": {
            "schema_version": pattern_defaults["schema_version"],
            "max_secondary": 2,
        },
        "material_library": material_library,
        "cli_contract": {
            "supports_job_preflight": True,
        },
        "job_schema": {
            "schema_version": 2,
            "required": [
                "job_id",
                "idea",
                "target_chapters",
                "words_per_chapter",
                "writer_style",
                "material_config",
                "pattern_config",
            ],
            "optional": ["pattern_seed", "material_seed"],
        },
    }


def _capabilities_markdown(capabilities: dict) -> str:
    lines = [
        "# AutoWrite CLI 能力表",
        "",
        f"生成时间：{capabilities['generated_at']}",
        "",
        "## 写手风格",
        "",
        "| 键 | 名称 |",
        "| --- | --- |",
    ]
    for item in capabilities["writer_styles"]:
        lines.append(f"| `{item['key']}` | {item['name']} |")

    lines.extend([
        "",
        "## 创作套路",
        "",
        "| 键 | 名称 | 强套路 | 兼容写手 | 结局键 |",
        "| --- | --- | --- | --- | --- |",
    ])
    for item in capabilities["story_patterns"]:
        compatible = ", ".join(item["compatible_styles"]) or "全部"
        endings = ", ".join(item["ending_options"]) or "-"
        lines.append(
            f"| `{item['id']}` | {item['name']} | "
            f"{'是' if item['strong'] else '否'} | {compatible} | {endings} |"
        )

    lines.extend([
        "",
        "## 素材库",
        "",
        "| 大类 | 子类数 | 素材数 |",
        "| --- | --- | --- |",
    ])
    for key, item in capabilities["material_library"]["groups"].items():
        lines.append(
            f"| `{key}` {item['name']} | {len(item['subcategories'])} | {item['count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def export_capabilities(path: str | os.PathLike) -> tuple[Path, Path]:
    capabilities = build_capabilities()
    json_path = _write_json(path, capabilities)
    markdown_path = json_path.with_suffix(".md")
    markdown_path.write_text(
        _capabilities_markdown(capabilities),
        encoding="utf-8",
    )
    return json_path, markdown_path


def _require_positive_int(value, field: str, issues: list[str]) -> int:
    if isinstance(value, bool):
        issues.append(f"{field} 必须是正整数")
        return 0
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        issues.append(f"{field} 必须是正整数")
        return 0
    if normalized <= 0:
        issues.append(f"{field} 必须大于 0")
    return normalized


def validate_job(job: dict) -> tuple[dict, list[str]]:
    issues = []
    if not isinstance(job, dict):
        return {}, ["任务文件根节点必须是 JSON 对象"]

    if job.get("schema_version") != 2:
        issues.append("schema_version 必须为 2")

    normalized = {
        "schema_version": 2,
        "job_id": str(job.get("job_id", "")).strip(),
        "idea": str(job.get("idea", "")).strip(),
        "target_chapters": _require_positive_int(
            job.get("target_chapters"), "target_chapters", issues
        ),
        "words_per_chapter": _require_positive_int(
            job.get("words_per_chapter"), "words_per_chapter", issues
        ),
        "writer_style": str(job.get("writer_style", "")).strip(),
        "material_config": normalize_material_config(job.get("material_config")),
        "pattern_config": normalize_pattern_config(job.get("pattern_config")),
        "pattern_seed": job.get("pattern_seed"),
        "material_seed": job.get("material_seed"),
    }
    if not normalized["job_id"]:
        issues.append("job_id 不能为空")
    if not normalized["idea"]:
        issues.append("idea 不能为空")
    if normalized["writer_style"] not in STYLE_KEYS:
        issues.append(
            "writer_style 无效，可选值为：" + ", ".join(sorted(STYLE_KEYS))
        )

    issues.extend(
        validate_pattern_config(
            normalized["pattern_config"], normalized["writer_style"]
        )
    )
    issues.extend(
        validate_material_config(
            normalized["material_config"],
            normalized["pattern_config"],
            require_items=False,
        )
    )

    for seed_field in ("pattern_seed", "material_seed"):
        if normalized[seed_field] in ("", None):
            normalized[seed_field] = None
            continue
        try:
            normalized[seed_field] = int(normalized[seed_field])
        except (TypeError, ValueError):
            issues.append(f"{seed_field} 必须是整数")
            normalized[seed_field] = None
    return normalized, list(dict.fromkeys(issues))


def _initial_state_from_job(
    job: dict,
    run_id: str,
    material_config: dict,
    pattern_config: dict,
) -> dict:
    return {
        "user_idea": job["idea"],
        "run_id": run_id,
        "material_config": material_config,
        "pattern_config": pattern_config,
        "target_chapters": job["target_chapters"],
        "words_per_chapter": job["words_per_chapter"],
        "writer_style": job["writer_style"],
        "continuity_state": "",
        "story_ledger": {},
        "ledger_delta": {},
        "continuity_report": {},
        "scene_plan": {},
        "draft_candidates": [],
        "current_chapter": 1,
        "iteration_count": 0,
    }


def _find_run_output(folder: str, run_id: str, suffix: str) -> str:
    output_dir = Path.cwd() / folder
    candidates = sorted(
        output_dir.glob(f"*_{run_id}{suffix}"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0].resolve()) if candidates else ""


def _safe_run_identifier(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "-", str(value or "").strip())
    return safe.strip(".-_")[:80] or "job"


def prepare_job(raw_job: dict) -> tuple[dict, dict, dict]:
    job, issues = validate_job(raw_job)
    if issues:
        raise ValueError("; ".join(issues))

    pattern_config = normalize_pattern_config(job["pattern_config"])
    primary_pattern = pattern_config["primary"]
    if is_strong_pattern(primary_pattern):
        ending_options = load_story_patterns()[primary_pattern].get(
            "ending_options", {}
        )
        requested_ending = pattern_config.get("manifest", {}).get("ending")
        ending = (
            requested_ending
            if requested_ending in ending_options
            else next(iter(ending_options), "default")
        )
        pattern_config["manifest"] = roll_pattern_manifest(
            primary_pattern,
            seed=job["pattern_seed"],
            ending=ending,
        )
    material_config = sample_materials(
        job["material_config"],
        pattern_config,
        seed=job["material_seed"],
    )
    return job, material_config, pattern_config


def validate_job_file(
    job_path: str | os.PathLike,
    result_path: str | os.PathLike | None,
) -> int:
    source_path = Path(job_path).expanduser().resolve()
    target_path = (
        Path(result_path).expanduser().resolve()
        if result_path
        else source_path.with_name("preflight.json")
    )
    result = {
        "schema_version": 2,
        "status": "failed",
        "job_file": str(source_path),
        "started_at": _now_iso(),
    }
    try:
        raw_job = json.loads(source_path.read_text(encoding="utf-8-sig"))
        job, material_config, pattern_config = prepare_job(raw_job)
        result.update({
            "status": "validated",
            "job_id": job["job_id"],
            "configuration": job,
            "material_config": material_config,
            "pattern_config": pattern_config,
        })
    except Exception as error:
        result.update({
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
    finally:
        result["finished_at"] = _now_iso()
        written_path = _write_json(target_path, result)
        print(f"PREFLIGHT_PATH={written_path}")
    return 0 if result["status"] == "validated" else 1


def run_job_file(
    job_path: str | os.PathLike,
    result_path: str | os.PathLike | None,
    auto_approve: bool,
) -> int:
    source_path = Path(job_path).expanduser().resolve()
    target_path = (
        Path(result_path).expanduser().resolve()
        if result_path
        else source_path.with_name("result.json")
    )
    started_at = _now_iso()
    result = {
        "schema_version": 2,
        "status": "failed",
        "job_file": str(source_path),
        "started_at": started_at,
    }
    try:
        raw_job = json.loads(source_path.read_text(encoding="utf-8-sig"))
        job, material_config, pattern_config = prepare_job(raw_job)
        result["job_id"] = job["job_id"]
        run_id = (
            f"cli-{_safe_run_identifier(job['job_id'])}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        config = {"configurable": {"thread_id": run_id}}
        initial_state = _initial_state_from_job(
            job,
            run_id,
            material_config,
            pattern_config,
        )
        result.update({
            "run_id": run_id,
            "configuration": job,
            "material_config": material_config,
            "pattern_config": pattern_config,
        })

        print(f"--- CLI任务 {job['job_id']}：生成大纲 ---")
        for output in app.stream(initial_state, config=config):
            for node_name in output:
                print(f"✅ 节点 [{node_name}] 执行完毕")

        state_snapshot = app.get_state(config)
        state_values = dict(state_snapshot.values or {})
        result["novel_title"] = state_values.get("novel_title", "")
        result["outline_file"] = _find_run_output(
            "Outline", run_id, ".json"
        )

        approved = auto_approve
        if not auto_approve:
            answer = input("大纲已生成，输入 Y 批准继续：").strip().upper()
            approved = answer == "Y"
        if not approved:
            raise RuntimeError("大纲未获批准，任务终止")

        print("--- 大纲已批准：开始全自动写作 ---")
        for output in app.stream(None, config=config):
            for node_name, node_state in output.items():
                if node_name == "writer":
                    print(
                        f"✍️ 写手产出第 {node_state.get('current_chapter')} 章"
                    )
                elif node_name == "summarizer":
                    print(
                        f"🗂️ 已保存第 {node_state.get('saved_chapter')} 章"
                    )

        final_snapshot = app.get_state(config)
        final_values = dict(final_snapshot.values or {})
        result.update({
            "status": "succeeded",
            "novel_title": final_values.get(
                "novel_title", result.get("novel_title", "")
            ),
            "saved_chapter": final_values.get("saved_chapter", 0),
            "outline_file": result.get("outline_file")
            or _find_run_output("Outline", run_id, ".json"),
            "novel_file": _find_run_output("Novel", run_id, ".txt"),
        })
        if not result["novel_file"]:
            raise RuntimeError("流水线结束但未找到小说输出文件")
    except Exception as error:
        result.update({
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        logger.error("❌ CLI任务失败：%s", error)
    finally:
        result["finished_at"] = _now_iso()
        written_path = _write_json(target_path, result)
        print(f"RESULT_PATH={written_path}")
    return 0 if result["status"] == "succeeded" else 1


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="AutoWrite 小说流水线命令行入口"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--describe-capabilities",
        metavar="PATH",
        help="导出 JSON 能力表，并在同目录生成同名 Markdown 文件",
    )
    mode.add_argument(
        "--job-file",
        metavar="PATH",
        help="读取 JSON 小说任务",
    )
    mode.add_argument(
        "--validate-job-file",
        metavar="PATH",
        help="只校验并试抽 JSON 小说任务，不调用写作模型",
    )
    parser.add_argument(
        "--result-file",
        metavar="PATH",
        help="写入 JSON 运行结果；默认与任务文件同目录",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="自动批准架构师大纲",
    )
    return parser.parse_args()


# ==========================================
# 8. 启动测试执行 (带交互式输入版)
# ==========================================
if __name__ == "__main__":
    cli_args = parse_cli_args()
    if cli_args.describe_capabilities:
        json_path, markdown_path = export_capabilities(
            cli_args.describe_capabilities
        )
        print(f"CAPABILITIES_JSON={json_path}")
        print(f"CAPABILITIES_MD={markdown_path}")
        sys.exit(0)
    if cli_args.job_file:
        sys.exit(
            run_job_file(
                cli_args.job_file,
                cli_args.result_file,
                cli_args.auto_approve,
            )
        )
    if cli_args.validate_job_file:
        sys.exit(
            validate_job_file(
                cli_args.validate_job_file,
                cli_args.result_file,
            )
        )
    if cli_args.result_file:
        raise SystemExit(
            "--result-file 必须与 --job-file 或 "
            "--validate-job-file 一起使用"
        )

    print("🚀 小说工业流水线 v4.0 (词库 + 篇幅自适应) 启动...\n")
    
    # ======== 步骤1: 输入灵感 ========
    print("-" * 50)
    my_idea = input("💡 请输入你的小说灵感/点子 (直接回车将使用默认设定)：\n> ")
    if not my_idea.strip():
        my_idea = "一个能在梦里修仙的现代程序员"
    print()
    
    # 素材需要服从主辅套路，因此在套路确认后统一抽取。
    material_config = default_material_config()
    
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
    print("✍️ 写手风格:")
    for index, item in enumerate(STYLE_OPTIONS, start=1):
        print(f"   [{index}] {item['name']}")
    style_input = input("   选择风格 (默认1): ").strip()
    try:
        writer_style = (
            STYLE_OPTIONS[int(style_input) - 1]["key"]
            if style_input
            else STYLE_OPTIONS[0]["key"]
        )
    except (ValueError, IndexError):
        writer_style = STYLE_OPTIONS[0]["key"]
    style_name = next(
        item["name"] for item in STYLE_OPTIONS if item["key"] == writer_style
    )
    print(f"   ✅ 写手风格: {style_name}")
    print()

    # ======== 步骤5: 创作套路 ========
    patterns = load_story_patterns()
    pattern_keys = list(patterns)
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
    print(f"   ✅ 主套路: {patterns.get(story_pattern, patterns['none']).get('name', '无套路')}")
    secondary_candidates = [
        key for key, value in patterns.items()
        if key not in {"none", "custom", story_pattern}
        and not value.get("strong")
    ]
    secondary_input = input(
        "   辅助套路键（最多2个，逗号分隔，回车跳过；可从能力表查看）: "
    ).strip()
    secondary_patterns = [
        item.strip()
        for item in secondary_input.replace("，", ",").split(",")
        if item.strip() in secondary_candidates
    ][:2]
    pattern_config = {
        "schema_version": 2,
        "primary": story_pattern,
        "secondary": secondary_patterns,
        "custom_instruction": custom_pattern,
        "manifest": {},
        "structure_plan": {},
    }
    pattern_manifest = {}
    if is_strong_pattern(story_pattern):
        ending_options = patterns.get(story_pattern, {}).get("ending_options", {})
        ending_keys = list(ending_options) or ["no_reunion"]
        print("   结局方向: " + "  ".join(
            f"[{index + 1}] {ending_options.get(key, key)}"
            for index, key in enumerate(ending_keys)
        ))
        ending_choice = input("   选择结局方向 (默认1): ").strip()
        try:
            ending = ending_keys[int(ending_choice) - 1] if ending_choice else ending_keys[0]
        except (ValueError, IndexError):
            ending = ending_keys[0]
        pattern_manifest = roll_pattern_manifest(story_pattern, ending=ending)
        pattern_config["manifest"] = pattern_manifest
        print("   🎲 已生成强套路契约：")
        print(format_pattern_manifest(pattern_manifest))
    pattern_issues = validate_pattern_config(pattern_config, writer_style)
    if pattern_issues:
        raise SystemExit("套路配置无效：" + "；".join(pattern_issues))

    print("   📚 正在按人物、冲突、舞台、剧情装置抽取4项素材...")
    material_config = sample_materials(material_config, pattern_config)
    for item in material_config["items"]:
        print(f"      [{item['slot']}] {item['text']}")
    print()
    
    config = {"configurable": {"thread_id": "novel_project_001"}}
    
    initial_state = {
        "user_idea": my_idea,
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
