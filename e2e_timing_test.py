import argparse
import json
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime

from langgraph.graph import END, StateGraph

import Nodes
from State import NovelState


IDEA = "末班地铁每晚驶入一座会遗忘乘客姓名的城市，失业地图师必须在十站内找回妹妹。"
CHAPTER_HEADING_RE = re.compile(r"(?m)^第(\d+)章 ([^\n]+)$")
SEPARATOR_RE = re.compile(r"(?m)^\s*[=\-_*~—]{3,}\s*$")


def parse_args():
    parser = argparse.ArgumentParser(description="Run an end-to-end novel generation timing test.")
    parser.add_argument("--chapters", type=int, default=10)
    parser.add_argument("--words", type=int, default=1500)
    parser.add_argument("--style", default="default")
    parser.add_argument("--pattern", default="none")
    parser.add_argument("--custom-pattern", default="")
    parser.add_argument("--pattern-seed", type=int)
    parser.add_argument("--ending", choices=["no_reunion", "costly_reunion"], default="no_reunion")
    return parser.parse_args()


def build_timed_graph(timings: list[dict], run_info: dict):
    def timed(name, func):
        def wrapper(state):
            chapter = state.get("current_chapter", 1)
            attempt = state.get("iteration_count", 0) + (1 if name == "writer" else 0)
            started = time.perf_counter()
            record = {
                "node": name,
                "chapter": chapter,
                "attempt": attempt,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            try:
                result = func(state)
                if name == "architect":
                    run_info["novel_title"] = result.get("novel_title", "")
                    run_info["pattern_plan"] = result.get("pattern_plan", {})
                    outlines = result.get("chapter_outlines", {})
                    run_info["outline_chapters"] = len(outlines)
                    run_info["outline_inspection"] = {
                        "chapters": [
                            {
                                "number": int(number),
                                "chars_no_whitespace": Nodes.outline_char_count(outline),
                            }
                            for number, outline in sorted(
                                outlines.items(), key=lambda item: int(item[0])
                            )
                        ],
                        "issues": Nodes.outline_validation_issues(
                            outlines, state.get("target_chapters", 1)
                        ),
                    }
                    novel_path = Nodes._build_output_path(run_info["novel_title"])
                    run_info["novel_start_offset"] = (
                        os.path.getsize(novel_path) if os.path.exists(novel_path) else 0
                    )
                if name == "reviewer":
                    record["audit_status"] = result.get("audit_report", {}).get("审核状态")
                    record["audit_warnings"] = result.get("audit_report", {}).get("警告", [])
                    record["pattern_status"] = result.get("audit_report", {}).get("套路执行状态")
                    record["pattern_issues"] = result.get("audit_report", {}).get("套路问题", [])
                    record["style_score"] = result.get("editor_report", {}).get("文风评分")
                    record["ai_trace_issues"] = result.get("editor_report", {}).get("AI痕迹问题", [])
                return result
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record["duration_seconds"] = round(time.perf_counter() - started, 3)
                timings.append(record)

        return wrapper

    workflow = StateGraph(NovelState)
    workflow.add_node("architect", timed("architect", Nodes.architect_node))
    workflow.add_node("writer", timed("writer", Nodes.writer_node))
    workflow.add_node("reviewer", timed("reviewer", Nodes.reviewer_node))
    workflow.add_node("summarizer", timed("summarizer", Nodes.summarizer_node))
    workflow.set_entry_point("architect")
    workflow.add_edge("architect", "writer")

    def route_after_writer(state):
        return "writer" if Nodes.should_retry_short_draft(state) else "reviewer"

    workflow.add_conditional_edges(
        "writer",
        route_after_writer,
        {"writer": "writer", "reviewer": "reviewer"},
    )

    def route_after_review(state):
        audit = state.get("audit_report", {})
        editor = state.get("editor_report", {})
        outlines = state.get("chapter_outlines", {})
        need_retry = (
            audit.get("审核状态") == "不通过"
            or editor.get("文风评分", 10) < Nodes.STYLE_PASS_SCORE
        )
        if need_retry and state.get("iteration_count", 1) < Nodes.MAX_REVIEW_ATTEMPTS:
            return "writer"
        if state.get("current_chapter", 1) <= len(outlines):
            return "summarizer"
        return END

    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"writer": "writer", "summarizer": "summarizer", END: END},
    )

    def route_after_summary(state):
        outlines = state.get("chapter_outlines", {})
        if state.get("current_chapter", 1) <= len(outlines):
            return "writer"
        return END

    workflow.add_conditional_edges(
        "summarizer",
        route_after_summary,
        {"writer": "writer", END: END},
    )
    return workflow.compile()


def inspect_novel(
    path: str, expected_chapters: int, max_chars: int, start_offset: int = 0
) -> dict:
    with open(path, "rb") as file:
        file.seek(start_offset)
        content = file.read().decode("utf-8")

    matches = list(CHAPTER_HEADING_RE.finditer(content))
    chapters = []
    for index, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        body = content[body_start:body_end].strip()
        chapters.append(
            {
                "number": int(match.group(1)),
                "title": match.group(2).strip(),
                "title_chars": len(match.group(2).strip()),
                "body_chars_no_whitespace": len(re.sub(r"\s+", "", body)),
            }
        )

    issues = []
    min_chars = int(max_chars * Nodes.MIN_CHAPTER_RATIO)
    numbers = [chapter["number"] for chapter in chapters]
    expected_numbers = list(range(1, expected_chapters + 1))
    if numbers != expected_numbers:
        issues.append(f"章节编号异常：期望 {expected_numbers}，实际 {numbers}")
    if SEPARATOR_RE.search(content):
        issues.append("检测到装饰性分割线")
    for chapter in chapters:
        if chapter["title_chars"] > 10:
            issues.append(f"第{chapter['number']}章标题超过十字")
        if chapter["body_chars_no_whitespace"] > max_chars:
            issues.append(
                f"第{chapter['number']}章正文超过{max_chars}字："
                f"{chapter['body_chars_no_whitespace']}字"
            )
        if chapter["body_chars_no_whitespace"] < min_chars:
            issues.append(
                f"第{chapter['number']}章正文低于建议下限{min_chars}字："
                f"{chapter['body_chars_no_whitespace']}字"
            )

    return {
        "path": path,
        "file_chars": len(content),
        "body_chars_no_whitespace_total": sum(
            chapter["body_chars_no_whitespace"] for chapter in chapters
        ),
        "chapters": chapters,
        "issues": issues,
    }


def summarize_timings(timings: list[dict]) -> dict:
    grouped = defaultdict(list)
    for timing in timings:
        grouped[timing["node"]].append(timing["duration_seconds"])
    return {
        node: {
            "calls": len(values),
            "total_seconds": round(sum(values), 3),
            "average_seconds": round(sum(values) / len(values), 3),
            "min_seconds": round(min(values), 3),
            "max_seconds": round(max(values), 3),
        }
        for node, values in grouped.items()
    }


def summarize_review_quality(timings: list[dict]) -> dict:
    reviews = [item for item in timings if item["node"] == "reviewer"]
    first_reviews = [item for item in reviews if item.get("attempt") == 1]
    final_by_chapter = {}
    for item in reviews:
        final_by_chapter[item.get("chapter")] = item
    return {
        "reviews": len(reviews),
        "first_draft_reviews": len(first_reviews),
        "first_draft_passes": sum(item.get("audit_status") == "通过" for item in first_reviews),
        "final_chapter_passes": sum(
            item.get("audit_status") == "通过" for item in final_by_chapter.values()
        ),
        "logic_warnings": sum(len(item.get("audit_warnings", [])) for item in reviews),
        "pattern_failures": sum(item.get("pattern_status") == "不通过" for item in reviews),
        "pattern_issues": sum(len(item.get("pattern_issues", [])) for item in reviews),
        "ai_trace_issues": sum(len(item.get("ai_trace_issues", [])) for item in reviews),
    }


def main():
    args = parse_args()
    timings = []
    pattern_manifest = (
        Nodes.roll_pattern_manifest(args.pattern, seed=args.pattern_seed, ending=args.ending)
        if Nodes.is_strong_pattern(args.pattern)
        else {}
    )
    if Nodes.is_strong_pattern(args.pattern):
        compatible_styles = Nodes.compatible_styles_for_pattern(args.pattern)
        if args.style not in compatible_styles:
            raise SystemExit(f"强套路不兼容写手风格 {args.style}；可用风格：{compatible_styles}")
    run_info = {
        "run_id": f"e2e-{uuid.uuid4().hex[:8]}",
        "idea": IDEA,
        "requested_chapters": args.chapters,
        "max_chars_per_chapter": args.words,
        "writer_style": args.style,
        "story_pattern": args.pattern,
        "pattern_manifest": pattern_manifest,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    graph = build_timed_graph(timings, run_info)
    initial_state = {
        "user_idea": IDEA,
        "keywords": [],
        "target_chapters": args.chapters,
        "words_per_chapter": args.words,
        "writer_style": args.style,
        "story_pattern": args.pattern,
        "custom_pattern": args.custom_pattern,
        "pattern_manifest": pattern_manifest,
        "pattern_plan": {},
        "continuity_state": "",
        "current_chapter": 1,
        "iteration_count": 0,
    }

    total_started = time.perf_counter()
    error = None
    try:
        for _ in graph.stream(initial_state, config={"recursion_limit": 200}):
            pass
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    total_seconds = round(time.perf_counter() - total_started, 3)

    run_info["finished_at"] = datetime.now().isoformat(timespec="seconds")
    run_info["total_seconds"] = total_seconds
    run_info["error"] = error
    run_info["timing_summary"] = summarize_timings(timings)
    run_info["review_quality"] = summarize_review_quality(timings)
    run_info["node_calls"] = timings
    run_info["execution_issues"] = []
    unexpected_calls = [
        timing
        for timing in timings
        if timing["node"] in {"writer", "reviewer"}
        and timing["chapter"] > args.chapters
    ]
    if unexpected_calls:
        run_info["execution_issues"].append(
            f"检测到目标章节之外的写作或审稿调用：{len(unexpected_calls)} 次"
        )
    review_counts = defaultdict(int)
    for timing in timings:
        if timing["node"] == "reviewer":
            review_counts[timing["chapter"]] += 1
    excessive_reviews = {
        chapter: count
        for chapter, count in review_counts.items()
        if count > Nodes.MAX_REVIEW_ATTEMPTS
    }
    if excessive_reviews:
        run_info["execution_issues"].append(
            f"检测到章节审核次数超过{Nodes.MAX_REVIEW_ATTEMPTS}次：{excessive_reviews}"
        )

    novel_title = run_info.get("novel_title")
    if novel_title:
        novel_path = Nodes._build_output_path(novel_title)
        if os.path.exists(novel_path):
            run_info["novel_inspection"] = inspect_novel(
                novel_path,
                args.chapters,
                args.words,
                run_info.get("novel_start_offset", 0),
            )
        else:
            run_info["novel_inspection"] = {
                "path": novel_path,
                "issues": ["小说输出文件不存在"],
            }

    os.makedirs("TestResults", exist_ok=True)
    report_path = os.path.join("TestResults", f"{run_info['run_id']}.json")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(run_info, file, ensure_ascii=False, indent=2)

    print(json.dumps(run_info, ensure_ascii=False, indent=2))
    print(f"\nREPORT_PATH={report_path}")
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
