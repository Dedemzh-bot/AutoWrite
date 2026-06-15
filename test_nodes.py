import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestLoadPrompt(unittest.TestCase):
    def setUp(self):
        from Nodes import load_prompt
        self.load_prompt = load_prompt
        self.temp_dir = tempfile.TemporaryDirectory()
        self.role_dir = os.path.join(self.temp_dir.name, "Role")
        os.makedirs(self.role_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_existing_prompt(self):
        test_content = "# 测试提示词"
        file_path = os.path.join(self.role_dir, "test.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(test_content)

        with patch("Nodes.os.path.dirname", return_value=self.temp_dir.name):
            with patch("Nodes.os.path.abspath", return_value=self.temp_dir.name):
                result = self.load_prompt("test.md")
                self.assertEqual(result, test_content)

    def test_load_missing_prompt_raises(self):
        with patch("Nodes.os.path.dirname", return_value=self.temp_dir.name):
            with patch("Nodes.os.path.abspath", return_value=self.temp_dir.name):
                with self.assertRaises(FileNotFoundError):
                    self.load_prompt("nonexistent.md")


class TestPydanticModels(unittest.TestCase):
    def test_architect_output_valid(self):
        from Nodes import ArchitectOutput
        data = ArchitectOutput(
            novel_title="绝世剑神",
            world_bible="测试世界观，不少于500字的内容填充" * 20,
            chapter_outlines={"1": "第1章剧情", "2": "第2章剧情", "3": "第3章剧情", "4": "第4章剧情", "5": "第5章剧情"},
            estimated_words=30000
        )
        self.assertEqual(data.novel_title, "绝世剑神")
        self.assertEqual(data.chapter_outlines["1"], "第1章剧情")
        self.assertIn("测试世界观", data.world_bible)

    def test_audit_report_pass(self):
        from Nodes import AuditReport
        report = AuditReport(
            审核状态="通过",
            发现的问题=[],
            修改建议="无"
        )
        self.assertEqual(report.审核状态, "通过")

    def test_audit_report_fail(self):
        from Nodes import AuditReport
        report = AuditReport(
            审核状态="不通过",
            发现的问题=["主角战力前后矛盾", "配角生死倒错"],
            修改建议="重新梳理战力体系，修正配角生死状态"
        )
        self.assertEqual(report.审核状态, "不通过")
        self.assertEqual(len(report.发现的问题), 2)

    def test_editor_report(self):
        from Nodes import EditorReport
        report = EditorReport(
            文风评分=7,
            改进建议="减少形容词堆砌，增加动作描写"
        )
        self.assertEqual(report.文风评分, 7)


class TestNovelState(unittest.TestCase):
    def test_state_fields_exist(self):
        from State import NovelState
        state: NovelState = {
            "user_idea": "测试",
            "world_bible": "",
            "chapter_outlines": {},
            "keywords": [],
            "target_chapters": 12,
            "words_per_chapter": 2500,
            "writer_style": "default",
            "current_chapter": 1,
            "current_draft": "",
            "audit_report": {},
            "editor_report": {},
            "iteration_count": 0,
            "saved_chapter": 0,
            "novel_title": "测试",
            "story_summary": "",
        }
        self.assertEqual(state["target_chapters"], 12)
        self.assertEqual(state["words_per_chapter"], 2500)
        self.assertEqual(state["writer_style"], "default")


class TestChapterOutputFormat(unittest.TestCase):
    def test_normalizes_heading_and_removes_separators(self):
        from Nodes import normalize_chapter_output

        content = """
====================
【第 12 章：这是一个超过十个字的章节名字】
--------------------

正文第一段。
========
正文第二段。
"""
        result = normalize_chapter_output(content, 12)

        self.assertEqual(result.splitlines()[0], "第12章 这是一个超过十个字的")
        self.assertNotIn("=", result)
        self.assertNotIn("【", result)

    def test_normalizes_markdown_heading_and_removes_duplicate_heading(self):
        from Nodes import normalize_chapter_output

        content = """
## 第三章 风起云涌

正文内容。

第3章 重复标题
"""
        result = normalize_chapter_output(content, 3)

        self.assertEqual(result.splitlines()[0], "第3章 风起云涌")
        self.assertEqual(result.count("第3章"), 1)

    def test_uses_fallback_title_when_heading_is_missing(self):
        from Nodes import normalize_chapter_output

        result = normalize_chapter_output("正文内容。", 5)

        self.assertEqual(result, "第5章 正文\n\n正文内容。")

    def test_limits_body_to_configured_maximum(self):
        from Nodes import MIN_CHAPTER_RATIO, normalize_chapter_output

        body = "第一句。" + ("内容" * 100) + "最后一句。"
        result = normalize_chapter_output(f"第1章 测试\n\n{body}", 1, 50)
        normalized_body = result.split("\n\n", 1)[1]
        body_chars = len("".join(normalized_body.split()))

        self.assertLessEqual(body_chars, 50)
        self.assertGreaterEqual(body_chars, int(50 * MIN_CHAPTER_RATIO))

    def test_short_draft_only_retries_before_second_draft(self):
        from Nodes import should_retry_short_draft

        state = {
            "current_draft": "第1章 测试\n\n太短。",
            "words_per_chapter": 100,
            "iteration_count": 1,
        }
        self.assertTrue(should_retry_short_draft(state))

        state["iteration_count"] = 2
        self.assertFalse(should_retry_short_draft(state))

    def test_normalizes_outline_count_to_requested_chapters(self):
        from Nodes import normalize_chapter_outlines

        trimmed = normalize_chapter_outlines({"1": "开端", "2": "多余"}, 1)
        self.assertEqual(trimmed, {"1": "开端"})

        filled = normalize_chapter_outlines({"1": "开端"}, 2)
        self.assertEqual(list(filled.keys()), ["1", "2"])
        self.assertIn("结局", filled["2"])

    def test_outline_validation_requires_two_hundred_chars(self):
        from Nodes import outline_validation_issues

        short_issues = outline_validation_issues({"1": "太短"}, 1)
        self.assertTrue(any("少于200字" in issue for issue in short_issues))
        self.assertTrue(outline_validation_issues(None, 1))
        self.assertTrue(outline_validation_issues({}, 0))

        detailed = "具体剧情推进与人物行动。" * 20
        self.assertEqual(outline_validation_issues({"1": detailed}, 1), [])


class TestModelRetry(unittest.TestCase):
    def test_invoke_retries_connection_errors(self):
        from Nodes import invoke_with_retry

        chain = MagicMock()
        chain.invoke.side_effect = [ConnectionError("Connection error"), "成功"]
        with patch("Nodes.time.sleep"):
            result = invoke_with_retry(chain, {}, "测试节点", max_attempts=2)

        self.assertEqual(result, "成功")
        self.assertEqual(chain.invoke.call_count, 2)

    def test_invoke_raises_clear_error_after_retries(self):
        from Nodes import invoke_with_retry

        chain = MagicMock()
        chain.invoke.side_effect = ConnectionError("Connection error")
        with patch("Nodes.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "已自动重试3次"):
                invoke_with_retry(chain, {}, "写手", max_attempts=3)


class TestWebPortSelection(unittest.TestCase):
    def test_available_preferred_port_is_not_treated_as_existing_server(self):
        from web_app import _is_existing_autowrite

        with patch("web_app._port_is_available", return_value=True):
            self.assertFalse(_is_existing_autowrite(8080))

    def test_reuses_existing_autowrite_server(self):
        from web_app import choose_web_port

        with patch("web_app._is_existing_autowrite", return_value=True):
            self.assertEqual(choose_web_port(8080), (8080, True))

    def test_uses_next_port_when_preferred_is_occupied(self):
        from web_app import choose_web_port

        with (
            patch("web_app._is_existing_autowrite", return_value=False),
            patch("web_app._port_is_available", side_effect=[False, True]),
        ):
            self.assertEqual(choose_web_port(8080), (8081, False))


class TestGraphStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from TheGraph import workflow, app
        cls.workflow = workflow
        cls.app = app

    def test_all_nodes_exist(self):
        nodes = list(self.workflow.nodes.keys())
        expected = {"architect", "writer", "reviewer", "summarizer"}
        self.assertEqual(set(nodes), expected)

    def test_entry_point(self):
        compiled = self.app
        self.assertIsNotNone(compiled)

    def test_architect_node_exists(self):
        from Nodes import architect_node
        self.assertTrue(callable(architect_node))

    def test_writer_node_exists(self):
        from Nodes import writer_node
        self.assertTrue(callable(writer_node))

    def test_auditor_node_exists(self):
        from Nodes import reviewer_node
        self.assertTrue(callable(reviewer_node))

    def test_editor_node_exists(self):
        from Nodes import reviewer_node
        self.assertTrue(callable(reviewer_node))

    def test_summarizer_node_exists(self):
        from Nodes import summarizer_node
        self.assertTrue(callable(summarizer_node))

    def test_summary_route_stops_after_last_chapter(self):
        from langgraph.graph import END
        from TheGraph import route_after_summary

        self.assertEqual(
            route_after_summary({"current_chapter": 11, "chapter_outlines": {"10": "结局"}}),
            END,
        )

    def test_writer_route_retries_short_draft_before_review(self):
        from TheGraph import route_after_writer

        state = {
            "current_draft": "第1章 测试\n\n太短。",
            "words_per_chapter": 100,
            "iteration_count": 1,
        }
        self.assertEqual(route_after_writer(state), "writer")

        state["iteration_count"] = 2
        self.assertEqual(route_after_writer(state), "reviewer")

    def test_review_route_accepts_style_score_seven(self):
        from TheGraph import route_after_review

        state = {
            "audit_report": {"审核状态": "通过"},
            "editor_report": {"文风评分": 7},
            "chapter_outlines": {"1": "剧情"},
            "current_chapter": 1,
            "iteration_count": 1,
        }
        self.assertEqual(route_after_review(state), "summarizer")

    def test_review_route_allows_only_one_rewrite(self):
        from TheGraph import route_after_review

        state = {
            "audit_report": {"审核状态": "不通过"},
            "editor_report": {"文风评分": 6},
            "chapter_outlines": {"1": "剧情"},
            "current_chapter": 1,
            "iteration_count": 1,
        }
        self.assertEqual(route_after_review(state), "writer")

        state["iteration_count"] = 2
        self.assertEqual(route_after_review(state), "summarizer")


class TestSummarizerBehavior(unittest.TestCase):
    def test_last_chapter_skips_summary_model(self):
        from Nodes import summarizer_node

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "novel.txt")
            state = {
                "current_chapter": 1,
                "current_draft": "第1章 结局\n\n故事结束。",
                "novel_title": "测试",
                "chapter_outlines": {"1": "结局"},
                "story_summary": "旧摘要",
                "words_per_chapter": 10,
                "audit_report": {"审核状态": "通过"},
                "editor_report": {"文风评分": 7},
            }
            with patch("Nodes._build_output_path", return_value=output_path):
                with patch("Nodes.load_prompt") as load_prompt:
                    result = summarizer_node(state)

            load_prompt.assert_not_called()
            self.assertTrue(result["summary_skipped"])
            self.assertEqual(result["story_summary"], "旧摘要")
            self.assertEqual(result["audit_report"], {})
            self.assertEqual(result["editor_report"], {})

    def test_summary_connection_failure_keeps_pipeline_running(self):
        from Nodes import summarizer_node

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "novel.txt")
            state = {
                "current_chapter": 1,
                "current_draft": "第1章 开端\n\n故事开始。",
                "novel_title": "测试",
                "chapter_outlines": {"1": "开端", "2": "后续"},
                "story_summary": "旧摘要",
                "words_per_chapter": 10,
                "audit_report": {"审核状态": "通过"},
                "editor_report": {"文风评分": 7},
            }
            with patch("Nodes._build_output_path", return_value=output_path):
                with patch("Nodes.invoke_with_retry", side_effect=RuntimeError("连接失败")):
                    result = summarizer_node(state)

            self.assertEqual(result["story_summary"], "旧摘要")
            self.assertTrue(
                any("摘要更新失败" in warning for warning in result["chapter_warnings"])
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
