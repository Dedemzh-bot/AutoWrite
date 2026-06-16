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
            "story_pattern": "none",
            "custom_pattern": "",
            "pattern_manifest": {},
            "pattern_plan": {},
            "continuity_state": "",
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
        self.assertEqual(state["story_pattern"], "none")


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

    def test_template_key_error_is_not_retried(self):
        from Nodes import invoke_with_retry

        chain = MagicMock()
        chain.invoke.side_effect = KeyError("未转义变量")

        with self.assertRaisesRegex(RuntimeError, "提示词模板变量缺失"):
            invoke_with_retry(chain, {}, "写手", max_attempts=3)

        self.assertEqual(chain.invoke.call_count, 1)


class TestStoryPatternsAndReviewPolicy(unittest.TestCase):
    def test_custom_pattern_is_added_to_each_stage(self):
        from Nodes import resolve_story_pattern

        pattern = resolve_story_pattern({
            "story_pattern": "custom",
            "custom_pattern": "每三章揭示一次假规则",
        })

        self.assertIn("每三章揭示一次假规则", pattern["architect"])
        self.assertIn("每三章揭示一次假规则", pattern["writer"])
        self.assertIn("每三章揭示一次假规则", pattern["auditor"])

    def test_unknown_pattern_falls_back_to_none(self):
        from Nodes import resolve_story_pattern

        pattern = resolve_story_pattern({"story_pattern": "missing"})

        self.assertEqual(pattern["key"], "none")
        self.assertEqual(pattern["name"], "无套路")

    def test_soft_warnings_do_not_trigger_audit_failure(self):
        from Nodes import normalize_audit_report

        report = normalize_audit_report({
            "审核状态": "不通过",
            "发现的问题": [],
            "警告": ["铺垫略少"],
            "修改建议": "补一处线索",
        })

        self.assertEqual(report["审核状态"], "通过")
        self.assertEqual(report["警告"], ["铺垫略少"])

    def test_hard_issues_always_trigger_audit_failure(self):
        from Nodes import normalize_audit_report

        report = normalize_audit_report({
            "审核状态": "通过",
            "发现的问题": ["已死亡角色重新出现"],
            "警告": [],
            "修改建议": "修正角色状态",
        })

        self.assertEqual(report["审核状态"], "不通过")

    def test_blank_hard_issue_does_not_trigger_failure(self):
        from Nodes import normalize_audit_report

        report = normalize_audit_report({
            "审核状态": "不通过",
            "发现的问题": [""],
            "警告": [],
            "修改建议": "无",
        })

        self.assertEqual(report["审核状态"], "通过")

    def test_pattern_issue_triggers_failure_but_soft_warning_does_not(self):
        from Nodes import normalize_audit_report

        failed = normalize_audit_report({
            "审核状态": "通过",
            "发现的问题": [],
            "警告": [],
            "套路执行状态": "通过",
            "套路问题": ["本章未完成女主心死转折"],
            "修改建议": "补足转折",
        })
        warned = normalize_audit_report({
            "审核状态": "不通过",
            "发现的问题": [],
            "警告": ["情绪浓度略弱"],
            "套路执行状态": "不通过",
            "套路问题": [],
            "修改建议": "增强对比",
        })

        self.assertEqual(failed["审核状态"], "不通过")
        self.assertEqual(failed["套路执行状态"], "不通过")
        self.assertEqual(warned["审核状态"], "通过")
        self.assertEqual(warned["套路执行状态"], "通过")

    def test_low_style_score_requires_specific_ai_trace_issue(self):
        from Nodes import normalize_editor_report

        no_evidence = normalize_editor_report({
            "文风评分": 4,
            "AI痕迹问题": [],
            "改进建议": "无",
        })
        with_evidence = normalize_editor_report({
            "文风评分": 4,
            "AI痕迹问题": ["连续三段使用同一句式"],
            "改进建议": "调整句式",
        })

        self.assertEqual(no_evidence["文风评分"], 7)
        self.assertEqual(with_evidence["文风评分"], 4)

    def test_auditor_receives_continuity_and_next_outline(self):
        from Nodes import _audit_inputs

        inputs = _audit_inputs({
            "current_chapter": 1,
            "world_bible": "规则不可违背",
            "continuity_state": "主角在车站",
            "chapter_outlines": {"1": "进入车站", "2": "离开车站"},
            "current_draft": "正文",
            "story_pattern": "rule_horror",
        })

        self.assertEqual(inputs["continuity_state"], "主角在车站")
        self.assertEqual(inputs["next_outline"], "离开车站")
        self.assertIn("规则", inputs["pattern_auditor"])

    def test_shared_writer_rules_exist(self):
        from Nodes import load_prompt

        rules = load_prompt("writer_common_rules.md")

        self.assertIn("句式去模板化", rules)
        self.assertIn("连续性优先", rules)

    def test_writer_system_prompts_have_no_unescaped_template_variables(self):
        import string
        from Nodes import load_prompt

        for file_name in [
            "writer_common_rules.md",
            "writer_system.md",
            "writer_system_hot_blood.md",
            "writer_system_literary.md",
            "writer_system_cold.md",
            "writer_system_humor.md",
            "writer_system_18xx.md",
        ]:
            variables = [
                field_name
                for _, field_name, _, _ in string.Formatter().parse(load_prompt(file_name))
                if field_name
            ]
            self.assertEqual(variables, [], file_name)

    def test_pattern_issues_are_preserved_as_final_chapter_warnings(self):
        from Nodes import chapter_quality_warnings

        warnings = chapter_quality_warnings({
            "current_chapter": 2,
            "current_draft": "第2章 测试\n\n" + "正文" * 50,
            "words_per_chapter": 100,
            "audit_report": {
                "审核状态": "不通过",
                "发现的问题": [],
                "警告": [],
                "套路问题": ["未完成最重伤害"],
            },
            "editor_report": {"文风评分": 7},
        })

        self.assertTrue(any("套路任务仍未完成" in warning for warning in warnings))


class TestFemaleAngstAwakeningPattern(unittest.TestCase):
    def test_strong_pattern_has_expected_style_constraints_and_rules(self):
        from Nodes import compatible_styles_for_pattern, is_strong_pattern, resolve_story_pattern

        self.assertTrue(is_strong_pattern("female_angst_awakening"))
        self.assertEqual(
            compatible_styles_for_pattern("female_angst_awakening"),
            ["default", "literary", "cold"],
        )
        pattern = resolve_story_pattern({"story_pattern": "female_angst_awakening"})
        self.assertIn("强制写作技巧", pattern["writer"])
        self.assertIn("套路审核规则", pattern["auditor"])

    def test_strong_pattern_allows_world_stages_but_blocks_conflicting_drivers(self):
        from Nodes import (
            filter_material_categories_for_pattern,
            validate_material_categories_for_pattern,
        )

        self.assertEqual(
            validate_material_categories_for_pattern(
                "female_angst_awakening", ["科幻", "修仙", "末日", "历史"]
            ),
            [],
        )
        issues = validate_material_categories_for_pattern(
            "female_angst_awakening", ["男频", "恐怖"]
        )
        self.assertEqual(len(issues), 2)
        self.assertEqual(
            filter_material_categories_for_pattern(
                "female_angst_awakening", ["科幻", "男频", "女频", "恐怖"]
            ),
            ["科幻", "女频"],
        )

    def test_manifest_is_reproducible_and_limits_reproductive_harm(self):
        from Nodes import roll_pattern_manifest, validate_pattern_manifest

        first = roll_pattern_manifest("female_angst_awakening", seed=20260616)
        second = roll_pattern_manifest("female_angst_awakening", seed=20260616)

        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first["conflicts"]), 2)
        self.assertLessEqual(len(first["conflicts"]), 3)
        self.assertLessEqual(
            sum(item["category"] == "reproductive" for item in first["conflicts"]),
            1,
        )
        self.assertEqual(validate_pattern_manifest(first), [])

    def test_paywall_target_uses_total_word_ratio_for_different_lengths(self):
        from Nodes import build_pattern_plan, roll_pattern_manifest

        manifest = roll_pattern_manifest("female_angst_awakening", seed=7)
        for chapters, words in [(5, 1500), (10, 1500), (20, 3000)]:
            plan = build_pattern_plan(manifest, chapters, words)
            paywalls = [task for task in plan.values() if task["is_paywall_turn"]]
            self.assertEqual(len(paywalls), 1)
            self.assertTrue(all(task["conflict_stage"] for task in plan.values()))
            ratio = paywalls[0]["paywall_target_word"] / (chapters * words)
            self.assertGreaterEqual(ratio, 0.45)
            self.assertLessEqual(ratio, 0.50)

    def test_plan_respects_selected_ending(self):
        from Nodes import build_pattern_plan, roll_pattern_manifest

        no_reunion = roll_pattern_manifest(
            "female_angst_awakening", seed=1, ending="no_reunion"
        )
        costly = roll_pattern_manifest(
            "female_angst_awakening", seed=1, ending="costly_reunion"
        )

        self.assertIn("不复合", build_pattern_plan(no_reunion, 10, 1500)["10"]["required_event"])
        self.assertIn("不可逆的代价", build_pattern_plan(costly, 10, 1500)["10"]["required_event"])

    def test_lexical_outline_checks_warn_without_rejecting_structured_plan(self):
        from Nodes import (
            attach_pattern_plan_to_outlines,
            build_pattern_plan,
            roll_pattern_manifest,
            strong_pattern_outline_content_warnings,
            strong_pattern_validation_issues,
        )

        manifest = roll_pattern_manifest("female_angst_awakening", seed=9)
        plan = build_pattern_plan(manifest, 5, 1500)
        outlines = {str(index): "普通剧情推进。" * 80 for index in range(1, 6)}
        warnings = strong_pattern_outline_content_warnings(manifest, plan, outlines, 5)
        attached = attach_pattern_plan_to_outlines(outlines, plan)
        blocking_issues = strong_pattern_validation_issues(manifest, plan, attached, 5)

        self.assertTrue(any("前300字" in warning for warning in warnings))
        self.assertTrue(any("心死" in warning for warning in warnings))
        self.assertTrue(any("默认结局" in warning for warning in warnings))
        self.assertTrue(any("虐点模块" in warning for warning in warnings))
        self.assertEqual(blocking_issues, [])

    def test_rewash_can_replace_old_attached_pattern_tasks(self):
        from Nodes import (
            attach_pattern_plan_to_outlines,
            build_pattern_plan,
            roll_pattern_manifest,
            strip_pattern_plan_from_outlines,
        )

        raw = {"1": "原始细纲内容"}
        manifest = roll_pattern_manifest("female_angst_awakening", seed=3)
        attached = attach_pattern_plan_to_outlines(raw, build_pattern_plan(manifest, 1, 1500))

        self.assertEqual(strip_pattern_plan_from_outlines(attached), raw)


class TestWebDefaults(unittest.TestCase):
    def test_default_estimate_is_fifteen_thousand(self):
        from web_app import HTML_PAGE

        self.assertIn('id="estWords">预估: 约 15,000 字', HTML_PAGE)
        self.assertIn("updateEstimate();", HTML_PAGE)

    def test_web_has_pattern_controls(self):
        from web_app import HTML_PAGE

        self.assertIn('id="storyPattern"', HTML_PAGE)
        self.assertIn('id="washStoryPattern"', HTML_PAGE)
        self.assertIn('value="rule_horror"', HTML_PAGE)
        self.assertIn("get_patterns", HTML_PAGE)
        self.assertIn("female_angst_awakening", HTML_PAGE)
        self.assertIn("roll_pattern_manifest", HTML_PAGE)
        self.assertIn('id="patternEnding"', HTML_PAGE)
        self.assertIn('id="washPatternEnding"', HTML_PAGE)
        self.assertIn("请先确认女频虐恋觉醒套路契约", HTML_PAGE)
        self.assertIn("随机素材库", HTML_PAGE)
        self.assertIn("isCategoryBlocked", HTML_PAGE)
        self.assertIn("updateMaterialHint", HTML_PAGE)


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

    def test_summary_updates_structured_continuity_state(self):
        from Nodes import summarizer_node

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "novel.txt")
            state = {
                "current_chapter": 1,
                "current_draft": "第1章 开端\n\n主角进入车站。",
                "novel_title": "测试",
                "chapter_outlines": {"1": "开端", "2": "后续"},
                "story_summary": "旧摘要",
                "continuity_state": "旧连续性",
                "words_per_chapter": 10,
                "audit_report": {"审核状态": "通过", "警告": []},
                "editor_report": {"文风评分": 7},
            }
            model_result = MagicMock(content="时间线：午夜\n角色位置：主角在车站")
            with patch("Nodes._build_output_path", return_value=output_path):
                with patch("Nodes.invoke_with_retry", return_value=model_result) as invoke:
                    result = summarizer_node(state)

            self.assertEqual(result["continuity_state"], model_result.content)
            self.assertEqual(result["story_summary"], model_result.content)
            self.assertEqual(invoke.call_args.args[1]["old_summary"], "旧连续性")


if __name__ == "__main__":
    unittest.main(verbosity=2)
