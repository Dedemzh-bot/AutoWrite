import os
import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

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
            novel_tags={
                "core": "玄幻升级",
                "情节": ["逆袭打脸", "重生"],
                "角色": ["天才主角"],
                "情绪": ["热血"],
                "背景": ["古代"],
            },
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


class TestArchitectFallback(unittest.TestCase):
    def setUp(self):
        import Nodes
        Nodes._PROBED_PRIORITY = "function_calling"
        Nodes._PERMANENT_FAILURES.clear()

    @staticmethod
    def payload():
        return {
            "novel_title": "回声井",
            "world_bible": "稳定世界观",
            "chapter_outlines": {"1": "第一章细纲"},
            "novel_tags": {
                "core": "规则怪谈",
                "情节": ["女性成长", "无限流"],
                "角色": ["大女主"],
                "情绪": ["惊悚"],
                "背景": ["副本世界"],
            },
            "estimated_words": 12000,
        }

    @staticmethod
    def message(content="", tool_calls=None, additional_kwargs=None):
        message = MagicMock()
        message.content = content
        message.tool_calls = tool_calls or []
        message.additional_kwargs = additional_kwargs or {}
        message.response_metadata = {"finish_reason": "stop"}
        return message

    def test_recovers_function_call_arguments_when_parsed_is_empty(self):
        from Nodes import invoke_architect_with_fallback

        response = {
            "raw": self.message(
                tool_calls=[{"name": "ArchitectOutput", "args": self.payload()}]
            ),
            "parsed": None,
            "parsing_error": ValueError("empty wrapper"),
        }
        with patch(
            "Nodes._invoke_architect_function_calling",
            return_value=response,
        ), patch("Nodes._invoke_architect_json_object") as json_strategy:
            result = invoke_architect_with_fallback(MagicMock(), {})

        self.assertEqual(result.novel_title, "回声井")
        json_strategy.assert_not_called()

    def test_empty_function_result_falls_back_to_json_object(self):
        from Nodes import invoke_architect_with_fallback

        json_response = self.message(
            "```json\n" + json.dumps(self.payload(), ensure_ascii=False) + "\n```"
        )
        with patch(
            "Nodes._invoke_architect_function_calling",
            return_value={"raw": self.message(), "parsed": None},
        ), patch(
            "Nodes._invoke_architect_json_object",
            return_value=json_response,
        ), patch("Nodes._invoke_architect_plain_text") as plain_strategy:
            result = invoke_architect_with_fallback(MagicMock(), {})

        self.assertEqual(result.estimated_words, 12000)
        plain_strategy.assert_not_called()

    def test_malformed_json_falls_back_to_plain_text_extraction(self):
        from Nodes import invoke_architect_with_fallback

        plain = "结果如下：\n" + json.dumps(self.payload(), ensure_ascii=False)
        with patch(
            "Nodes._invoke_architect_function_calling",
            side_effect=RuntimeError("transport failed"),
        ), patch(
            "Nodes._invoke_architect_json_object",
            return_value=self.message("{bad json"),
        ), patch(
            "Nodes._invoke_architect_plain_text",
            return_value=self.message(plain),
        ):
            result = invoke_architect_with_fallback(MagicMock(), {})

        self.assertEqual(result.chapter_outlines["1"], "第一章细纲")

    def test_all_strategies_report_separate_failures(self):
        from Nodes import invoke_architect_with_fallback

        with patch(
            "Nodes._invoke_architect_function_calling",
            return_value={"raw": self.message(), "parsed": None},
        ), patch(
            "Nodes._invoke_architect_json_object",
            return_value=self.message("[]"),
        ), patch(
            "Nodes._invoke_architect_plain_text",
            return_value=self.message("not json"),
        ):
            with self.assertRaises(RuntimeError) as raised:
                invoke_architect_with_fallback(MagicMock(), {})

        detail = str(raised.exception)
        self.assertIn("function_calling", detail)
        self.assertIn("json_object", detail)
        self.assertIn("plain_text_json_extract", detail)


class TestStructuredOutputFallback(unittest.TestCase):
    @staticmethod
    def message(content):
        message = MagicMock()
        message.content = content
        message.tool_calls = []
        message.additional_kwargs = {}
        message.response_metadata = {"finish_reason": "stop"}
        return message

    def test_editor_empty_function_call_falls_back_to_json_object(self):
        from Nodes import EditorReport, invoke_structured_with_fallback

        payload = {
            "文风评分": 8,
            "AI痕迹问题": [],
            "AI痕迹警告": [],
            "改进建议": "无",
        }
        with patch(
            "Nodes.invoke_with_retry",
            side_effect=[None, self.message(json.dumps(payload, ensure_ascii=False))],
        ) as invoke:
            result = invoke_structured_with_fallback(
                MagicMock(), {}, MagicMock(), MagicMock(), EditorReport, "editor"
            )

        self.assertEqual(result.文风评分, 8)
        self.assertEqual(invoke.call_count, 2)

    def test_continuity_falls_back_to_plain_text_json_extraction(self):
        from Nodes import ContinuityReview, invoke_structured_with_fallback

        payload = {
            "new_immutable_facts": [],
            "state_updates": [],
            "new_foreshadowing": [],
            "resolved_foreshadowing_ids": [],
            "chapter_ending": "故事结束",
            "next_handoff": "",
            "conflicts": [],
            "warnings": [],
            "status": "pass",
        }
        with patch(
            "Nodes.invoke_with_retry",
            side_effect=[
                None,
                self.message("not json"),
                self.message("结果：" + json.dumps(payload, ensure_ascii=False)),
            ],
        ) as invoke:
            result = invoke_structured_with_fallback(
                MagicMock(), {}, MagicMock(), MagicMock(),
                ContinuityReview, "第8章连续性台账"
            )

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.chapter_ending, "故事结束")
        self.assertEqual(invoke.call_count, 3)

class TestNovelState(unittest.TestCase):
    def test_state_fields_exist(self):
        from State import NovelState
        state: NovelState = {
            "user_idea": "测试",
            "world_bible": "",
            "novel_tags": {},
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


class TestNovelTagLibrary(unittest.TestCase):
    @staticmethod
    def library():
        return {
            "schema_version": 1,
            "core_tags": ["规则怪谈", "都市情感"],
            "secondary_tags": {
                "情节": ["女性成长", "无限流"],
                "角色": ["大女主", "群像"],
                "情绪": ["惊悚", "治愈"],
                "背景": ["现代都市", "副本世界"],
            },
        }

    def test_loads_valid_library_without_cache(self):
        from LibraryV2 import load_novel_tag_library

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tags.json"
            path.write_text(
                json.dumps(self.library(), ensure_ascii=False),
                encoding="utf-8",
            )
            with patch("LibraryV2.NOVEL_TAG_LIBRARY_PATH", path):
                first = load_novel_tag_library()
                updated = self.library()
                updated["core_tags"].append("悬疑推理")
                path.write_text(
                    json.dumps(updated, ensure_ascii=False),
                    encoding="utf-8",
                )
                second = load_novel_tag_library()

        self.assertNotIn("悬疑推理", first["core_tags"])
        self.assertIn("悬疑推理", second["core_tags"])

    def test_rejects_missing_category_duplicate_empty_and_too_few_tags(self):
        from LibraryV2 import validate_novel_tag_library

        invalid_cases = {}

        missing = self.library()
        missing["secondary_tags"].pop("背景")
        invalid_cases["缺少分类"] = missing

        duplicate = self.library()
        duplicate["secondary_tags"]["角色"][0] = "女性成长"
        invalid_cases["重复词"] = duplicate

        empty = self.library()
        empty["core_tags"].append(" ")
        invalid_cases["空词"] = empty

        too_few = self.library()
        too_few["secondary_tags"] = {
            "情节": ["女性成长"],
            "角色": ["大女主"],
            "情绪": ["惊悚"],
            "背景": ["副本世界"],
        }
        invalid_cases["词数不足"] = too_few

        for name, library in invalid_cases.items():
            with self.subTest(name=name):
                self.assertTrue(validate_novel_tag_library(library))

    def test_validates_selection_membership_category_uniqueness_and_count(self):
        from LibraryV2 import validate_novel_tag_selection

        library = self.library()
        valid = {
            "core": "规则怪谈",
            "情节": ["女性成长", "无限流"],
            "角色": ["大女主"],
            "情绪": ["惊悚"],
            "背景": ["副本世界"],
        }
        self.assertEqual(validate_novel_tag_selection(valid, library), [])

        outside = {**valid, "core": "自由生成词"}
        self.assertTrue(validate_novel_tag_selection(outside, library))

        wrong_category = {**valid, "角色": ["女性成长"]}
        self.assertTrue(validate_novel_tag_selection(wrong_category, library))

        duplicate = {**valid, "角色": ["大女主", "大女主"]}
        self.assertTrue(validate_novel_tag_selection(duplicate, library))

        too_few = {**valid, "情节": ["女性成长"]}
        too_few["背景"] = []
        self.assertTrue(validate_novel_tag_selection(too_few, library))

        too_many = {
            "core": "规则怪谈",
            "情节": ["女性成长", "无限流"],
            "角色": ["大女主", "群像"],
            "情绪": ["惊悚", "治愈"],
            "背景": ["现代都市", "副本世界"],
        }
        self.assertTrue(validate_novel_tag_selection(too_many, library))


class TestArchitectNovelTags(unittest.TestCase):
    def test_tags_are_selected_only_after_outline_acceptance(self):
        from Nodes import ArchitectOutput, architect_node

        library = TestNovelTagLibrary.library()
        candidate = ArchitectOutput(
            novel_title="病患之眼",
            world_bible="完整世界观",
            chapter_outlines={"1": "【开场状态】剧情"},
            estimated_words=1500,
        )
        valid_tags = {
            "core": "规则怪谈",
            "情节": ["女性成长", "无限流"],
            "角色": ["大女主"],
            "情绪": ["惊悚"],
            "背景": ["副本世界"],
        }
        pattern_config = {
            "primary": "none",
            "secondary": [],
            "custom_instruction": "",
            "manifest": {},
            "structure_plan": {},
        }
        dump = MagicMock()

        with patch("Nodes.load_novel_tag_library", return_value=library), patch(
            "Nodes.normalize_pattern_config", return_value=pattern_config
        ), patch("Nodes.normalize_material_config", return_value={}), patch(
            "Nodes.validate_pattern_config", return_value=[]
        ), patch("Nodes.validate_material_config", return_value=[]), patch(
            "Nodes.resolve_story_pattern",
            return_value={"strong": False, "name": "无套路", "architect": ""},
        ), patch("Nodes.format_selected_materials", return_value="无"), patch(
            "Nodes.invoke_architect_with_fallback", return_value=candidate
        ) as invoke, patch(
            "Nodes.select_novel_tags", return_value=valid_tags
        ) as select_tags, patch(
            "Nodes.normalize_chapter_outlines",
            side_effect=lambda outlines, chapters: outlines,
        ), patch(
            "Nodes.normalize_outline_structures",
            side_effect=lambda outlines, chapters: outlines,
        ), patch(
            "Nodes.outline_validation_issues",
            side_effect=lambda outlines, chapters: [],
        ), patch(
            "Nodes.finale_outline_validation_issues", return_value=[]
        ), patch(
            "Nodes.build_chapter_contracts", return_value={"1": {}}
        ), patch("Nodes.build_finale_contract", return_value={}), patch(
            "Nodes.load_prompt", return_value=""
        ), patch("Nodes.os.makedirs"), patch(
            "builtins.open", mock_open()
        ), patch("Nodes.json.dump", dump):
            result = architect_node({
                "user_idea": "规则怪谈中的女医生成长故事",
                "target_chapters": 1,
                "words_per_chapter": 1500,
                "writer_style": "default",
                "pattern_config": {},
                "material_config": {},
            })

        self.assertEqual(invoke.call_count, 1)
        self.assertNotIn("novel_tag_library", invoke.call_args.args[1])
        select_tags.assert_called_once()
        self.assertEqual(result["novel_tags"], valid_tags)
        saved_payload = dump.call_args.args[0]
        self.assertEqual(saved_payload["schema_version"], 2)
        self.assertEqual(saved_payload["novel_tags"], valid_tags)

    def test_old_schema_v2_outline_without_tags_still_loads_and_lists(self):
        from Nodes import list_outline_files, load_outline_json

        old_outline = {
            "schema_version": 2,
            "title": "旧大纲",
            "world_bible": "设定",
            "chapter_outlines": {"1": "细纲"},
            "created_at": "2026-01-01T00:00:00",
        }
        reader = mock_open(read_data=json.dumps(old_outline, ensure_ascii=False))
        with patch("Nodes.os.makedirs"), patch(
            "Nodes.os.listdir", return_value=["old.json"]
        ), patch("builtins.open", reader):
            listed = list_outline_files()
            loaded = load_outline_json("old.json")

        self.assertEqual(listed[0]["core_tag"], "")
        self.assertNotIn("novel_tags", loaded)


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

    def test_preserves_complete_body_instead_of_hard_truncating(self):
        from Nodes import normalize_chapter_output

        body = "第一句。" + ("内容" * 100) + "最后一句。"
        result = normalize_chapter_output(f"第1章 测试\n\n{body}", 1, 50)
        normalized_body = result.split("\n\n", 1)[1]

        self.assertEqual(normalized_body, body)
        self.assertTrue(normalized_body.endswith("最后一句。"))

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
        self.assertTrue(any("缺少规范标签【开场状态】" in issue for issue in short_issues))
        self.assertTrue(outline_validation_issues(None, 1))
        self.assertTrue(outline_validation_issues({}, 0))

        detailed = (
            "【开场状态】主角被困塔顶，城市核心失控，外部救援被风暴隔绝。"
            "【核心冲突】主角必须在核心爆炸前关闭装置，同时证明自己没有背叛同伴。"
            "【关键行动】主角取得旧钥匙，破解三道机关，关闭失控核心，救出受伤同伴。"
            "【人物关系变化】两人从互相怀疑转为重新信任，并确认继续并肩调查。"
            "【重要信息或伏笔】旧钥匙其实是父亲留下的，钥匙背面刻着下个地点。"
            "【全书结局】城市恢复正常，主角公开真相，反派被审判，父亲遗留的秘密完成闭环。"
            "具体剧情推进与人物行动。" * 8
        )
        self.assertEqual(outline_validation_issues({"1": detailed}, 1), [])

    def test_normalizes_outline_labels_to_brackets(self):
        from Nodes import normalize_outline_structure

        outline = (
            "开场：主角被困塔顶。核心冲突：必须关闭失控核心。"
            "关键行动：主角取得钥匙；主角关闭核心；主角救出同伴。"
            "人物关系变化：两人解除误会。重要伏笔：旧钥匙是父亲留下的。"
            "结尾：城市恢复正常，主角回家。钩子：黑匣子传来新的求救信号。"
        )

        normalized = normalize_outline_structure(outline)

        self.assertIn("【开场状态】主角被困塔顶", normalized)
        self.assertIn("【重要信息或伏笔】旧钥匙是父亲留下的", normalized)
        self.assertIn("【结尾结果】城市恢复正常，主角回家", normalized)
        self.assertIn("【下一章钩子】黑匣子传来新的求救信号", normalized)
        self.assertNotIn("开场：", normalized)

    def test_normalizes_title_prefixed_colon_outline_labels(self):
        from Nodes import normalize_outline_structure, outline_validation_issues

        filler = "具体剧情推进与人物行动都清楚落地，冲突推进没有摘要跳过。"
        outline = (
            "第一章 母女裂痕 开场状态：林秀兰在出租屋整理账本，发现大学学费和房租已经压到同一天，女儿却在门口等她给毕业旅行转账。"
            "核心冲突：女儿坚持把手机、电脑、手表和旅行称作正常需求，林秀兰必须在沉默忍让和守住学费底线之间做选择。"
            "关键行动：林秀兰拿出存折解释家里只剩八千元；女儿偷拍视频发到网上；邻居王婶看到热搜后决定保留工厂打卡和旧账本证据。"
            "人物关系变化：母女从隐忍依赖转为公开撕裂，林秀兰第一次没有追出去哄女儿，女儿也第一次把母亲当成可以攻击的对象。"
            "重要信息或伏笔：旧账本里夹着女儿小时候写给母亲的作文，后续会成为舆论反转时最刺痛她的证据。"
            "全书结局：真相曝光后女儿终于理解母亲多年牺牲，林秀兰没有立刻原谅，而是把学费托人转交，搬进工厂宿舍重新开始。"
            + filler * 8
        )

        normalized = normalize_outline_structure(outline, is_final=True)

        self.assertTrue(normalized.startswith("【开场状态】"))
        self.assertIn("【核心冲突】", normalized)
        self.assertIn("【全书结局】", normalized)
        self.assertEqual(outline_validation_issues({"1": normalized}, 1), [])

    def test_normalizes_nested_outline_dict_values(self):
        from Nodes import (
            ArchitectOutput,
            normalize_chapter_outlines,
            normalize_outline_structures,
            outline_validation_issues,
        )

        filler = "具体剧情推进与人物行动都清楚落地，冲突推进没有摘要跳过。"
        raw_outline = {
            "开场": "林秀兰在工厂宿舍外接到女儿电话，手里还攥着刚退掉的理疗药单。",
            "core_conflict": "女儿要求母亲公开道歉，林秀兰必须决定是否继续替女儿遮掩家庭真实经济状况。",
            "required_events": [
                "林秀兰整理存折和旧账本",
                "王婶把工厂打卡记录交给她",
                "女儿在直播里继续控诉母亲",
            ],
            "relationship_change": "林秀兰从一味补偿转为设立边界，女儿从理直气壮转为第一次感到心虚。",
            "重要信息/伏笔": "旧作文和退药单会在最终舆论反转时同时出现。",
            "final_ending": "真相公布后女儿向母亲道歉，林秀兰仍选择把学费留下但不再共同生活。" + filler * 8,
        }

        normalized = normalize_chapter_outlines({"1": raw_outline}, 1)
        normalized = normalize_outline_structures(normalized, 1)
        parsed = ArchitectOutput.model_validate({
            "novel_title": "高考后的账本",
            "world_bible": "现实家庭伦理故事，围绕单亲母亲、女儿、邻居和同城舆论展开。",
            "chapter_outlines": {"1": raw_outline},
            "novel_tags": {
                "core": "都市情感",
                "情节": ["女性成长"],
                "角色": ["大女主", "成长型主角"],
                "情绪": ["治愈"],
                "背景": ["现代都市"],
            },
            "estimated_words": 2500,
        })

        self.assertIn("【开场状态】", normalized["1"])
        self.assertIn("【关键行动】", normalized["1"])
        self.assertIn("【全书结局】", normalized["1"])
        self.assertIn("【开场状态】", parsed.chapter_outlines["1"])
        self.assertEqual(outline_validation_issues(normalized, 1), [])

    def test_normalizes_slash_foreshadowing_and_inline_hook(self):
        from Nodes import normalize_outline_structure

        outline = (
            "【开场状态】主角被困塔顶。【核心冲突】必须关闭失控核心。"
            "【关键行动】主角取得钥匙；主角关闭核心；主角救出同伴。"
            "【人物关系变化】两人解除误会。【重要信息/伏笔】旧钥匙是父亲留下的。"
            "【结尾结果】城市恢复正常。下章钩子：黑匣子传来新的求救信号。"
        )

        normalized = normalize_outline_structure(outline)

        self.assertIn("【重要信息或伏笔】旧钥匙是父亲留下的", normalized)
        self.assertIn("【结尾结果】城市恢复正常", normalized)
        self.assertIn("【下一章钩子】黑匣子传来新的求救信号", normalized)
        self.assertNotIn("【重要信息/伏笔】", normalized)
        self.assertNotIn("下章钩子：", normalized)

    def test_normalizes_final_outline_ending_label(self):
        from Nodes import normalize_outline_structure, build_chapter_contracts

        outline = (
            "【开场状态】主角被困塔顶。【核心冲突】必须关闭失控核心。"
            "【关键行动】主角取得钥匙；主角关闭核心；主角救出同伴。"
            "【人物关系变化】两人解除误会。【重要信息或伏笔】旧钥匙是父亲留下的。"
            "【结尾结果】城市恢复正常，主角回家。"
        )

        normalized = normalize_outline_structure(outline, is_final=True)
        contracts = build_chapter_contracts({"1": normalized})

        self.assertIn("【全书结局】城市恢复正常", normalized)
        self.assertNotIn("【结尾结果】", normalized)
        self.assertIn("城市恢复正常", contracts["1"]["ending_state"])

    def test_builds_structured_contract_from_labeled_outline(self):
        from Nodes import build_chapter_contracts, build_finale_contract

        outline = (
            "【开场状态】主角被困塔顶。【核心冲突】必须关闭失控核心。"
            "【关键行动】主角取得钥匙；主角关闭核心；主角救出同伴。"
            "【人物关系变化】两人解除误会。【重要信息或伏笔】旧钥匙是父亲留下的。"
            "【全书结局】城市恢复正常，主角回家。"
        )
        contracts = build_chapter_contracts({"1": outline})
        finale = build_finale_contract(contracts)

        self.assertGreaterEqual(len(contracts["1"]["required_events"]), 3)
        self.assertTrue(
            any("关闭核心" in event for event in contracts["1"]["required_events"])
        )
        self.assertLess(
            max(map(len, contracts["1"]["required_events"])),
            len(outline),
        )
        self.assertTrue(contracts["1"]["is_final"])
        self.assertIn("城市恢复正常", finale["required_resolution"])

    def test_finale_length_uses_soft_range_and_hard_guardrail(self):
        from Nodes import chapter_length_assessment, chapter_length_limits

        limits = chapter_length_limits(1500, final_chapter=True)
        self.assertEqual(limits["recommended_min"], 1200)
        self.assertEqual(limits["recommended_max"], 2100)
        self.assertEqual(limits["hard_min"], 900)
        self.assertEqual(limits["hard_max"], 2400)

        state = {
            "current_chapter": 1,
            "chapter_outlines": {"1": "结局"},
            "words_per_chapter": 1500,
        }
        acceptable = chapter_length_assessment(
            state, "第1章 结局\n\n" + ("字" * 1000)
        )
        excessive = chapter_length_assessment(
            state, "第1章 结局\n\n" + ("字" * 2500)
        )
        self.assertFalse(acceptable["blocking"])
        self.assertTrue(excessive["blocking"])

    def test_story_ledger_context_is_capped_at_fifteen_hundred_chars(self):
        from Nodes import render_story_ledger

        ledger = {
            "immutable_facts": [
                {
                    "id": f"F-C{i}-01",
                    "fact_key": f"fact_{i}",
                    "chapter": i,
                    "category": "history",
                    "subject": f"角色{i}",
                    "statement": f"角色{i}在医院楼梯确认了关键事实。" + ("细节" * 30),
                    "source_evidence": "正文",
                    "keywords": ["医院", "楼梯"],
                }
                for i in range(1, 30)
            ],
            "current_states": {
                "location:主角": {
                    "state_key": "location:主角",
                    "chapter": 29,
                    "category": "location",
                    "subject": "主角",
                    "value": "医院",
                    "source_evidence": "正文",
                }
            },
            "last_chapter_ending": "主角站在医院门口",
            "next_handoff": "调查医院楼梯事故",
        }
        context = render_story_ledger(
            ledger,
            {"core_conflict": "医院楼梯事故"},
            {},
            max_chars=1500,
        )

        self.assertLessEqual(len(context), 1500)
        self.assertIn("医院楼梯", context)
        self.assertIn("当前状态", context)
        self.assertIn("章节交接", context)

    def test_mutable_state_updates_without_conflicting_with_history(self):
        from Nodes import merge_story_ledger

        merged = merge_story_ledger(
            {
                "current_states": {
                    "location:沈念": {
                        "state_key": "location:沈念",
                        "chapter": 1,
                        "category": "location",
                        "subject": "沈念",
                        "value": "北京",
                        "source_evidence": "第1章",
                    }
                }
            },
            {
                "state_updates": [{
                    "state_key": "location:沈念",
                    "category": "location",
                    "subject": "沈念",
                    "value": "丽江",
                    "source_evidence": "沈念抵达丽江",
                }]
            },
            5,
            {"status": "pass", "conflicts": []},
        )

        self.assertEqual(
            merged["current_states"]["location:沈念"]["value"],
            "丽江",
        )

    def test_financial_fact_cannot_be_rewritten_as_private_trust(self):
        from Nodes import LedgerMergeError, merge_story_ledger

        ledger = {
            "immutable_facts": [{
                "id": "F-C2-01",
                "fact_key": "company_500k_transaction",
                "chapter": 2,
                "category": "legal_financial",
                "subject": "陆氏公司50万异常支出",
                "statement": "公司账目出现50万异常支出，沈念被伪造授权签名",
                "source_evidence": "第2章会议",
                "keywords": ["公司", "50万", "授权签名"],
            }]
        }
        delta = {
            "new_immutable_facts": [{
                "fact_key": "company_500k_transaction",
                "category": "legal_financial",
                "subject": "陆氏公司50万异常支出",
                "statement": "沈念贪污了私人遗产信托资金",
                "source_evidence": "后续回忆",
                "keywords": ["信托", "贪污"],
            }]
        }
        conflict = {
            "fact_id": "F-C2-01",
            "established_fact": ledger["immutable_facts"][0]["statement"],
            "draft_claim": delta["new_immutable_facts"][0]["statement"],
        }
        with self.assertRaisesRegex(LedgerMergeError, "连续性审核未通过"):
            merge_story_ledger(
                ledger,
                delta,
                8,
                {"status": "fail", "conflicts": [conflict]},
            )

    def test_same_character_can_have_multiple_distinct_history_facts(self):
        from Nodes import merge_story_ledger

        merged = merge_story_ledger(
            {},
            {
                "new_immutable_facts": [
                    {
                        "fact_key": "rebirth_event",
                        "category": "identity",
                        "subject": "沈月瑶",
                        "statement": "沈月瑶在七月十四满月之夜重生回十八岁生日前三天的夜晚。",
                        "source_evidence": "第1章开场",
                        "keywords": ["沈月瑶", "重生"],
                    },
                    {
                        "fact_key": "past_life_sacrifice",
                        "category": "identity",
                        "subject": "沈月瑶",
                        "statement": "前世沈月瑶被献祭给噬梦井，灵魂在井中煎熬十年。",
                        "source_evidence": "第1章回忆",
                        "keywords": ["沈月瑶", "献祭", "噬梦井"],
                    },
                ]
            },
            1,
            {"status": "pass", "conflicts": []},
        )

        self.assertEqual(len(merged["immutable_facts"]), 2)
        self.assertEqual(
            {item["fact_key"] for item in merged["immutable_facts"]},
            {"rebirth_event", "past_life_sacrifice"},
        )


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
            with self.assertRaisesRegex(RuntimeError, "已尝试3次"):
                invoke_with_retry(chain, {}, "写手", max_attempts=3)

    def test_invalid_json_mode_request_is_not_retried(self):
        from Nodes import invoke_with_retry

        chain = MagicMock()
        chain.invoke.side_effect = RuntimeError(
            "invalid_request_error: Prompt must contain the word 'json' "
            "to use response_format"
        )
        with patch("Nodes.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "已尝试1次"):
                invoke_with_retry(chain, {}, "连续性台账", max_attempts=3)

        self.assertEqual(chain.invoke.call_count, 1)
        sleep.assert_not_called()

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
            "pattern_config": {
                "schema_version": 2,
                "primary": "custom",
                "secondary": [],
                "custom_instruction": "每三章揭示一次假规则",
                "manifest": {},
                "structure_plan": {},
            },
        })

        self.assertIn("每三章揭示一次假规则", pattern["architect"])
        self.assertIn("每三章揭示一次假规则", pattern["writer"])
        self.assertIn("每三章揭示一次假规则", pattern["auditor"])

    def test_unknown_pattern_falls_back_to_none(self):
        from Nodes import resolve_story_pattern

        pattern = resolve_story_pattern({
            "pattern_config": {
                "schema_version": 2,
                "primary": "missing",
                "secondary": [],
            },
        })

        self.assertEqual(pattern["key"], "none")
        self.assertEqual(pattern["name"], "无固定主套路")

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

    def test_deterministic_ai_trace_flags_repeated_sentence_templates(self):
        from Nodes import apply_deterministic_ai_trace_checks

        draft = "第1章 测试\n\n他知道这不是终点而是开始。\n她知道这不是结束而是选择。\n我知道这不是退路而是代价。"
        report = apply_deterministic_ai_trace_checks({"文风评分": 8}, draft)

        self.assertLess(report["文风评分"], 7)
        self.assertTrue(any("不是……而是" in issue for issue in report["AI痕迹问题"]))

    def test_deterministic_ai_trace_warns_single_stock_expression(self):
        from Nodes import apply_deterministic_ai_trace_checks

        draft = "第1章 测试\n\n空气仿佛凝固。她推开门，递出证据。"
        report = apply_deterministic_ai_trace_checks({"文风评分": 8}, draft)

        self.assertEqual(report["文风评分"], 8)
        self.assertFalse(report["AI痕迹问题"])
        self.assertTrue(any("空气仿佛凝固" in warning for warning in report["AI痕迹警告"]))

    def test_deterministic_ai_trace_flags_action_crutch_density(self):
        from Nodes import apply_deterministic_ai_trace_checks

        body = "\n".join(["他沉默片刻，深吸一口气，握紧那张纸。" for _ in range(5)])
        report = apply_deterministic_ai_trace_checks({"文风评分": 8}, "第1章 测试\n\n" + body)

        self.assertLess(report["文风评分"], 7)
        self.assertTrue(any("动作拐杖" in issue for issue in report["AI痕迹问题"]))

    def test_default_novel_length_is_six_by_twenty_five_hundred(self):
        from Nodes import DEFAULT_CHAPTERS, DEFAULT_WORDS_PER_CHAPTER

        self.assertEqual(DEFAULT_CHAPTERS, 6)
        self.assertEqual(DEFAULT_WORDS_PER_CHAPTER, 2500)

    def test_auditor_receives_continuity_and_next_outline(self):
        from Nodes import _audit_inputs

        inputs = _audit_inputs({
            "current_chapter": 1,
            "world_bible": "规则不可违背",
            "chapter_outlines": {"1": "进入车站", "2": "离开车站"},
            "story_ledger": {
                "immutable_facts": [{
                    "id": "F-C0-01",
                    "fact_key": "station_entry",
                    "chapter": 0,
                    "category": "history",
                    "subject": "主角",
                    "statement": "主角已经进入车站",
                    "source_evidence": "前情",
                    "keywords": ["主角", "车站"],
                }],
            },
            "current_draft": "正文",
            "pattern_config": {
                "schema_version": 2,
                "primary": "rule_horror",
                "secondary": [],
                "manifest": {},
                "structure_plan": {},
            },
        })

        self.assertIn("F-C0-01", inputs["continuity_state"])
        self.assertIn("主角已经进入车站", inputs["continuity_state"])
        self.assertEqual(inputs["next_outline"], "离开车站")
        self.assertIn("规则", inputs["pattern_auditor"])

    def test_reviewer_turns_continuity_conflict_into_blocking_issue(self):
        from Nodes import reviewer_node

        state = {
            "current_chapter": 6,
            "current_draft": "第6章 回忆\n\n事故发生在祠堂。",
            "chapter_outlines": {str(i): "剧情" for i in range(1, 7)},
            "words_per_chapter": 20,
            "iteration_count": 1,
            "draft_candidates": [],
        }
        audit_result = {
            "audit_report": {
                "审核状态": "通过",
                "发现的问题": [],
                "警告": [],
                "套路问题": [],
                "阻断问题": [],
                "未完成事件": [],
                "结局问题": [],
                "大纲完成度": 90,
                "连续性评分": 90,
                "衔接评分": 90,
                "结局完整性": True,
                "修改建议": "无",
            }
        }
        continuity_result = {
            "ledger_delta": {},
            "continuity_report": {
                "status": "fail",
                "conflicts": [{
                    "fact_id": "F-C3-02",
                    "established_fact": "流产发生在医院楼梯",
                    "draft_claim": "流产发生在祠堂",
                    "draft_evidence": "事故发生在祠堂",
                    "repair_instruction": "恢复医院楼梯经过",
                }],
                "warnings": [],
            },
        }
        with (
            patch("Nodes._auditor_internal", return_value=audit_result),
            patch(
                "Nodes._editor_internal",
                return_value={
                    "editor_report": {
                        "文风评分": 8,
                        "AI痕迹问题": [],
                        "改进建议": "无",
                    }
                },
            ),
            patch("Nodes._continuity_internal", return_value=continuity_result),
        ):
            result = reviewer_node(state)

        self.assertEqual(result["audit_report"]["审核状态"], "不通过")
        self.assertEqual(result["audit_report"]["连续性评分"], 0)
        self.assertTrue(
            any(
                "F-C3-02" in issue
                for issue in result["audit_report"]["阻断问题"]
            )
        )
        self.assertEqual(
            result["draft_candidates"][0]["continuity_report"]["status"],
            "fail",
        )

    def test_continuity_accepts_deepseek_flat_payload(self):
        from Nodes import (
            ContinuityReview,
            normalize_continuity_report,
            normalize_ledger_delta,
        )

        payload = {
            "new_immutable_facts": [{
                "fact_key": "rebirth_event",
                "description": "井晴重生回到十八岁生日前三天。",
                "source": "井晴猛地睁开眼，前世记忆涌来。",
            }],
            "state_updates": [{
                "entity": "井晴",
                "location": "井家老宅自己房间",
                "possessions": ["古井玉佩（藏于床底夹层）"],
                "relationships": {"井玄冥": "恭敬但内心警惕"},
                "physical_state": "夜晚惊醒，手心冒汗",
                "awareness": "记得前世所有经历",
            }],
            "new_foreshadowing": [{
                "id": "foreshadow_001",
                "type": "诅咒转移",
                "description": "井月也梦见古井，诅咒可能已经转移。",
                "source": "井月提到梦境",
            }],
            "resolved_foreshadowing_ids": [],
            "chapter_ending": {
                "time": "深夜",
                "scene": "井晴惊醒，发现首饰盒盖打开",
            },
            "next_handoff": {
                "required_state": "第二天井月描述更真实的梦境",
            },
            "conflicts": [],
        }
        parsed = ContinuityReview.model_validate(payload).model_dump()
        delta = normalize_ledger_delta(parsed)
        report = normalize_continuity_report(parsed)

        self.assertEqual(
            delta["new_immutable_facts"][0]["statement"],
            "井晴重生回到十八岁生日前三天。",
        )
        self.assertEqual(
            delta["new_immutable_facts"][0]["source_evidence"],
            "井晴猛地睁开眼，前世记忆涌来。",
        )
        self.assertTrue(
            any(
                item["state_key"].startswith("location:井晴")
                for item in delta["state_updates"]
            )
        )
        self.assertTrue(
            any(
                item["state_key"] == "relationship:井晴-井玄冥"
                for item in delta["state_updates"]
            )
        )
        self.assertIn("深夜", delta["chapter_ending"])
        self.assertEqual(report["status"], "pass")

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
            "writer_system_suspense.md",
            "writer_system_emotional_tension.md",
            "writer_system_sweet_romcom.md",
            "writer_system_ancient_elegant.md",
            "writer_system_realist_ensemble.md",
            "writer_system_business.md",
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
            [
                "default",
                "literary",
                "cold",
                "emotional_tension",
                "realist_ensemble",
                "ancient_elegant",
            ],
        )
        pattern = resolve_story_pattern({
            "pattern_config": {
                "schema_version": 2,
                "primary": "female_angst_awakening",
                "secondary": [],
                "manifest": {},
                "structure_plan": {},
            },
        })
        self.assertIn("强制写作技巧", pattern["writer"])
        self.assertIn("套路审核规则", pattern["auditor"])

    def test_strong_pattern_allows_world_stages_but_blocks_conflicting_drivers(self):
        from LibraryV2 import (
            load_material_library,
            material_pattern_conflict_reason,
        )

        config = {
            "schema_version": 2,
            "primary": "female_angst_awakening",
            "secondary": [],
        }
        entries = load_material_library()["entries"]
        world_item = next(
            item for item in entries if item["category"] == "world_stage"
        )
        cheat_item = next(
            item for item in entries if item["category"] == "cheat_device"
        )
        self.assertEqual(
            material_pattern_conflict_reason(world_item, config), ""
        )
        self.assertIn(
            "禁止素材大类",
            material_pattern_conflict_reason(cheat_item, config),
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

    def test_new_strong_patterns_generate_valid_manifests_and_plans(self):
        from Nodes import (
            build_pattern_plan,
            load_story_patterns,
            roll_pattern_manifest,
            validate_pattern_manifest,
        )

        strong_keys = [
            key
            for key, item in load_story_patterns().items()
            if item.get("strong")
        ]
        self.assertEqual(len(strong_keys), 13)
        for key in strong_keys:
            with self.subTest(key=key):
                manifest = roll_pattern_manifest(key, seed=42)
                self.assertEqual(validate_pattern_manifest(manifest), [])
                plan = build_pattern_plan(manifest, 6, 1500)
                self.assertEqual(len(plan), 6)
                self.assertEqual(len([task for task in plan.values() if task["is_paywall_turn"]]), 1)
                self.assertTrue(all(task["protagonist_state"] for task in plan.values()))


class TestContentLibrariesV2(unittest.TestCase):
    def test_library_sizes_and_strong_pattern_schema(self):
        from LibraryV2 import (
            load_material_library,
            load_pattern_library,
            validate_material_library,
            validate_pattern_library,
        )

        materials = load_material_library()
        patterns = load_pattern_library()
        self.assertEqual(validate_material_library(materials), [])
        self.assertEqual(validate_pattern_library(patterns), [])
        self.assertGreaterEqual(len(materials["entries"]), 500)
        self.assertGreaterEqual(
            sum(
                not item.get("strong") and key not in {"none", "custom"}
                for key, item in patterns["patterns"].items()
            ),
            60,
        )
        self.assertEqual(
            sum(item.get("strong") for item in patterns["patterns"].values()),
            13,
        )

    def test_group_quota_sampling_and_single_item_reroll(self):
        from LibraryV2 import (
            default_material_config,
            default_pattern_config,
            resample_material_item,
            sample_materials,
        )

        config = sample_materials(
            default_material_config(),
            default_pattern_config(),
            seed=11,
        )
        self.assertEqual(
            [item["category"] for item in config["items"]],
            ["world_stage", "protagonist", "cheat_device", "core_conflict"],
        )
        before = {
            item["selection_key"]: item["id"] for item in config["items"]
        }
        rerolled = resample_material_item(
            config,
            default_pattern_config(),
            "core_conflict:1",
            seed=12,
        )
        after = {
            item["selection_key"]: item["id"] for item in rerolled["items"]
        }
        self.assertEqual(before["world_stage:1"], after["world_stage:1"])
        self.assertEqual(before["protagonist:1"], after["protagonist:1"])
        self.assertEqual(before["cheat_device:1"], after["cheat_device:1"])
        self.assertNotEqual(
            before["core_conflict:1"], after["core_conflict:1"]
        )

    def test_zero_material_quota_is_valid_without_changing_default(self):
        from LibraryV2 import (
            DEFAULT_MATERIAL_COUNT,
            default_material_config,
            default_pattern_config,
            material_library_metadata,
            sample_materials,
            validate_material_config,
        )

        self.assertEqual(DEFAULT_MATERIAL_COUNT, 4)
        self.assertEqual(material_library_metadata()["count_range"], [0, 8])
        config = default_material_config()
        config["group_counts"] = {
            group_id: 0 for group_id in config["group_counts"]
        }
        sampled = sample_materials(
            config,
            default_pattern_config(),
            seed=11,
        )
        self.assertEqual(sampled["count"], 0)
        self.assertEqual(sampled["items"], [])
        self.assertEqual(
            validate_material_config(sampled, default_pattern_config()),
            [],
        )

    def test_same_group_items_have_independent_selection_keys(self):
        from LibraryV2 import (
            default_material_config,
            default_pattern_config,
            resample_material_item,
            sample_materials,
        )

        raw = default_material_config()
        raw["group_counts"]["cheat_device"] = 2
        config = sample_materials(raw, default_pattern_config(), seed=21)
        cheats = [
            item for item in config["items"]
            if item["category"] == "cheat_device"
        ]
        self.assertEqual(
            [item["selection_key"] for item in cheats],
            ["cheat_device:1", "cheat_device:2"],
        )
        before = {item["selection_key"]: item for item in config["items"]}
        rerolled = resample_material_item(
            config,
            default_pattern_config(),
            "cheat_device:2",
            seed=22,
        )
        after = {item["selection_key"]: item for item in rerolled["items"]}
        self.assertEqual(
            before["cheat_device:1"]["id"],
            after["cheat_device:1"]["id"],
        )
        self.assertNotEqual(
            before["cheat_device:2"]["id"],
            after["cheat_device:2"]["id"],
        )

    def test_group_limits_reject_double_world_but_allow_double_cheat(self):
        from LibraryV2 import (
            default_material_config,
            default_pattern_config,
            sample_materials,
            validate_material_config,
        )

        invalid = default_material_config()
        invalid["group_counts"]["world_stage"] = 2
        invalid["group_counts"]["protagonist"] = 0
        issues = validate_material_config(
            invalid,
            default_pattern_config(),
            require_items=False,
        )
        self.assertTrue(any("世界舞台最多选择1项" in issue for issue in issues))

        valid = default_material_config()
        valid["group_counts"]["cheat_device"] = 2
        sampled = sample_materials(
            valid,
            default_pattern_config(),
            seed=23,
        )
        self.assertEqual(
            sum(
                item["category"] == "cheat_device"
                for item in sampled["items"]
            ),
            2,
        )

    def test_writer_style_registry_is_shared_and_has_twelve_styles(self):
        from WriterStyles import WRITER_STYLES

        self.assertEqual(len(WRITER_STYLES), 12)
        self.assertEqual(len({item["key"] for item in WRITER_STYLES}), 12)
        for item in WRITER_STYLES:
            prompt = Path("Role", item["prompt_file"])
            self.assertTrue(prompt.exists(), item["key"])
            self.assertTrue(item["editor_focus"], item["key"])

    def test_primary_secondary_and_material_hard_conflicts(self):
        from LibraryV2 import (
            load_material_library,
            material_pattern_conflict_reason,
            validate_pattern_config,
        )

        issues = validate_pattern_config({
            "schema_version": 2,
            "primary": "strong_rule_horror",
            "secondary": ["marriage_first"],
        })
        self.assertTrue(any("硬冲突" in issue for issue in issues))

        cheat = next(
            item
            for item in load_material_library()["entries"]
            if item["category"] == "cheat_device"
        )
        reason = material_pattern_conflict_reason(
            cheat,
            {
                "schema_version": 2,
                "primary": "female_angst_awakening",
                "secondary": [],
            },
        )
        self.assertIn("禁止素材大类", reason)

    def test_outline_migration_preview_and_apply(self):
        from migrate_outlines import migrate_directory

        with tempfile.TemporaryDirectory() as directory:
            outline_dir = Path(directory) / "Outline"
            outline_dir.mkdir()
            old_path = outline_dir / "old.json"
            old_path.write_text(json.dumps({
                "title": "旧大纲",
                "world_bible": "设定",
                "chapter_outlines": {"1": "细纲"},
                "story_pattern": "rule_horror",
                "custom_pattern": "",
                "pattern_manifest": {},
                "pattern_plan": {},
                "keywords": ["旧关键词"],
            }, ensure_ascii=False), encoding="utf-8")

            preview = migrate_directory(outline_dir, apply=False)
            self.assertEqual(len(preview["converted"]), 1)
            self.assertNotIn("schema_version", json.loads(old_path.read_text(encoding="utf-8")))

            applied = migrate_directory(outline_dir, apply=True)
            self.assertEqual(applied["failed"], [])
            migrated = json.loads(old_path.read_text(encoding="utf-8"))
            self.assertEqual(migrated["schema_version"], 2)
            self.assertEqual(migrated["pattern_config"]["primary"], "rule_horror")
            self.assertEqual(
                migrated["material_config"]["legacy_import"],
                ["旧关键词"],
            )
            self.assertTrue(Path(applied["backup_directory"]).exists())

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
    def test_default_estimate_is_twelve_thousand(self):
        from web_app import HTML_PAGE

        self.assertIn('id="estWords">预估: 约 12,000 字', HTML_PAGE)
        self.assertIn("updateEstimate();", HTML_PAGE)

    def test_web_has_pattern_controls(self):
        from web_app import HTML_PAGE

        self.assertIn('id="storyPattern"', HTML_PAGE)
        self.assertIn('id="washStoryPattern"', HTML_PAGE)
        self.assertIn('value="rule_horror"', HTML_PAGE)
        self.assertIn("get_patterns", HTML_PAGE)
        self.assertIn("female_angst_awakening", HTML_PAGE)
        self.assertIn("strong_rule_horror", HTML_PAGE)
        self.assertIn("male_angst_awakening", HTML_PAGE)
        self.assertIn("roll_pattern_manifest", HTML_PAGE)

    def test_web_has_grouped_material_quota_and_reroll_controls(self):
        from web_app import HTML_PAGE

        self.assertIn("material-groups", HTML_PAGE)
        self.assertIn("changeGroupCount", HTML_PAGE)
        self.assertIn("locked_item_keys", HTML_PAGE)
        self.assertIn("randomize_types", HTML_PAGE)
        self.assertIn("换类型重抽", HTML_PAGE)
        self.assertIn("命中", HTML_PAGE)
        self.assertIn('id="patternEnding"', HTML_PAGE)
        self.assertIn('id="washPatternEnding"', HTML_PAGE)
        self.assertIn("请先确认强套路契约", HTML_PAGE)
        self.assertIn("结构化素材库", HTML_PAGE)
        self.assertIn("renderSecondaryPatterns", HTML_PAGE)
        self.assertIn("resample_material", HTML_PAGE)


    def test_writer_style_pattern_material_order_is_stable(self):
        from web_app import HTML_PAGE

        create_start = HTML_PAGE.index('<div id="createPanel">')
        wash_start = HTML_PAGE.index('<div id="washPanel"')
        create_html = HTML_PAGE[create_start:wash_start]
        wash_html = HTML_PAGE[wash_start:HTML_PAGE.index('  </div><!-- /panel -->')]

        self.assertLess(create_html.index('id="writerStyle"'), create_html.index('id="storyPattern"'))
        self.assertLess(create_html.index('id="storyPattern"'), create_html.index('id="kwSection"'))
        self.assertLess(wash_html.index('id="washWriterStyle"'), wash_html.index('id="washStoryPattern"'))
        self.assertLess(wash_html.index('id="washStoryPattern"'), wash_html.index('id="washMaterialCards"'))
        self.assertNotIn("select.value='default'", HTML_PAGE)

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

    def test_review_route_uses_four_drafts_before_selecting_best(self):
        from TheGraph import route_after_review

        state = {
            "audit_report": {"审核状态": "不通过"},
            "editor_report": {"文风评分": 6},
            "chapter_outlines": {"1": "剧情", "2": "结局"},
            "current_chapter": 1,
            "iteration_count": 1,
        }
        self.assertEqual(route_after_review(state), "writer")

        state["iteration_count"] = 2
        self.assertEqual(route_after_review(state), "writer")

        state["iteration_count"] = 3
        self.assertEqual(route_after_review(state), "writer")

        state["iteration_count"] = 4
        state["draft_candidates"] = [{
            "chapter": 1,
            "draft": "第1章 候选\n\n可继续使用。",
            "score": 80,
            "continuity_report": {"status": "pass", "conflicts": []},
        }]
        self.assertEqual(route_after_review(state), "summarizer")

    def test_all_conflicting_drafts_receive_two_continuity_repairs(self):
        from Nodes import route_after_review_decision

        state = {
            "audit_report": {"审核状态": "不通过"},
            "editor_report": {"文风评分": 7},
            "continuity_report": {
                "status": "fail",
                "conflicts": [{"fact_id": "F-C3-02"}],
            },
            "chapter_outlines": {"1": "当前章", "2": "后续"},
            "current_chapter": 1,
            "iteration_count": 4,
            "draft_candidates": [
                {
                    "chapter": 1,
                    "draft": f"冲突稿{i}",
                    "score": 90 - i,
                    "continuity_report": {
                        "status": "fail",
                        "conflicts": [{"fact_id": "F-C3-02"}],
                    },
                }
                for i in range(4)
            ],
        }
        self.assertEqual(route_after_review_decision(state), "writer")
        state["iteration_count"] = 5
        self.assertEqual(route_after_review_decision(state), "writer")
        state["iteration_count"] = 6
        with self.assertRaisesRegex(RuntimeError, "不可变事实冲突"):
            route_after_review_decision(state)

    def test_finale_gets_two_extra_repairs_and_requires_complete_candidate(self):
        from Nodes import route_after_review_decision

        state = {
            "audit_report": {
                "审核状态": "不通过",
                "阻断问题": ["结局未完成"],
                "结局完整性": False,
            },
            "editor_report": {"文风评分": 7},
            "chapter_outlines": {"1": "结局"},
            "current_chapter": 1,
            "iteration_count": 4,
            "draft_candidates": [],
        }
        self.assertEqual(route_after_review_decision(state), "writer")

        state["iteration_count"] = 6
        with self.assertRaisesRegex(RuntimeError, "最终章连续6稿"):
            route_after_review_decision(state)

        state["draft_candidates"] = [{
            "chapter": 1,
            "draft": "第1章 结局\n\n故事完整结束。",
            "score": 90,
            "audit_report": {
                "审核状态": "通过",
                "结局完整性": True,
                "阻断问题": [],
            },
            "editor_report": {"文风评分": 7},
        }]
        self.assertEqual(route_after_review_decision(state), "summarizer")


class TestSummarizerBehavior(unittest.TestCase):
    def test_failed_fourth_draft_saves_highest_scoring_candidate(self):
        from Nodes import summarizer_node

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "novel.txt")
            best = "第1章 最佳稿\n\n最佳剧情完整推进。"
            state = {
                "current_chapter": 1,
                "current_draft": "第1章 第四稿\n\n仍有问题。",
                "novel_title": "测试",
                "chapter_outlines": {"1": "开端", "2": "结局"},
                "story_summary": "",
                "continuity_state": "",
                "words_per_chapter": 10,
                "iteration_count": 4,
                "audit_report": {"审核状态": "不通过", "结局完整性": True},
                "editor_report": {"文风评分": 7},
                "continuity_report": {
                    "status": "fail",
                    "conflicts": [{"fact_id": "F-C0-01"}],
                },
                "draft_candidates": [
                    {
                        "chapter": 1,
                        "iteration": 2,
                        "draft": best,
                        "score": 92,
                        "audit_report": {"审核状态": "通过", "结局完整性": True},
                        "editor_report": {"文风评分": 8},
                        "ledger_delta": {
                            "new_immutable_facts": [{
                                "fact_key": "station_entry",
                                "category": "history",
                                "subject": "主角",
                                "statement": "主角进入车站",
                                "source_evidence": "主角推门进入车站",
                                "keywords": ["主角", "车站"],
                            }],
                            "state_updates": [],
                            "new_foreshadowing": [],
                            "resolved_foreshadowing_ids": [],
                            "chapter_ending": "主角站在车站大厅",
                            "next_handoff": "继续调查车站",
                        },
                        "continuity_report": {
                            "status": "pass",
                            "conflicts": [],
                            "warnings": [],
                        },
                    },
                    {
                        "chapter": 1,
                        "iteration": 4,
                        "draft": "第1章 第四稿\n\n仍有问题。",
                        "score": 60,
                        "audit_report": {"审核状态": "不通过", "结局完整性": True},
                        "editor_report": {"文风评分": 7},
                        "ledger_delta": {},
                        "continuity_report": {
                            "status": "fail",
                            "conflicts": [{"fact_id": "F-C0-01"}],
                        },
                    },
                ],
            }
            with patch("Nodes._build_output_path", return_value=output_path):
                result = summarizer_node(state)

            with open(output_path, encoding="utf-8") as saved_file:
                self.assertEqual(saved_file.read(), best)
            self.assertEqual(result["current_draft"], best)
            self.assertEqual(
                result["story_ledger"]["immutable_facts"][0]["statement"],
                "主角进入车站",
            )
            self.assertEqual(result["draft_candidates"], [])

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
                "ledger_delta": {
                    "new_immutable_facts": [{
                        "fact_key": "story_resolution",
                        "category": "irreversible_relationship",
                        "subject": "故事结局",
                        "statement": "主线冲突已经解决",
                        "source_evidence": "故事结束",
                        "keywords": ["结局"],
                    }],
                    "chapter_ending": "故事结束",
                    "next_handoff": "",
                },
                "continuity_report": {"status": "pass", "conflicts": []},
            }
            with patch("Nodes._build_output_path", return_value=output_path):
                with patch("Nodes.load_prompt") as load_prompt:
                    result = summarizer_node(state)

            load_prompt.assert_not_called()
            self.assertTrue(result["summary_skipped"])
            self.assertIn("主线冲突已经解决", result["story_summary"])
            self.assertEqual(result["audit_report"], {})
            self.assertEqual(result["editor_report"], {})

    def test_continuity_model_failure_stops_review(self):
        from Nodes import _continuity_internal

        with patch("Nodes.invoke_with_retry", side_effect=RuntimeError("连接失败")):
            result = _continuity_internal({
                "current_chapter": 1,
                "chapter_outlines": {"1": "开端"},
                "current_draft": "第1章 开端\n\n故事开始。",
            })
        self.assertEqual(result["continuity_report"]["status"], "pass")
        self.assertEqual(result["continuity_report"]["conflicts"], [])
        self.assertEqual(result["ledger_delta"], {})

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
                "ledger_delta": {
                    "new_immutable_facts": [],
                    "state_updates": [{
                        "state_key": "location:主角",
                        "category": "location",
                        "subject": "主角",
                        "value": "车站",
                        "source_evidence": "主角进入车站",
                    }],
                    "new_foreshadowing": [],
                    "resolved_foreshadowing_ids": [],
                    "chapter_ending": "主角站在车站",
                    "next_handoff": "离开车站",
                },
                "continuity_report": {"status": "pass", "conflicts": []},
            }
            with patch("Nodes._build_output_path", return_value=output_path):
                result = summarizer_node(state)

            self.assertEqual(
                result["story_ledger"]["current_states"]["location:主角"]["value"],
                "车站",
            )
            self.assertIn("主角location：车站", result["continuity_state"])

    def test_conflicting_immutable_fact_is_not_written(self):
        from Nodes import LedgerMergeError, summarizer_node

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "novel.txt")
            state = {
                "current_chapter": 6,
                "current_draft": "第6章 回忆\n\n事故发生在祠堂。",
                "novel_title": "测试",
                "chapter_outlines": {str(i): "剧情" for i in range(1, 7)},
                "words_per_chapter": 10,
                "audit_report": {"审核状态": "通过"},
                "editor_report": {"文风评分": 7},
                "story_ledger": {
                    "immutable_facts": [{
                        "id": "F-C3-02",
                        "fact_key": "pregnancy_loss_event",
                        "chapter": 3,
                        "category": "reproductive",
                        "subject": "沈念流产事故",
                        "statement": "沈念在医院楼梯摔倒，路人呼救后被送入急诊并流产",
                        "source_evidence": "第3章",
                        "keywords": ["沈念", "医院楼梯", "流产"],
                    }],
                },
                "ledger_delta": {
                    "new_immutable_facts": [{
                        "fact_key": "pregnancy_loss_event",
                        "category": "reproductive",
                        "subject": "沈念流产事故",
                        "statement": "沈念在祠堂摔倒且现场无人，自己拨打120",
                        "source_evidence": "第6章回忆",
                        "keywords": ["沈念", "祠堂", "流产"],
                    }],
                },
                "continuity_report": {
                    "status": "fail",
                    "conflicts": [{
                        "fact_id": "F-C3-02",
                        "established_fact": "沈念在医院楼梯摔倒后流产",
                        "draft_claim": "沈念在祠堂摔倒后流产",
                    }],
                },
            }
            with patch("Nodes._build_output_path", return_value=output_path):
                with self.assertRaisesRegex(LedgerMergeError, "没有可安全入库"):
                    summarizer_node(state)
            self.assertFalse(os.path.exists(output_path))

    def test_best_draft_excludes_high_scoring_continuity_conflict(self):
        from Nodes import select_best_draft

        selected = select_best_draft({
            "current_chapter": 2,
            "draft_candidates": [
                {
                    "chapter": 2,
                    "draft": "冲突稿",
                    "score": 99,
                    "continuity_report": {
                        "status": "fail",
                        "conflicts": [{"fact_id": "F-C1-01"}],
                    },
                },
                {
                    "chapter": 2,
                    "draft": "一致稿",
                    "score": 80,
                    "continuity_report": {"status": "pass", "conflicts": []},
                },
            ],
        })
        self.assertEqual(selected["draft"], "一致稿")


class TestGenerationReliabilityChanges(unittest.TestCase):
    def test_tag_fallback_normalizes_aliases_truncates_and_fills(self):
        from LibraryV2 import load_novel_tag_library, validate_novel_tag_selection
        from Nodes import finalize_novel_tag_selection

        library = load_novel_tag_library()
        result = finalize_novel_tag_selection(
            {
                "core": "空",
                "情节": [
                    "末世求生",
                    "末日求生",
                    "基地建设",
                    "系统",
                    "推理",
                    "直播",
                    "科幻",
                    "升级流",
                ],
                "角色": ["女医生"],
                "情绪": ["惊悚"],
                "背景": ["现代"],
            },
            library,
            "末世爆发后，女医生在现代城市带队求生并调查病毒来源。",
        )
        auxiliary = sum(
            (result[category] for category in ("情节", "角色", "情绪", "背景")),
            [],
        )
        self.assertEqual(validate_novel_tag_selection(result, library), [])
        self.assertGreaterEqual(len(auxiliary), 5)
        self.assertLessEqual(len(auxiliary), 7)
        self.assertIn("末日求生", result["情节"])
        self.assertIn("医生", result["角色"])

    def test_tag_model_failure_uses_local_selection(self):
        from LibraryV2 import load_novel_tag_library, validate_novel_tag_selection
        from Nodes import select_novel_tags

        library = load_novel_tag_library()
        with patch("Nodes._safe_structured_invoke", return_value=None):
            result = select_novel_tags(
                "末日值班室",
                "女医生在末日医院保护幸存者。",
                {"1": "调查病毒来源", "2": "关闭感染源，城市开始重建。"},
                {},
                {},
            )
        self.assertEqual(validate_novel_tag_selection(result, library), [])

    def test_open_finale_outline_is_rejected(self):
        from Nodes import finale_outline_validation_issues

        bad = (
            "【开场状态】终战开始【核心冲突】击败首领"
            "【关键行动】主角完成决战【人物关系变化】众人和解"
            "【重要信息或伏笔】真相公开"
            "【全书结局】旧敌倒下，但下一个目标出现，这只是开始"
        )
        good = bad.replace(
            "旧敌倒下，但下一个目标出现，这只是开始",
            "旧敌倒下，主要人物各归其位，风波尘埃落定，故事结束",
        )
        self.assertTrue(finale_outline_validation_issues({"1": bad}, 1))
        self.assertEqual(finale_outline_validation_issues({"1": good}, 1), [])

    def test_compatible_fact_detail_uses_derived_key(self):
        from Nodes import merge_story_ledger

        ledger = {
            "immutable_facts": [{
                "id": "F-C2-01",
                "fact_key": "blood_letter",
                "chapter": 2,
                "category": "evidence",
                "subject": "血书",
                "statement": "血书记载了幕后人的姓氏",
                "source_evidence": "第2章",
                "keywords": ["血书"],
            }],
        }
        delta = {
            "new_immutable_facts": [{
                "fact_key": "blood_letter",
                "category": "evidence",
                "subject": "血书",
                "statement": "血书还记录了交接时间和地点",
                "source_evidence": "第5章",
                "keywords": ["血书", "时间"],
            }],
        }
        merged = merge_story_ledger(
            ledger,
            delta,
            5,
            {"status": "pass", "conflicts": []},
        )
        self.assertEqual(merged["immutable_facts"][0]["statement"], ledger["immutable_facts"][0]["statement"])
        detail = merged["immutable_facts"][1]
        self.assertEqual(detail["derived_from_fact_key"], "blood_letter")
        self.assertTrue(detail["fact_key"].startswith("blood_letter__detail_c5"))

    def test_chapters_stage_before_atomic_publish(self):
        from Nodes import publish_staged_novel, stage_accepted_chapter

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            work = root / "work" / "run-1"
            final = root / "Novel" / "book_run-1.txt"
            partial = stage_accepted_chapter(work, 1, "第一章")
            self.assertFalse(final.exists())
            partial = stage_accepted_chapter(work, 2, "第二章")
            self.assertEqual(Path(partial).read_text(encoding="utf-8"), "第一章\n\n第二章")
            with patch("Nodes._build_output_path", return_value=str(final)):
                published = publish_staged_novel("书", "run-1", partial)
            self.assertEqual(Path(published), final)
            self.assertEqual(final.read_text(encoding="utf-8"), "第一章\n\n第二章")

    def test_failed_fifth_chapter_stays_staged_until_retry_succeeds(self):
        from Nodes import LedgerMergeError, publish_staged_novel, summarizer_node

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            work = root / "job" / "work" / "run-1"
            published = root / "Novel" / "完整小说_run-1.txt"
            state = {
                "run_id": "run-1",
                "artifact_dir": str(work),
                "novel_title": "完整小说",
                "chapter_outlines": {str(index): "细纲" for index in range(1, 6)},
                "chapter_contracts": {},
                "words_per_chapter": 5,
                "story_ledger": {},
                "draft_candidates": [],
                "current_chapter": 1,
            }

            for chapter in range(1, 5):
                state.update({
                    "current_chapter": chapter,
                    "current_draft": f"第{chapter}章 正文\n\n第{chapter}章正文",
                    "audit_report": {"审核状态": "通过", "结局完整性": True},
                    "editor_report": {"文风评分": 8},
                    "ledger_delta": {},
                    "continuity_report": {"status": "pass", "conflicts": []},
                })
                state.update(summarizer_node(state))

            self.assertFalse(published.exists())
            state.update({
                "current_chapter": 5,
                "current_draft": "第5章 正文\n\n冲突版本",
                "audit_report": {"审核状态": "通过", "结局完整性": True},
                "editor_report": {"文风评分": 8},
                "continuity_report": {
                    "status": "fail",
                    "conflicts": [{"fact_id": "F-C3-01"}],
                },
                "ledger_delta": {},
            })
            with self.assertRaises(LedgerMergeError):
                summarizer_node(state)
            self.assertFalse(published.exists())
            self.assertEqual(len(list((work / "chapters").glob("*.txt"))), 4)

            state.update({
                "current_draft": "第5章 正文\n\n第5章正文",
                "audit_report": {"审核状态": "通过", "结局完整性": True},
                "editor_report": {"文风评分": 8},
                "continuity_report": {"status": "pass", "conflicts": []},
                "ledger_delta": {},
                "draft_candidates": [],
            })
            state.update(summarizer_node(state))
            partial = Path(state["partial_novel_file"])
            with patch("Nodes._build_output_path", return_value=str(published)):
                publish_staged_novel("完整小说", "run-1", partial)
            content = published.read_text(encoding="utf-8")
            self.assertEqual(len(list((work / "chapters").glob("*.txt"))), 5)
            for chapter in range(1, 6):
                self.assertEqual(content.count(f"第{chapter}章正文"), 1)

    def test_candidate_files_are_never_overwritten(self):
        from Nodes import _persist_draft_candidate

        with tempfile.TemporaryDirectory() as temp_dir:
            state = {
                "artifact_dir": temp_dir,
                "run_id": "run-1",
                "current_chapter": 5,
                "iteration_count": 5,
            }
            candidate = {
                "chapter": 5,
                "iteration": 5,
                "draft": "候选正文",
                "audit_report": {"结局问题": ["人物命运未交代"]},
                "editor_report": {"文风评分": 8},
                "continuity_report": {"status": "pass", "conflicts": []},
                "ledger_delta": {},
                "score": 88,
            }
            first = _persist_draft_candidate(state, candidate)
            second = _persist_draft_candidate(state, candidate)
            self.assertNotEqual(first, second)
            self.assertTrue(Path(first).is_file())
            self.assertTrue(Path(second).is_file())

    def test_finale_attempts_five_and_six_rebuild_from_best_candidate(self):
        from Nodes import writer_node

        best = {
            "chapter": 1,
            "iteration": 3,
            "draft": "评分最高且连续性干净的终局底稿",
            "score": 91,
            "audit_report": {
                "审核状态": "不通过",
                "未完成事件": ["交代主角最终选择"],
                "结局问题": ["主要人物命运未交代"],
                "阻断问题": [],
                "发现的问题": [],
                "套路问题": [],
                "修改建议": "只补齐终局缺口",
            },
            "editor_report": {"文风评分": 8},
            "continuity_report": {"status": "pass", "conflicts": []},
        }
        base_state = {
            "current_chapter": 1,
            "current_draft": "最近但评分较低的稿件",
            "chapter_outlines": {"1": "【全书结局】故事结束"},
            "chapter_contracts": {"1": {}},
            "finale_contract": {},
            "world_bible": "设定",
            "story_ledger": {},
            "draft_candidates": [best],
            "audit_report": {"审核状态": "不通过"},
            "editor_report": {"文风评分": 6},
            "continuity_report": {"status": "pass", "conflicts": []},
            "words_per_chapter": 100,
            "writer_style": "default",
            "pattern_config": {},
        }
        response = MagicMock(content="第1章 终局\n\n完整正文")
        for previous_iteration in (4, 5):
            with self.subTest(attempt=previous_iteration + 1), patch(
                "Nodes._build_scene_plan",
                return_value={"scenes": ["重建后的终局场景"]},
            ) as planner, patch(
                "Nodes.invoke_with_retry", return_value=response
            ) as invoke:
                state = {**base_state, "iteration_count": previous_iteration}
                writer_node(state)
                inputs = invoke.call_args.args[1]
                self.assertEqual(inputs["previous_draft"], best["draft"])
                self.assertIn("主要人物命运未交代", inputs["feedback"])
                self.assertEqual(
                    json.loads(inputs["scene_plan"])["scenes"],
                    ["重建后的终局场景"],
                )
                planner.assert_called_once()

    def test_checkpoint_roundtrip_and_capability_flags(self):
        from TheGraph import (
            _job_hash,
            _load_resume_checkpoint,
            _save_resume_checkpoint,
            build_capabilities,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            work_root = Path(temp_dir) / "work"
            run_dir = work_root / "run-1"
            (run_dir / "chapters").mkdir(parents=True)
            (run_dir / "chapters" / "0001.txt").write_text("第一章", encoding="utf-8")
            job_hash = _job_hash({"job_id": "job-1", "idea": "测试"})
            checkpoint = run_dir / "resume.json"
            _save_resume_checkpoint(
                checkpoint,
                job_hash,
                "run-1",
                {
                    "run_id": "run-1",
                    "saved_chapter": 1,
                    "current_chapter": 2,
                    "chapter_outlines": {"1": "a", "2": "b"},
                },
                {},
                {},
                "summarizer",
            )
            payload, loaded_path = _load_resume_checkpoint(work_root, job_hash)
            self.assertEqual(payload["next_chapter"], 2)
            self.assertEqual(loaded_path, checkpoint.resolve())
            missing, missing_path = _load_resume_checkpoint(work_root, "wrong")
            self.assertIsNone(missing)
            self.assertIsNone(missing_path)

        contract = build_capabilities()["cli_contract"]
        self.assertTrue(contract["supports_job_resume"])
        self.assertTrue(contract["publishes_novel_on_success_only"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
