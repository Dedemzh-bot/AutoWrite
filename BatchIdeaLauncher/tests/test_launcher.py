import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from BatchIdeaLauncher.core import (
    AutoWriteCLI,
    BatchRunner,
    DEFAULT_CONFIG,
    initialize_batch,
    load_batch_config,
    load_ideas,
    make_job_payload,
    read_json,
    validate_selection,
)


def sample_capabilities():
    return {
        "schema_version": 2,
        "pattern_library": {"schema_version": 2, "max_secondary": 2},
        "cli_contract": {"supports_job_preflight": True},
        "writer_styles": [
            {"key": "default", "name": "默认"},
            {"key": "hot_blood", "name": "热血爽文"},
            {"key": "literary", "name": "文艺细腻"},
            {"key": "cold", "name": "冷峻纪实"},
            {"key": "humor", "name": "轻松搞笑"},
            {"key": "18xx", "name": "18XX"},
            {"key": "suspense", "name": "悬疑压迫"},
            {"key": "emotional_tension", "name": "情感拉扯"},
            {"key": "sweet_romcom", "name": "甜宠轻喜"},
            {"key": "ancient_elegant", "name": "古言雅致"},
            {"key": "realist_ensemble", "name": "现实群像"},
            {"key": "business", "name": "商战职场"},
        ],
        "story_patterns": [
            {
                "id": "none",
                "name": "无固定主套路",
                "strong": False,
                "hard_conflicts": [],
                "forbidden_material_categories": [],
                "forbidden_material_tags": [],
                "compatible_styles": [],
                "ending_options": {},
            },
            {
                "id": "strong_rule_horror",
                "name": "强规则怪谈",
                "strong": True,
                "hard_conflicts": ["marriage_first"],
                "forbidden_material_categories": ["cheat_device"],
                "forbidden_material_tags": ["甜宠"],
                "compatible_styles": ["default", "cold", "literary"],
                "ending_options": {
                    "escape_truth": "逃离",
                    "become_rule": "成为规则",
                },
            },
            {
                "id": "marriage_first",
                "name": "先婚后爱",
                "strong": False,
                "hard_conflicts": ["strong_rule_horror"],
                "forbidden_material_categories": [],
                "forbidden_material_tags": [],
                "compatible_styles": [],
                "ending_options": {},
            },
            {
                "id": "custom",
                "name": "自定义套路",
                "strong": False,
                "hard_conflicts": [],
                "forbidden_material_categories": [],
                "forbidden_material_tags": [],
                "compatible_styles": [],
                "ending_options": {},
            },
        ],
        "material_library": {
            "schema_version": 2,
            "count_range": [2, 8],
            "default_group_counts": {
                "world_stage": 1,
                "protagonist": 1,
                "supporting_role": 0,
                "cheat_device": 1,
                "plot_event": 0,
                "core_conflict": 1,
                "career_resource": 0,
                "atmosphere": 0,
            },
            "group_limits": {
                "world_stage": 1, "protagonist": 1,
                "supporting_role": 2, "cheat_device": 2,
                "plot_event": 2, "core_conflict": 2,
                "career_resource": 2, "atmosphere": 2,
            },
            "groups": {
                "world_stage": {
                    "name": "世界舞台",
                    "subcategories": [
                        {
                            "id": "modern_city",
                            "name": "现代都市",
                            "tags": ["现实", "都市"],
                            "count": 12,
                        }
                    ],
                },
                "cheat_device": {
                    "name": "金手指",
                    "subcategories": [
                        {
                            "id": "system",
                            "name": "任务系统",
                            "tags": ["系统"],
                            "count": 12,
                        }
                    ],
                },
            },
        },
    }


def write_config(path: Path):
    path.write_text(
        json.dumps({
            "length": {
                "preferred_chapters": 10,
                "min_chapters": 6,
                "max_chapters": 14,
                "preferred_words_per_chapter": 1500,
                "min_words_per_chapter": 1200,
                "max_words_per_chapter": 2000,
            },
            "job_timeout_seconds": 60,
            "selector": {
                "model": "fake-model",
                "temperature": 0,
                "max_retries": 0,
                "request_timeout_seconds": 5,
            },
        }, ensure_ascii=False),
        encoding="utf-8",
    )


def write_fake_cli(path: Path, capabilities: dict):
    script = f"""
import argparse
import json
from pathlib import Path

CAPABILITIES = {capabilities!r}

parser = argparse.ArgumentParser()
parser.add_argument("--describe-capabilities")
parser.add_argument("--job-file")
parser.add_argument("--validate-job-file")
parser.add_argument("--result-file")
parser.add_argument("--auto-approve", action="store_true")
args = parser.parse_args()

if args.describe_capabilities:
    target = Path(args.describe_capabilities)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(CAPABILITIES, ensure_ascii=False), encoding="utf-8")
    target.with_suffix(".md").write_text("# fake", encoding="utf-8")
    raise SystemExit(0)

cwd = Path.cwd()
if args.validate_job_file:
    job = json.loads(Path(args.validate_job_file).read_text(encoding="utf-8"))
    marker = cwd / ".preflight-rejected-once"
    if "preflight-repair" in job["idea"] and not marker.exists():
        marker.write_text("1", encoding="utf-8")
        Path(args.result_file).write_text(
            json.dumps({{"status": "failed", "error": "preflight rejected"}}),
            encoding="utf-8",
        )
        raise SystemExit(1)
    Path(args.result_file).write_text(
        json.dumps({{
            "status": "validated",
            "material_config": job["material_config"],
            "pattern_config": job["pattern_config"],
        }}, ensure_ascii=False),
        encoding="utf-8",
    )
    raise SystemExit(0)

job = json.loads(Path(args.job_file).read_text(encoding="utf-8"))
with (cwd.parent / "order.log").open("a", encoding="utf-8") as stream:
    stream.write(job["job_id"] + "\\n")

marker = cwd / ".failed-once"
if job["job_id"] == "idea-002" and not marker.exists():
    marker.write_text("1", encoding="utf-8")
    Path(args.result_file).write_text(
        json.dumps({{"status": "failed", "error": "simulated failure"}}),
        encoding="utf-8",
    )
    raise SystemExit(1)

(cwd / "Novel").mkdir(exist_ok=True)
(cwd / "Outline").mkdir(exist_ok=True)
novel = (cwd / "Novel" / (job["job_id"] + ".txt")).resolve()
outline = (cwd / "Outline" / (job["job_id"] + ".json")).resolve()
novel.write_text("正文", encoding="utf-8")
outline.write_text("{{}}", encoding="utf-8")
Path(args.result_file).write_text(
    json.dumps({{
        "status": "succeeded",
        "run_id": "fake-" + job["job_id"],
        "novel_file": str(novel),
        "outline_file": str(outline),
        "material_config": job["material_config"],
        "pattern_config": job["pattern_config"],
    }}, ensure_ascii=False),
    encoding="utf-8",
)
"""
    path.write_text(script, encoding="utf-8")


class StaticSelector:
    def choose(self, idea, constraints, capabilities, repair=None):
        return {
            "target_chapters": constraints["preferred_chapters"],
            "words_per_chapter": constraints[
                "preferred_words_per_chapter"
            ],
            "writer_style": "default",
            "material_config": {
                "schema_version": 2,
                "filters": {
                    "categories": ["world_stage"],
                    "subcategories": ["modern_city"],
                    "tags": [],
                },
                "group_counts": {
                    "world_stage": 1,
                    "protagonist": 1,
                    "supporting_role": 0,
                    "cheat_device": 1,
                    "plot_event": 0,
                    "core_conflict": 1,
                    "career_resource": 0,
                    "atmosphere": 0,
                },
                "items": [],
                "locked_item_keys": [],
                "auto_selected_subcategories": [],
            },
            "pattern_config": {
                "schema_version": 2,
                "primary": "none",
                "secondary": [],
                "custom_instruction": "",
                "manifest": {},
                "structure_plan": {},
            },
            "rationale": "模拟选择",
        }


class RepairingSelector(StaticSelector):
    def __init__(self):
        self.calls = []

    def choose(self, idea, constraints, capabilities, repair=None):
        self.calls.append(repair)
        if repair is None:
            return {
                "target_chapters": 999,
                "words_per_chapter": 1,
                "writer_style": "unknown",
                "material_config": {
                    "filters": {
                        "categories": ["不存在"],
                        "subcategories": ["不存在"],
                    },
                    "count": 99,
                },
                "pattern_config": {
                    "primary": "missing",
                    "secondary": ["missing"],
                },
            }
        return super().choose(idea, constraints, capabilities, repair)


class PreflightRepairSelector(StaticSelector):
    def __init__(self):
        self.calls = []

    def choose(self, idea, constraints, capabilities, repair=None):
        self.calls.append(repair)
        return super().choose(idea, constraints, capabilities, repair)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.temp = Path(self.temp_directory.name)
        self.capabilities = sample_capabilities()

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_default_batch_length_is_eight_chapters(self):
        self.assertEqual(DEFAULT_CONFIG["length"]["preferred_chapters"], 8)

    def test_txt_inherits_config_and_csv_can_override(self):
        config_path = self.temp / "config.json"
        write_config(config_path)
        config = load_batch_config(config_path)

        text_path = self.temp / "ideas.txt"
        text_path.write_text("点子甲\n点子乙\n", encoding="utf-8")
        text_items = load_ideas(text_path)
        self.assertEqual(
            [item["job_id"] for item in text_items],
            ["idea-001", "idea-002"],
        )
        self.assertEqual(text_items[0]["length_overrides"], {})

        csv_path = self.temp / "ideas.csv"
        csv_path.write_text(
            "job_id,idea,preferred_chapters,min_chapters,max_chapters,"
            "preferred_words_per_chapter,min_words_per_chapter,"
            "max_words_per_chapter\n"
            "special,点子丙,8,7,9,1800,1600,1900\n",
            encoding="utf-8",
        )
        csv_items = load_ideas(csv_path)
        self.assertEqual(
            csv_items[0]["length_overrides"]["preferred_chapters"], "8"
        )
        self.assertEqual(config["length"]["preferred_chapters"], 10)

    def test_normal_strong_and_custom_selections(self):
        constraints = {
            "preferred_chapters": 10,
            "min_chapters": 6,
            "max_chapters": 14,
            "preferred_words_per_chapter": 1500,
            "min_words_per_chapter": 1200,
            "max_words_per_chapter": 2000,
        }
        selections = [
            {
                "target_chapters": 10,
                "words_per_chapter": 1500,
                "writer_style": "18xx",
                "material_config": {
                    "filters": {"categories": ["world_stage"], "subcategories": ["modern_city"], "tags": []},
                    "count": 4,
                },
                "pattern_config": {
                    "primary": "none",
                    "secondary": ["marriage_first"],
                },
            },
            {
                "target_chapters": 8,
                "words_per_chapter": 1400,
                "writer_style": "cold",
                "material_config": {
                    "filters": {"categories": ["world_stage"], "subcategories": ["modern_city"], "tags": []},
                    "count": 4,
                },
                "pattern_config": {
                    "primary": "strong_rule_horror",
                    "secondary": [],
                    "manifest": {"ending": "escape_truth"},
                },
            },
            {
                "target_chapters": 12,
                "words_per_chapter": 1800,
                "writer_style": "literary",
                "material_config": {
                    "filters": {"categories": [], "subcategories": [], "tags": []},
                    "count": 4,
                },
                "pattern_config": {
                    "primary": "custom",
                    "secondary": [],
                    "custom_instruction": "每三章完成一次身份反转",
                },
            },
        ]
        for selection in selections:
            with self.subTest(pattern=selection["pattern_config"]["primary"]):
                normalized, issues = validate_selection(
                    selection, constraints, self.capabilities
                )
                self.assertEqual(issues, [])
                job = make_job_payload(
                    "batch-test",
                    {"job_id": "idea-001", "idea": "测试"},
                    normalized,
                )
                self.assertEqual(job["schema_version"], 2)
                self.assertIsInstance(job["pattern_seed"], int)
                self.assertIsInstance(job["material_seed"], int)

    def test_strong_pattern_rejects_style_and_material(self):
        constraints = {
            "preferred_chapters": 10,
            "min_chapters": 6,
            "max_chapters": 14,
            "preferred_words_per_chapter": 1500,
            "min_words_per_chapter": 1200,
            "max_words_per_chapter": 2000,
        }
        _, issues = validate_selection({
            "target_chapters": 10,
            "words_per_chapter": 1500,
            "writer_style": "18xx",
            "material_config": {
                "filters": {
                    "categories": ["cheat_device"],
                    "subcategories": ["system"],
                    "tags": [],
                },
                "count": 4,
            },
            "pattern_config": {
                "primary": "strong_rule_horror",
                "secondary": [],
                "manifest": {"ending": "escape_truth"},
            },
        }, constraints, self.capabilities)
        self.assertTrue(any("不兼容写手" in issue for issue in issues))
        self.assertTrue(any("冲突" in issue for issue in issues))

    def test_failure_continue_retry_and_output_isolation(self):
        config_path = self.temp / "config.json"
        write_config(config_path)
        config = load_batch_config(config_path)
        ideas_path = self.temp / "ideas.txt"
        ideas_path.write_text("第一条点子\n第二条点子\n", encoding="utf-8")
        fake_cli_path = self.temp / "fake_autowrite.py"
        write_fake_cli(fake_cli_path, self.capabilities)

        autowrite = AutoWriteCLI(
            entry=fake_cli_path,
            python_command=sys.executable,
        )
        batch_dir = initialize_batch(
            ideas_path,
            config,
            autowrite,
            runs_dir=self.temp / "runs",
            batch_id="batch-test",
        )
        first_summary = BatchRunner(
            batch_dir, autowrite, selector=StaticSelector()
        ).process()
        self.assertEqual(first_summary["counts"]["succeeded"], 1)
        self.assertEqual(first_summary["counts"]["failed"], 1)
        manifest = read_json(batch_dir / "batch.json")
        self.assertEqual(manifest["jobs"][0]["attempts"], 1)
        self.assertEqual(manifest["jobs"][1]["attempts"], 1)
        self.assertTrue(
            (batch_dir / "idea-001" / "Novel" / "idea-001.txt").exists()
        )
        self.assertFalse(
            (batch_dir / "idea-002" / "Novel" / "idea-002.txt").exists()
        )

        second_summary = BatchRunner(
            batch_dir, autowrite, selector=StaticSelector()
        ).process(statuses={"failed"}, reuse_selection=True)
        self.assertEqual(second_summary["counts"]["succeeded"], 2)
        self.assertEqual(second_summary["counts"]["failed"], 0)
        manifest = read_json(batch_dir / "batch.json")
        self.assertEqual(manifest["jobs"][0]["attempts"], 1)
        self.assertEqual(manifest["jobs"][1]["attempts"], 2)
        self.assertTrue(
            (batch_dir / "idea-002" / "Novel" / "idea-002.txt").exists()
        )
        self.assertTrue((batch_dir / "summary.json").exists())
        self.assertTrue((batch_dir / "summary.csv").exists())
        self.assertEqual(
            (batch_dir / "order.log").read_text(
                encoding="utf-8"
            ).splitlines(),
            ["idea-001", "idea-002", "idea-002"],
        )

    def test_invalid_agent_output_is_repaired_once(self):
        config_path = self.temp / "config.json"
        write_config(config_path)
        config = load_batch_config(config_path)
        ideas_path = self.temp / "ideas.txt"
        ideas_path.write_text("需要修复选型的点子\n", encoding="utf-8")
        fake_cli_path = self.temp / "fake_autowrite.py"
        write_fake_cli(fake_cli_path, self.capabilities)
        autowrite = AutoWriteCLI(
            entry=fake_cli_path,
            python_command=sys.executable,
        )
        batch_dir = initialize_batch(
            ideas_path,
            config,
            autowrite,
            runs_dir=self.temp / "runs",
            batch_id="repair-test",
        )
        selector = RepairingSelector()
        summary = BatchRunner(
            batch_dir, autowrite, selector=selector
        ).process()
        self.assertEqual(summary["counts"]["succeeded"], 1)
        self.assertEqual(len(selector.calls), 2)
        self.assertIsNone(selector.calls[0])
        self.assertTrue(selector.calls[1]["validation_errors"])

    def test_body_preflight_failure_is_repaired_before_writing(self):
        config_path = self.temp / "config.json"
        write_config(config_path)
        config = load_batch_config(config_path)
        ideas_path = self.temp / "ideas.txt"
        ideas_path.write_text("preflight-repair\n", encoding="utf-8")
        fake_cli_path = self.temp / "fake_autowrite.py"
        write_fake_cli(fake_cli_path, self.capabilities)
        autowrite = AutoWriteCLI(
            entry=fake_cli_path,
            python_command=sys.executable,
        )
        batch_dir = initialize_batch(
            ideas_path,
            config,
            autowrite,
            runs_dir=self.temp / "runs",
            batch_id="preflight-repair-test",
        )
        selector = PreflightRepairSelector()
        summary = BatchRunner(
            batch_dir, autowrite, selector=selector
        ).process()
        self.assertEqual(summary["counts"]["succeeded"], 1)
        self.assertEqual(len(selector.calls), 2)
        self.assertIsNone(selector.calls[0])
        self.assertIn("preflight rejected", str(selector.calls[1]))
        job_dir = batch_dir / "idea-001"
        self.assertEqual(read_json(job_dir / "preflight.json")["status"], "validated")
        self.assertTrue((job_dir / "Novel" / "idea-001.txt").exists())

    def test_real_cli_exports_twelve_styles_and_launcher_has_no_web_stack(self):
        project_root = Path(__file__).resolve().parents[2]
        destination = self.temp / "capabilities.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "TheGraph.py"),
                "--describe-capabilities",
                str(destination),
            ],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(destination.with_suffix(".md").exists())
        exported = read_json(destination)
        self.assertEqual(len(exported["writer_styles"]), 12)
        self.assertTrue(
            exported["cli_contract"]["supports_job_preflight"]
        )
        self.assertTrue(all(
            "forbidden_pattern_tags" in item
            for item in exported["story_patterns"]
        ))
        self.assertEqual(exported["pattern_library"]["max_secondary"], 2)
        self.assertEqual(
            exported["material_library"]["count_range"], [2, 8]
        )
        self.assertGreaterEqual(
            {item["key"] for item in exported["writer_styles"]},
            {
                "default",
                "hot_blood",
                "literary",
                "cold",
                "humor",
                "18xx",
                "suspense",
                "emotional_tension",
                "sweet_romcom",
                "ancient_elegant",
                "realist_ensemble",
                "business",
            },
        )
        source = (
            project_root / "BatchIdeaLauncher" / "core.py"
        ).read_text(encoding="utf-8").lower()
        self.assertNotIn("fastapi", source)
        self.assertNotIn("uvicorn", source)
        self.assertNotIn("websocket", source)

    def test_real_cli_rejects_invalid_job_before_model_execution(self):
        project_root = Path(__file__).resolve().parents[2]
        job_dir = self.temp / "invalid-job"
        job_dir.mkdir()
        job_path = job_dir / "job.json"
        result_path = job_dir / "result.json"
        job_path.write_text(json.dumps({
            "schema_version": 2,
            "job_id": "invalid-001",
            "idea": "规则怪谈点子",
            "target_chapters": 10,
            "words_per_chapter": 1500,
            "writer_style": "18xx",
            "material_config": {
                "schema_version": 2,
                "filters": {
                    "categories": ["cheat_device"],
                    "subcategories": ["system"],
                    "tags": [],
                },
                "group_counts": {
                    "world_stage": 1,
                    "protagonist": 1,
                    "supporting_role": 0,
                    "cheat_device": 1,
                    "plot_event": 0,
                    "core_conflict": 1,
                    "career_resource": 0,
                    "atmosphere": 0,
                },
                "items": [],
                "locked_item_keys": [],
                "auto_selected_subcategories": [],
            },
            "pattern_config": {
                "schema_version": 2,
                "primary": "strong_rule_horror",
                "secondary": [],
                "custom_instruction": "",
                "manifest": {"ending": "escape_truth"},
                "structure_plan": {},
            },
            "pattern_seed": 123,
            "material_seed": 456,
        }, ensure_ascii=False), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "TheGraph.py"),
                "--validate-job-file",
                str(job_path),
                "--result-file",
                str(result_path),
                "--auto-approve",
            ],
            cwd=str(job_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 1)
        result = read_json(result_path)
        self.assertEqual(result["status"], "failed")
        self.assertIn("仅支持写手风格", result["error"])
        self.assertFalse((job_dir / "Novel").exists())
        self.assertFalse((job_dir / "Outline").exists())


if __name__ == "__main__":
    unittest.main()
