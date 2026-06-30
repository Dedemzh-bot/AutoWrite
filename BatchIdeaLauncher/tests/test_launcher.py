import csv
import json
import os
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
                "id": "strong_entertainment_reveal",
                "name": "Entertainment Reveal",
                "strong": True,
                "hard_conflicts": [],
                "forbidden_material_categories": [],
                "forbidden_material_tags": [],
                "compatible_styles": [
                    "default",
                    "humor",
                    "hot_blood",
                    "business",
                    "sweet_romcom",
                ],
                "ending_options": {"award_reveal": "Award reveal"},
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
            "count_range": [0, 8],
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
import csv
import json
from pathlib import Path

CAPABILITIES = {capabilities!r}

parser = argparse.ArgumentParser()
parser.add_argument("--describe-capabilities")
parser.add_argument("--job-file")
parser.add_argument("--validate-job-file")
parser.add_argument("--result-file")
parser.add_argument("--auto-approve", action="store_true")
parser.add_argument("--restart", action="store_true")
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
job_dir = Path(args.job_file).resolve().parent
with (job_dir.parent / "order.log").open("a", encoding="utf-8") as stream:
    stream.write(job["job_id"] + "\\n")

marker = job_dir / ".failed-once"
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
        if not hasattr(self, "ideas"):
            self.ideas = []
        self.ideas.append(idea)
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

class FakeRefiner:
    def __init__(self, failures=None):
        self.calls = []
        self.failures = set(failures or [])

    def __call__(self, idea):
        self.calls.append(idea)
        if idea in self.failures:
            raise RuntimeError("refine failed")
        return f"精炼后：{idea}"

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

    def test_default_batch_length_is_quality_short_story(self):
        self.assertEqual(DEFAULT_CONFIG["length"]["preferred_chapters"], 6)
        self.assertEqual(DEFAULT_CONFIG["max_concurrent_jobs"], 2)
        self.assertEqual(DEFAULT_CONFIG["length"]["preferred_words_per_chapter"], 2500)
        self.assertEqual(DEFAULT_CONFIG["length"]["min_chapters"], 5)
        self.assertEqual(DEFAULT_CONFIG["length"]["max_chapters"], 7)

    def test_batch_concurrency_config_is_validated(self):
        config_path = self.temp / "config.json"
        write_config(config_path)
        config = load_batch_config(config_path)
        self.assertEqual(config["max_concurrent_jobs"], 2)

        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["max_concurrent_jobs"] = 3
        config_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        config = load_batch_config(config_path)
        self.assertEqual(config["max_concurrent_jobs"], 3)

        payload["max_concurrent_jobs"] = 0
        config_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        with self.assertRaisesRegex(Exception, "max_concurrent_jobs"):
            load_batch_config(config_path)

    def test_cli_accepts_worker_overrides(self):
        from BatchIdeaLauncher import launcher

        run_args = launcher.build_parser().parse_args([
            "run",
            "--ideas",
            "ideas.txt",
            "--config",
            "config.json",
            "--workers",
            "2",
        ])
        self.assertEqual(run_args.workers, 2)

        retry_args = launcher.build_parser().parse_args([
            "retry",
            "--batch-id",
            "batch-test",
            "--failed-only",
            "--workers",
            "1",
            "--restart-failed",
        ])
        self.assertEqual(retry_args.workers, 1)
        self.assertTrue(retry_args.restart_failed)

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
        self.assertEqual(text_items[0]["source_idea"], "点子甲")
        self.assertFalse(text_items[0]["refine_idea"])

        csv_path = self.temp / "ideas.csv"
        csv_path.write_text(
            "job_id,idea,refine_idea,preferred_chapters,min_chapters,max_chapters,"
            "preferred_words_per_chapter,min_words_per_chapter,"
            "max_words_per_chapter\n"
            "special,点子丙,1,8,7,9,1800,1600,1900\n",
            encoding="utf-8",
        )
        csv_items = load_ideas(csv_path)
        self.assertEqual(
            csv_items[0]["length_overrides"]["preferred_chapters"], "8"
        )
        self.assertTrue(csv_items[0]["refine_idea"])
        self.assertEqual(csv_items[0]["source_idea"], "点子丙")
        self.assertEqual(csv_items[0]["idea"], "点子丙")
        bad_csv_path = self.temp / "bad_ideas.csv"
        bad_csv_path.write_text(
            "job_id,idea,refine_idea\ninvalid,点子丁,yes\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(Exception, "refine_idea"):
            load_ideas(bad_csv_path)
        jsonl_path = self.temp / "ideas.jsonl"
        jsonl_path.write_text(
            json.dumps({"job_id": "json-1", "idea": "点子戊", "refine_idea": True}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        self.assertTrue(load_ideas(jsonl_path)[0]["refine_idea"])
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

    def test_entertainment_reveal_locks_emotion_writer(self):
        constraints = {
            "preferred_chapters": 10,
            "min_chapters": 6,
            "max_chapters": 14,
            "preferred_words_per_chapter": 1500,
            "min_words_per_chapter": 1200,
            "max_words_per_chapter": 2000,
        }
        for writer_style in ("emotional_tension", "emotion"):
            with self.subTest(writer_style=writer_style):
                normalized, issues = validate_selection({
                    "target_chapters": 10,
                    "words_per_chapter": 1500,
                    "writer_style": writer_style,
                    "material_config": {
                        "filters": {
                            "categories": [],
                            "subcategories": [],
                            "tags": [],
                        },
                        "count": 4,
                    },
                    "pattern_config": {
                        "primary": "strong_entertainment_reveal",
                        "secondary": [],
                        "manifest": {"ending": "award_reveal"},
                    },
                }, constraints, self.capabilities)
                self.assertEqual(issues, [])
                self.assertEqual(normalized["writer_style"], "sweet_romcom")
                self.assertEqual(
                    normalized["selection_locks"][0]["from"], writer_style
                )

    def test_batch_runner_processes_four_jobs_with_two_workers(self):
        config_path = self.temp / "config.json"
        write_config(config_path)
        config = load_batch_config(config_path)
        ideas_path = self.temp / "ideas.jsonl"
        ideas = [
            {"job_id": f"job-{index}", "idea": f"point {index}"}
            for index in range(1, 5)
        ]
        ideas_path.write_text(
            "\n".join(
                json.dumps(item, ensure_ascii=False) for item in ideas
            ) + "\n",
            encoding="utf-8",
        )
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
            batch_id="parallel-test",
        )
        summary = BatchRunner(
            batch_dir, autowrite, selector=StaticSelector()
        ).process(max_workers=2)
        self.assertEqual(summary["counts"]["succeeded"], 4)
        self.assertEqual(summary["counts"]["failed"], 0)
        manifest = read_json(batch_dir / "batch.json")
        self.assertEqual(manifest["config"]["max_concurrent_jobs"], 2)
        for item in ideas:
            job_dir = batch_dir / item["job_id"]
            self.assertTrue((self.temp / "Novel" / f"{item['job_id']}.txt").exists())
            self.assertTrue((self.temp / "Outline" / f"{item['job_id']}.json").exists())
            self.assertEqual(read_json(job_dir / "status.json")["status"], "succeeded")
        order_lines = (batch_dir / "order.log").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertCountEqual(order_lines, [item["job_id"] for item in ideas])

    def test_refine_idea_runs_before_selection_and_writing(self):
        config_path = self.temp / "config.json"
        write_config(config_path)
        config = load_batch_config(config_path)
        ideas_path = self.temp / "ideas.csv"
        ideas_path.write_text(
            "job_id,idea,refine_idea\n"
            "refined,原始点子,1\n"
            "plain,普通点子,0\n",
            encoding="utf-8",
        )
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
            batch_id="refine-test",
        )
        selector = StaticSelector()
        refiner = FakeRefiner()
        summary = BatchRunner(
            batch_dir, autowrite, selector=selector, refiner=refiner
        ).process(max_workers=1)
        self.assertEqual(summary["counts"]["succeeded"], 2)
        self.assertEqual(refiner.calls, ["原始点子"])
        self.assertEqual(selector.ideas, ["精炼后：原始点子", "普通点子"])

        manifest = read_json(batch_dir / "batch.json")
        refined_job = next(job for job in manifest["jobs"] if job["job_id"] == "refined")
        plain_job = next(job for job in manifest["jobs"] if job["job_id"] == "plain")
        self.assertTrue(refined_job["refine_idea"])
        self.assertEqual(refined_job["source_idea"], "原始点子")
        self.assertEqual(refined_job["idea"], "精炼后：原始点子")
        self.assertFalse(plain_job["refine_idea"])
        self.assertEqual(plain_job["idea"], "普通点子")
        self.assertEqual(
            read_json(batch_dir / "refined" / "job.json")["idea"],
            "精炼后：原始点子",
        )
        refinement = read_json(batch_dir / "refined" / "idea_refinement.json")
        self.assertEqual(refinement["status"], "succeeded")
        self.assertEqual(refinement["source_idea"], "原始点子")
        self.assertEqual(refinement["refined_idea"], "精炼后：原始点子")
        report = read_json(batch_dir / "summary.json")
        report_job = next(job for job in report["jobs"] if job["job_id"] == "refined")
        self.assertEqual(report_job["refine_idea"], 1)
        self.assertEqual(report_job["source_idea"], "原始点子")
        self.assertEqual(report_job["used_idea"], "精炼后：原始点子")
        with (batch_dir / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        row = next(item for item in rows if item["job_id"] == "refined")
        self.assertEqual(row["refine_idea"], "1")
        self.assertEqual(row["source_idea"], "原始点子")
        self.assertEqual(row["used_idea"], "精炼后：原始点子")

    def test_refine_idea_failure_isolated_and_retryable(self):
        config_path = self.temp / "config.json"
        write_config(config_path)
        config = load_batch_config(config_path)
        ideas_path = self.temp / "ideas.csv"
        ideas_path.write_text(
            "job_id,idea,refine_idea\n"
            "needs-refine,坏点子,1\n"
            "plain,普通点子,0\n",
            encoding="utf-8",
        )
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
            batch_id="refine-failure-test",
        )
        first_summary = BatchRunner(
            batch_dir,
            autowrite,
            selector=StaticSelector(),
            refiner=FakeRefiner(failures={"坏点子"}),
        ).process(max_workers=1)
        self.assertEqual(first_summary["counts"]["succeeded"], 1)
        self.assertEqual(first_summary["counts"]["failed"], 1)
        failed_refinement = read_json(
            batch_dir / "needs-refine" / "idea_refinement.json"
        )
        self.assertEqual(failed_refinement["status"], "failed")

        selector = StaticSelector()
        second_summary = BatchRunner(
            batch_dir,
            autowrite,
            selector=selector,
            refiner=FakeRefiner(),
        ).process(statuses={"failed"}, reuse_selection=True, max_workers=1)
        self.assertEqual(second_summary["counts"]["succeeded"], 2)
        self.assertEqual(second_summary["counts"]["failed"], 0)
        self.assertEqual(selector.ideas, ["精炼后：坏点子"])
        manifest = read_json(batch_dir / "batch.json")
        retried = next(job for job in manifest["jobs"] if job["job_id"] == "needs-refine")
        self.assertEqual(retried["idea"], "精炼后：坏点子")

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
            (self.temp / "Novel" / "idea-001.txt").exists()
        )
        self.assertFalse(
            (self.temp / "Novel" / "idea-002.txt").exists()
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
            (self.temp / "Novel" / "idea-002.txt").exists()
        )
        self.assertTrue((batch_dir / "summary.json").exists())
        self.assertTrue((batch_dir / "summary.csv").exists())
        order_lines = (batch_dir / "order.log").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(order_lines.count("idea-001"), 1)
        self.assertEqual(order_lines.count("idea-002"), 2)

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
        self.assertTrue((self.temp / "Novel" / "idea-001.txt").exists())

    def test_real_cli_exports_twelve_styles_and_launcher_has_no_web_stack(self):
        project_root = Path(__file__).resolve().parents[2]
        destination = self.temp / "capabilities.json"
        env = dict(os.environ)
        env["OPENAI_MODEL"] = env.get("OPENAI_MODEL") or "test-model"
        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "TheGraph.py"),
                "--describe-capabilities",
                str(destination),
            ],
            cwd=str(project_root),
            env=env,
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
            exported["material_library"]["count_range"], [0, 8]
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
        invalid_env = dict(os.environ)
        invalid_env["OPENAI_MODEL"] = invalid_env.get("OPENAI_MODEL") or "test-model"
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
            env=invalid_env,
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
