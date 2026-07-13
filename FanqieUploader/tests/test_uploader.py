import json
import tempfile
import unittest
from pathlib import Path

from FanqieUploader.catalog import Catalog, MIN_BODY_CHARS


def make_outline(title: str, run_id: str = "run-1", tags=None):
    return {
        "schema_version": 2,
        "title": title,
        "run_id": run_id,
        "novel_tags": tags
        or {
            "core": "男生生活",
            "情节": ["重生", "末日求生"],
            "角色": ["特种兵"],
            "情绪": ["爽文"],
            "背景": ["现代"],
        },
        "chapter_outlines": {},
    }


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "Novel").mkdir()
        (self.root / "Outline").mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def write_pair(self, stem="测试小说_run-1", title="测试小说"):
        (self.root / "Novel" / f"{stem}.txt").write_text(
            "正文" * (MIN_BODY_CHARS // 2 + 20),
            encoding="utf-8",
        )
        (self.root / "Outline" / f"{stem}.json").write_text(
            json.dumps(make_outline(title), ensure_ascii=False),
            encoding="utf-8",
        )

    def test_exact_pair_loads_title_body_and_tags(self):
        self.write_pair()
        item = Catalog(self.root).scan()[0]
        self.assertTrue(item.valid)
        self.assertEqual(item.title, "测试小说")
        self.assertEqual(item.matched_by, "exact")
        self.assertEqual(item.core_tag, "男生生活")
        self.assertEqual(len(item.all_tags), 6)
        self.assertGreaterEqual(item.body_chars, MIN_BODY_CHARS)

    def test_unique_run_id_fallback(self):
        (self.root / "Novel" / "不同文件名_run-1.txt").write_text(
            "正文" * 3100,
            encoding="utf-8",
        )
        (self.root / "Outline" / "大纲文件.json").write_text(
            json.dumps(make_outline("回退匹配"), ensure_ascii=False),
            encoding="utf-8",
        )
        item = Catalog(self.root).scan()[0]
        self.assertTrue(item.valid)
        self.assertEqual(item.matched_by, "run_id")
        self.assertEqual(item.title, "回退匹配")

    def test_short_body_and_too_many_or_duplicate_tags_are_blocked(self):
        stem = "不合格_run-1"
        (self.root / "Novel" / f"{stem}.txt").write_text(
            "太短",
            encoding="utf-8",
        )
        tags = {
            "core": "主类",
            "情节": ["一", "二", "三"],
            "角色": ["四", "五"],
            "情绪": ["六", "七"],

            "背景": ["主类"],
        }
        (self.root / "Outline" / f"{stem}.json").write_text(
            json.dumps(
                make_outline("不合格", tags=tags),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        item = Catalog(self.root).scan()[0]
        self.assertFalse(item.valid)
        self.assertTrue(any("少于6000" in error for error in item.errors))
        self.assertTrue(any("超过网站上限8" in error for error in item.errors))
        self.assertTrue(any("Tag重复" in error for error in item.errors))

    def test_missing_and_ambiguous_outline_are_blocked(self):
        novel = self.root / "Novel" / "待匹配小说_run-1.txt"
        novel.write_text("正文" * 3100, encoding="utf-8")

        missing = Catalog(self.root).scan()[0]
        self.assertFalse(missing.valid)
        self.assertEqual(missing.matched_by, "missing")

        for name in ("大纲甲.json", "大纲乙.json"):
            (self.root / "Outline" / name).write_text(
                json.dumps(make_outline(name, run_id="run-1"), ensure_ascii=False),
                encoding="utf-8",
            )
        ambiguous = Catalog(self.root).scan()[0]
        self.assertFalse(ambiguous.valid)
        self.assertEqual(ambiguous.matched_by, "ambiguous")




if __name__ == "__main__":
    unittest.main()
