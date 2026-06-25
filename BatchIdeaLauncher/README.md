# 独立 AI 批量小说启动器

该目录是独立进程工具，不导入 AutoWrite 的 `Nodes.py`、`State.py` 或 Web
代码。它只通过能力表、任务 JSON、CLI 子进程和输出文件与原工具通信，也不会
启动 Web 端口。

## 配置

1. 如需独立环境，运行 `python -m pip install -r requirements.txt`。
2. 将 `.env.example` 复制为 `.env`，填写启动器自己的模型密钥、地址和模型。
3. 将 `batch_config.example.json` 复制为你的批次配置。
4. 准备 TXT、CSV 或 JSONL 点子清单。

TXT 每个非空行是一条点子，全部继承批次篇幅配置。CSV/JSONL 可使用以下字段
覆盖单篇配置：

- `job_id`
- `preferred_chapters`、`min_chapters`、`max_chapters`
- `preferred_words_per_chapter`
- `min_words_per_chapter`、`max_words_per_chapter`
- `refine_idea`：CSV/JSONL 可选；`1` 表示该点子写作前自动 AI 精炼，`0` 或空表示保持原始点子。TXT 点子不精炼。

## 命令

在本目录运行：

```powershell
python launcher.py catalog
python launcher.py run --ideas ideas.example.csv --config batch_config.example.json --workers 2
python launcher.py status --batch-id batch-20260622-120000
python launcher.py retry --batch-id batch-20260622-120000 --failed-only --workers 2
```

不想每次输命令时，直接双击项目根目录的 `启动批量工具.bat`，或在本目录运行
`batch_console.bat`。菜单里可以开始新批次、查看状态、重试失败任务、续跑未完成任务和
刷新能力表；新批次默认按 `workers=2` 双并发执行。是否精炼点子由 CSV/JSONL 每行的 `refine_idea` 决定，不需要额外菜单项。

`run` 和 `retry` 默认按 `max_concurrent_jobs: 2` 同时跑两篇；需要串行时传 `--workers 1`。单篇失败会记录错误并继续，成功任务在 `retry` 时自动跳过。每篇任务拥有独立的 `Outline/`、`Novel/`、日志和结果文件。

## schema v2 选型

选型 Agent 不再使用旧版 `keyword_categories` 和单一 `story_pattern`。每篇任务
会生成：

- `material_config`：素材大类/子类筛选、各大类配额及最终抽取结果。
- `pattern_config.primary`：唯一主套路，负责全书结构和硬审稿。
- `pattern_config.secondary`：普通辅助套路，只补充局部桥段；数量上限读取能力表。
- `pattern_config.manifest`：强主套路的角色、冲突模块和结局契约。

写手、套路、素材数量范围和每类上限均从本体实时导出的能力表读取，不在启动器
中固定维护。每篇正式写作前，启动器还会调用本体的 `--validate-job-file`：由
本体使用当前 `LibraryV2` 规则复核主辅套路、写手兼容并实际试抽素材。预检失败
时，错误会交给选型 Agent 修复一次；第二次仍失败才会将该篇标记为失败。因此
本体继续扩充内容库或调整约束时，启动器无需同步复制整套校验代码。
