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

## 命令

在本目录运行：

```powershell
python launcher.py catalog
python launcher.py run --ideas ideas.example.csv --config batch_config.example.json
python launcher.py status --batch-id batch-20260622-120000
python launcher.py retry --batch-id batch-20260622-120000 --failed-only
```

`run` 会顺序执行所有点子。单篇失败会记录错误并继续；成功任务在 `retry` 时
自动跳过。每篇任务拥有独立的 `Outline/`、`Novel/`、日志和结果文件。

## schema v2 选型

选型 Agent 不再使用旧版 `keyword_categories` 和单一 `story_pattern`。每篇任务
会生成：

- `material_config`：素材大类/子类筛选、2—6 个均衡槽位及最终抽取结果。
- `pattern_config.primary`：唯一主套路，负责全书结构和硬审稿。
- `pattern_config.secondary`：最多两个普通辅助套路，只补充局部桥段。
- `pattern_config.manifest`：强主套路的角色、冲突模块和结局契约。

主辅套路、素材大类、子类和标签均通过同一份能力表进行硬冲突校验。被禁止的
组合不会进入 AutoWrite CLI。
