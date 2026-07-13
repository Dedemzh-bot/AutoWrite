# FanqieUploader

这是 AutoWrite 的独立番茄短故事「批量存草稿」工具。它读取项目根目录的 `Novel/` 与 `Outline/`，通过 Playwright 直接控制真实 Chrome 浏览器操作番茄作者后台，逐篇填写标题/正文/封面/AI/标签并存入草稿。

方案演进：

- **第一套（已废弃）**：本地管理页 + Chrome 扩展（server.py / database.py / extension/）。已删除。
- **第二套（当前）**：Playwright 自动化（executor.py + playwright_batch.py + playwright_selftest.py）。直接操作真实 Chrome，无 CSP/隔离世界问题。

## 安全边界

- 只访问 `fanqienovel.com` 和 `127.0.0.1`。
- 不读取或保存番茄账号密码。
- 不绕过验证码；检测到验证或登录失效会立即暂停。
- 只点击「存草稿」，不设置试读比例、不勾发布协议、不执行正式发布。

## 前置条件

1. **所有 Chrome 窗口必须关闭**——脚本需要复制 User Data 到临时目录，锁定文件会导致失败。
2. **番茄作者后台已登录**——首次使用脚本会打开 Chrome 并检测登录；如果未登录，会等待你手动登录。
3. **Novel/ 和 Outline/ 目录有有效配对文件**——配对规则见下文「文件规则」。

## 使用方式

### 单篇校准（推荐首次使用）

```bash
python -m FanqieUploader.playwright_selftest
```

流程：

1. 扫描 catalog 取第一篇 valid 小说。
2. 启动 Chrome（复制用户登录态到临时目录）。
3. 检查番茄登录 → 执行校准流程（填标题/正文/封面/AI/标签）。
4. **停在「存草稿」前**，等待人工核对。
5. 输出报告到 `FanqieUploader/data/playwright-selftest-latest.json`。

人工确认页面内容无误后，再进行批量存草稿。

### 批量存草稿

```bash
python -m FanqieUploader.playwright_batch
```

流程：

1. 扫描 catalog 取所有 valid 小说。
2. 启动 Chrome → 逐篇执行：新建短故事 → 填标题/正文/封面/AI/标签 → 点击存草稿 → 等待保存成功提示 → 关闭编辑器页面 → 下一篇。
3. 汇总报告保存到 `FanqieUploader/data/playwright-batch-latest.json`。

## 文件规则

- 优先精确配对 `Novel/<stem>.txt` 与 `Outline/<stem>.json`。
- 精确配对失败时，仅接受唯一 `run_id` 匹配。
- 标题读取大纲的 `title`，正文原样读取 TXT。
- `novel_tags.core` 映射为主分类，其余只接受 `情节/角色/情绪/背景`。
- 正文有效字符少于 6000、标签缺失/重复、总数超过 8、文件歧义或缺失都会阻止入队。

## 故障处理

- 验证码或登录失效：人工完成验证/登录后，关闭 Chrome 窗口，重新运行脚本。
- 封面超过 180 秒：任务暂停，保留当前页面。
- 缺失 Tag 或标签核对不一致：修正大纲或站点分类后重试。
- 页面结构变化：重新执行单篇校准。
- Chrome 窗口被锁定：确保所有 Chrome 窗口已关闭再重新运行。

## 技术架构

- `executor.py`：核心执行器 `FanqieExecutor`，封装了填写标题、正文、封面制作、AI 选择、标签选择、存草稿等操作。
- `launch_chrome_with_login()`：复制 Chrome User Data（Cookies / Login Data / Local State）到临时目录，启动带登录态的独立 Chrome 实例。
- `playwright_selftest.py`：单篇校准入口，停在存草稿前。
- `playwright_batch.py`：批量存草稿入口，自动处理所有可上传小说。
- `catalog.py`：目录扫描模块（两套共用），负责扫描 Novel/ 与 Outline/ 目录，返回可上传的小说列表。