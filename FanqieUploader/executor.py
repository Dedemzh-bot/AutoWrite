"""Playwright 执行器 — 替代 Chrome 扩展，直接控制本地 Chrome 操作番茄网站。

工作方式：
1. 连接已经启动的 Chrome（需带 --remote-debugging-port=9222）
2. 打开番茄后台 → 新建短故事 → 填标题/正文/封面/AI/Tag → 停在存草稿前（校准）或点击存草稿（批量）
3. 通过真实浏览器点击，React 完全响应，不存在隔离世界/CSP/fiber 问题
"""
from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright, Page, Browser, TimeoutError as PWTimeout

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
MANAGE_URL = "https://fanqienovel.com/main/writer/short-manage"
CDP_URL = "http://127.0.0.1:9222"


def clean(value: str) -> str:
    return re.sub(r"\s+", "", str(value or ""))


class FanqieExecutor:
    """番茄短故事自动化执行器。"""

    def __init__(self, page: Page, mode: str = "calibration"):
        self.page = page
        self.mode = mode
        self.logs: list[str] = []

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.logs.append(line)
        print(line)

    def wait_text(self, text: str, timeout: int = 20000) -> None:
        """等待页面出现指定文本。"""
        self.page.wait_for_selector(f"text={text}", timeout=timeout)

    def click_text(self, text: str, timeout: int = 15000) -> None:
        """点击精确匹配文本的元素。"""
        el = self.page.get_by_text(text, exact=True).first
        el.wait_for(state="visible", timeout=timeout)
        el.click()
        self.log(f"已点击：{text}")

    def click_button(self, text: str, timeout: int = 15000) -> None:
        """点击按钮（含模糊匹配）。"""
        btn = self.page.locator(f"button:has-text('{text}')").first
        btn.wait_for(state="visible", timeout=timeout)
        btn.click()
        self.log(f"已点击按钮：{text}")

    def fill_title(self, title: str) -> None:
        """填写标题。"""
        # 番茄标题输入框 placeholder 含"请输入短故事名称"
        sel = "textarea[placeholder*='短故事名称'], input[placeholder*='短故事名称'], [contenteditable][data-placeholder*='短故事名称']"
        el = self.page.locator(sel).first
        el.wait_for(state="visible", timeout=15000)
        el.click()
        el.fill(title)
        self.log(f"标题已填写：{title}")

    def fill_body(self, body: str) -> None:
        """填写正文（保留换行）。"""
        # 番茄正文是 contenteditable div（ProseMirror）
        editor = self.page.locator("[contenteditable='true']").last
        editor.wait_for(state="visible", timeout=15000)
        editor.click()
        # 用 keyboard.type 逐字输入太慢；用 fill + 换行处理
        # ProseMirror 需要 <p> 标签分段，直接 innerHTML 更可靠
        # 把 \n 转成 </p><p> 并包裹
        html_body = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        paragraphs = html_body.split("\n")
        html = "".join(f"<p>{p}</p>" for p in paragraphs if p)
        self.page.evaluate(
            "(args) => { const editor = document.querySelector('[contenteditable=\"true\"]:last-of-type') || [...document.querySelectorAll('[contenteditable=\"true\"]')].pop(); editor.innerHTML = args.html; editor.dispatchEvent(new Event('input', {bubbles:true})); }",
            {"html": html},
        )
        # 验证字数
        actual = clean(editor.inner_text())
        expected = clean(body)
        if len(actual) < max(6000, int(len(expected) * 0.95)):
            raise RuntimeError(f"正文写入校验失败：预期{len(expected)}字，实际{len(actual)}字")
        self.log(f"正文已填写（{len(actual)}字）")

    def make_cover(self) -> None:
        """封面制作：选最后一个模板 → 完成制作。"""
        # 点击"封面制作"
        self.click_text("封面制作", timeout=10000)

        # 等待模板列表加载，直接用 .story-template-list-item 选择器
        self.log("等待封面模板加载...")
        template_locator = self.page.locator(".story-template-list-item")
        try:
            template_locator.first.wait_for(state="visible", timeout=30000)
        except PWTimeout:
            self.log("⚠️ 模板列表加载超时")
            return
        count = template_locator.count()
        self.log(f"模板数：{count}")
        if count == 0:
            self.log("⚠️ 没有封面模板，跳过封面")
            return
        # 滚动到最后一个模板并点击
        last_template = template_locator.nth(count - 1)
        last_template.scroll_into_view_if_needed()
        time.sleep(0.5)
        last_template.click()
        self.log(f"已点击最后一个模板（第 {count} 个）")

        # 等待 AI 封面图片生成完成（最后一个模板是 AI 生成，需要时间）
        self.log("等待封面图片生成...")
        deadline = time.time() + 120
        cover_ready = False
        while time.time() < deadline:
            cover_ready = self.page.evaluate("""() => {
                // 检测封面预览：canvas 有内容 或 出现预览 img
                const canvas = document.querySelector('.cover-canvas, canvas.upper-canvas');
                if (canvas && canvas.width > 100 && canvas.height > 100) {
                    // 检查 canvas 是否真的有像素内容（非全透明/全白）
                    try {
                        const ctx = canvas.getContext('2d');
                        const data = ctx.getImageData(canvas.width/2, canvas.height/2, 1, 1).data;
                        if (data[3] > 0) return true; // alpha > 0 表示有内容
                    } catch(e) {}
                }
                // 也检查是否有预览图片
                const preview = document.querySelector('.story-template-list-item.selected img, .cover-preview img, .ai-story-cover-preview img');
                if (preview && preview.complete && preview.naturalWidth > 50) return true;
                return false;
            }""")
            if cover_ready:
                break
            time.sleep(1)
        if cover_ready:
            self.log("封面图片已生成")
        else:
            self.log("⚠️ 封面图片生成超时（120秒），继续尝试完成制作")

        # 等待"完成制作"按钮可用并点击
        finish = self.page.get_by_role("button", name="完成制作")
        try:
            finish.wait_for(state="visible", timeout=60000)
            # 等 disabled 属性消失，额外等 2 秒确保稳定
            for _ in range(120):
                if not finish.is_disabled():
                    break
                time.sleep(0.5)
            time.sleep(2)  # 多等 2 秒确保 React 状态稳定
            # 诊断按钮状态
            btn_state = self.page.evaluate("""() => {
                const btn = [...document.querySelectorAll('button')].find(b => b.textContent.includes('完成制作'));
                if (!btn) return {found: false};
                return {
                    found: true,
                    disabled: btn.disabled,
                    ariaDisabled: btn.getAttribute('aria-disabled'),
                    class: btn.className.slice(0, 80),
                    loading: btn.className.includes('loading') || btn.querySelector('.arco-icon-loading') !== null,
                };
            }""")
            self.log(f"完成制作按钮状态：{btn_state}")
            # 用 evaluate 发完整鼠标事件序列（React 合成事件需要）
            click_result = self.page.evaluate("""() => {
                const btn = [...document.querySelectorAll('button')].find(b => b.textContent.includes('完成制作'));
                if (!btn) return {ok: false};
                const rect = btn.getBoundingClientRect();
                const x = rect.left + rect.width / 2;
                const y = rect.top + rect.height / 2;
                const opts = {bubbles: true, cancelable: true, clientX: x, clientY: y, view: window};
                btn.dispatchEvent(new PointerEvent('pointerdown', opts));
                btn.dispatchEvent(new MouseEvent('mousedown', opts));
                btn.dispatchEvent(new PointerEvent('pointerup', opts));
                btn.dispatchEvent(new MouseEvent('mouseup', opts));
                btn.dispatchEvent(new MouseEvent('click', opts));
                return {ok: true, class: btn.className.slice(0, 60)};
            }""")
            self.log(f"已点击完成制作：{click_result}")
            # 等 3 秒后检查按钮是否进入 loading 状态（表示保存中）
            time.sleep(3)
            loading_state = self.page.evaluate("""() => {
                const btn = [...document.querySelectorAll('button')].find(b => b.textContent.includes('完成制作'));
                if (!btn) return {found: false};
                return {
                    loading: btn.className.includes('loading') || btn.querySelector('.arco-icon-loading') !== null,
                    class: btn.className.slice(0, 80),
                };
            }""")
            self.log(f"完成制作按钮点击后状态：{loading_state}")
        except PWTimeout:
            self.log("⚠️ 完成制作按钮等待超时，跳过封面")

        # 检测"上传成功"提示（番茄会弹一个 toast 提示）
        self.log("等待封面上传成功提示...")
        deadline = time.time() + 120
        upload_success = False
        while time.time() < deadline:
            upload_success = self.page.evaluate("""() => {
                // 检测页面上是否有"上传成功"文字（toast/message/通知）
                const allText = document.body.innerText;
                if (allText.includes('上传成功')) return true;
                // 也检测"保存成功"
                if (allText.includes('保存成功') && allText.includes('封面')) return true;
                // 检查 Arco Message/Toast 组件
                const messages = document.querySelectorAll('.arco-message, .arco-notification, [class*="message"], [class*="toast"], [class*="notice"]');
                for (const msg of messages) {
                    if (msg.offsetParent !== null && msg.textContent.includes('成功')) return true;
                }
                return false;
            }""")
            if upload_success:
                break
            time.sleep(1)

        if upload_success:
            self.log("✅ 封面上传成功")
        else:
            self.log("⚠️ 120秒未检测到上传成功提示，继续")

        # 用 Escape 正常关闭 drawer（不暴力移除 DOM）
        self.page.keyboard.press("Escape")
        time.sleep(2)
        # 如果 drawer 还在，再试一次
        drawer_still_open = self.page.evaluate("""() => {
            const drawer = document.querySelector('.byte-drawer-wrapper');
            if (!drawer) return false;
            const mask = drawer.querySelector('.byte-drawer-mask');
            const content = drawer.querySelector('.byte-drawer-content');
            const maskVisible = mask && getComputedStyle(mask).display !== 'none';
            const contentVisible = content && getComputedStyle(content).display !== 'none';
            return maskVisible || contentVisible;
        }""")
        if drawer_still_open:
            self.log("Escape 未关闭，尝试点击关闭按钮")
            try:
                close_btn = self.page.locator(".byte-drawer-wrapper [class*='close'], .byte-drawer-wrapper button[class*='close']").first
                close_btn.click(timeout=3000)
                time.sleep(1)
            except Exception:
                self.page.keyboard.press("Escape")
                time.sleep(1)
        self.log("封面弹窗已关闭")

    def select_ai(self) -> None:
        """选择"是否使用AI：是"。
        用 evaluate 直接点击，避免被遮罩层拦截。
        """
        clicked = self.page.evaluate("""() => {
            const radios = [...document.querySelectorAll('.arco-radio, [role=radio], .arco-radio-text')];
            const target = radios.find(r => r.textContent.trim() === '是');
            if (!target) return {ok: false, count: radios.length};
            const radio = target.closest('.arco-radio') || target;
            const input = radio.querySelector('input[type=radio]');
            if (input) {
                input.click();
            } else {
                radio.click();
            }
            return {ok: true};
        }""")
        self.log(f"选择使用AI：{clicked}")
        if not clicked.get("ok"):
            try:
                self.page.get_by_text("是", exact=True).first.click(force=True, timeout=5000)
                self.log("已选择使用AI（force click）")
            except Exception:
                self.log("⚠️ 未找到AI选项'是'")
        else:
            self.log("已选择使用AI")

    def dismiss_popup(self) -> None:
        """检测并关闭真正的模态弹窗（有遮罩层阻止页面交互的那种）。
        不碰页面内嵌元素（如底部工具栏）。
        """
        popup_info = self.page.evaluate("""() => {
            const results = [];
            // 只检测真正的模态弹窗：有遮罩层（mask/overlay）阻止页面交互
            // 1. Arco Modal（有 .arco-modal-mask 遮罩）
            document.querySelectorAll('.arco-modal-wrapper').forEach(el => {
                if (el.offsetParent === null) return;
                const mask = document.querySelector('.arco-modal-mask');
                if (!mask || getComputedStyle(mask).display === 'none') return;
                const text = el.textContent.slice(0, 150);
                results.push({type: 'arco-modal', text});
                // 找按钮点击
                const btns = el.querySelectorAll('button');
                for (const btn of btns) {
                    if (btn.textContent.includes('继续编辑')) { btn.click(); return; }
                }
                for (const btn of btns) {
                    if (btn.className.includes('arco-btn-primary')) { btn.click(); return; }
                }
            });
            // 2. 任何有 fixed 定位 + 遮罩层的弹窗
            document.querySelectorAll('div').forEach(el => {
                if (el.offsetParent === null) return;
                const style = getComputedStyle(el);
                if (style.position !== 'fixed') return;
                // 必须有遮罩层特征：z-index 高 + 全屏覆盖
                const z = parseInt(style.zIndex) || 0;
                if (z < 100) return;
                const rect = el.getBoundingClientRect();
                // 遮罩层通常覆盖大部分屏幕
                if (rect.width < window.innerWidth * 0.5 || rect.height < window.innerHeight * 0.3) return;
                const text = el.textContent.trim().slice(0, 150);
                if (!text) return;
                // 必须包含"草稿"或"继续编辑"或"提示"
                if (!text.includes('草稿') && !text.includes('继续编辑') && !text.includes('提示')) return;
                // 不能是编辑器工具栏（工具栏不会有"提示"或"继续编辑"）
                if (text.includes('存草稿') && text.includes('下一步') && !text.includes('提示')) return;
                results.push({type: 'fixed-modal', text, z});
                const btns = el.querySelectorAll('button');
                for (const btn of btns) {
                    if (btn.textContent.includes('继续编辑')) { btn.click(); return; }
                }
                for (const btn of btns) {
                    if (btn.textContent.includes('确定') || btn.textContent.includes('确认')) { btn.click(); return; }
                }
            });
            return results;
        }""")
        if popup_info and len(popup_info) > 0:
            self.log(f"⚠️ 检测到模态弹窗：{popup_info}")
            time.sleep(1)

    def select_tags(self, core_tag: str, tag_groups: dict) -> None:
        """选择作品分类和标签。番茄用 Arco Cascader 级联选择器。"""
        self.log("开始选择分类和标签")

        # 诊断分类区域 DOM
        cat_info = self.page.evaluate("""() => {
            const els = [...document.querySelectorAll('*')];
            return els.filter(el =>
                el.offsetParent !== null &&
                /cascad|select|category|分类/i.test((el.className||'').toString()) &&
                el.children.length < 50
            ).slice(0, 10).map(el => ({
                tag: el.tagName, cls: (el.className||'').toString().slice(0,60),
                text: el.textContent.slice(0,40),
            }));
        }""")
        self.log(f"分类元素：{cat_info}")

        # 点击分类触发器（番茄自定义 class：publish-short-category-select）
        trigger_clicked = False
        for sel in [
            ".publish-short-category-select",
            "[class*='publish-short-category-select']",
            "input.arco-cascader",
            ".arco-select-view",
        ]:
            try:
                el = self.page.locator(sel).first
                el.wait_for(state="visible", timeout=3000)
                el.click()
                trigger_clicked = True
                self.log(f"打开分类下拉（{sel}）")
                break
            except Exception:
                continue
        if not trigger_clicked:
            self.page.evaluate("""() => {
                const el = document.querySelector('.publish-short-category-select');
                if (el) el.click();
            }""")
            self.log("打开分类下拉（evaluate 兜底）")
        time.sleep(2)

        # 诊断下拉面板结构（等面板出现后再查）
        panel_info = self.page.evaluate("""() => {
            // 找弹出的下拉面板
            const panels = [...document.querySelectorAll('[class*="publish-short-category"][class*="panel"], [class*="publish-short-category"][class*="list"], [class*="category-option"], [class*="category-list"]')]
                .filter(el => el.offsetParent !== null);
            if (panels.length === 0) {
                // 退回通用：找所有新出现的可见元素
                return [...document.querySelectorAll('[class*="option"], [class*="list-item"], [role=option]')]
                    .filter(el => el.offsetParent !== null && el.textContent.trim().length < 25)
                    .slice(0, 15)
                    .map(el => ({cls:(el.className||'').toString().slice(0,50), text:el.textContent.trim().slice(0,20)}));
            }
            return panels.slice(0, 5).map(p => ({
                cls: (p.className||'').toString().slice(0,60),
                text: p.textContent.slice(0,80),
                childTags: [...p.querySelectorAll('*')].slice(0,3).map(c=>c.tagName),
            }));
        }""")
        self.log(f"下拉面板：{panel_info}")

        # 通用选项点击
        def click_option(text, timeout=10000):
            deadline = time.time() + timeout / 1000
            while time.time() < deadline:
                r = self.page.evaluate("""(text) => {
                    // 番茄自定义：选项在 .publish-short-category-select-item-list > div 里
                    // 标签（主分类/情节等）在 .publish-short-category-select-label-list > div 里
                    const sels = '.publish-short-category-select-item-list > div, .publish-short-category-select-label-list > div, [class*="publish-short-category-select"] [class*="item"], [class*="publish-short-category-select"] [class*="label"]';
                    const opts = [...document.querySelectorAll(sels)]
                        .filter(el => el.offsetParent !== null && el.textContent.trim().length < 25);
                    const t = opts.find(o => o.textContent.trim() === text || o.textContent.trim().includes(text));
                    if (!t) return {ok:false, cnt:opts.length, sample: opts.slice(0,3).map(o=>o.textContent.trim().slice(0,10))};
                    t.click();
                    return {ok:true, text:t.textContent.trim().slice(0,20)};
                }""", text)
                if r.get("ok"):
                    return r
                time.sleep(0.3)
            return {"ok": False, "err": f"timeout: {text}"}

        def ensure_panel_open():
            """确保分类下拉面板是打开的，如果收起了就重新打开。"""
            panel_visible = self.page.evaluate("""() => {
                const labelList = document.querySelector('.publish-short-category-select-label-list');
                const itemList = document.querySelector('.publish-short-category-select-item-list');
                const labelVisible = labelList && labelList.offsetParent !== null;
                const itemVisible = itemList && itemList.offsetParent !== null;
                return labelVisible || itemVisible;
            }""")
            if not panel_visible:
                self.log("分类面板已收起，重新打开...")
                self.page.evaluate("""() => {
                    const el = document.querySelector('.publish-short-category-select');
                    if (el) el.click();
                }""")
                time.sleep(1)

        # 分类标签点击（主分类/情节/角色/情绪/背景）：精确匹配 label-list 单个子 div
        def click_label(text, timeout=8000):
            deadline = time.time() + timeout / 1000
            while time.time() < deadline:
                ensure_panel_open()
                r = self.page.evaluate("""(text) => {
                    // label-list 的直接子 div，每个是一个分类标签
                    const labels = [...document.querySelectorAll('.publish-short-category-select-label-list > div')];
                    const t = labels.find(d => d.textContent.trim() === text);
                    if (!t) return {ok:false, cnt:labels.length, texts: labels.map(d=>d.textContent.trim().slice(0,6))};
                    t.click();
                    return {ok:true};
                }""", text)
                if r.get("ok"):
                    return r
                time.sleep(0.3)
            return {"ok": False, "err": f"label timeout: {text}", "labels": r.get("texts", [])}

        # 选项点击（item-list 里的具体标签）
        def click_item(text, timeout=8000):
            deadline = time.time() + timeout / 1000
            while time.time() < deadline:
                ensure_panel_open()
                r = self.page.evaluate("""(text) => {
                    const items = [...document.querySelectorAll('.publish-short-category-select-item-list > div')];
                    const t = items.find(d => d.textContent.trim() === text || d.textContent.trim().includes(text));
                    if (!t) return {ok:false, cnt:items.length, sample: items.slice(0,5).map(d=>d.textContent.trim().slice(0,8))};
                    t.click();
                    return {ok:true, text:t.textContent.trim().slice(0,15)};
                }""", text)
                if r.get("ok"):
                    return r
                time.sleep(0.3)
            return {"ok": False, "err": f"item timeout: {text}", "items": r.get("sample", [])}

        # 主分类 → 选 core_tag
        r = click_label("主分类"); self.log(f"切主分类标签：{r}"); time.sleep(0.5)
        r = click_item(core_tag); self.log(f"选{core_tag}：{r}"); time.sleep(0.5)

        # 辅助标签：每个 group 先切标签再选项
        for group in ["情节", "角色", "情绪", "背景"]:
            for tag in (tag_groups.get(group) or []):
                r = click_label(group); self.log(f"切{group}标签：{r}"); time.sleep(0.3)
                r = click_item(tag); self.log(f"选{tag}：{r}"); time.sleep(0.3)
        self.page.keyboard.press("Escape")
        time.sleep(0.5)
        self.log("标签选择完成")

    def save_draft(self) -> None:
        """点击存草稿（仅批量模式）。用 evaluate 直接点击，避免被遮罩层拦截。"""
        clicked = self.page.evaluate("""() => {
            const btn = document.querySelector('[data-apm-action="core_chain_short_story_save_draft"]')
                || [...document.querySelectorAll('button')].find(b => b.textContent.includes('存草稿'));
            if (!btn) return {ok: false};
            btn.click();
            return {ok: true};
        }""")
        self.log(f"点击存草稿：{clicked}")
        if not clicked.get("ok"):
            try:
                self.page.locator("button:has-text('存草稿')").first.click(force=True, timeout=5000)
                self.log("存草稿（force click）")
            except Exception:
                self.log("⚠️ 未找到存草稿按钮")
                return
        # 等待保存成功
        try:
            self.page.wait_for_selector("text=保存成功", timeout=20000)
            self.log("✅ 草稿已保存")
        except PWTimeout:
            self.log("⚠️ 未检测到保存成功提示")

    def execute(self, task: dict) -> dict:
        """执行完整流程。返回结果字典。"""
        try:
            # 1. 进入短故事管理页
            self.log(f"开始执行：{task['title']}")
            self.page.goto(MANAGE_URL, wait_until="networkidle")

            # 2. 点击"新建短故事"——先诊断页面按钮
            context = self.page.context
            # 监听新标签页打开
            popup_page_holder = []
            def on_popup(popup):
                popup_page_holder.append(popup)
            context.on("page", on_popup)

            # 诊断：列出所有按钮文本
            btn_info = self.page.evaluate("""() => {
                const btns = [...document.querySelectorAll('button, [role=button], a')];
                return btns.filter(b => b.offsetParent !== null).slice(0, 20).map(b => ({
                    tag: b.tagName, text: b.textContent.slice(0,30), cls: b.className.slice(0,40),
                    rect: {x: Math.round(b.getBoundingClientRect().x), y: Math.round(b.getBoundingClientRect().y), w: Math.round(b.getBoundingClientRect().width)}
                }));
            }""")
            self.log(f"页面按钮数：{len(btn_info)}")
            for b in btn_info:
                if "新建" in b.get("text","") or "短故事" in b.get("text",""):
                    self.log(f"  新建按钮：{b}")

            # 用 Playwright 原生 click（模拟真实鼠标移动+点击）
            btn = self.page.locator("button.arco-btn-primary:has-text('新建短故事')").first
            btn.wait_for(state="visible", timeout=10000)
            # 方案1: locator.click()（Playwright 原生，最可靠）
            try:
                btn.click(timeout=5000)
                self.log("已点击新建短故事按钮（locator.click）")
            except Exception:
                # 方案2: 坐标点击
                box = btn.bounding_box()
                if box:
                    self.page.mouse.click(box["x"] + box["width"]/2, box["y"] + box["height"]/2)
                    self.log(f"已点击新建短故事按钮（坐标 {box['x']:.0f},{box['y']:.0f}）")

            self.log("等待编辑器页面")

            # 3. 等待编辑器页面：可能新标签页或同页跳转或弹窗内加载
            # 番茄是 SPA，可能客户端路由跳转但 URL 不含 publish-short
            # 用多种信号检测：新 tab、URL 变化、标题输入框出现
            context = self.page.context
            new_page = None
            deadline = time.time() + 30
            while time.time() < deadline:
                # 检查 popup 监听捕获的新标签页
                if popup_page_holder:
                    new_page = popup_page_holder[0]
                    break
                # 检查所有标签页
                for p in context.pages:
                    if "/publish-short/" in p.url or "/writer/publish" in p.url:
                        new_page = p
                        break
                if new_page:
                    break
                try:
                    cur_url = self.page.url
                    if "/publish-short/" in cur_url or "/writer/publish" in cur_url:
                        new_page = self.page
                        break
                except Exception:
                    pass
                # 检查当前页是否出现了标题输入框
                try:
                    if self.page.locator(
                        "textarea[placeholder*='短故事名称'], [contenteditable][data-placeholder*='短故事名称']"
                    ).is_visible(timeout=500):
                        new_page = self.page
                        self.log("当前页已出现标题输入框")
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            try:
                context.remove_listener("page", on_popup)
            except Exception:
                pass

            # 诊断：截图 + 列出所有 tab
            self.log(f"当前 tab 数：{len(context.pages)}")
            for i, p in enumerate(context.pages):
                self.log(f"  tab[{i}]：{p.url}")
            # 截图当前页看发生了什么
            try:
                shot_path = str(Path(tempfile.gettempdir()) / "fanqie-after-click.png")
                self.page.screenshot(path=shot_path, full_page=True)
                self.log(f"截图已保存：{shot_path}")
            except Exception as e:
                self.log(f"截图失败：{e}")
            # 检查页面是否有弹窗/对话框（如"有刚刚更新的草稿，是否继续编辑？"）
            dialogs = self.page.locator("[role='dialog'], [aria-modal='true'], .arco-modal, [class*='modal'], [class*='dialog']")
            try:
                dialog_count = dialogs.count()
                self.log(f"弹窗数：{dialog_count}")
                if dialog_count > 0:
                    for i in range(min(dialog_count, 3)):
                        text = dialogs.nth(i).inner_text()[:200]
                        self.log(f"  弹窗[{i}]：{text}")
                    # 尝试点击弹窗中的确认按钮
                    for i in range(dialog_count):
                        dlg = dialogs.nth(i)
                        if "草稿" in dlg.inner_text() or "继续编辑" in dlg.inner_text():
                            # 找确认按钮（通常是"继续编辑"或"确定"）
                            confirm_btn = dlg.locator("button:has-text('继续编辑'), button:has-text('确定'), button:has-text('确认'), .arco-btn-primary").first
                            try:
                                confirm_btn.click(timeout=3000)
                                self.log(f"已点击弹窗确认按钮")
                                time.sleep(1)
                            except Exception:
                                # 兜底：用 evaluate 点所有可见按钮
                                self.page.evaluate("""() => {
                                    const modal = document.querySelector('.arco-modal-wrapper:not([style*="display: none"])');
                                    if (modal) {
                                        const btns = modal.querySelectorAll('button');
                                        for (const btn of btns) {
                                            if (btn.textContent.includes('继续编辑') || btn.textContent.includes('确定') || btn.className.includes('arco-btn-primary')) {
                                                btn.click();
                                                break;
                                            }
                                        }
                                    }
                                }""")
                                self.log("已点击弹窗确认按钮（evaluate）")
                                time.sleep(1)
            except Exception as e:
                self.log(f"弹窗检测异常：{e}")

            if new_page:
                self.page = new_page
                self.page.wait_for_load_state("networkidle", timeout=20000)
                self.log(f"编辑器页面已就绪：{self.page.url}")

                # 编辑器加载后也检查草稿确认弹窗
                time.sleep(2)
                try:
                    modal = self.page.locator(".arco-modal-wrapper:visible")
                    if modal.count() > 0:
                        modal_text = modal.first.inner_text()[:100]
                        self.log(f"编辑器页检测到弹窗：{modal_text}")
                        if "草稿" in modal_text or "继续编辑" in modal_text:
                            btn = modal.locator("button:has-text('继续编辑'), button:has-text('确定'), .arco-btn-primary").first
                            btn.click(timeout=3000)
                            self.log("已点击编辑器弹窗确认按钮")
                            time.sleep(1)
                except Exception:
                    pass
            else:
                # 没找到 publish-short，尝试在当前页找标题输入框
                self.log("未检测到 publish-short URL，尝试在当前页找编辑器")
                try:
                    self.page.wait_for_selector(
                        "textarea[placeholder*='短故事名称'], [contenteditable][data-placeholder*='短故事名称']",
                        timeout=5000,
                    )
                    self.log("当前页找到标题输入框，继续")
                except Exception:
                    raise RuntimeError("点击新建短故事后未跳转到编辑器，也未找到标题输入框")

            # 等待标题输入框出现
            self.page.wait_for_selector(
                "textarea[placeholder*='短故事名称'], [contenteditable][data-placeholder*='短故事名称']",
                timeout=20000,
            )

            # 4. 填写标题和正文
            self.fill_title(task["title"])
            self.fill_body(task["body"])

            # 5. 封面
            self.make_cover()

            # 6. AI
            self.select_ai()

            # 7. 标签
            self.select_tags(task["core_tag"], task.get("tag_groups", {}))

            # 8. 校准模式停在这里，批量模式点存草稿
            if self.mode == "calibration":
                self.log("✅ 校准模式：已停在存草稿前，请人工核对")
                return {"status": "awaiting_calibration", "url": self.page.url}
            else:
                self.save_draft()
                return {"status": "completed", "url": self.page.url}

        except Exception as e:
            self.log(f"❌ 执行失败：{e}")
            return {"status": "failed", "error": str(e)}


def connect_chrome() -> Browser:
    """连接到已启动的 Chrome（需带 --remote-debugging-port=9222）。"""
    pw = sync_playwright().start()
    browser = pw.connect_over_cdp(CDP_URL)
    return browser


def launch_chrome_with_login() -> tuple:
    """复制用户登录态到临时目录，启动 Chrome。
    Chrome 安全策略禁止在默认 User Data 上开 DevTools，必须用独立目录。
    """
    import os
    import shutil
    import tempfile

    pw = sync_playwright().start()
    src_user_data = os.path.join(
        os.environ["LOCALAPPDATA"], "Google", "Chrome", "User Data"
    )
    temp_user_data = os.path.join(tempfile.gettempdir(), "fanqie-chrome-debug")
    # 只在临时目录不存在时才复制（后续复用已有登录态）
    need_copy = not os.path.exists(os.path.join(temp_user_data, "Local State"))
    if need_copy:
        print("复制登录态到临时目录...")
        if os.path.exists(temp_user_data):
            shutil.rmtree(temp_user_data, ignore_errors=True)
        os.makedirs(temp_user_data, exist_ok=True)
        # 复制 Local State（含加密密钥）
        local_state = os.path.join(src_user_data, "Local State")
        if os.path.exists(local_state):
            shutil.copy2(local_state, os.path.join(temp_user_data, "Local State"))
        # 复制 Default profile
        src_default = os.path.join(src_user_data, "Default")
        dst_default = os.path.join(temp_user_data, "Default")
        os.makedirs(dst_default, exist_ok=True)
        for fname in [
            "Login Data", "Login Data-journal",
            "Web Data", "Web Data-journal",
            "Preferences",
        ]:
            src = os.path.join(src_default, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst_default, fname))
        # Chrome 120+ Cookies 在 Default/Network/ 目录下
        src_network = os.path.join(src_default, "Network")
        dst_network = os.path.join(dst_default, "Network")
        if os.path.isdir(src_network):
            os.makedirs(dst_network, exist_ok=True)
            for fname in os.listdir(src_network):
                src = os.path.join(src_network, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(dst_network, fname))
        print("登录态复制完成")
    else:
        print("使用已有临时登录态")

    context = pw.chromium.launch_persistent_context(
        user_data_dir=temp_user_data,
        channel="chrome",
        headless=False,
        args=[
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-popup-blocking",
        ],
        viewport={"width": 1280, "height": 800},
    )
    return pw, context


def run_task(task: dict, mode: str = "calibration") -> dict:
    """启动带登录态的 Chrome 并执行单个任务。"""
    pw, context = launch_chrome_with_login()
    try:
        page = context.new_page()
        executor = FanqieExecutor(page, mode)
        result = executor.execute(task)
        result["logs"] = executor.logs
        return result
    finally:
        context.close()
        pw.stop()


if __name__ == "__main__":
    # 手动测试入口
    from .catalog import Catalog

    catalog = Catalog(PROJECT_ROOT)
    items = [i for i in catalog.scan() if i.valid]
    if not items:
        print("没有可上传的小说")
        raise SystemExit(1)

    item = items[0]
    print(f"使用第一篇小说：{item.title}")

    task = {
        "title": item.title,
        "body": catalog.body(item.item_id),
        "core_tag": item.core_tag,
        "tag_groups": item.tag_groups,
    }

    result = run_task(task, mode="calibration")
    print(f"\n结果：{result.get('status')}")
    print(f"URL：{result.get('url', '')}")
    if "error" in result:
        print(f"错误：{result['error']}")
