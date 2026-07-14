@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
echo ========================================
echo   番茄小说批量录入草稿 — 启动
echo ========================================
echo.
echo 前提条件：
echo   1. 所有 Chrome 窗口必须先关闭（避免登录态锁定）
echo   2. 番茄作者后台已登录过（登录态保存在本地）
echo   3. Novel/ 和 Outline/ 目录有有效配对文件
echo.
echo 正在关闭已打开的 Chrome...
taskkill /F /IM chrome.exe >nul 2>&1
timeout /t 2 /nobreak >nul
echo.
echo 启动批量录入...
python -m FanqieUploader.playwright_batch
echo.
echo ========================================
echo   录入完成，按任意键退出
echo ========================================
pause >nul