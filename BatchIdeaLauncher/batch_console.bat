@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul || (
  echo 无法进入批量工具目录：%SCRIPT_DIR%
  pause
  exit /b 1
)

if defined AUTOWRITE_PYTHON (
  set "PY=%AUTOWRITE_PYTHON%"
) else (
  set "PY=python"
)

"%PY%" --version >nul 2>nul
if errorlevel 1 (
  echo 找不到 Python：%PY%
  echo 请先安装 Python，或设置 AUTOWRITE_PYTHON 指向 python.exe。
  echo.
  pause
  popd
  exit /b 1
)

:menu
cls
echo ================================================
echo AutoWrite 批量创作控制台
echo ================================================
echo.
echo  1. 开始新批次（默认双并发）
echo  2. 查看批次状态
echo  3. 只重试失败任务
echo  4. 续跑未完成/失败任务
echo  5. 刷新能力表 catalog
echo  0. 退出
echo.
set "CHOICE="
set /p "CHOICE=请选择："
if errorlevel 1 goto done

if "%CHOICE%"=="1" goto run_batch
if "%CHOICE%"=="2" goto status_batch
if "%CHOICE%"=="3" goto retry_failed
if "%CHOICE%"=="4" goto retry_all
if "%CHOICE%"=="5" goto catalog
if "%CHOICE%"=="0" goto done
goto menu

:run_batch
call :default_paths
echo.
echo 点子文件默认：%DEFAULT_IDEAS%
set "IDEAS="
set /p "IDEAS=点子文件路径（回车使用默认）："
if errorlevel 1 goto done
if "%IDEAS%"=="" set "IDEAS=%DEFAULT_IDEAS%"
set "IDEAS=%IDEAS:"=%"

echo.
echo 配置文件默认：%DEFAULT_CONFIG%
set "CONFIG="
set /p "CONFIG=配置文件路径（回车使用默认）："
if errorlevel 1 goto done
if "%CONFIG%"=="" set "CONFIG=%DEFAULT_CONFIG%"
set "CONFIG=%CONFIG:"=%"

echo.
set "WORKERS=2"
set "WORKERS_INPUT="
set /p "WORKERS_INPUT=并发篇数（回车=2，串行=1）："
if errorlevel 1 goto done
if not "%WORKERS_INPUT%"=="" set "WORKERS=%WORKERS_INPUT%"

if not exist "%IDEAS%" (
  echo.
  echo 找不到点子文件：%IDEAS%
  call :hold
  goto menu
)
if not exist "%CONFIG%" (
  echo.
  echo 找不到配置文件：%CONFIG%
  call :hold
  goto menu
)

echo.
echo 即将启动新批次，workers=%WORKERS%
echo.
"%PY%" launcher.py run --ideas "%IDEAS%" --config "%CONFIG%" --workers "%WORKERS%"
call :after_command
goto menu

:status_batch
call :ask_batch_id
if errorlevel 1 goto menu
echo.
"%PY%" launcher.py status --batch-id "%BATCH_ID%"
call :after_command
goto menu

:retry_failed
call :ask_batch_id
if errorlevel 1 goto menu
call :ask_workers
echo.
"%PY%" launcher.py retry --batch-id "%BATCH_ID%" --failed-only --workers "%WORKERS%"
call :after_command
goto menu

:retry_all
call :ask_batch_id
if errorlevel 1 goto menu
call :ask_workers
echo.
"%PY%" launcher.py retry --batch-id "%BATCH_ID%" --workers "%WORKERS%"
call :after_command
goto menu

:catalog
echo.
"%PY%" launcher.py catalog
call :after_command
goto menu

:default_paths
set "DEFAULT_IDEAS=%SCRIPT_DIR%ideas.csv"
if not exist "%DEFAULT_IDEAS%" set "DEFAULT_IDEAS=%SCRIPT_DIR%ideas.example.csv"
set "DEFAULT_CONFIG=%SCRIPT_DIR%batch_config.json"
if not exist "%DEFAULT_CONFIG%" set "DEFAULT_CONFIG=%SCRIPT_DIR%batch_config.example.json"
exit /b 0

:ask_batch_id
echo.
set "BATCH_ID="
set /p "BATCH_ID=批次 ID（例如 batch-20260622-120000）："
if errorlevel 1 goto done
if "%BATCH_ID%"=="" (
  echo 批次 ID 不能为空。
  call :hold
  exit /b 1
)
exit /b 0

:ask_workers
echo.
set "WORKERS=2"
set "WORKERS_INPUT="
set /p "WORKERS_INPUT=并发篇数（回车=2，串行=1）："
if errorlevel 1 goto done
if not "%WORKERS_INPUT%"=="" set "WORKERS=%WORKERS_INPUT%"
exit /b 0

:after_command
set "LAST_CODE=%ERRORLEVEL%"
echo.
if not "%LAST_CODE%"=="0" echo 命令退出码：%LAST_CODE%
call :hold
exit /b 0

:hold
echo.
pause
exit /b 0

:done
popd
endlocal
exit /b 0
