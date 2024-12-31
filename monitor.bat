REM ======================
REM monitor.bat
REM 用于在 Windows 上启动监控脚本
REM ======================

:: 切换到当前脚本所在目录（可避免路径不一致导致的执行错误）
cd /d "%~dp0"

REM 关闭快速编辑模式
reg add "HKCU\Console" /v QuickEdit /t REG_DWORD /d 0 /f

echo [INFO] Starting monitor script...
D:\dlquant3\venv\Scripts\python main.py

if %ERRORLEVEL% NEQ 0 (
echo [ERROR] The monitor script exited with an error code %ERRORLEVEL%.
pause
exit /b %ERRORLEVEL%
)

echo [INFO] Monitor script finished successfully.
pause

