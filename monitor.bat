REM ======================  
REM monitor.bat  
REM 监控脚本的Windows启动器  
REM ======================  

:: 切换到当前脚本所在目录（可避免路径不一致导致的执行错误）  
cd /d "%~dp0"  

REM 关闭快速编辑模式  
reg add "HKCU\Console" /v QuickEdit /t REG_DWORD /d 0 /f  

echo [信息] 正在启动量化交易脚本...  
"%~dp0venv\Scripts\python" main.py  

if %ERRORLEVEL% NEQ 0 (  
echo [错误] 量化交易脚本异常退出，错误代码: %ERRORLEVEL%  
pause  
exit /b %ERRORLEVEL%  
)  

echo [信息] 量化交易脚本已成功完成。  
pause