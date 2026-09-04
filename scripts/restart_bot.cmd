@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0restart_bot.ps1"
set "restart_exit=%ERRORLEVEL%"
echo.
if not "%restart_exit%"=="0" echo Restart failed. Review the error and log path above.
pause
exit /b %restart_exit%
