@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_model2.ps1"
set "model2_exit=%ERRORLEVEL%"
echo.
if not "%model2_exit%"=="0" echo Model 2 startup failed. Review the error and log path above.
pause
exit /b %model2_exit%
