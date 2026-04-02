@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\stop_windows.ps1" %*
exit /b %ERRORLEVEL%