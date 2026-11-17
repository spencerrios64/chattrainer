@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe goto noinstall
.venv\Scripts\python.exe -m chattrainer gui
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo chattrainer failed to start. Read the error above and send it to me.
pause
exit /b 1
:noinstall
echo Run install.bat first (it must finish with a SUCCESS message).
pause
exit /b 1
