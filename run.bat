@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe goto noinstall
start "" .venv\Scripts\pythonw.exe -m chattrainer gui
exit /b 0
:noinstall
echo Run install.bat first.
pause
