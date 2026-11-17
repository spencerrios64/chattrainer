@echo off
cd /d "%~dp0"
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" 2>nul
if errorlevel 1 goto nopython
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip
.venv\Scripts\python.exe -m pip install -e .
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\chattrainer.lnk');$s.TargetPath='%~dp0run.bat';$s.WorkingDirectory='%~dp0';$s.Save()"
echo.
echo Installed. A chattrainer shortcut is on your desktop (or double-click run.bat).
echo In the app, open the Train tab and press "Install training components" once.
pause
exit /b 0
:nopython
echo Python 3.11 or newer is required. Install it with:  winget install Python.Python.3.12
echo or from https://www.python.org/downloads/ (tick "Add python.exe to PATH"), then run this again.
pause
exit /b 1
