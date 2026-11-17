@echo off
setlocal
cd /d "%~dp0"
set PYCMD=
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul && set PYCMD=py -3
if not defined PYCMD python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul && set PYCMD=python
if not defined PYCMD goto nopython
echo Using Python: %PYCMD%
%PYCMD% -m venv .venv || goto failed
.venv\Scripts\python.exe -m pip install -U pip || goto failed
.venv\Scripts\python.exe -m pip install -e . || goto failed
.venv\Scripts\python.exe -c "import tkinter" >nul 2>nul || goto notk
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\chattrainer.lnk');$s.TargetPath='%~dp0run.bat';$s.WorkingDirectory='%~dp0';$s.Save()"
echo.
echo ==================================================
echo  SUCCESS: chattrainer is installed.
echo  Start it from the "chattrainer" shortcut on your desktop,
echo  or double-click run.bat in this folder.
echo  Then open the Train tab and press "Install training components" once.
echo ==================================================
pause
exit /b 0
:failed
echo.
echo FAILED. Scroll up to read the error above, then send it to me.
pause
exit /b 1
:notk
echo.
echo FAILED: this Python has no Tk (needed for the app window).
echo Re-run the Python installer from python.org, choose Modify, and tick "tcl/tk and IDLE".
pause
exit /b 1
:nopython
echo Python 3.11 or newer was not found. Install it with:  winget install Python.Python.3.12
echo or from https://www.python.org/downloads/ (tick "Add python.exe to PATH"), then run install.bat again.
pause
exit /b 1
