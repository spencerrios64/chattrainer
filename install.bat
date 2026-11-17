@echo off
setlocal
cd /d "%~dp0"
rem If this file was downloaded on its own, fetch the whole project first.
if exist pyproject.toml goto havecode
echo Project files not found next to install.bat - downloading chattrainer...
set DEST=%USERPROFILE%\chattrainer
if exist "%DEST%\pyproject.toml" goto usedest
powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest https://github.com/spencerrios64/chattrainer/archive/refs/heads/main.zip -OutFile $env:TEMP\chattrainer.zip; Expand-Archive -Force $env:TEMP\chattrainer.zip $env:TEMP\chattrainer_x; if (Test-Path $env:USERPROFILE\chattrainer) { Remove-Item -Recurse -Force $env:USERPROFILE\chattrainer }; Move-Item $env:TEMP\chattrainer_x\chattrainer-main $env:USERPROFILE\chattrainer" || goto failed
:usedest
cd /d "%DEST%"
:havecode
set PYCMD=
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul && set PYCMD=py -3
if not defined PYCMD python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul && set PYCMD=python
if not defined PYCMD goto nopython
echo Using Python: %PYCMD%
echo Installing in: %CD%
if exist .venv rmdir /s /q .venv
%PYCMD% -m venv .venv || goto failed
.venv\Scripts\python.exe -m pip install -U pip || goto failed
.venv\Scripts\python.exe -m pip install -e . || goto failed
.venv\Scripts\python.exe -c "import tkinter" >nul 2>nul || goto notk
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\chattrainer.lnk');$s.TargetPath='%CD%\run.bat';$s.WorkingDirectory='%CD%';$s.Save()"
echo.
echo ==================================================
echo  SUCCESS: chattrainer is installed in %CD%
echo  Start it from the "chattrainer" shortcut on your desktop.
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
