@echo off
setlocal
:: start.bat — Launch the Apollo Web UI (Windows)

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "VENV_DIR=%SCRIPT_DIR%\.venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [start] ERROR: Virtual environment not found. Run install.bat first.
    pause & exit /b 1
)

echo.
echo   Apollo Web UI -- starting
echo   Open http://127.0.0.1:5000 in your browser
echo.

"%PYTHON%" "%SCRIPT_DIR%\webui.py"
endlocal
