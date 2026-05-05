@echo off
:: @claude last-modified: 2026-05-05T06:40:00Z
:: @claude last-commit: chore: remove web UI, start scripts launch TUI
setlocal

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "VENV_DIR=%SCRIPT_DIR%\.venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [start] ERROR: Virtual environment not found. Run install.bat first.
    pause & exit /b 1
)

"%PYTHON%" "%SCRIPT_DIR%\tui.py"
endlocal
