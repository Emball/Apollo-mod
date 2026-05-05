@echo off
setlocal EnableDelayedExpansion
:: install.bat — Apollo installer (Windows)

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "VENV_DIR=%SCRIPT_DIR%\.venv"

echo.
echo   +==================================+
echo   ^|     Apollo -- Installer          ^|
echo   +==================================+
echo.

:: ── 1. Locate or install uv ─────────────────────────────────────────────────
set "UV_BIN="
where uv >nul 2>&1
if %errorlevel% == 0 (
    for /f "delims=" %%i in ('where uv') do set "UV_BIN=%%i" & goto :uv_found
)
if exist "%USERPROFILE%\.local\bin\uv.exe"  ( set "UV_BIN=%USERPROFILE%\.local\bin\uv.exe" & goto :uv_found )
if exist "%USERPROFILE%\.cargo\bin\uv.exe"  ( set "UV_BIN=%USERPROFILE%\.cargo\bin\uv.exe" & goto :uv_found )
if exist "%LOCALAPPDATA%\uv\uv.exe"         ( set "UV_BIN=%LOCALAPPDATA%\uv\uv.exe"        & goto :uv_found )

echo [install] uv not found -- installing via PowerShell...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if %errorlevel% neq 0 ( echo [install] ERROR: Could not install uv. & pause & exit /b 1 )

if exist "%USERPROFILE%\.local\bin\uv.exe" ( set "UV_BIN=%USERPROFILE%\.local\bin\uv.exe" & goto :uv_found )
if exist "%LOCALAPPDATA%\uv\uv.exe"        ( set "UV_BIN=%LOCALAPPDATA%\uv\uv.exe"        & goto :uv_found )
where uv >nul 2>&1
if %errorlevel% == 0 ( for /f "delims=" %%i in ('where uv') do set "UV_BIN=%%i" & goto :uv_found )
echo [install] ERROR: uv installed but not found. Restart terminal and retry.
pause & exit /b 1

:uv_found
echo [install] Found uv: %UV_BIN%

:: ── 2. Create virtual environment ───────────────────────────────────────────
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [install] Virtual environment already exists -- skipping
) else (
    echo [install] Creating virtual environment with Python 3.11...
    "%UV_BIN%" venv "%VENV_DIR%" --python 3.11
    if %errorlevel% neq 0 ( echo [install] ERROR: venv creation failed. & pause & exit /b 1 )
)

:: ── 3. Install dependencies ──────────────────────────────────────────────────
echo [install] Installing dependencies...
"%UV_BIN%" pip install ^
    --python "%VENV_DIR%\Scripts\python.exe" ^
    "setuptools<71" ^
    pyyaml ^
    -r "%SCRIPT_DIR%\requirements.txt"
if %errorlevel% neq 0 ( echo [install] ERROR: Dependency install failed. & pause & exit /b 1 )
echo [install] All dependencies installed

echo.
echo   Installation complete!
echo   Run start.bat to launch the Apollo Web UI
echo.
pause
endlocal
