@echo off
setlocal EnableDelayedExpansion
:: apollo.bat — install (if needed) then run an Apollo command or drop into a shell

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "VENV_DIR=%SCRIPT_DIR%\.venv"

:: 1. Locate or install uv
set "UV_BIN="
where uv >nul 2>&1 && for /f "delims=" %%i in ('where uv') do set "UV_BIN=%%i" & goto :uv_found
if exist "%USERPROFILE%\.local\bin\uv.exe"  set "UV_BIN=%USERPROFILE%\.local\bin\uv.exe" & goto :uv_found
if exist "%USERPROFILE%\.cargo\bin\uv.exe"  set "UV_BIN=%USERPROFILE%\.cargo\bin\uv.exe" & goto :uv_found
if exist "%LOCALAPPDATA%\uv\uv.exe"         set "UV_BIN=%LOCALAPPDATA%\uv\uv.exe"        & goto :uv_found

echo [apollo] uv not found -- installing...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if %errorlevel% neq 0 ( echo [apollo] ERROR: Could not install uv. & pause & exit /b 1 )
if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_BIN=%USERPROFILE%\.local\bin\uv.exe" & goto :uv_found
if exist "%LOCALAPPDATA%\uv\uv.exe"        set "UV_BIN=%LOCALAPPDATA%\uv\uv.exe"        & goto :uv_found
where uv >nul 2>&1 && for /f "delims=" %%i in ('where uv') do set "UV_BIN=%%i" & goto :uv_found
echo [apollo] ERROR: uv installed but not found. Restart terminal and retry.
pause & exit /b 1

:uv_found
echo [apollo] uv: %UV_BIN%

:: 2. Create venv if missing
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [apollo] venv exists
) else (
    echo [apollo] Creating virtual environment (Python 3.11^)...
    "%UV_BIN%" venv "%VENV_DIR%" --python 3.11
    if %errorlevel% neq 0 ( echo [apollo] ERROR: venv creation failed. & pause & exit /b 1 )
)

:: 3. Install / sync dependencies
echo [apollo] Installing PyTorch (CUDA 12.1^)...
"%UV_BIN%" pip install ^
    --python "%VENV_DIR%\Scripts\python.exe" ^
    --index-url https://download.pytorch.org/whl/cu121 ^
    torch==2.1.2+cu121 torchaudio==2.1.2+cu121
if %errorlevel% neq 0 ( echo [apollo] ERROR: PyTorch install failed. & pause & exit /b 1 )
echo [apollo] PyTorch installed

echo [apollo] Syncing remaining dependencies...
"%UV_BIN%" pip install ^
    --python "%VENV_DIR%\Scripts\python.exe" ^
    "setuptools<71" pyyaml ^
    "bitsandbytes>=0.43.0" ^
    -r "%SCRIPT_DIR%\requirements.txt"
if %errorlevel% neq 0 ( echo [apollo] ERROR: Dependency install failed. & pause & exit /b 1 )
echo [apollo] Dependencies up to date

:: 4. If arguments given, treat first as the script name and run it
if not "%*"=="" (
    for /f "tokens=1,* delims= " %%a in ("%*") do set "CMD=%%a" & set "REST=%%b"
    if /i "!CMD!"=="train"     set "SCRIPT=train.py"
    if /i "!CMD!"=="inference" set "SCRIPT=inference.py"
    if /i "!CMD!"=="test"      set "SCRIPT=test.py"
    if defined SCRIPT (
        "%VENV_DIR%\Scripts\python.exe" "%SCRIPT_DIR%\!SCRIPT!" !REST!
    ) else if /i "!CMD!"=="python" (
        "%VENV_DIR%\Scripts\python.exe" !REST!
    ) else (
        "%VENV_DIR%\Scripts\python.exe" "%SCRIPT_DIR%\!CMD!" !REST!
    )
    exit /b %errorlevel%
)

:: 5. No arguments — drop into an activated cmd session with command aliases
set "INIT_FILE=%TEMP%\apollo_init_%RANDOM%.bat"
(
    echo @echo off
    echo doskey train=python "%SCRIPT_DIR%\train.py" $*
    echo doskey inference=python "%SCRIPT_DIR%\inference.py" $*
    echo doskey test=python "%SCRIPT_DIR%\test.py" $*
    echo cd /d "%SCRIPT_DIR%"
) > "%INIT_FILE%"

echo.
echo   Apollo ready. Type 'exit' to leave.
echo   Commands: train, inference, test (or python directly)
echo.

call "%VENV_DIR%\Scripts\activate.bat"
cmd /k "%INIT_FILE%"