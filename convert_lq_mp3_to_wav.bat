@echo off
setlocal enabledelayedexpansion

REM Trim 1057 samples of iTunes encoder delay from MP3 LQ files and save as WAV.
REM Run from the Apollo-mod directory:
REM   convert_lq_mp3_to_wav.bat data\apollo_stfl2\train\LQ

set DELAY_SAMPLES=1057
set SR=44100

if "%~1"=="" (
    echo Usage: convert_lq_mp3_to_wav.bat ^<folder_containing_mp3s^>
    exit /b 1
)

set IN_DIR=%~1

for %%f in ("%IN_DIR%\*.mp3") do (
    set "OUT=%%~dpf%%~nf.wav"
    echo %%~nxf -^> %%~nf.wav
    ffmpeg -y -i "%%f" -af "atrim=start_sample=%DELAY_SAMPLES%" -ar %SR% -c:a pcm_s16le "!OUT!" -loglevel error
)

echo Done. You can now delete the MP3s from the folder and set align_data: false in your config.
