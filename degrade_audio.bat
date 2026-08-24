@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: degrade_audio.bat
:: Synthetic audio degradation pipeline for training data gen
::
:: Single file:  degrade_audio.bat "input.flac" "C:\output"
:: Bulk folder:  degrade_audio.bat "C:\input_folder" "C:\output" /bulk
:: ============================================================

:: --- CONFIG ---
set FFMPEG=ffmpeg
set ACMENC=C:\Portable\acmenc\acmenc.exe
set WMA_BITRATE=128k
set MP3_PASS1_Q=5
set MP3_PASS3_Q=5
set MP3_PASS5_Q=2
set MP3_FINAL_BITRATE=192
:: --------------

if "%~1"=="" ( echo Usage: degrade_audio.bat "input" "output" [/bulk] & pause & exit /b 1 )
if "%~2"=="" ( echo Usage: degrade_audio.bat "input" "output" [/bulk] & pause & exit /b 1 )

:: Check for /bulk flag
if /i "%~3"=="/bulk" (
    echo.
    echo ============================================================
    echo  BULK MODE -- processing all audio files in:
    echo  %~f1
    echo  Output: %~f2
    echo ============================================================
    echo.
    for %%F in ("%~f1\*.flac" "%~f1\*.wav" "%~f1\*.mp3" "%~f1\*.aiff" "%~f1\*.aif") do (
        echo.
        echo [BULK] Processing: %%~nxF
        call "%~f0" "%%~fF" "%~f2"
    )
    echo.
    echo ============================================================
    echo  BULK MODE complete.
    echo ============================================================
    pause
    exit /b 0
)

set INPUT=%~f1
set OUTDIR=%~f2

:: Sanitize basename
set BASENAME=%~n1
set BASENAME=%BASENAME:(=%
set BASENAME=%BASENAME:)=%
set BASENAME=%BASENAME:&=%
set BASENAME=%BASENAME: =_%

set TMPDIR=%OUTDIR%\tmp_%BASENAME%
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
if not exist "%TMPDIR%" mkdir "%TMPDIR%"

for %%I in ("%TMPDIR%") do set TMPDIR_S=%%~sI
for %%I in ("%OUTDIR%") do set OUTDIR_S=%%~sI

echo.
echo ============================================================
echo  Degradation pipeline starting
echo  Input:    %INPUT%
echo  Output:   %OUTDIR%
echo  Temp dir: %TMPDIR%
echo ============================================================
echo.

echo [1/10] WMA %WMA_BITRATE% encode...
"%FFMPEG%" -y -i "%INPUT%" -vn -c:a wmav2 -b:a %WMA_BITRATE% "%TMPDIR%\gen1_wma.wma"
if errorlevel 1 ( echo ERROR: WMA encode failed & pause & exit /b 1 )
echo    Done.

echo [2/10] MP3 pass 1 (LAME q=%MP3_PASS1_Q%)...
"%FFMPEG%" -y -i "%TMPDIR%\gen1_wma.wma" -vn -c:a libmp3lame -q:a %MP3_PASS1_Q% "%TMPDIR%\gen2_mp3.mp3"
if errorlevel 1 ( echo ERROR: MP3 pass 1 failed & pause & exit /b 1 )
echo    Done.

echo [3/10] Decode for FhG pass 2...
"%FFMPEG%" -y -i "%TMPDIR%\gen2_mp3.mp3" -vn -c:a pcm_s16le "%TMPDIR%\gen2_dec.wav"
if errorlevel 1 ( echo ERROR: Decode before FhG pass 2 failed & pause & exit /b 1 )

echo [4/10] MP3 pass 2 (FhG %MP3_FINAL_BITRATE%k)...
"%ACMENC%" -c "Fraunhofer IIS MPEG Layer-3 Codec (professional)" --enc-delay 672 -b%MP3_FINAL_BITRATE% %TMPDIR_S%\gen2_dec.wav %TMPDIR_S%\gen3_mp3.mp3
if errorlevel 1 ( echo ERROR: FhG pass 2 failed & pause & exit /b 1 )
echo    Done.

echo [5/10] MP3 pass 3 (LAME q=%MP3_PASS3_Q%)...
"%FFMPEG%" -y -i "%TMPDIR%\gen3_mp3.mp3" -vn -c:a libmp3lame -q:a %MP3_PASS3_Q% "%TMPDIR%\gen4_mp3.mp3"
if errorlevel 1 ( echo ERROR: MP3 pass 3 failed & pause & exit /b 1 )
echo    Done.

echo [6/10] Decode for FhG pass 4...
"%FFMPEG%" -y -i "%TMPDIR%\gen4_mp3.mp3" -vn -c:a pcm_s16le "%TMPDIR%\gen4_dec.wav"
if errorlevel 1 ( echo ERROR: Decode before FhG pass 4 failed & pause & exit /b 1 )

echo [7/10] MP3 pass 4 (FhG %MP3_FINAL_BITRATE%k)...
"%ACMENC%" -c "Fraunhofer IIS MPEG Layer-3 Codec (professional)" --enc-delay 672 -b%MP3_FINAL_BITRATE% %TMPDIR_S%\gen4_dec.wav %TMPDIR_S%\gen5_mp3.mp3
if errorlevel 1 ( echo ERROR: FhG pass 4 failed & pause & exit /b 1 )
echo    Done.

echo [8/10] MP3 pass 5 (LAME q=%MP3_PASS5_Q%)...
"%FFMPEG%" -y -i "%TMPDIR%\gen5_mp3.mp3" -vn -c:a libmp3lame -q:a %MP3_PASS5_Q% "%TMPDIR%\gen6_mp3.mp3"
if errorlevel 1 ( echo ERROR: MP3 pass 5 failed & pause & exit /b 1 )
echo    Done.

echo [9/10] Decode for final FhG encode...
"%FFMPEG%" -y -i "%TMPDIR%\gen6_mp3.mp3" -vn -c:a pcm_s16le "%TMPDIR%\gen7_dec.wav"
if errorlevel 1 ( echo ERROR: Decode for final encode failed & pause & exit /b 1 )
echo    Done.

echo [10/10] Final encode to %MP3_FINAL_BITRATE%k MP3 (Fraunhofer IIS)...
"%ACMENC%" -c "Fraunhofer IIS MPEG Layer-3 Codec (professional)" --enc-delay 672 -b%MP3_FINAL_BITRATE% %TMPDIR_S%\gen7_dec.wav %OUTDIR_S%\%BASENAME%_degraded.mp3
if errorlevel 1 ( echo ERROR: Final FhG encode failed & pause & exit /b 1 )
echo    Done.

echo Cleaning up temp files...
rmdir /s /q "%TMPDIR%"

echo.
echo ============================================================
echo  Complete.
echo  Output: %OUTDIR%\%BASENAME%_degraded.mp3
echo ============================================================
echo.
