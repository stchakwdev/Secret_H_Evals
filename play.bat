@echo off
REM Smart Secret Hitler Game Launcher for Windows
REM Automatically detects Python environment and handles dependencies

setlocal EnableDelayedExpansion

echo 🎮 Secret Hitler - Smart Launcher
echo =================================

REM Get script directory
set SCRIPT_DIR=%~dp0
set PYTHON_SCRIPT=%SCRIPT_DIR%play_hybrid.py
set REQUIREMENTS=%SCRIPT_DIR%requirements.txt

echo 🔍 Detecting Python environment...

REM Array of Python commands to try (in order of preference)
set PYTHON_CANDIDATES=python python3 py "C:\Python311\python.exe" "C:\Python310\python.exe" "C:\Python39\python.exe"

set FOUND_PYTHON=
set NEED_INSTALL=0

REM Test each Python candidate
for %%p in (%PYTHON_CANDIDATES%) do (
    where %%p >nul 2>&1
    if !errorlevel! equ 0 (
        REM Test if Python has required modules
        %%p -c "import asyncio, websockets, json, logging" >nul 2>&1
        if !errorlevel! equ 0 (
            set FOUND_PYTHON=%%p
            echo ✅ Found working Python: %%p
            goto :found_python
        ) else (
            REM Python exists but missing modules
            %%p -c "import sys; exit(0 if sys.version_info >= (3,7) else 1)" >nul 2>&1
            if !errorlevel! equ 0 (
                set FOUND_PYTHON=%%p
                set NEED_INSTALL=1
                echo ⚠️  Found Python with missing modules: %%p
                goto :found_python
            )
        )
    )
)

echo ❌ No suitable Python installation found
echo    Please install Python 3.7+ and websockets module
echo    Try: pip install websockets
exit /b 1

:found_python

REM Install dependencies if needed
if %NEED_INSTALL% equ 1 (
    echo 📦 Installing missing dependencies...
    %FOUND_PYTHON% -m pip install -r "%REQUIREMENTS%" --quiet --user
    if !errorlevel! neq 0 (
        pip install -r "%REQUIREMENTS%" --quiet
        if !errorlevel! neq 0 (
            echo ❌ Installation failed
            echo    Please manually install: pip install -r requirements.txt
            exit /b 1
        )
    )
    echo ✅ Dependencies installed successfully
    
    REM Test again after installation
    %FOUND_PYTHON% -c "import asyncio, websockets, json, logging" >nul 2>&1
    if !errorlevel! neq 0 (
        echo ❌ Installation failed or incomplete
        echo    Please manually install: pip install -r requirements.txt
        exit /b 1
    )
)

REM Check for .env file
if not exist "%SCRIPT_DIR%.env" (
    echo ⚠️  No .env file found
    echo    Please create .env with your OPENROUTER_API_KEY
    echo    Example: OPENROUTER_API_KEY=sk-or-v1-your-key-here
)

REM Launch the game
echo 🚀 Launching Secret Hitler Hybrid Game...
echo    Using: %FOUND_PYTHON%
echo.

REM Pass all arguments to the Python script
%FOUND_PYTHON% "%PYTHON_SCRIPT%" %*