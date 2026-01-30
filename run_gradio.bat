디@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python.
    pause
    exit /b 1
)

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing/Updating requirements...
pip install -r requirements.txt

echo Starting Gradio App...
echo Open your browser at http://localhost:7860 if it doesn't open automatically.
python gradio_app.py

pause
