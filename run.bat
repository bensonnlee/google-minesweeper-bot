@echo off
cd /d "%~dp0"

:: Find Python 3
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python 3 is required but not found.
    echo Install it from https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version 2>&1 | findstr /r "Python 3\." >nul
if %errorlevel% neq 0 (
    echo Error: Python 3 is required. Found a different version.
    echo Install Python 3 from https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo Using %%i

:: Create venv if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

:: Activate venv
call .venv\Scripts\activate.bat

:: Install/update dependencies
pip install -q -r requirements.txt

echo Starting bot...
echo.
python bot.py %*
