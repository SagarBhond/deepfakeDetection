@echo off
echo ========================================
echo Deepfake Detection System Installation
echo ========================================
echo.

REM Check if Python is installed
py --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Checking version...
py -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"

REM Create virtual environment
echo.
echo Creating virtual environment...
py -m venv deepfake_env
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call deepfake_env\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
py -m pip install --upgrade pip

REM Install PyTorch (CPU version for compatibility)
echo Installing PyTorch...
py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Install other dependencies
echo Installing other dependencies...
py -m pip install -r requirements.txt

REM Run setup script
echo Running setup script...
py setup.py

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo To activate the environment in the future, run:
echo deepfake_env\Scripts\activate.bat
echo.
echo To start the web interface:
echo python web_app.py
echo.
pause
