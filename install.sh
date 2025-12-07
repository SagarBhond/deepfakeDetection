#!/bin/bash

echo "========================================"
echo "Deepfake Detection System Installation"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

echo "Python found. Checking version..."
python3 -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"

# Create virtual environment
echo
echo "Creating virtual environment..."
python3 -m venv deepfake_env
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source deepfake_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Check for CUDA
echo "Checking for CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing CUDA version of PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "CUDA not detected. Installing CPU version of PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Make scripts executable
chmod +x setup.py
chmod +x test_model.py

# Run setup script
echo "Running setup script..."
python setup.py

echo
echo "========================================"
echo "Installation completed successfully!"
echo "========================================"
echo
echo "To activate the environment in the future, run:"
echo "source deepfake_env/bin/activate"
echo
echo "To start the web interface:"
echo "python web_app.py"
echo
