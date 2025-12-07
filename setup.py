#!/usr/bin/env python3
"""
Setup script for Deepfake Detection System
Handles installation, configuration, and initial setup
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse


def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"Running: {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description or command} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description or command} failed:")
        print(f"  Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_cuda():
    """Check CUDA availability"""
    print("Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA not available. Will use CPU (slower training)")
            return False
    except ImportError:
        print("⚠ PyTorch not installed yet. Will check after installation.")
        return False


def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    
    directories = [
        'models/checkpoints',
        'data/raw/real',
        'data/raw/fake',
        'data/processed/real',
        'data/processed/fake',
        'data/frames/real',
        'data/frames/fake',
        'results/plots',
        'results/predictions',
        'uploads',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("✗ requirements.txt not found")
        return False
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    return True


def install_pytorch():
    """Install PyTorch with appropriate CUDA support"""
    print("Installing PyTorch...")
    
    # Check CUDA availability
    cuda_available = check_cuda()
    
    if cuda_available:
        # Install CUDA version
        pytorch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        # Install CPU version
        pytorch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(pytorch_command, "Installing PyTorch")


def download_sample_data():
    """Download or create sample data"""
    print("Setting up sample data...")
    
    # Run the test script to create sample data
    if os.path.exists('test_model.py'):
        return run_command("python test_model.py", "Creating sample data")
    else:
        print("⚠ test_model.py not found. Skipping sample data creation.")
        return True


def test_installation():
    """Test the installation"""
    print("Testing installation...")
    
    if os.path.exists('test_model.py'):
        return run_command("python test_model.py", "Running installation tests")
    else:
        print("⚠ test_model.py not found. Skipping installation tests.")
        return True


def create_config_file():
    """Create configuration file"""
    print("Creating configuration file...")
    
    config = {
        "model": {
            "sequence_length": 16,
            "hidden_size": 256,
            "num_lstm_layers": 2,
            "num_classes": 2
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "epochs": 50,
            "early_stopping_patience": 10
        },
        "inference": {
            "confidence_threshold": 0.5,
            "device": "auto"
        },
        "web": {
            "host": "0.0.0.0",
            "port": 5000,
            "debug": False
        }
    }
    
    import json
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Configuration file created: config.json")


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Prepare your training data:")
    print("   - Place real videos in: data/raw/real/")
    print("   - Place fake videos in: data/raw/fake/")
    print("\n2. Train the model:")
    print("   python train.py --real_paths data/raw/real/*.mp4 --fake_paths data/raw/fake/*.mp4")
    print("\n3. Run inference:")
    print("   python inference.py --model_path models/checkpoints/best_model.pth --video_path your_video.mp4")
    print("\n4. Launch web interface:")
    print("   python web_app.py")
    print("   Then open: http://localhost:5000")
    print("\n5. For more information, see README.md")
    print("\nHappy deepfake detecting! 🚀")


def main():
    parser = argparse.ArgumentParser(description='Setup Deepfake Detection System')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--skip-tests', action='store_true', help='Skip installation tests')
    parser.add_argument('--skip-sample-data', action='store_true', help='Skip sample data creation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DEEPFAKE DETECTION SYSTEM SETUP")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            print("✗ Dependency installation failed")
            sys.exit(1)
        
        if not install_pytorch():
            print("✗ PyTorch installation failed")
            sys.exit(1)
    
    # Create configuration
    create_config_file()
    
    # Create sample data
    if not args.skip_sample_data:
        if not download_sample_data():
            print("⚠ Sample data creation failed, but continuing...")
    
    # Test installation
    if not args.skip_tests:
        if not test_installation():
            print("⚠ Installation tests failed, but continuing...")
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
