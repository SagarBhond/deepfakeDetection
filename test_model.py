#!/usr/bin/env python3
"""
Test script for the Deepfake Detection Model
Tests model functionality, data loading, and inference
"""

import torch
import numpy as np
import cv2
import os
import sys
from pathlib import Path
import time
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.resnext_lstm import ResNextLSTMDeepfakeDetector, create_model
from data.preprocessing import get_transforms, preprocess_video_for_inference
from inference import DeepfakeDetector


def test_model_creation():
    """Test model creation and basic functionality"""
    print("Testing model creation...")
    
    try:
        # Test model creation
        model = create_model(sequence_length=16, num_classes=2, device='cpu')
        print("✓ Model created successfully")
        
        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        batch_size = 2
        sequence_length = 16
        channels = 3
        height = 224
        width = 224
        
        test_input = torch.randn(batch_size, sequence_length, channels, height, width)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✓ Forward pass successful. Output shape: {output.shape}")
        
        # Test feature extraction
        features = model.extract_features(test_input)
        print(f"✓ Feature extraction successful. Feature shape: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_data_preprocessing():
    """Test data preprocessing functionality"""
    print("\nTesting data preprocessing...")
    
    try:
        # Test transforms
        train_transform = get_transforms(is_training=True)
        val_transform = get_transforms(is_training=False)
        print("✓ Transforms created successfully")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test training transform
        transformed = train_transform(image=dummy_image)
        train_tensor = transformed['image']
        print(f"✓ Training transform successful. Output shape: {train_tensor.shape}")
        
        # Test validation transform
        transformed = val_transform(image=dummy_image)
        val_tensor = transformed['image']
        print(f"✓ Validation transform successful. Output shape: {val_tensor.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data preprocessing failed: {e}")
        return False


def test_inference_pipeline():
    """Test inference pipeline"""
    print("\nTesting inference pipeline...")
    
    try:
        # Create a dummy model file for testing
        model = create_model(sequence_length=16, num_classes=2, device='cpu')
        dummy_model_path = 'models/checkpoints/test_model.pth'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(dummy_model_path), exist_ok=True)
        
        # Save dummy model
        torch.save(model.state_dict(), dummy_model_path)
        print("✓ Dummy model saved")
        
        # Test detector initialization
        detector = DeepfakeDetector(dummy_model_path, device='cpu')
        print("✓ Detector initialized successfully")
        
        # Test frame sequence prediction
        dummy_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
        
        is_fake, confidence, details = detector.predict_frame_sequence(dummy_frames)
        print(f"✓ Frame sequence prediction successful")
        print(f"  Prediction: {details['prediction']}")
        print(f"  Confidence: {confidence:.3f}")
        
        # Clean up
        os.remove(dummy_model_path)
        print("✓ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference pipeline failed: {e}")
        return False


def test_video_preprocessing():
    """Test video preprocessing functionality"""
    print("\nTesting video preprocessing...")
    
    try:
        # Create a dummy video file for testing
        dummy_video_path = 'test_video.mp4'
        
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(dummy_video_path, fourcc, 30.0, (640, 480))
        
        for i in range(100):  # 100 frames
            # Create a simple colored frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        print("✓ Dummy video created")
        
        # Test video preprocessing
        video_tensor = preprocess_video_for_inference(dummy_video_path, sequence_length=16)
        print(f"✓ Video preprocessing successful. Tensor shape: {video_tensor.shape}")
        
        # Clean up
        os.remove(dummy_video_path)
        print("✓ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Video preprocessing failed: {e}")
        return False


def test_performance():
    """Test model performance"""
    print("\nTesting model performance...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = create_model(sequence_length=16, num_classes=2, device=device)
        model.eval()
        
        # Test inference speed
        batch_size = 4
        test_input = torch.randn(batch_size, 16, 3, 224, 224).to(device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = model(test_input)
        
        # Time inference
        start_time = time.time()
        num_iterations = 20
        
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(test_input)
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        fps = batch_size / avg_time
        
        print(f"✓ Performance test completed")
        print(f"  Average inference time: {avg_time:.4f}s")
        print(f"  Throughput: {fps:.2f} videos/second")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


def create_sample_data():
    """Create sample data for testing"""
    print("\nCreating sample data...")
    
    try:
        # Create directories
        os.makedirs('data/raw/real', exist_ok=True)
        os.makedirs('data/raw/fake', exist_ok=True)
        os.makedirs('data/frames/real', exist_ok=True)
        os.makedirs('data/frames/fake', exist_ok=True)
        
        # Create sample videos
        for category in ['real', 'fake']:
            for i in range(3):  # Create 3 sample videos
                video_path = f'data/raw/{category}/sample_{i}.mp4'
                
                # Create a simple test video
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
                
                for frame_idx in range(90):  # 3 seconds at 30fps
                    # Create frames with different patterns for real vs fake
                    if category == 'real':
                        # Real videos: natural color patterns
                        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
                    else:
                        # Fake videos: more artificial patterns
                        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    
                    out.write(frame)
                
                out.release()
                print(f"✓ Created sample video: {video_path}")
        
        # Create sample frames
        for category in ['real', 'fake']:
            for i in range(10):  # Create 10 sample frames
                frame_path = f'data/frames/{category}/frame_{i:03d}.jpg'
                
                if category == 'real':
                    frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
                else:
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                cv2.imwrite(frame_path, frame)
                print(f"✓ Created sample frame: {frame_path}")
        
        print("✓ Sample data creation completed")
        return True
        
    except Exception as e:
        print(f"✗ Sample data creation failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("DEEPFAKE DETECTION MODEL TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Data Preprocessing", test_data_preprocessing),
        ("Inference Pipeline", test_inference_pipeline),
        ("Video Preprocessing", test_video_preprocessing),
        ("Performance", test_performance),
        ("Sample Data Creation", create_sample_data),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:.<40} {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
