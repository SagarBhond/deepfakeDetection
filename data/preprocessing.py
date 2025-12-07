import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import random
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split


class DeepfakeDataset(Dataset):
    """
    Dataset class for deepfake detection
    Handles both real and fake video frames
    """
    
    def __init__(self, 
                 real_paths: List[str], 
                 fake_paths: List[str],
                 sequence_length: int = 16,
                 transform=None,
                 is_training: bool = True):
        """
        Initialize dataset
        
        Args:
            real_paths: List of paths to real video frames
            fake_paths: List of paths to fake video frames
            sequence_length: Number of frames in each sequence
            transform: Albumentations transform pipeline
            is_training: Whether this is training data
        """
        self.real_paths = real_paths
        self.fake_paths = fake_paths
        self.sequence_length = sequence_length
        self.transform = transform
        self.is_training = is_training
        
        # Create labels (0 for real, 1 for fake)
        self.data = []
        self.labels = []
        
        # Add real data
        for path in real_paths:
            self.data.append(path)
            self.labels.append(0)
            
        # Add fake data
        for path in fake_paths:
            self.data.append(path)
            self.labels.append(1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_path = self.data[idx]
        label = self.labels[idx]
        
        # Load video frames
        frames = self.load_video_frames(video_path)
        
        # Apply transforms if provided
        if self.transform:
            frames = self.apply_transforms(frames)
        
        # Convert to tensor
        frames = torch.stack(frames)
        
        return frames, torch.tensor(label, dtype=torch.long)
    
    def load_video_frames(self, video_path: str) -> List[torch.Tensor]:
        """
        Load frames from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frame tensors
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame
            frame = cv2.resize(frame, (224, 224))
            
            frames.append(frame)
        
        cap.release()
        
        # If we don't have enough frames, repeat the last frame
        while len(frames) < self.sequence_length:
            frames.append(frames[-1])
        
        # Take only the required number of frames
        frames = frames[:self.sequence_length]
        
        return frames
    
    def apply_transforms(self, frames: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Apply transforms to frames
        
        Args:
            frames: List of frame arrays
            
        Returns:
            List of transformed frame tensors
        """
        transformed_frames = []
        
        for frame in frames:
            if self.transform:
                transformed = self.transform(image=frame)
                frame = transformed['image']
            else:
                frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
            
            transformed_frames.append(frame)
        
        return transformed_frames


class FrameDataset(Dataset):
    """
    Dataset class for frame-based deepfake detection
    """
    
    def __init__(self, 
                 frame_paths: List[str], 
                 labels: List[int],
                 transform=None):
        """
        Initialize frame dataset
        
        Args:
            frame_paths: List of paths to frame images
            labels: List of corresponding labels
            transform: Albumentations transform pipeline
        """
        self.frame_paths = frame_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        label = self.labels[idx]
        
        # Load frame
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=frame)
            frame = transformed['image']
        else:
            frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
        
        return frame, torch.tensor(label, dtype=torch.long)


def get_transforms(is_training: bool = True):
    """
    Get data augmentation transforms
    
    Args:
        is_training: Whether to apply training augmentations
        
    Returns:
        Albumentations transform pipeline
    """
    if is_training:
        transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transform


def create_data_loaders(real_paths: List[str], 
                       fake_paths: List[str],
                       batch_size: int = 8,
                       sequence_length: int = 16,
                       test_size: float = 0.2,
                       random_state: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        real_paths: List of paths to real videos
        fake_paths: List of paths to fake videos
        batch_size: Batch size for data loaders
        sequence_length: Number of frames in sequence
        test_size: Fraction of data for validation
        random_state: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Split data
    train_real, val_real = train_test_split(
        real_paths, test_size=test_size, random_state=random_state
    )
    train_fake, val_fake = train_test_split(
        fake_paths, test_size=test_size, random_state=random_state
    )
    
    # Get transforms
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        train_real, train_fake, sequence_length, train_transform, is_training=True
    )
    val_dataset = DeepfakeDataset(
        val_real, val_fake, sequence_length, val_transform, is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader


def extract_frames_from_video(video_path: str, 
                             output_dir: str, 
                             max_frames: Optional[int] = None) -> List[str]:
    """
    Extract frames from video file
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of paths to extracted frames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if max_frames and frame_count >= max_frames:
            break
        
        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        
        frame_count += 1
    
    cap.release()
    return frame_paths


def preprocess_video_for_inference(video_path: str, 
                                  sequence_length: int = 16) -> torch.Tensor:
    """
    Preprocess video for inference
    
    Args:
        video_path: Path to video file
        sequence_length: Number of frames in sequence
        
    Returns:
        Preprocessed video tensor
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame
        frame = cv2.resize(frame, (224, 224))
        
        frames.append(frame)
    
    cap.release()
    
    # If we don't have enough frames, repeat the last frame
    while len(frames) < sequence_length:
        frames.append(frames[-1])
    
    # Take only the required number of frames
    frames = frames[:sequence_length]
    
    # Apply transforms
    transform = get_transforms(is_training=False)
    transformed_frames = []
    
    for frame in frames:
        transformed = transform(image=frame)
        frame_tensor = transformed['image']
        transformed_frames.append(frame_tensor)
    
    # Stack frames
    video_tensor = torch.stack(transformed_frames)
    
    # Add batch dimension
    video_tensor = video_tensor.unsqueeze(0)
    
    return video_tensor


def create_sample_data_structure():
    """
    Create sample data directory structure
    """
    directories = [
        'data/raw/real',
        'data/raw/fake',
        'data/processed/real',
        'data/processed/fake',
        'data/frames/real',
        'data/frames/fake',
        'models/checkpoints',
        'results/plots',
        'results/predictions'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing data preprocessing pipeline...")
    
    # Create sample data structure
    create_sample_data_structure()
    
    # Test transforms
    transform = get_transforms(is_training=True)
    print("Transforms created successfully")
    
    # Test frame extraction (if video file exists)
    sample_video = "sample_video.mp4"
    if os.path.exists(sample_video):
        frames = extract_frames_from_video(sample_video, "data/frames/test")
        print(f"Extracted {len(frames)} frames from {sample_video}")
    
    print("Data preprocessing pipeline test completed!")
