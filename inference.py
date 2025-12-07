import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import time
from typing import List, Tuple, Optional
import json

from models.resnext_lstm import ResNextLSTMDeepfakeDetector, load_pretrained_model
from data.preprocessing import preprocess_video_for_inference, get_transforms


class DeepfakeDetector:
    """Deepfake detection inference class"""
    
    def __init__(self, model_path: str, device: str = 'auto', confidence_threshold: float = 0.5):
        """
        Initialize the deepfake detector
        
        Args:
            model_path: Path to the trained model
            device: Device to run inference on
            confidence_threshold: Confidence threshold for predictions
        """
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Get transforms
        self.transform = get_transforms(is_training=False)
        
        print("Deepfake detector initialized successfully!")
    
    def load_model(self, model_path: str) -> ResNextLSTMDeepfakeDetector:
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            model = load_pretrained_model(model_path, self.device)
            print(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create a new model if loading fails
            model = ResNextLSTMDeepfakeDetector()
            model = model.to(self.device)
            print("Created new model instance")
            return model
    
    def predict_video(self, video_path: str) -> Tuple[bool, float, dict]:
        """
        Predict if a video is deepfake
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (is_fake, confidence, details)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Preprocess video
        video_tensor = preprocess_video_for_inference(video_path)
        video_tensor = video_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(video_tensor)
            inference_time = time.time() - start_time
            
            # Get probabilities
            probabilities = F.softmax(outputs, dim=1)
            confidence = probabilities.max().item()
            prediction = outputs.argmax(dim=1).item()
            
            # Determine if fake
            is_fake = prediction == 1 and confidence > self.confidence_threshold
            
            details = {
                'prediction': 'Fake' if prediction == 1 else 'Real',
                'confidence': confidence,
                'real_probability': probabilities[0][0].item(),
                'fake_probability': probabilities[0][1].item(),
                'inference_time': inference_time,
                'threshold': self.confidence_threshold
            }
            
            return is_fake, confidence, details
    
    def predict_frame_sequence(self, frames: List[np.ndarray]) -> Tuple[bool, float, dict]:
        """
        Predict if a sequence of frames is deepfake
        
        Args:
            frames: List of frame arrays
            
        Returns:
            Tuple of (is_fake, confidence, details)
        """
        if len(frames) == 0:
            raise ValueError("No frames provided")
        
        # Preprocess frames
        processed_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            processed_frames.append(frame)
        
        # Apply transforms
        transformed_frames = []
        for frame in processed_frames:
            transformed = self.transform(image=frame)
            frame_tensor = transformed['image']
            transformed_frames.append(frame_tensor)
        
        # Stack frames and add batch dimension
        video_tensor = torch.stack(transformed_frames).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(video_tensor)
            inference_time = time.time() - start_time
            
            # Get probabilities
            probabilities = F.softmax(outputs, dim=1)
            confidence = probabilities.max().item()
            prediction = outputs.argmax(dim=1).item()
            
            # Determine if fake
            is_fake = prediction == 1 and confidence > self.confidence_threshold
            
            details = {
                'prediction': 'Fake' if prediction == 1 else 'Real',
                'confidence': confidence,
                'real_probability': probabilities[0][0].item(),
                'fake_probability': probabilities[0][1].item(),
                'inference_time': inference_time,
                'threshold': self.confidence_threshold,
                'num_frames': len(frames)
            }
            
            return is_fake, confidence, details
    
    def predict_realtime(self, video_source: int = 0) -> None:
        """
        Real-time deepfake detection from webcam
        
        Args:
            video_source: Video source (0 for webcam)
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Starting real-time deepfake detection...")
        print("Press 'q' to quit")
        
        frame_buffer = []
        buffer_size = 16  # Number of frames to analyze
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add frame to buffer
            frame_buffer.append(frame.copy())
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)
            
            # Make prediction if buffer is full
            if len(frame_buffer) == buffer_size:
                try:
                    is_fake, confidence, details = self.predict_frame_sequence(frame_buffer)
                    
                    # Display prediction on frame
                    color = (0, 0, 255) if is_fake else (0, 255, 0)
                    label = f"{details['prediction']} ({confidence:.2f})"
                    
                    cv2.putText(frame, label, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"Confidence: {confidence:.3f}", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
            
            # Display frame
            cv2.imshow('Deepfake Detection', frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def batch_predict(self, video_paths: List[str], output_file: Optional[str] = None) -> List[dict]:
        """
        Predict on multiple videos
        
        Args:
            video_paths: List of video file paths
            output_file: Optional output JSON file
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"Processing {len(video_paths)} videos...")
        
        for i, video_path in enumerate(video_paths):
            print(f"Processing {i+1}/{len(video_paths)}: {video_path}")
            
            try:
                is_fake, confidence, details = self.predict_video(video_path)
                
                result = {
                    'video_path': video_path,
                    'is_fake': is_fake,
                    'confidence': confidence,
                    'details': details
                }
                
                results.append(result)
                
                print(f"  Result: {details['prediction']} (Confidence: {confidence:.3f})")
                
            except Exception as e:
                print(f"  Error processing {video_path}: {e}")
                results.append({
                    'video_path': video_path,
                    'error': str(e)
                })
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--video_path', type=str,
                       help='Path to video file for prediction')
    parser.add_argument('--batch_paths', type=str, nargs='+',
                       help='Paths to multiple video files')
    parser.add_argument('--realtime', action='store_true',
                       help='Run real-time detection from webcam')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    parser.add_argument('--output_file', type=str,
                       help='Output file for batch predictions')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DeepfakeDetector(
        model_path=args.model_path,
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )
    
    # Run inference based on arguments
    if args.realtime:
        detector.predict_realtime()
    elif args.video_path:
        is_fake, confidence, details = detector.predict_video(args.video_path)
        print(f"\nPrediction Results:")
        print(f"Video: {args.video_path}")
        print(f"Prediction: {details['prediction']}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Real Probability: {details['real_probability']:.3f}")
        print(f"Fake Probability: {details['fake_probability']:.3f}")
        print(f"Inference Time: {details['inference_time']:.3f}s")
    elif args.batch_paths:
        results = detector.batch_predict(args.batch_paths, args.output_file)
        
        # Print summary
        total = len(results)
        successful = len([r for r in results if 'error' not in r])
        fake_count = len([r for r in results if r.get('is_fake', False)])
        
        print(f"\nBatch Prediction Summary:")
        print(f"Total videos: {total}")
        print(f"Successful predictions: {successful}")
        print(f"Fake videos detected: {fake_count}")
        print(f"Real videos detected: {successful - fake_count}")
    else:
        print("Please specify --video_path, --batch_paths, or --realtime")
        parser.print_help()


if __name__ == "__main__":
    main()
