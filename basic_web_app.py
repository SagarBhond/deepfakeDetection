#!/usr/bin/env python3
"""
Basic Web App for Deepfake Detection System
Enhanced with proper video frame analysis
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import json
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
import numpy as np
import time
import uuid

# S3 Integration
try:
    from s3_storage import S3Storage
    S3_ENABLED = os.environ.get('S3_ENABLED', 'true').lower() == 'true'
    S3_BUCKET = os.environ.get('S3_BUCKET', 'deepfakeddetection')
    s3_storage = S3Storage(bucket_name=S3_BUCKET) if S3_ENABLED else None
except ImportError:
    S3_ENABLED = False
    s3_storage = None
    print("Warning: S3 storage not available. Install boto3 to enable S3 integration.")

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['S3_ENABLED'] = S3_ENABLED
app.config['S3_BUCKET'] = S3_BUCKET

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_video_basic(video_path):
    """Enhanced video analysis using OpenCV to detect deepfake artifacts"""
    try:
        start_time = time.time()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, 0.0, {"error": "Could not open video file"}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        file_size = os.path.getsize(video_path)
        
        if frame_count == 0:
            cap.release()
            return None, 0.0, {"error": "Video has no frames"}
        
        # Analyze frames for deepfake artifacts
        fake_indicators = []
        frames_to_analyze = min(30, frame_count)  # Analyze up to 30 frames
        frame_interval = max(1, frame_count // frames_to_analyze)
        
        prev_frame = None
        frame_num = 0
        analyzed_count = 0
        
        while analyzed_count < frames_to_analyze:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % frame_interval == 0:
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 1. Check for excessive blur (deepfakes often have blur artifacts)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var < 100:  # Low variance indicates blur
                    fake_indicators.append('blur')
                
                # 2. Check for edge artifacts (common in deepfakes)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (width * height)
                if edge_density < 0.05 or edge_density > 0.3:  # Unusual edge patterns
                    fake_indicators.append('edge_artifact')
                
                # 3. Check for color inconsistencies
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                color_variance = np.var(hsv[:, :, 0])  # Hue variance
                if color_variance < 500:  # Low color variance might indicate manipulation
                    fake_indicators.append('color_inconsistency')
                
                # 4. Check for temporal inconsistencies (frame-to-frame differences)
                if prev_frame is not None:
                    frame_diff = cv2.absdiff(gray, prev_frame)
                    diff_mean = np.mean(frame_diff)
                    if diff_mean < 5:  # Very little change (possible static face swap)
                        fake_indicators.append('temporal_inconsistency')
                    elif diff_mean > 50:  # Excessive change (possible artifacts)
                        fake_indicators.append('temporal_jump')
                
                prev_frame = gray.copy()
                analyzed_count += 1
            
            frame_num += 1
        
        cap.release()
        
        # Calculate fake probability based on indicators
        indicator_count = len(fake_indicators)
        total_checks = analyzed_count * 4  # 4 checks per frame
        
        # Base probability calculation
        if indicator_count == 0:
            fake_probability = 0.2  # Low probability if no indicators
        else:
            # Higher indicator ratio = higher fake probability
            indicator_ratio = indicator_count / total_checks
            fake_probability = min(0.95, 0.3 + (indicator_ratio * 0.65))
        
        # Additional heuristics
        # Very short videos might be suspicious
        if frame_count < 30:
            fake_probability += 0.1
        
        # Very low resolution might indicate manipulation
        if width < 320 or height < 240:
            fake_probability += 0.1
        
        # Normalize probability
        fake_probability = min(0.95, max(0.1, fake_probability))
        real_probability = 1 - fake_probability
        
        # Determine if fake (threshold at 0.5)
        is_fake = fake_probability > 0.5
        confidence = fake_probability if is_fake else real_probability
        
        inference_time = time.time() - start_time
        
        details = {
            'prediction': 'Fake' if is_fake else 'Real',
            'real_probability': round(real_probability, 3),
            'fake_probability': round(fake_probability, 3),
            'inference_time': round(inference_time, 3),
            'file_info': {
                'file_size': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'frame_count': frame_count,
                'fps': round(fps, 2),
                'resolution': f"{width}x{height}",
                'frames_analyzed': analyzed_count
            },
            'indicators': {
                'total_indicators': indicator_count,
                'indicator_types': list(set(fake_indicators)) if fake_indicators else []
            }
        }
        
        return is_fake, confidence, details
        
    except Exception as e:
        return None, 0.0, {"error": str(e)}

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a video file.'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        # Make prediction (basic analysis)
        is_fake, confidence, details = analyze_video_basic(filepath)
        
        if is_fake is None:
            return jsonify({'error': f'Analysis failed: {details.get("error", "Unknown error")}'}), 500
        
        # Prepare response
        result = {
            'success': True,
            'filename': filename,
            'prediction': details['prediction'],
            'is_fake': is_fake,
            'confidence': confidence,
            'real_probability': details['real_probability'],
            'fake_probability': details['fake_probability'],
            'inference_time': details['inference_time'],
            'file_info': details.get('file_info', {}),
            'timestamp': datetime.now().isoformat(),
            'analysis_method': 'OpenCV-based frame analysis with artifact detection'
        }
        
        # Save result locally
        os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{timestamp}_result.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Upload to S3 if enabled
        if S3_ENABLED and s3_storage:
            try:
                # Upload video to S3
                video_id = str(uuid.uuid4())
                s3_video_key = s3_storage.upload_video(filepath, video_id)
                if s3_video_key:
                    result['s3_video_key'] = s3_video_key
                    result['s3_video_url'] = s3_storage.get_presigned_url(s3_video_key)
                
                # Upload result to S3
                result_id = f"{timestamp}_{video_id}"
                s3_result_key = s3_storage.upload_result(result, result_id)
                if s3_result_key:
                    result['s3_result_key'] = s3_result_key
                    result['s3_result_url'] = s3_storage.get_presigned_url(s3_result_key)
                
                result['s3_enabled'] = True
            except Exception as s3_error:
                print(f"S3 upload error: {s3_error}")
                result['s3_enabled'] = False
                result['s3_error'] = str(s3_error)
        else:
            result['s3_enabled'] = False
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/status')
def get_status():
    """Get system status"""
    status = {
        'model_loaded': True,  # Using OpenCV-based analysis
        'device': 'CPU (OpenCV Analysis)',
        'mode': 'Enhanced Frame Analysis',
        'analysis_method': 'OpenCV-based artifact detection',
        's3_enabled': S3_ENABLED,
        's3_bucket': S3_BUCKET if S3_ENABLED else None,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(status)

@app.route('/api/history')
def get_history():
    """Get prediction history"""
    try:
        results_dir = app.config['RESULTS_FOLDER']
        if not os.path.exists(results_dir):
            return jsonify({'history': []})
        
        history = []
        for filename in os.listdir(results_dir):
            if filename.endswith('_result.json'):
                filepath = os.path.join(results_dir, filename)
                with open(filepath, 'r') as f:
                    result = json.load(f)
                    history.append(result)
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'error': f'Failed to load history: {str(e)}'}), 500

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear prediction history"""
    try:
        results_dir = app.config['RESULTS_FOLDER']
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if filename.endswith('_result.json'):
                    os.remove(os.path.join(results_dir, filename))
        
        return jsonify({'success': True, 'message': 'History cleared'})
    except Exception as e:
        return jsonify({'error': f'Failed to clear history: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    
    print("Starting Deepfake Detection Web App")
    print("=" * 60)
    print("Enhanced with OpenCV-based frame analysis")
    print("Detecting artifacts: blur, edges, color inconsistencies, temporal patterns")
    print("=" * 60)
    print("Web interface will be available at: http://localhost:5000")
    print("Upload videos to analyze for deepfake detection")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
