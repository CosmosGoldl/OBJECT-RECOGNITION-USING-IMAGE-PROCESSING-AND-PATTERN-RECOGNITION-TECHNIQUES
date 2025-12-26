from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os
import sys
import subprocess
from datetime import datetime
import cv2
import threading

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import server configuration
from config.server_config import server_config
from config.app_config import app_config

# Import Windows configuration helper
try:
    from config.windows_config import windows_config
    # Initialize Windows environment if on Windows
    if windows_config.is_windows():
        print("🔧 Setting up Windows environment...")
        windows_config.setup_windows_environment()
        print("✓ Windows setup complete")
except ImportError:
    print("Windows config helper not available, using default configuration")

from object_models import object_detection_upload
from object_models import object_detection_live


app = Flask(__name__, static_folder='static')

# Configure Flask using app_config
app.config['UPLOAD_FOLDER'] = app_config.UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = app_config.AUDIO_FOLDER
app.config['MAX_CONTENT_LENGTH'] = app_config.MAX_CONTENT_LENGTH

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# Remove global camera initialization to avoid conflicts
# camera = cv2.VideoCapture(0)


@app.route('/')
def home():
    return render_template('home-page.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/upload-video')
def upload_video():
    return render_template('object-detection-upload.html')


@app.route('/video-detection', methods=['POST'])
def video_detection():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Use single model (no selection)
        detection_result, audio_path, timestap = object_detection_upload.detect_objects(file_path)
        
        # Get paths for displaying results
        audio_file = os.path.basename(audio_path)
        
        return render_template('object-detection-upload.html', detection_result=detection_result, audio_file=audio_file, timestap=timestap)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['AUDIO_FOLDER'], filename)

@app.route('/start-live-detection', methods=['POST'])
def start_live_detection():
    try:
        # Lấy thông tin từ form
        use_camera = request.form.get('use_camera', 'true').lower() == 'true'
        camera_id = int(request.form.get('camera_id', 0))
        
        # Xử lý video file nếu có
        video_path = None
        if 'video_file' in request.files:
            video_file = request.files['video_file']
            if video_file.filename != '':
                filename = secure_filename(video_file.filename)
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                video_file.save(video_path)
                use_camera = False
                print(f"Video uploaded: {video_path}")
        
        # Gọi detection function
        if use_camera:
            print(f"Starting camera detection (Camera {camera_id})...")
            object_detection_live.start_detection_with_camera(camera_id)
        else:
            if video_path:
                print(f"Starting video detection: {video_path}")
                object_detection_live.start_detection_with_video(video_path)
            else:
                # Sử dụng default video nếu có
                print("Starting detection with default video...")
                object_detection_live.start_detection(use_cam=False)
        
        # Return success response (detection has completed)
        return jsonify({
            'status': 'completed',
            'message': 'Detection session completed successfully',
            'source': 'camera' if use_camera else 'video',
            'camera_id': camera_id if use_camera else None,
            'video_path': video_path if not use_camera else None
        })
        
    except Exception as e:
        print(f"Detection failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/system-info')
def system_info():
    return render_template('system-info.html')


if __name__ == '__main__':
    # Use server configuration to start the app
    server_config.start_server(app)
