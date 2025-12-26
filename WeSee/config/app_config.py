#!/usr/bin/env python3
"""
Application Configuration Settings
All configurable parameters for WeSee application
"""

class AppConfig:
    """Application configuration settings"""
    
    # Server Settings
    SERVER_HOST = '127.0.0.1'
    SERVER_PORT = 5001
    DEBUG_MODE = False
    AUTO_RELOAD = False
    
    # Flask Settings
    UPLOAD_FOLDER = 'static/captured'
    AUDIO_FOLDER = 'static/audio'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file upload
    
    # YOLO Model Settings (simplified to single model)
    DEFAULT_YOLO_MODEL = 'yolov5s'
    MODEL_FILE = 'yolov5s.pt'
    
    # Detection Settings
    DETECTION_TIMEOUT = 300  # 5 minutes max detection time
    MAX_DETECTION_FRAMES = 300
    CAMERA_FPS = 30
    
    # UI Settings
    ENABLE_SPEECH_FEEDBACK = True
    GLASSMORPHISM_ENABLED = True
    
    # Logging Settings
    SUPPRESS_FLASK_WARNINGS = True
    LOG_LEVEL = 'ERROR'
    
# Global app configuration instance
app_config = AppConfig()
