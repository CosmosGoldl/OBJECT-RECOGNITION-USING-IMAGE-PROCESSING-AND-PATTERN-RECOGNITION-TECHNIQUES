#!/usr/bin/env python3
"""
Shared utilities for YOLO model loading and object detection
"""

import os
import torch
from config.app_config import app_config


def load_yolo_model(model_name=None):
    """
    Load YOLO model with comprehensive fallback options
    
    Args:
        model_name: Model name from config (e.g., 'yolov5s', 'yolov5su')
        
    Returns:
        Tuple of (model, model_type, model_info) or (None, None, None) if failed
    """
    if model_name is None:
        model_name = app_config.DEFAULT_YOLO_MODEL
    
    model_info = app_config.get_model_info(model_name)
    model_filename = model_info['file']
    
    print(f"Loading model: {model_name} ({model_filename})")
    
    # Method 1: Try ultralytics first (most stable)
    try:
        from ultralytics import YOLO
        model = YOLO(model_filename)
        print(f"✓ Successfully loaded {model_filename} using ultralytics")
        return model, 'ultralytics', model_info
    except Exception as e:
        print(f"✗ Ultralytics failed: {e}")
    
    # Method 2: Try local YOLOv5 repo
    try:
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolov5'))
        weights_path = os.path.join(repo_dir, model_filename)
        if os.path.exists(weights_path):
            model = torch.hub.load(repo_dir, 'custom', path=weights_path, source='local')
            print(f"✓ Successfully loaded {weights_path} using local YOLOv5")
            return model, 'local', model_info
    except Exception as e:
        print(f"✗ Local YOLOv5 failed: {e}")
    
    # Method 3: Try torch hub as final fallback
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        print("✓ Successfully loaded YOLOv5s using torch hub (fallback)")
        return model, 'hub', model_info
    except Exception as e:
        print(f"✗ Torch hub failed: {e}")
    
    print("✗ All model loading methods failed")
    return None, None, None


def extract_objects_from_results(results, model, model_type):
    """
    Extract detected objects from YOLO results regardless of model type
    
    Args:
        results: YOLO detection results
        model: YOLO model instance
        model_type: Type of model ('ultralytics', 'local', 'hub')
        
    Returns:
        List of detected object names
    """
    try:
        if model_type == 'ultralytics':
            # For ultralytics YOLO
            if results and len(results) > 0 and results[0].boxes is not None:
                detected_objects = [results[0].names[int(box.cls[0])] for box in results[0].boxes]
                return list(set(detected_objects))  # Get unique objects
            else:
                return []
        else:
            # For local YOLOv5 or torch hub
            if hasattr(results, 'pandas'):
                df = results.pandas().xyxy[0]
                return df['name'].unique().tolist() if not df.empty else []
            elif hasattr(results, 'xyxy'):
                if len(results.xyxy[0]) > 0:
                    return [model.names[int(x)] for x in results.xyxy[0][:, -1]]
                else:
                    return []
            else:
                return []
    except Exception as parse_error:
        print(f"Results parsing error: {parse_error}")
        return []


def get_annotated_frame(results, frame, model_type):
    """
    Get annotated frame with bounding boxes and labels
    
    Args:
        results: YOLO detection results
        frame: Original frame/image
        model_type: Type of model ('ultralytics', 'local', 'hub')
        
    Returns:
        Annotated frame or original frame if annotation fails
    """
    try:
        if model_type == 'ultralytics' and results:
            # For ultralytics, use built-in plot function
            return results[0].plot()
        else:
            # For other types, could implement custom annotation here
            # For now, return original frame
            return frame
    except Exception as e:
        print(f"Annotation error: {e}")
        return frame


def generate_detection_description(detected_objects):
    """
    Generate natural language description from detected objects
    
    Args:
        detected_objects: List of detected object names
        
    Returns:
        String description
    """
    if not detected_objects:
        return "No objects detected."
    
    unique_objects = list(set(detected_objects))
    objects_list = ", ".join(unique_objects)
    return f"The detected objects are: {objects_list}."
