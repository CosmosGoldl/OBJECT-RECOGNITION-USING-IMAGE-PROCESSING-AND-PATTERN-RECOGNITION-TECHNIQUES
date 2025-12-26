#!/usr/bin/env python3
"""
Simple camera utilities - basic functionality only
"""

import cv2


def initialize_camera(camera_id=0):
    """
    Simple camera initialization
    
    Args:
        camera_id: Camera index (default 0)
        
    Returns:
        tuple: (cv2.VideoCapture object, error_message)
    """
    try:
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            return None, f"Could not open camera {camera_id}"
        
        # Basic camera settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Test camera read
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            cap.release()
            return None, f"Could not read from camera {camera_id}"
        
        return cap, None
        
    except Exception as e:
        if 'cap' in locals() and cap:
            cap.release()
        return None, f"Camera error: {str(e)}"


def get_camera_info(cap):
    """Get basic camera information"""
    if not cap or not cap.isOpened():
        return {}
    
    try:
        return {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS))
        }
    except:
        return {}
