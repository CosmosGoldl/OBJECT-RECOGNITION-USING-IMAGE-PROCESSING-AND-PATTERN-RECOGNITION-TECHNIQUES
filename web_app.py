import warnings
import os

from flask import Flask, render_template, request, jsonify, Response
import cv2
import threading
import time
import json
from datetime import datetime
import numpy as np
import subprocess
from queue import Queue

# Try to import onnxruntime, fallback if not available
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("⚠️ onnxruntime not available. Running in simulation mode only.")
    ONNX_AVAILABLE = False

app = Flask(__name__)

detect_every_n = 10

class WeSeeWebApp:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.detection_thread = None
        self.droidcam_ip = ""
        self.current_mode = None
        self.video_path = None
        self.status = "🔴 Ready"
        self.frame_count = 0
        self.has_models = False
        
        # Dual Model configurations
        self.session_coco = None
        self.session_custom = None
        self.class_names_coco = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.class_names_custom = ['door', 'trash_can']
        
        # Gamma correction for preprocessing
        gamma = 0.8
        invGamma = 1.0 / gamma
        self.gamma_lut = np.array([((i / 255.0) ** invGamma) * 255
                                    for i in range(256)]).astype("uint8")
        
        # Distance warning settings
        self.warning_distance_cm = 200  # 2 meters
        self.last_warning_time = 0
        self.warning_cooldown = 5  # seconds
        
        # Audio warning queue
        self.audio_queue = Queue()
        self.audio_is_speaking = False
        self.setup_audio_system()
        
        # Performance optimization - cache detections
        self.last_detections = []
        self.detection_cache_count = 0
        
        # Comprehensive class sizes for accurate distance estimation
        self.class_avg_sizes = {
            # Custom objects
            "door": {"width_ratio": 0.8},
            "trash_can": {"width_ratio": 2.5},
            # People and animals
            "person": {"width_ratio": 2.5},
            "bird": {"width_ratio": 8.0},
            "cat": {"width_ratio": 1.9},
            "dog": {"width_ratio": 1.5},
            "horse": {"width_ratio": 0.8},
            "sheep": {"width_ratio": 1.2},
            "cow": {"width_ratio": 0.6},
            "elephant": {"width_ratio": 0.3},
            "bear": {"width_ratio": 0.9},
            "zebra": {"width_ratio": 0.8},
            "giraffe": {"width_ratio": 0.4},
            # Vehicles
            "bicycle": {"width_ratio": 2.3},
            "car": {"width_ratio": 0.37},
            "motorcycle": {"width_ratio": 2.4},
            "airplane": {"width_ratio": 0.1},
            "bus": {"width_ratio": 0.3},
            "train": {"width_ratio": 0.2},
            "truck": {"width_ratio": 0.25},
            "boat": {"width_ratio": 0.5},
            # Traffic
            "traffic light": {"width_ratio": 2.95},
            "fire hydrant": {"width_ratio": 3.0},
            "stop sign": {"width_ratio": 2.55},
            "parking meter": {"width_ratio": 4.0},
            # Large objects
            "bench": {"width_ratio": 1.6},
            "chair": {"width_ratio": 2.2},
            "couch": {"width_ratio": 1.0},
            "bed": {"width_ratio": 0.8},
            "dining table": {"width_ratio": 1.2},
            "toilet": {"width_ratio": 2.8},
            "tv": {"width_ratio": 1.8},
            # Medium objects
            "backpack": {"width_ratio": 3.5},
            "umbrella": {"width_ratio": 2.8},
            "handbag": {"width_ratio": 4.0},
            "tie": {"width_ratio": 8.0},
            "suitcase": {"width_ratio": 2.0},
            "laptop": {"width_ratio": 3.2},
            "book": {"width_ratio": 6.0},
            "clock": {"width_ratio": 3.0},
            "vase": {"width_ratio": 3.5},
            # Sports
            "frisbee": {"width_ratio": 4.5},
            "skis": {"width_ratio": 6.0},
            "snowboard": {"width_ratio": 3.5},
            "sports ball": {"width_ratio": 5.5},
            "kite": {"width_ratio": 2.5},
            "baseball bat": {"width_ratio": 4.8},
            "baseball glove": {"width_ratio": 4.2},
            "skateboard": {"width_ratio": 3.8},
            "surfboard": {"width_ratio": 2.8},
            "tennis racket": {"width_ratio": 3.2},
            # Small objects
            "bottle": {"width_ratio": 8.0},
            "wine glass": {"width_ratio": 8.5},
            "cup": {"width_ratio": 9.0},
            "fork": {"width_ratio": 12.0},
            "knife": {"width_ratio": 10.0},
            "spoon": {"width_ratio": 11.0},
            "bowl": {"width_ratio": 6.0},
            "banana": {"width_ratio": 10.0},
            "apple": {"width_ratio": 8.5},
            "sandwich": {"width_ratio": 5.0},
            "orange": {"width_ratio": 9.0},
            "broccoli": {"width_ratio": 7.0},
            "carrot": {"width_ratio": 12.0},
            "hot dog": {"width_ratio": 8.0},
            "pizza": {"width_ratio": 4.0},
            "donut": {"width_ratio": 9.0},
            "cake": {"width_ratio": 4.5},
            # Household
            "potted plant": {"width_ratio": 3.0},
            "mouse": {"width_ratio": 10.0},
            "remote": {"width_ratio": 7.0},
            "keyboard": {"width_ratio": 3.8},
            "cell phone": {"width_ratio": 8.0},
            "microwave": {"width_ratio": 1.8},
            "oven": {"width_ratio": 1.5},
            "toaster": {"width_ratio": 3.5},
            "sink": {"width_ratio": 1.4},
            "refrigerator": {"width_ratio": 0.9},
            "scissors": {"width_ratio": 8.0},
            "teddy bear": {"width_ratio": 3.2},
            "hair drier": {"width_ratio": 5.0},
            "toothbrush": {"width_ratio": 12.0},
        }
        
        self.load_model()
    
    def load_model(self):
        """Load ONNX models - COCO and Custom"""
        if not ONNX_AVAILABLE:
            print("⚠️ ONNX Runtime not available. Running in simulation mode.")
            self.session_coco = None
            self.session_custom = None
            self.has_models = False
            return
            
        try:
            # Load COCO model (yolov10s.onnx)
            coco_path = "models/yolov10s.onnx"
            if os.path.exists(coco_path):
                self.session_coco = ort.InferenceSession(coco_path, providers=['CPUExecutionProvider'])
                print("✓ COCO Model loaded: yolov10s.onnx")
            else:
                print("⚠️ COCO model not found: yolov10s.onnx")
                self.session_coco = None
            
            # Load Custom model (best.onnx)
            custom_path = "models/best.onnx"
            if os.path.exists(custom_path):
                self.session_custom = ort.InferenceSession(custom_path, providers=['CPUExecutionProvider'])
                print("✓ Custom Model loaded: best.onnx")
            else:
                print("⚠️ Custom model not found: best.onnx")
                self.session_custom = None
            
            # Check if at least one model is loaded
            self.has_models = (self.session_coco is not None) or (self.session_custom is not None)
            
            if self.has_models:
                print("🚀 DUAL MODEL SYSTEM READY!")
            else:
                print("⚠️ No models found. Running in simulation mode.")
            
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            print("📝 Running in simulation mode")
            self.session_coco = None
            self.session_custom = None
            self.has_models = False
    
    def preprocess_image(self, image):
        """Preprocess image for ONNX model with gamma correction"""
        # Resize image to model input size (640x640 for YOLOv10)
        input_size = 640
        original_height, original_width = image.shape[:2]
        
        # Calculate scaling factors
        scale = min(input_size / original_width, input_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image with letterbox
        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        pad_x = (input_size - new_width) // 2
        pad_y = (input_size - new_height) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
        
        # Apply gamma correction
        padded = cv2.LUT(padded, self.gamma_lut)
        
        # Normalize
        padded = padded.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to CHW format
        padded = np.transpose(padded, (2, 0, 1))
        padded = np.expand_dims(padded, axis=0)
        
        return padded, scale, pad_x, pad_y, original_width, original_height
    
    def postprocess_detections(self, outputs, scale, pad_x, pad_y, original_width, original_height, class_names, conf_threshold=0.5):
        """Process model outputs to get final detections"""
        detections = []
        
        try:
            # Get the output tensor
            output = outputs[0][0]  # Shape should be (num_detections, 6) for YOLOv10
            
            for detection in output:
                if len(detection) >= 6:
                    x1, y1, x2, y2, conf, class_id = detection[:6]
                    
                    if conf > conf_threshold:
                        # Convert coordinates back to original image space
                        x1 = (x1 - pad_x) / scale
                        y1 = (y1 - pad_y) / scale
                        x2 = (x2 - pad_x) / scale
                        y2 = (y2 - pad_y) / scale
                        
                        # Clip coordinates to image bounds
                        x1 = max(0, min(x1, original_width))
                        y1 = max(0, min(y1, original_height))
                        x2 = max(0, min(x2, original_width))
                        y2 = max(0, min(y2, original_height))
                        
                        # Calculate width and height
                        width = x2 - x1
                        height = y2 - y1
                        
                        if width > 0 and height > 0:
                            class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else "unknown"
                            
                            detections.append({
                                'bbox': [int(x1), int(y1), int(width), int(height)],
                                'confidence': float(conf),
                                'class': class_name,
                                'class_id': int(class_id)
                            })
            
        except Exception as e:
            print(f"Error in postprocessing: {e}")
        
        return detections
    
    def calculate_distance(self, obj_w, frame_w, label):
        """Calculate distance based on object width - supports all object classes"""
        try:
            if label in self.class_avg_sizes:
                ratio = self.class_avg_sizes[label].get("width_ratio", 2.0)
            else:
                ratio = 2.0
            obj_w *= ratio
            distance = (frame_w * 0.5) / np.tan(np.radians(70 / 2)) / (obj_w + 1e-6)
            return round(distance, 2)
        except:
            return 5.0
    
    def get_position(self, frame_w, cx):
        """Get object position: LEFT, FORWARD, or RIGHT"""
        if cx < frame_w / 3:
            return "LEFT"
        elif cx < frame_w * 2 / 3:
            return "FORWARD"
        else:
            return "RIGHT"
    
    def validate_trash_can(self, x1, y1, x2, y2, orig_h, orig_w):
        """Validate trash_can detection to reduce false positives"""
        try:
            box_height = y2 - y1
            box_width = x2 - x1
            box_center_y = (y1 + y2) // 2
            box_area = box_width * box_height
            frame_area = orig_w * orig_h
            
            # Reject if too high in frame
            if box_center_y < orig_h * 0.3:
                return False
            
            # Check aspect ratio
            aspect_ratio = box_height / (box_width + 1e-6)
            if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                return False
            
            # Reject if too large (likely misdetection)
            area_ratio = box_area / frame_area
            if area_ratio > 0.7:
                return False
            
            # Reject if too small
            min_area = frame_area * 0.001
            if box_area < min_area:
                return False
            
            return True
        except:
            return False
    
    def blur_person_face(self, frame, x1, y1, x2, y2):
        """Apply Gaussian blur to person's face area for privacy"""
        try:
            h = y2 - y1
            face_h = int(0.08 * h)
            if face_h > 0 and y1 + face_h <= frame.shape[0]:
                face = frame[y1:y1 + face_h, x1:x2]
                if face.size > 0:
                    frame[y1:y1 + face_h, x1:x2] = cv2.GaussianBlur(face, (15, 15), 0)
            return frame
        except Exception as e:
            print(f"Face blur error: {e}")
            return frame
    
    def setup_audio_system(self):
        """Initialize audio warning system with PowerShell TTS for Windows"""
        def audio_worker():
            try:
                last_announcement = 0
                
                while True:
                    current_time = time.time()
                    
                    if not self.audio_queue.empty() and (current_time - last_announcement) > self.warning_cooldown:
                        self.audio_is_speaking = True
                        
                        # Collect objects from queue
                        objects_by_position = {"LEFT": [], "FORWARD": [], "RIGHT": []}
                        seen_objects = set()
                        
                        while not self.audio_queue.empty():
                            try:
                                label, distance, position = self.audio_queue.get_nowait()
                                obj_key = f"{label}_{position}"
                                if obj_key not in seen_objects:
                                    objects_by_position[position].append(label)
                                    seen_objects.add(obj_key)
                            except:
                                break
                        
                        # Create announcement
                        message_parts = []
                        for position in ["LEFT", "FORWARD", "RIGHT"]:
                            if objects_by_position[position]:
                                # Take top 1 object per position
                                top_objects = objects_by_position[position][:1]
                                objects_list = ", ".join(top_objects)
                                message_parts.append(f"{objects_list} on {position}")
                        
                        # Speak using PowerShell TTS
                        if message_parts:
                            full_message = ". ".join(message_parts)
                            try:
                                ps_cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{full_message}")'
                                subprocess.Popen(['powershell', '-Command', ps_cmd],
                                               stdout=subprocess.DEVNULL,
                                               stderr=subprocess.DEVNULL)
                                print(f"🔊 Audio: {full_message}")
                            except Exception as e:
                                print(f"TTS error: {e}")
                            
                            last_announcement = time.time()
                        
                        self.audio_is_speaking = False
                    else:
                        time.sleep(0.05)
            except Exception as e:
                print(f"Audio worker error: {e}")
                self.audio_is_speaking = False
        
        # Start audio thread
        audio_thread = threading.Thread(target=audio_worker, daemon=True)
        audio_thread.start()
    
    def detect_objects(self, frame):
        """Perform object detection using dual models (COCO + Custom)"""
        if not self.has_models:
            # Simulation mode - return fake detections
            height, width = frame.shape[:2]
            fake_detections = [
                {
                    'bbox': [width//4, height//4, width//4, height//4],
                    'confidence': 0.85,
                    'class': 'person',
                    'class_id': 0,
                    'model': 'simulation'
                }
            ]
            return fake_detections
        
        all_detections = []
        
        try:
            # Preprocess image
            input_tensor, scale, pad_x, pad_y, orig_width, orig_height = self.preprocess_image(frame)
            
            # Run COCO model (every 3 frames for web performance)
            if self.session_coco is not None and self.frame_count % detect_every_n == 0:
                try:
                    input_name = self.session_coco.get_inputs()[0].name
                    outputs = self.session_coco.run(None, {input_name: input_tensor})
                    coco_detections = self.postprocess_detections(
                        outputs, scale, pad_x, pad_y, orig_width, orig_height,
                        self.class_names_coco, conf_threshold=0.35
                    )
                    for det in coco_detections:
                        det['model'] = 'coco'
                    all_detections.extend(coco_detections)
                except Exception as e:
                    print(f"COCO model error: {e}")
            
            # Run Custom model (every 4 frames to save performance)
            if self.session_custom is not None and self.frame_count % 4 == 0:
                try:
                    input_name = self.session_custom.get_inputs()[0].name
                    outputs = self.session_custom.run(None, {input_name: input_tensor})
                    custom_detections = self.postprocess_detections(
                        outputs, scale, pad_x, pad_y, orig_width, orig_height,
                        self.class_names_custom, conf_threshold=0.35
                    )
                    for det in custom_detections:
                        det['model'] = 'custom'
                    all_detections.extend(custom_detections)
                except Exception as e:
                    print(f"Custom model error: {e}")
            
            return all_detections
            
        except Exception as e:
            print(f"Error in detection: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels with distance, position, and audio warnings"""
        danger_objects = []
        orig_h, orig_w = frame.shape[:2]
        
        # Count persons for performance optimization
        person_count = sum(1 for d in detections if d.get('class') == 'person')
        apply_face_blur = person_count <= 3  # Only blur if 3 or fewer people
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            x, y, w, h = bbox
            cx = x + w // 2
            
            # Validate trash_can detections
            if class_name == 'trash_can':
                if not self.validate_trash_can(x, y, x + w, y + h, orig_h, orig_w):
                    continue
                if confidence < 0.4:  # Higher threshold for trash_can
                    continue
            
            # Calculate distance and position
            distance = self.calculate_distance(w, orig_w, class_name)
            position = self.get_position(orig_w, cx)
            
            # Track dangerous objects (< 2m)
            if distance < 2.0:
                danger_objects.append((class_name, distance, position))
            
            # Apply face blur for persons (skip if too many for performance)
            if class_name == 'person' and apply_face_blur:
                frame = self.blur_person_face(frame, x, y, x + w, y + h)
            
            # Color based on distance
            if distance < 2.0:
                color = (0, 0, 255)  # Red for danger
            else:
                color = (0, 255, 0)  # Green for safe
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label text
            label = f"{class_name} {confidence:.2f} {distance:.1f}m {position}"
            
            # Calculate label size and position
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y, label_height + 10)
            
            # Draw label background
            cv2.rectangle(frame, (x, label_y - label_height - 10), 
                         (x + label_width, label_y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, label_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Send dangerous objects to audio queue (top 5 closest)
        if danger_objects:
            sorted_objects = sorted(danger_objects, key=lambda x: x[1])[:5]
            for label, dist, pos in sorted_objects:
                self.audio_queue.put((label, dist, pos))
        
        return frame

    def generate_frames(self):
        """Generator function for video streaming with real-time detection"""
        while self.is_running and self.cap is not None and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Perform object detection (with caching for performance)
                # Run detection every 3 frames, use cached results otherwise
                if self.frame_count % detect_every_n == 0:
                    detections = self.detect_objects(frame)
                    self.last_detections = detections
                    self.detection_cache_count = 0
                else:
                    detections = self.last_detections
                    self.detection_cache_count += 1
                
                # Draw detections on frame
                frame = self.draw_detections(frame, detections)
                
                # Add mode info overlay
                mode_text = "DUAL MODEL: COCO + CUSTOM" if self.has_models else "SIMULATION MODE"
                cv2.putText(frame, f"{mode_text}", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Show detection stats with cache indicator
                danger_count = sum(1 for d in detections if 'bbox' in d and self.calculate_distance(d['bbox'][2], frame.shape[1], d['class']) < 2.0)
                cache_indicator = "[CACHED]" if self.detection_cache_count > 0 else "[LIVE]"
                stats_text = f"Frame: {self.frame_count} | Objects: {len(detections)} | Danger: {danger_count} {cache_indicator}"
                cv2.putText(frame, stats_text, (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show audio status
                audio_status = "SPEAKING" if self.audio_is_speaking else "READY"
                cv2.putText(frame, f"Audio: {audio_status}", (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Encode frame to JPEG with lower quality for better performance
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # No sleep - let it run at natural speed
            except Exception as e:
                print(f"Frame generation error: {e}")
                break
    
    def add_simulation_overlays(self, frame):
        """Add fake detection overlays for simulation (kept for backward compatibility)"""
        return frame  # Now handled by detect_objects method
    
    def start_camera_detection(self):
        """Start real-time camera detection"""
        try:
            self.current_mode = "Camera"
            self.status = "🟡 Initializing camera..."
            
            self.cap = None
            
            # Try DroidCam first if IP is configured
            if self.droidcam_ip and self.droidcam_ip.strip():
                droidcam_url = f"http://{self.droidcam_ip.strip()}:4747/video"
                print(f"Trying DroidCam URL: {droidcam_url}")
                self.cap = cv2.VideoCapture(droidcam_url)
                if self.cap.isOpened():
                    print("✓ Connected to DroidCam successfully")
                else:
                    print("⚠️ DroidCam connection failed, trying default camera...")
                    self.cap.release()
                    self.cap = None

            # Fallback to default camera if DroidCam failed or not configured
            if self.cap is None or not self.cap.isOpened():
                print("Trying default camera (index 0)...")
                self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():
                raise Exception("Cannot connect to camera or DroidCam")
            
            self.is_running = True
            self.status = "🟢 Camera detection running"
            self.frame_count = 0
            
            return {"success": True, "message": "Camera detection started"}
            
        except Exception as e:
            self.stop_detection()
            return {"success": False, "message": f"Camera error: {str(e)}"}
    
    def start_video_detection(self, video_path):
        """Start video file detection"""
        try:
            if not os.path.exists(video_path):
                return {"success": False, "message": "Video file not found"}
            
            self.current_mode = "Video"
            self.video_path = video_path
            self.status = f"🟡 Loading video: {os.path.basename(video_path)}"
            
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise Exception(f"Cannot open video file: {video_path}")
            
            self.is_running = True
            self.status = "🟢 Video detection running"
            self.frame_count = 0
            
            return {"success": True, "message": f"Video detection started: {os.path.basename(video_path)}"}
            
        except Exception as e:
            self.stop_detection()
            return {"success": False, "message": f"Video error: {str(e)}"}
    
    def stop_detection(self):
        """Stop current detection"""
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.current_mode = None
        self.video_path = None
        self.status = "🔴 Detection stopped"
        self.frame_count = 0
        
        return {"success": True, "message": "Detection stopped"}
    
    def get_status(self):
        """Get current status"""
        return {
            "status": self.status,
            "is_running": self.is_running,
            "current_mode": self.current_mode,
            "droidcam_ip": self.droidcam_ip,
            "has_models": self.has_models,
            "frame_count": self.frame_count
        }

# Global app instance
wesee_app = WeSeeWebApp()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/status')
def status():
    """Get current status"""
    return jsonify(wesee_app.get_status())

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera detection"""
    if wesee_app.is_running:
        return jsonify({"success": False, "message": "Detection is already running"})
    
    result = wesee_app.start_camera_detection()
    return jsonify(result)

@app.route('/start_video', methods=['POST'])
def start_video():
    """Start video detection"""
    if wesee_app.is_running:
        return jsonify({"success": False, "message": "Detection is already running"})
    
    data = request.get_json()
    video_path = data.get('video_path', '')
    
    if not video_path:
        return jsonify({"success": False, "message": "No video path provided"})
    
    result = wesee_app.start_video_detection(video_path)
    return jsonify(result)

@app.route('/stop', methods=['POST'])
def stop():
    """Stop detection"""
    result = wesee_app.stop_detection()
    return jsonify(result)

@app.route('/set_droidcam_ip', methods=['POST'])
def set_droidcam_ip():
    """Set DroidCam IP address"""
    data = request.get_json()
    new_ip = data.get('ip', '').strip()
    
    if new_ip:
        wesee_app.droidcam_ip = new_ip
        return jsonify({"success": True, "message": f"DroidCam IP updated to: {new_ip}"})
    else:
        return jsonify({"success": False, "message": "Invalid IP address"})

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    if not wesee_app.is_running:
        return Response("No active video stream", mimetype='text/plain')
    
    return Response(wesee_app.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video file upload"""
    if 'video' not in request.files:
        return jsonify({"success": False, "message": "No video file uploaded"})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"success": False, "message": "No video file selected"})
    
    # Save uploaded file
    upload_dir = 'uploads'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)
    
    return jsonify({
        "success": True, 
        "message": f"Video uploaded successfully: {filename}",
        "filepath": filepath
    })

if __name__ == '__main__':
    print("🚀 Starting WeSee Web Application...")
    print("📱 Access at: http://127.0.0.1:5001")
    print("📹 Real-time Detection: Camera/DroidCam support")
    print("🎬 Video Detection: Upload and process video files")
    print("⚙️ Settings: Configure DroidCam IP address")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    app.run(debug=False, host='127.0.0.1', port=5001, threaded=True)
