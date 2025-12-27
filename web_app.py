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
from collections import deque

##########################################################################################
# CUDA/GPU SETUP
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
cudnn_bin = r"C:\Program Files\NVIDIA\CUDNN\v9.17\bin\12.9"
if os.path.exists(cuda_bin): 
    os.add_dll_directory(cuda_bin)
    print(f"✓ Added CUDA bin directory: {cuda_bin}")
if os.path.exists(cudnn_bin): 
    os.add_dll_directory(cudnn_bin)
    print(f"✓ Added cuDNN bin directory: {cudnn_bin}")

# Try to import onnxruntime with GPU support, fallback if not available
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    
    # Check available providers
    providers = ort.get_available_providers()
    print(f"🔍 Available ONNX providers: {providers}")
    
    # Setup providers with GPU priority
    if 'CUDAExecutionProvider' in providers:
        EXECUTION_PROVIDERS = [('CUDAExecutionProvider', {}), 'CPUExecutionProvider']
        print("🚀 GPU (CUDA) support enabled for inference")
    else:
        EXECUTION_PROVIDERS = ['CPUExecutionProvider']
        print("⚠️ GPU not available, using CPU for inference")
except ImportError:
    print("⚠️ onnxruntime not available. Running in simulation mode only.")
    ONNX_AVAILABLE = False
    EXECUTION_PROVIDERS = ['CPUExecutionProvider']

app = Flask(__name__)

##########################################################################################
# GAMMA CORRECTION LUT (Pre-calculated for performance)
gamma = 0.8
invGamma = 1.0 / gamma
GLOBAL_GAMMA_LUT = np.array([((i / 255.0) ** invGamma) * 255
                            for i in range(256)]).astype("uint8")

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
        
        # Use global gamma LUT for better performance
        self.gamma_lut = GLOBAL_GAMMA_LUT
        
        # Distance warning settings
        self.warning_distance_cm = 200  # 2 meters
        self.last_warning_time = 0
        self.warning_cooldown = 5  # seconds
        
        # Audio warning queue
        self.audio_queue = Queue()
        self.audio_is_speaking = False
        self.setup_audio_system()
        
        # Performance optimization - remove caching variables since we run detection every frame
        self.last_detections = []
        
        # FPS tracking for accurate measurement
        self.fps_history = deque(maxlen=30)  # Track last 30 frames
        self.total_frames = 0
        
        # Pre-allocate buffer for better performance
        self.jpeg_buffer = None
        
        # Frame skipping for performance
        self.skip_detection_counter = 0
        self.last_detections_cache = []
        
        # Desktop mode option
        self.desktop_mode = False
        self.desktop_window_name = "DIPR Desktop View"
        
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
            print("\n=== LOADING MODELS ===")
            # Load COCO model (yolov10s.onnx) with GPU support
            coco_path = "yolov10s.onnx"  # Check root directory first
            if not os.path.exists(coco_path):
                coco_path = "models/yolov10s.onnx"  # Fallback to models directory
            
            if os.path.exists(coco_path):
                self.session_coco = ort.InferenceSession(coco_path, providers=EXECUTION_PROVIDERS)
                print(f"✓ COCO Model (yolov10s.onnx) loaded with providers: {self.session_coco.get_providers()}")
            else:
                print("⚠️ COCO model not found: yolov10s.onnx")
                self.session_coco = None
            
            # Load Custom model (best.onnx) with GPU support
            custom_path = "best.onnx"  # Check root directory first
            if not os.path.exists(custom_path):
                custom_path = "models/best.onnx"  # Fallback to models directory
            
            if os.path.exists(custom_path):
                self.session_custom = ort.InferenceSession(custom_path, providers=EXECUTION_PROVIDERS)
                print(f"✓ Custom Model (best.onnx) loaded with providers: {self.session_custom.get_providers()}")
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
    
    def preprocess_frame(self, frame, orig_w, orig_h):
        """Resize + Letterbox + Gamma correction - matches provided code exactly"""
        r = min(640 / orig_w, 640 / orig_h)
        new_size = (int(orig_w * r), int(orig_h * r))

        resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)

        dw = (640 - new_size[0]) // 2
        dh = (640 - new_size[1]) // 2
        canvas[dh:dh + new_size[1], dw:dw + new_size[0]] = resized

        canvas = cv2.LUT(canvas, self.gamma_lut)

        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        return blob, r, dw, dh
    
    def scale_coords(self, x1, y1, x2, y2, r, dw, dh, w, h):
        """Scale coordinates from model space back to original image space"""
        return (
            max(0, int((x1 - dw) / r)),
            max(0, int((y1 - dh) / r)),
            min(w, int((x2 - dw) / r)),
            min(h, int((y2 - dh) / r))
        )

    def postprocess_detections(self, outputs, r, dw, dh, original_width, original_height, class_names, conf_threshold=0.35):
        """Process model outputs to get final detections using improved coordinate scaling"""
        detections = []
        
        try:
            # Get the output tensor
            output = outputs[0][0]  # Shape should be (num_detections, 6) for YOLOv10
            
            for detection in output:
                if len(detection) >= 6:
                    x1, y1, x2, y2, conf, class_id = detection[:6]
                    
                    if conf > conf_threshold:
                        # Scale coordinates back to original image space
                        ix1, iy1, ix2, iy2 = self.scale_coords(x1, y1, x2, y2, r, dw, dh, original_width, original_height)
                        
                        # Calculate width and height
                        width = ix2 - ix1
                        height = iy2 - iy1
                        
                        if width > 0 and height > 0:
                            class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else "unknown"
                            
                            detections.append({
                                'bbox': [int(ix1), int(iy1), int(width), int(height)],
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
            # Get frame dimensions
            orig_height, orig_width = frame.shape[:2]
            
            # Preprocess image using improved method
            input_tensor, r, dw, dh = self.preprocess_frame(frame, orig_width, orig_height)
            
            # Run COCO model every frame (like standalone script)
            if self.session_coco is not None:
                try:
                    input_name = self.session_coco.get_inputs()[0].name
                    outputs = self.session_coco.run(None, {input_name: input_tensor})
                    coco_detections = self.postprocess_detections(
                        outputs, r, dw, dh, orig_width, orig_height,
                        self.class_names_coco, conf_threshold=0.45
                    )
                    for det in coco_detections:
                        det['model'] = 'coco'
                    all_detections.extend(coco_detections)
                except Exception as e:
                    print(f"COCO model error: {e}")
            
            # Run Custom model every 2nd frame (like standalone script)
            if self.session_custom is not None and self.frame_count % 2 == 0:
                try:
                    input_name = self.session_custom.get_inputs()[0].name
                    outputs = self.session_custom.run(None, {input_name: input_tensor})
                    custom_detections = self.postprocess_detections(
                        outputs, r, dw, dh, orig_width, orig_height,
                        self.class_names_custom, conf_threshold=0.45
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
                # CLEAR CAMERA BUFFER to reduce delay (grab latest frame only)
                for _ in range(2):  # Skip 2 buffered frames
                    self.cap.grab()
                    
                start_time = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # EXTREME PERFORMANCE: Skip detection every 3rd frame
                self.skip_detection_counter += 1
                if self.skip_detection_counter % 3 == 0:
                    # Run full detection
                    detections = self.detect_objects(frame)
                    self.last_detections_cache = detections
                else:
                    # Reuse previous detections for 3x speed boost
                    detections = self.last_detections_cache
                
                # Draw detections on frame
                frame = self.draw_detections(frame, detections)
                
                # Calculate accurate FPS with moving average
                frame_time = time.time() - start_time
                self.fps_history.append(frame_time)
                self.total_frames += 1
                
                # Use moving average for stable FPS reading
                if len(self.fps_history) > 5:
                    avg_frame_time = sum(self.fps_history) / len(self.fps_history)
                    fps = 1 / avg_frame_time
                else:
                    fps = 1 / frame_time
                
                # Display info every 5th frame to reduce text rendering overhead
                if self.skip_detection_counter % 5 == 0:
                    info_text = f"FPS: {fps:.1f} | DUAL MODEL"
                    cv2.putText(frame, info_text, (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Count dangerous objects
                    danger_count = sum(1 for d in detections if 'bbox' in d and self.calculate_distance(d['bbox'][2], frame.shape[1], d['class']) < 2.0)
                    stats_text = f"Objects < 2m: {danger_count}"
                    cv2.putText(frame, stats_text, (20, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Desktop view option (cv2.imshow alongside web streaming)
                if self.desktop_mode:
                    cv2.imshow(self.desktop_window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop_detection()
                        break
                
                # EXTREME OPTIMIZATION: Small frame + lowest quality
                small_frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_NEAREST)  # Nearest = fastest
                ret, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # No sleep - let it run at natural speed
            except Exception as e:
                print(f"Frame generation error: {e}")
                break
    
    def enable_desktop_view(self):
        """Enable cv2.imshow window alongside web streaming"""
        self.desktop_mode = True
        print("🖥️  Desktop view enabled - cv2.imshow window will open")
        
    def disable_desktop_view(self):
        """Disable cv2.imshow window"""
        self.desktop_mode = False
        cv2.destroyWindow(self.desktop_window_name)
        print("🌐 Desktop view disabled - web only mode")

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
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow for Windows - faster

            if not self.cap.isOpened():
                raise Exception("Cannot connect to camera or DroidCam")
            
            # OPTIMIZE CAMERA FOR LOW DELAY
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer = less delay  
            self.cap.set(cv2.CAP_PROP_FPS, 30)        # Set to 30fps
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower res = faster
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Lower res = faster
            print("✓ Camera optimized for minimal delay")
            
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

@app.route('/desktop/enable')
def enable_desktop():
    """Enable cv2.imshow window for desktop viewing"""
    app_instance.enable_desktop_view()
    return jsonify({"status": "success", "message": "Desktop view enabled"})

@app.route('/desktop/disable') 
def disable_desktop():
    """Disable cv2.imshow window"""
    app_instance.disable_desktop_view()
    return jsonify({"status": "success", "message": "Desktop view disabled"})

if __name__ == '__main__':
    print("🚀 Starting WeSee Web Application with GPU Support...")
    print("📱 Access at: http://127.0.0.1:5001")
    print("📹 Real-time Detection: Camera/DroidCam support")
    print("🎬 Video Detection: Upload and process video files")
    print("⚙️ Settings: Configure DroidCam IP address")
    print("🎮 GPU Acceleration: CUDA + cuDNN optimized")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    app.run(debug=False, host='127.0.0.1', port=5001, threaded=True)
