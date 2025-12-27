from flask import Flask, render_template, request, jsonify, Response
import cv2
import threading
import time
import json
import os
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

class WeSeeWebApp:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.detection_thread = None
        self.droidcam_ip = "169.254.96.95"
        self.current_mode = None
        self.video_path = None
        self.status = "🔴 Ready"
        self.frame_count = 0
        self.has_models = False
        
        # Model configurations
        self.model_path = None
        self.session = None
        self.class_names = [
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
        
        # Distance warning settings
        self.warning_distance_cm = 100
        self.last_warning_time = 0
        self.warning_cooldown = 2  # seconds
        
        self.load_model()
    
    def load_model(self):
        """Load ONNX model"""
        if not ONNX_AVAILABLE:
            print("⚠️ ONNX Runtime not available. Running in simulation mode.")
            self.session = None
            self.has_models = False
            return
            
        try:
            # Try loading the best model first
            model_path = "models/best.onnx"
            if os.path.exists(model_path):
                self.model_path = model_path
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                print("✓ Model loaded successfully: best.onnx")
                self.has_models = True
                return
            
            # Fallback to yolov10s model
            model_path = "models/yolov10s.onnx"
            if os.path.exists(model_path):
                self.model_path = model_path
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                print("✓ Model loaded successfully: yolov10s.onnx")
                self.has_models = True
                return
            
            print("⚠️ No model found. Running in simulation mode.")
            self.session = None
            self.has_models = False
            
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            print("📝 Running in simulation mode")
            self.session = None
            self.has_models = False
    
    def preprocess_image(self, image):
        """Preprocess image for ONNX model"""
        # Resize image to model input size (640x640 for YOLOv10)
        input_size = 640
        original_height, original_width = image.shape[:2]
        
        # Calculate scaling factors
        scale = min(input_size / original_width, input_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height))
        
        # Create padded image
        padded = np.full((input_size, input_size, 3), 128, dtype=np.uint8)
        
        # Calculate padding offsets
        pad_x = (input_size - new_width) // 2
        pad_y = (input_size - new_height) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
        
        # Convert to RGB and normalize
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to CHW format
        padded = np.transpose(padded, (2, 0, 1))
        padded = np.expand_dims(padded, axis=0)
        
        return padded, scale, pad_x, pad_y, original_width, original_height
    
    def postprocess_detections(self, outputs, scale, pad_x, pad_y, original_width, original_height, conf_threshold=0.5):
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
                            class_name = self.class_names[int(class_id)] if int(class_id) < len(self.class_names) else "unknown"
                            
                            detections.append({
                                'bbox': [int(x1), int(y1), int(width), int(height)],
                                'confidence': float(conf),
                                'class': class_name,
                                'class_id': int(class_id)
                            })
            
        except Exception as e:
            print(f"Error in postprocessing: {e}")
        
        return detections
    
    def estimate_distance(self, bbox, image_height):
        """Estimate distance to object based on bounding box size"""
        try:
            bbox_height = bbox[3]
            # Simple distance estimation based on bbox height
            # Assuming average person height is 170cm
            if bbox_height > 0:
                # This is a rough estimation - you might want to calibrate this
                distance_cm = (170 * image_height) / (bbox_height * 10)
                return min(distance_cm, 500)  # Cap at 5 meters
            return 500
        except:
            return 500
    
    def play_audio_warning_macos(self, message):
        """Play audio warning on macOS using 'say' command"""
        try:
            current_time = time.time()
            if current_time - self.last_warning_time > self.warning_cooldown:
                subprocess.run(['say', message], check=False)
                self.last_warning_time = current_time
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def detect_objects(self, frame):
        """Perform object detection on a frame"""
        if self.session is None:
            # Simulation mode - return fake detections
            height, width = frame.shape[:2]
            fake_detections = [
                {
                    'bbox': [width//4, height//4, width//4, height//4],
                    'confidence': 0.85,
                    'class': 'person',
                    'class_id': 0
                }
            ]
            return fake_detections
        
        try:
            # Preprocess image
            input_tensor, scale, pad_x, pad_y, orig_width, orig_height = self.preprocess_image(frame)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            
            # Postprocess detections
            detections = self.postprocess_detections(outputs, scale, pad_x, pad_y, orig_width, orig_height)
            
            return detections
            
        except Exception as e:
            print(f"Error in detection: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            x, y, w, h = bbox
            
            # Estimate distance
            distance_cm = self.estimate_distance(bbox, frame.shape[0])
            
            # Check if warning is needed
            if class_name == 'person' and distance_cm < self.warning_distance_cm:
                color = (0, 0, 255)  # Red for warning
                self.play_audio_warning_macos(f"Warning: Person detected at {distance_cm:.0f} centimeters")
            else:
                color = (0, 255, 0)  # Green for normal detection
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"
            if distance_cm < 500:
                label += f" ({distance_cm:.0f}cm)"
            
            # Calculate label size and position
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y, label_height + 10)
            
            # Draw label background
            cv2.rectangle(frame, (x, label_y - label_height - 10), 
                         (x + label_width, label_y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, label_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def generate_frames(self):
        """Generator function for video streaming with real-time detection"""
        while self.is_running and self.cap is not None and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Perform object detection
                detections = self.detect_objects(frame)
                
                # Draw detections on frame
                frame = self.draw_detections(frame, detections)
                
                # Add mode info overlay
                mode_text = "AI DETECTION" if self.has_models else "SIMULATION MODE"
                cv2.putText(frame, f"{mode_text} - {self.current_mode}", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Frame: {self.frame_count} | Detections: {len(detections)}", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
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
            
            # Try default camera first
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                # Try DroidCam
                droidcam_url = f"http://{self.droidcam_ip}:4747/video"
                print(f"Trying DroidCam URL: {droidcam_url}")
                self.cap = cv2.VideoCapture(droidcam_url)
            
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
    print("📱 Access at: http://127.0.0.1:5000")
    print("📹 Real-time Detection: Camera/DroidCam support")
    print("🎬 Video Detection: Upload and process video files")
    print("⚙️ Settings: Configure DroidCam IP address")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)
