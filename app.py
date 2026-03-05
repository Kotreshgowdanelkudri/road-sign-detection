"""
Flask Web Application for Road Sign Detection
Supports webcam, image upload, and video upload
"""
import os
import base64
import cv2
import numpy as np
import threading
import pyttsx3
from flask import Flask, render_template, Response, request, jsonify, send_file
from werkzeug.utils import secure_filename

from config import (
    MODEL_PATH, UPLOAD_FOLDER, MAX_UPLOAD_SIZE,
    ALLOWED_IMAGE_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS,
    IMAGE_SIZE, CONFIDENCE_THRESHOLD, MIN_DISPLAY_CONFIDENCE,
    FRAME_SKIP, FRAME_WIDTH, FRAME_HEIGHT
)
from utils.preprocessing import preprocess_frame_for_inference, enhance_image_for_detection
from utils.roi_detection import (
    detect_roi_color_based, extract_roi, draw_detection, non_max_suppression,
    get_smart_region_candidates
)
from utils.label_mapping import load_label_mapping

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
model = None
label_mapping = None
camera = None
camera_lock = threading.Lock()
detection_active = False
# Per-source detection labels — kept separate so each page's Speak button
# only announces its own result, not a stale result from another page.
last_detected_label_live  = "No detection yet"
last_detected_label_video = "No detection yet"
last_detected_label_image = "No detection yet"
# Backward-compat alias used by process_frame (live/video both write here as well)
last_detected_label = "No detection yet"
current_video_path = None
tts_engine = None
tts_lock = threading.Lock()


def initialize_model():
    """Load the trained model and label mapping"""
    global model, label_mapping
    
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model not found at {MODEL_PATH}")
        print("Please train the model first using train_model.py")
        return False
    
    try:
        # Use custom model loader to handle TensorFlow version compatibility
        from utils.model_loader import load_compatible_model
        
        # Determine number of classes from label mapping
        label_mapping = load_label_mapping()
        num_classes = len(label_mapping)
        print(f"Label mapping loaded: {num_classes} classes")
        
        # Load model with custom loader
        model = load_compatible_model(MODEL_PATH, num_classes)
        
        if model is None:
            print("Failed to load model")
            return False
        
        print(f"✓ Model loaded successfully!")
        print(f"  Output shape: {model.output_shape}")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def initialize_tts():
    """Initialize text-to-speech engine (validates pyttsx3 is available)"""
    global tts_engine
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        engine.stop()
        tts_engine = True  # Flag that TTS is available; we reinit per-call
        print("TTS engine initialized")
    except Exception as e:
        tts_engine = None
        print(f"Error initializing TTS: {e}")


def _speak_text(text):
    """Speak text by creating a fresh pyttsx3 engine each call.

    pyttsx3 on Windows enters a broken state after runAndWait() completes
    and cannot be reliably reused. Reinitializing solves repeated-press silence.
    Also replaces underscores with spaces so dataset folder names are read naturally.
    """
    # Replace underscores with spaces for natural speech
    text = text.replace('_', ' ')
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"TTS speak error: {e}")


def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def predict_sign(roi, enhance=False):
    """
    Predict traffic sign from ROI
    Args:
        roi: region of interest (BGR image)
        enhance: if True, apply sharpening/contrast for blurry images
    Returns: (label_name, confidence)
    """
    global model, label_mapping
    
    if model is None or label_mapping is None:
        return "Model not loaded", 0.0
    
    try:
        # Preprocess ROI (optionally enhance for blurry/difficult images)
        processed = preprocess_frame_for_inference(roi, IMAGE_SIZE, enhance=enhance)
        
        # Predict
        predictions = model.predict(processed, verbose=0)
        class_id = np.argmax(predictions[0])
        confidence = float(predictions[0][class_id])
        
        # Get label name
        label_name = label_mapping.get(int(class_id), f"Unknown {class_id}")
        
        return label_name, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0


def process_frame(frame, use_smart_fallback=True, source='live'):
    """
    Process a single frame for detection
    Args:
        frame: input frame (BGR)
        use_smart_fallback: if True, use multi-region candidates when color ROI fails
        source: 'live' or 'video' — determines which per-source label is updated
    Returns: annotated frame, detected_label
    """
    global last_detected_label, last_detected_label_live, last_detected_label_video
    
    # Resize frame for performance
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    
    # Detect ROIs (color-based for red/blue/yellow signs)
    bboxes = detect_roi_color_based(frame)
    bboxes = non_max_suppression(bboxes)
    used_smart_fallback = False
    
    # FALLBACK: If no color ROIs, use smart multi-region (center + grid)
    if len(bboxes) == 0 and use_smart_fallback:
        bboxes = get_smart_region_candidates(frame, max_candidates=12)
        used_smart_fallback = True
    elif len(bboxes) == 0:
        h, w = frame.shape[:2]
        bboxes = [(0, 0, w, h)]
    
    detected_label = None
    best_confidence = 0.0
    best_bbox = None
    
    # Process each ROI
    for bbox in bboxes:
        roi = extract_roi(frame, bbox)
        if roi.size == 0:
            continue
        
        # Use blur enhancement when we fell back to smart regions (often difficult images)
        enhance = used_smart_fallback
        label, confidence = predict_sign(roi, enhance=enhance)
        
        # Track best detection (use MIN_DISPLAY_CONFIDENCE for showing best guess)
        if confidence > best_confidence:
            best_confidence = confidence
            detected_label = label
            best_bbox = bbox
        
        # Draw when above strong threshold
        if confidence > CONFIDENCE_THRESHOLD:
            frame = draw_detection(frame, bbox, label, confidence, draw_bbox=True)
    
    # Show best guess even if below threshold (helps with difficult/blurry images)
    if detected_label and best_confidence >= MIN_DISPLAY_CONFIDENCE and best_confidence < CONFIDENCE_THRESHOLD and best_bbox:
        frame = draw_detection(frame, best_bbox, f"{detected_label}?", best_confidence, draw_bbox=True)
    
    if detected_label:
        last_detected_label = detected_label  # shared alias
        if source == 'video':
            last_detected_label_video = detected_label
        else:
            last_detected_label_live = detected_label
    
    return frame, detected_label


class VideoCamera:
    """Video camera handler for webcam or video file"""
    
    def __init__(self, source=0):
        """
        Initialize camera
        Args:
            source: 0 for webcam, or path to video file
        """
        self.source = source
        self.video = cv2.VideoCapture(source)
        self.frame_count = 0
        
        if not self.video.isOpened():
            raise ValueError(f"Could not open video source: {source}")
    
    def release(self):
        """Explicitly release camera resources"""
        if hasattr(self, 'video') and self.video.isOpened():
            self.video.release()
    
    def __del__(self):
        """Release camera on destruction"""
        self.release()
    
    def get_frame(self):
        """
        Get next frame from camera
        Returns: JPEG encoded frame
        """
        success, frame = self.video.read()
        
        if not success:
            return None
        
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_count % FRAME_SKIP != 0:
            # Just encode without processing
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        
        # Process frame
        if detection_active:
            frame, _ = process_frame(frame)
        
        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


def generate_frames():
    """Generator for video streaming. Yields placeholder when camera inactive."""
    global camera
    
    while True:
        with camera_lock:
            if camera is None:
                # Yield placeholder so img doesn't show broken image
                placeholder = get_placeholder_frame()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
                break
            
            frame = camera.get_frame()
        
        if frame is None:
            break
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def get_placeholder_frame():
    """Generate a placeholder image (640x480) when camera is not active"""
    img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    img[:] = (40, 40, 60)  # Dark gray-blue
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Click 'Start Detection' to begin"
    (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
    x = (FRAME_WIDTH - tw) // 2
    y = FRAME_HEIGHT // 2
    cv2.putText(img, text, (x, y), font, 0.8, (255, 255, 255), 2)
    ret, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes()


def create_placeholder_response():
    """Create Flask response for placeholder image"""
    import io
    data = get_placeholder_frame()
    return send_file(
        io.BytesIO(data),
        mimetype='image/jpeg',
        as_attachment=False
    )


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/video_placeholder')
def video_placeholder():
    """Static placeholder image (use when camera is not active - avoids fetch errors)"""
    return create_placeholder_response()


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start', methods=['POST'])
def start_detection():
    """Start webcam detection"""
    global camera, detection_active, current_video_path
    
    try:
        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None
            
            # Determine source
            source = 0  # Default to webcam
            if current_video_path and os.path.exists(current_video_path):
                source = current_video_path
            
            # Start camera
            camera = VideoCamera(source)
            detection_active = True
        
        return jsonify({'status': 'success', 'message': 'Detection started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/stop', methods=['POST'])
def stop_detection():
    """Stop detection"""
    global camera, detection_active
    
    try:
        with camera_lock:
            detection_active = False
            if camera is not None:
                camera.release()
                camera = None
        
        return jsonify({'status': 'success', 'message': 'Detection stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload and detection"""
    global last_detected_label_image
    
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    
    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400
    
    try:
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'status': 'error', 'message': 'Could not read image'}), 400
        
        # Process image (source='image' keeps it isolated from live/video labels)
        processed_image, detected_label = process_frame(image, source='image')
        
        # Encode annotated image as base64 for display
        ret, buffer = cv2.imencode('.jpg', processed_image)
        image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8') if ret else None
        
        # Update image-specific label only
        if detected_label:
            last_detected_label_image = detected_label
        
        return jsonify({
            'status': 'success',
            'label': detected_label or 'No sign detected',
            'last_label': last_detected_label_image,
            'image_preview': f"data:image/jpeg;base64,{image_base64}" if image_base64 else None
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload"""
    global current_video_path
    
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    
    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400
    
    try:
        # Save video file (use unique name to avoid overwrite)
        base = secure_filename(file.filename)
        name, ext = os.path.splitext(base)
        filename = f"{name}_{os.urandom(4).hex()}{ext}" if name else f"video_{os.urandom(6).hex()}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        current_video_path = filepath
        
        return jsonify({
            'status': 'success',
            'message': 'Video uploaded successfully',
            'filename': base or filename
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    """
    Analyze uploaded video by sampling 50 frames evenly across its duration.
    Returns detection results for each sampled frame.
    """
    global current_video_path

    if not current_video_path or not os.path.exists(current_video_path):
        return jsonify({'status': 'error', 'message': 'No video uploaded. Please upload a video first.'}), 400

    try:
        cap = cv2.VideoCapture(current_video_path)
        if not cap.isOpened():
            return jsonify({'status': 'error', 'message': 'Could not open video file.'}), 500

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0:
            fps = 25.0
        num_samples = 50

        if total_frames <= 0:
            cap.release()
            return jsonify({'status': 'error', 'message': 'Video has no frames or could not be read.'}), 500

        # Compute evenly spaced frame indices across the video
        if total_frames <= num_samples:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / num_samples
            frame_indices = [int(i * step) for i in range(num_samples)]

        detections = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Run detection (source='video' so it updates last_detected_label_video)
            _, detected_label = process_frame(frame, source='video')

            # Get confidence by running predictions on same regions as process_frame
            frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            bboxes = detect_roi_color_based(frame_resized)
            bboxes = non_max_suppression(bboxes)
            if len(bboxes) == 0:
                bboxes = get_smart_region_candidates(frame_resized, max_candidates=12)

            best_conf = 0.0
            best_label = detected_label or 'No sign detected'
            # Use blur enhancement for video frames (often motion blur)
            for bbox in bboxes:
                roi = extract_roi(frame_resized, bbox)
                if roi.size == 0:
                    continue
                lbl, conf = predict_sign(roi, enhance=True)
                if conf > best_conf and conf >= MIN_DISPLAY_CONFIDENCE:
                    best_conf = conf
                    best_label = lbl

            timestamp = round(idx / fps, 2)
            detections.append({
                'frame_number': idx,
                'timestamp': timestamp,
                'label': best_label,
                'confidence': round(best_conf, 4)
            })

        cap.release()

        return jsonify({
            'status': 'success',
            'frames_analyzed': len(detections),
            'total_frames': total_frames,
            'detections': detections
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/speak', methods=['POST'])
def speak():
    """Trigger voice alert for the detected sign on a specific page.

    Expects JSON body: { "source": "live" | "video" | "image" }
    Falls back to the shared last_detected_label when source is not specified.
    """
    global last_detected_label_live, last_detected_label_video, last_detected_label_image
    global last_detected_label, tts_engine

    if tts_engine is None:
        return jsonify({'status': 'error', 'message': 'TTS not initialized'}), 500

    # Determine which label to speak based on the caller's source
    data = request.get_json(silent=True) or {}
    source = data.get('source', 'live')  # default to live for backward compat

    if source == 'image':
        label_to_speak = last_detected_label_image
    elif source == 'video':
        label_to_speak = last_detected_label_video
    else:
        label_to_speak = last_detected_label_live

    # Check if there's a valid detection to speak
    if not label_to_speak or label_to_speak in ("No detection yet", "No sign detected"):
        return jsonify({
            'status': 'error',
            'message': f'No sign detected yet on the {source} page. Please detect a sign first.'
        }), 400

    try:
        speak_label = label_to_speak  # capture for closure
        def speak_async():
            with tts_lock:
                _speak_text(f"Detected: {speak_label}")

        thread = threading.Thread(target=speak_async)
        thread.daemon = True
        thread.start()

        return jsonify({
            'status': 'success',
            'message': f'Speaking: {speak_label}'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/tts_test', methods=['POST'])
def tts_test():
    """Play a short test phrase to verify speakers"""
    global tts_engine

    if tts_engine is None:
        return jsonify({'status': 'error', 'message': 'TTS not initialized'}), 500

    try:
        def speak_async():
            with tts_lock:
                _speak_text("This is a test of the road sign voice alert system.")

        thread = threading.Thread(target=speak_async)
        thread.daemon = True
        thread.start()

        return jsonify({
            'status': 'success',
            'message': 'Playing test voice message'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/status', methods=['GET'])
def get_status():
    """Get current detection status"""
    return jsonify({
        'detection_active': detection_active,
        'last_detected_label': last_detected_label,          # live/video shared (backward compat)
        'last_detected_label_live':  last_detected_label_live,
        'last_detected_label_video': last_detected_label_video,
        'last_detected_label_image': last_detected_label_image,
        'model_loaded': model is not None,
        'label_count': len(label_mapping) if label_mapping else 0
    })


if __name__ == '__main__':
    print("Initializing Road Sign Detection Application...")

    # Initialize model
    if not initialize_model():
        print("WARNING: Running without model. Please train the model first.")

    # Initialize TTS
    initialize_tts()

    print("\nStarting Flask server...")
    print("Open http://localhost:5000 in your browser")

    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
