"""
Flask Web Application for Road Sign Detection
Supports webcam (live), image upload, and video upload

How this app works:
 - Flask handles HTTP requests from the browser
 - When the user starts detection, we open a camera or video file using OpenCV
 - Each frame is analyzed: first we look for sign-colored regions (ROI detection)
 - The cropped sign region is passed through the trained CNN model for classification
 - Results are streamed back to the browser as a live video feed (MJPEG)
 - A separate "Speak" button triggers pyttsx3 to read the detected sign name aloud
"""
import os
import base64
import cv2
import numpy as np
import threading
import pyttsx3
from flask import Flask, render_template, Response, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Pull in all the settings from config.py
from config import (
    MODEL_PATH, UPLOAD_FOLDER, MAX_UPLOAD_SIZE,
    ALLOWED_IMAGE_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS,
    IMAGE_SIZE, CONFIDENCE_THRESHOLD, MIN_DISPLAY_CONFIDENCE,
    FRAME_SKIP, FRAME_WIDTH, FRAME_HEIGHT
)
# Utility functions for image preparation and sign region finding
from utils.preprocessing import preprocess_frame_for_inference, enhance_image_for_detection
from utils.roi_detection import (
    detect_roi_color_based, extract_roi, draw_detection, non_max_suppression,
    get_smart_region_candidates
)
from utils.label_mapping import load_label_mapping

# Create the Flask app — this is the main web server object
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE

# Make sure the uploads folder exists when the app starts
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL VARIABLES
# These are shared across all requests — accessed by multiple threads
# ──────────────────────────────────────────────────────────────────────────────

model = None           # The loaded Keras CNN model
label_mapping = None   # Dictionary: {0: "Stop Sign", 1: "Speed Limit"...}
camera = None          # Current camera/video capture object
camera_lock = threading.Lock()   # Prevents two threads from touching the camera at once
detection_active = False         # Is detection currently running?

# We keep separate label variables for each page/mode so the Speak button
# only speaks about what's detected on THAT page — not a stale result from another page
last_detected_label_live  = "No detection yet"
last_detected_label_video = "No detection yet"
last_detected_label_image = "No detection yet"

# Shared alias — also updated by live/video for backward compatibility
last_detected_label = "No detection yet"

current_video_path = None  # Path to the currently uploaded video file
tts_engine = None          # Flag indicating pyttsx3 is available
tts_lock = threading.Lock()  # Prevents two voice alerts from playing at the same time


def initialize_model():
    """
    Loads the trained model from disk when the app starts.
    
    We use a custom loader (model_loader.py) to handle different TensorFlow
    versions that might save models in slightly different formats.
    Also loads the label mapping so we can convert class IDs to sign names.
    """
    global model, label_mapping

    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model not found at {MODEL_PATH}")
        print("Please train the model first using train_model.py")
        return False

    try:
        from utils.model_loader import load_compatible_model

        # Load the label mapping first to know how many output classes exist
        label_mapping = load_label_mapping()
        num_classes = len(label_mapping)
        print(f"Label mapping loaded: {num_classes} classes")

        # Load the model — the custom loader handles version compatibility issues
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
    """
    Checks that pyttsx3 (text-to-speech) is available and working.
    We set tts_engine to True as a flag — actual speech uses a fresh engine per call.
    """
    global tts_engine
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        engine.stop()
        tts_engine = True  # Flag: TTS is available
        print("TTS engine initialized")
    except Exception as e:
        tts_engine = None
        print(f"Error initializing TTS: {e}")


def _speak_text(text):
    """
    Speaks a piece of text out loud using pyttsx3.
    
    Why we create a new engine every time:
      pyttsx3 on Windows becomes unreliable after the first call — subsequent
      calls produce no audio. Creating a fresh engine each time fixes this.
    
    We also replace underscores with spaces because sign names come from
    folder names like "Speed_Limit_50", and we want it to be read naturally.
    """
    # Replace underscores with spaces — "Speed_Limit" → "Speed Limit"
    text = text.replace('_', ' ')
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        engine.runAndWait()  # Block until finished speaking
        engine.stop()        # Clean up
    except Exception as e:
        print(f"TTS speak error: {e}")


def allowed_file(filename, allowed_extensions):
    """
    Checks if an uploaded file has an accepted extension.
    Prevents users from uploading random or dangerous file types.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def predict_sign(roi, enhance=False):
    """
    Runs the CNN model on a cropped region of interest (ROI) to classify the sign.
    
    Steps:
    1. Optionally enhance the image (if it's blurry or dark)
    2. Resize and normalize the image to match training format
    3. Run the model and get probabilities for each class
    4. Return the class name with the highest probability and its confidence score
    
    Returns: (label_name, confidence) — e.g. ("Stop Sign", 0.94)
    """
    global model, label_mapping

    # If the model hasn't loaded yet, we can't predict
    if model is None or label_mapping is None:
        return "Model not loaded", 0.0

    try:
        # Preprocess the ROI (same format used during training)
        processed = preprocess_frame_for_inference(roi, IMAGE_SIZE, enhance=enhance)

        # Run prediction — model returns an array of probabilities, one per class
        predictions = model.predict(processed, verbose=0)
        class_id = np.argmax(predictions[0])         # Pick the highest probability class
        confidence = float(predictions[0][class_id]) # That class's probability

        # Convert class ID number (e.g. 14) to a human-readable name (e.g. "Stop Sign")
        label_name = label_mapping.get(int(class_id), f"Unknown {class_id}")

        return label_name, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0


def process_frame(frame, use_smart_fallback=True, source='live'):
    """
    The main detection pipeline for a single frame.
    
    Steps:
    1. Resize the frame for consistent processing
    2. Try color-based ROI detection (look for red/blue/yellow sign areas)
    3. If no colored regions found, fall back to smart multi-region scanning
    4. For each candidate region, crop it out and run the CNN model
    5. Track the highest-confidence detection across all regions
    6. Draw bounding box and label on the frame for display
    7. Update the global label variable for the Speak button
    
    The 'source' parameter ('live', 'video', or 'image') ensures each mode
    tracks its own detections independently.
    """
    global last_detected_label, last_detected_label_live, last_detected_label_video

    # Resize to standard dimensions before processing
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Step 1: Try to find sign-colored regions in the frame
    bboxes = detect_roi_color_based(frame)
    bboxes = non_max_suppression(bboxes)  # Remove overlapping boxes
    used_smart_fallback = False

    # Step 2: If color detection finds nothing, use the smart multi-region approach
    if len(bboxes) == 0 and use_smart_fallback:
        bboxes = get_smart_region_candidates(frame, max_candidates=12)
        used_smart_fallback = True
    elif len(bboxes) == 0:
        # Last resort: just use the whole frame
        h, w = frame.shape[:2]
        bboxes = [(0, 0, w, h)]

    detected_label = None
    best_confidence = 0.0
    best_bbox = None

    # Step 3: Run the model on each candidate region, keep the best result
    for bbox in bboxes:
        roi = extract_roi(frame, bbox)
        if roi.size == 0:
            continue  # Skip if the crop ended up empty

        # When we're using the fallback (not color-based), images may be blurry
        # so we apply enhancement to improve model accuracy
        enhance = used_smart_fallback
        label, confidence = predict_sign(roi, enhance=enhance)

        # Track the detection with the highest confidence
        if confidence > best_confidence:
            best_confidence = confidence
            detected_label = label
            best_bbox = bbox

        # Only draw a solid box if the model is very confident (above threshold)
        if confidence > CONFIDENCE_THRESHOLD:
            frame = draw_detection(frame, bbox, label, confidence, draw_bbox=True)

    # Step 4: If the best detection is uncertain (between 10% and 50% confidence),
    # still show it with a "?" to give the user a hint of what the sign might be
    if detected_label and best_confidence >= MIN_DISPLAY_CONFIDENCE and best_confidence < CONFIDENCE_THRESHOLD and best_bbox:
        frame = draw_detection(frame, best_bbox, f"{detected_label}?", best_confidence, draw_bbox=True)

    # Step 5: Save the detected label so the Speak button knows what to say
    if detected_label:
        last_detected_label = detected_label  # shared legacy alias
        if source == 'video':
            last_detected_label_video = detected_label
        else:
            last_detected_label_live = detected_label  # live and image both use this

    return frame, detected_label


class VideoCamera:
    """
    Wraps OpenCV's VideoCapture to manage camera/video access.
    
    Can handle:
    - Live webcam (source=0 means the first connected camera)
    - Uploaded video files (source = file path string)
    
    frame_count is used with FRAME_SKIP to only process every 5th frame
    instead of every frame — this improves performance on live feeds.
    """

    def __init__(self, source=0):
        self.source = source
        self.video = cv2.VideoCapture(source)
        self.frame_count = 0

        if not self.video.isOpened():
            raise ValueError(f"Could not open video source: {source}")

    def release(self):
        """Releases the camera/video resource back to the system."""
        if hasattr(self, 'video') and self.video.isOpened():
            self.video.release()

    def __del__(self):
        """Automatically release the camera when this object is garbage collected."""
        self.release()

    def get_frame(self):
        """
        Reads the next frame from the camera or video.
        
        - If detection is off: just encode and return the raw frame
        - If detection is on: process every 5th frame through the CNN pipeline
        - Returns: JPEG-encoded bytes ready to send to the browser, or None at end of video
        """
        success, frame = self.video.read()

        if not success:
            return None  # End of video or camera disconnected

        self.frame_count += 1

        # Skip frames to improve speed — only process every FRAME_SKIP-th frame
        if self.frame_count % FRAME_SKIP != 0:
            # Just send the raw frame without running detection
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

        # Process this frame through the full detection pipeline
        if detection_active:
            frame, _ = process_frame(frame)

        # Encode the annotated frame as JPEG for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


def generate_frames():
    """
    Generator function for MJPEG video streaming.
    
    Flask uses this as a streaming response — it keeps yielding new frames
    to the browser indefinitely while the camera is active.
    
    Each frame is wrapped in a multipart HTTP boundary format that browsers
    understand as a continuous video stream (MJPEG).
    """
    global camera

    while True:
        with camera_lock:
            if camera is None:
                # Camera stopped — send a placeholder and end the stream
                placeholder = get_placeholder_frame()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
                break

            frame = camera.get_frame()

        if frame is None:
            break  # End of video

        # Yield the frame in MJPEG multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def get_placeholder_frame():
    """
    Creates a dark placeholder image shown before detection starts.
    Displays a message telling the user to click 'Start Detection'.
    """
    img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    img[:] = (40, 40, 60)  # Dark gray-blue background
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Click 'Start Detection' to begin"
    (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
    # Center the text horizontally and vertically
    x = (FRAME_WIDTH - tw) // 2
    y = FRAME_HEIGHT // 2
    cv2.putText(img, text, (x, y), font, 0.8, (255, 255, 255), 2)
    ret, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes()


def create_placeholder_response():
    """Wraps the placeholder image as a Flask response for the browser."""
    import io
    data = get_placeholder_frame()
    return send_file(
        io.BytesIO(data),
        mimetype='image/jpeg',
        as_attachment=False
    )


# ──────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES — these handle browser requests
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serves the main HTML page of the web app."""
    return render_template('index.html')


@app.route('/video_placeholder')
def video_placeholder():
    """Returns the static placeholder image when the camera is not running."""
    return create_placeholder_response()


@app.route('/video_feed')
def video_feed():
    """
    Streams live video frames to the browser as MJPEG.
    The browser's <img> tag continuously receives and displays new frames.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start', methods=['POST'])
def start_detection():
    """
    Called when the user clicks 'Start Detection'.
    Opens the webcam (or the uploaded video file if one exists).
    """
    global camera, detection_active, current_video_path

    try:
        with camera_lock:
            # Release any existing camera first
            if camera is not None:
                camera.release()
                camera = None

            # Use the uploaded video if one exists, otherwise use live webcam
            source = 0  # 0 = default webcam
            if current_video_path and os.path.exists(current_video_path):
                source = current_video_path

            camera = VideoCamera(source)
            detection_active = True

        return jsonify({'status': 'success', 'message': 'Detection started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/stop', methods=['POST'])
def stop_detection():
    """
    Called when the user clicks 'Stop Detection'.
    Releases the camera and stops processing frames.
    """
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
    """
    Handles image upload from the Image Detection page.
    
    The uploaded image is:
    1. Read into memory (not saved to disk)
    2. Processed through the full detection pipeline
    3. The annotated image is sent back as base64 so the browser can display it
    4. The detected label is returned for the Speak button to use
    """
    global last_detected_label_image

    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400

    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400

    try:
        # Read the uploaded file bytes directly into a numpy array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'status': 'error', 'message': 'Could not read image'}), 400

        # Run the detection — use 'image' as source so it doesn't mix with live labels
        processed_image, detected_label = process_frame(image, source='image')

        # Convert the annotated image to base64 so we can embed it directly in JSON
        # The browser can display it as <img src="data:image/jpeg;base64,...">
        ret, buffer = cv2.imencode('.jpg', processed_image)
        image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8') if ret else None

        # Update the image-specific label for the Speak button
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
    """
    Saves an uploaded video file to the uploads folder.
    
    The video is not processed here — the user will click 'Start Detection'
    or 'Analyze Video' separately to process it.
    
    A random hex suffix is added to the filename to avoid overwrite conflicts
    if the user uploads two videos with the same name.
    """
    global current_video_path

    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400

    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400

    try:
        # Sanitize the filename (removes dangerous characters like ../ etc.)
        base = secure_filename(file.filename)
        name, ext = os.path.splitext(base)

        # Add 4 random hex characters to make filename unique
        filename = f"{name}_{os.urandom(4).hex()}{ext}" if name else f"video_{os.urandom(6).hex()}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        current_video_path = filepath  # Remember this path for detection

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
    Analyzes a full uploaded video by sampling 50 frames evenly across its length.
    
    Instead of processing every frame (slow), we pick 50 evenly spaced frames.
    For each frame, we run detection and return the results as a list.
    The frontend then displays a table of timestamps and detected signs.
    
    Returns a JSON list of {frame_number, timestamp, label, confidence} for each sample.
    """
    global current_video_path

    if not current_video_path or not os.path.exists(current_video_path):
        return jsonify({'status': 'error', 'message': 'No video uploaded. Please upload a video first.'}), 400

    try:
        cap = cv2.VideoCapture(current_video_path)
        if not cap.isOpened():
            return jsonify({'status': 'error', 'message': 'Could not open video file.'}), 500

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # Default to 25fps if unknown
        if fps <= 0:
            fps = 25.0
        num_samples = 50  # We analyze exactly 50 frames

        if total_frames <= 0:
            cap.release()
            return jsonify({'status': 'error', 'message': 'Video has no frames or could not be read.'}), 500

        # Calculate evenly spaced frame numbers across the full video
        if total_frames <= num_samples:
            frame_indices = list(range(total_frames))  # Video shorter than 50 frames
        else:
            step = total_frames / num_samples
            frame_indices = [int(i * step) for i in range(num_samples)]

        detections = []
        for idx in frame_indices:
            # Seek to the specific frame — faster than reading every frame sequentially
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Run detection and update the video label tracker
            _, detected_label = process_frame(frame, source='video')

            # Run predictions again on the same frame to get the best confidence score
            # (process_frame doesn't return confidence, only the label)
            frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            bboxes = detect_roi_color_based(frame_resized)
            bboxes = non_max_suppression(bboxes)
            if len(bboxes) == 0:
                bboxes = get_smart_region_candidates(frame_resized, max_candidates=12)

            best_conf = 0.0
            best_label = detected_label or 'No sign detected'

            # Video frames often have motion blur — use enhancement for better results
            for bbox in bboxes:
                roi = extract_roi(frame_resized, bbox)
                if roi.size == 0:
                    continue
                lbl, conf = predict_sign(roi, enhance=True)
                if conf > best_conf and conf >= MIN_DISPLAY_CONFIDENCE:
                    best_conf = conf
                    best_label = lbl

            # Calculate the timestamp in seconds from the frame number
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
    """
    Triggered when the user clicks the 'Speak' button on any page.
    
    The browser sends a JSON body with {"source": "live"/"video"/"image"}
    so we know which page's detected label to read aloud.
    
    Speech runs in a background thread so the HTTP response returns immediately
    and the browser doesn't freeze waiting for the audio to finish.
    """
    global last_detected_label_live, last_detected_label_video, last_detected_label_image
    global last_detected_label, tts_engine

    if tts_engine is None:
        return jsonify({'status': 'error', 'message': 'TTS not initialized'}), 500

    # Read which page is calling (live, video, or image)
    data = request.get_json(silent=True) or {}
    source = data.get('source', 'live')

    # Pick the appropriate label for this page
    if source == 'image':
        label_to_speak = last_detected_label_image
    elif source == 'video':
        label_to_speak = last_detected_label_video
    else:
        label_to_speak = last_detected_label_live

    # Don't speak if nothing has been detected yet
    if not label_to_speak or label_to_speak in ("No detection yet", "No sign detected"):
        return jsonify({
            'status': 'error',
            'message': f'No sign detected yet on the {source} page. Please detect a sign first.'
        }), 400

    try:
        speak_label = label_to_speak  # Capture in closure for the background thread
        def speak_async():
            # Lock prevents two voices from speaking at the same time
            with tts_lock:
                _speak_text(f"Detected: {speak_label}")

        # Run in a background thread so this endpoint returns immediately
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
    """
    Plays a short test phrase to verify the speakers and TTS are working.
    Useful before a demo to make sure the audio works.
    """
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
    """
    Returns the current state of the app as JSON.
    The frontend uses this to update the UI — model loaded, detection active, etc.
    Separate labels for each source are exposed so each page can show its own result.
    """
    return jsonify({
        'detection_active': detection_active,
        'last_detected_label': last_detected_label,           # live/video shared (legacy)
        'last_detected_label_live':  last_detected_label_live,
        'last_detected_label_video': last_detected_label_video,
        'last_detected_label_image': last_detected_label_image,
        'model_loaded': model is not None,
        'label_count': len(label_mapping) if label_mapping else 0
    })


# ──────────────────────────────────────────────────────────────────────────────
# STARTUP
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Initializing Road Sign Detection Application...")

    # Load the trained model into memory
    if not initialize_model():
        print("WARNING: Running without model. Please train the model first.")

    # Set up text-to-speech
    initialize_tts()

    print("\nStarting Flask server...")
    print("Open http://localhost:5000 in your browser")

    # Start the web server
    # threaded=True allows multiple browser tabs to connect at once
    # host='0.0.0.0' makes it accessible from other devices on the same network
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
