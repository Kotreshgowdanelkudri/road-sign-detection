"""
Flask Web Application for Road Sign Detection
Supports webcam (live), image upload, and video upload

How this app works:
 - Flask handles HTTP requests from the browser
 - When the user starts detection, we open a camera or video file using OpenCV
 - Each frame is analyzed: first we look for sign-colored regions (ROI detection)
 - The cropped sign region is passed through the trained CNN model for classification
 - Results are streamed back to the browser as a live video feed (MJPEG)
 - The "Speak" button uses the browser's built-in Web Speech API (no server audio needed)
   This makes the app fully compatible with cloud deployments like Render.
"""
import os

# Suppress TensorFlow CPU optimization warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import base64
import cv2
import numpy as np
import threading
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
# NOTE: No server-side TTS engine needed — speech is handled by the browser's Web Speech API


def initialize_model():
    """
    Loads the trained model from disk when the app starts.

    Uses model_loader.py which auto-detects the Keras version (2 or 3)
    and applies the correct loading strategy for this TF installation.
    Also loads the label mapping so we can convert class IDs to sign names.
    """
    global model, label_mapping
    import traceback

    print(f"[INIT] MODEL_PATH = {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"[INIT] WARNING: Model file not found at {MODEL_PATH}")
        print("[INIT] Please train the model first with: python train_model.py")
        return False

    try:
        from utils.model_loader import load_model_for_inference
        model = load_model_for_inference(MODEL_PATH)
        print(f"[INIT] Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"[INIT] ERROR loading model: {e}")
        traceback.print_exc()
        model = None
        return False

    try:
        label_mapping = load_label_mapping()
        if label_mapping:
            print(f"[INIT] Label mapping loaded successfully ({len(label_mapping)} classes).")
        else:
            print("[INIT] WARNING: label_mapping.json is empty or missing.")
    except Exception as e:
        print(f"[INIT] ERROR loading label mapping: {e}")
        traceback.print_exc()
        label_mapping = None
        return False

    print("DEBUG: initialize_model completed successfully")
    return True
    
    return True


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

    'detection_source' tracks whether this is 'live' or 'video' so that
    detections are stored in the correct label variable for each page.
    """

    def __init__(self, source=0, detection_source='live'):
        self.source = source
        self.detection_source = detection_source  # 'live' or 'video'
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
        - If detection is on: process every FRAME_SKIP-th frame through the CNN pipeline
        - detection_source ensures detection results go to the right label (live vs video)
        - Returns: JPEG-encoded bytes ready to send to the browser, or None at end of video
        """
        success, frame = self.video.read()

        if not success:
            return None  # End of video or camera disconnected

        self.frame_count += 1

        # For video files, process every frame (no skip needed — slower but more detections)
        # For live camera, skip frames to improve speed
        skip = FRAME_SKIP if self.detection_source == 'live' else 2
        if self.frame_count % skip != 0:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

        # Process this frame through the full detection pipeline
        # Pass the correct source so the label goes to the right variable
        if detection_active:
            frame, _ = process_frame(frame, source=self.detection_source)

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
    global camera, detection_active
    
    # Check if a video filename was passed from the frontend
    video_filename = request.args.get('video')
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename) if video_filename else None

    try:
        with camera_lock:
            # Release any existing camera first
            if camera is not None:
                camera.release()
                camera = None

            # Use the uploaded video if one exists, otherwise use live webcam
            is_video_file = bool(video_path and os.path.exists(video_path))
            source = video_path if is_video_file else 0
            # IMPORTANT: pass the correct detection_source so labels update the
            # right variable ('video' panel vs 'live' panel)
            detection_src = 'video' if is_video_file else 'live'

            camera = VideoCamera(source, detection_source=detection_src)
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
    # We don't need a global path anymore, the frontend remembers the name

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

        # Return the filename so the frontend can send it back in future requests

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
    video_filename = request.args.get('video')
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename) if video_filename else None

    if not video_path or not os.path.exists(video_path):
        return jsonify({'status': 'error', 'message': 'No video uploaded. Please upload a video first.'}), 400

    try:
        cap = cv2.VideoCapture(video_path)
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
                try:
                    lbl, conf = predict_sign(roi, enhance=True)
                    if conf is not None and conf > best_conf and conf >= MIN_DISPLAY_CONFIDENCE:
                        best_conf = conf
                        best_label = lbl or 'No sign detected'
                except Exception as e:
                    print(f"Prediction skipped for ROI due to error: {e}")
                    continue

            # Calculate the timestamp in seconds from the frame number
            timestamp = round(idx / fps, 2)
            detections.append({
                'frame_number': idx,
                'timestamp': timestamp,
                'label': best_label,
                'confidence': round(best_conf, 4)
            })

        try:
            cap.release()
        except:
            pass

        return jsonify({
            'status': 'success',
            'frames_analyzed': len(detections),
            'total_frames': total_frames,
            'detections': detections
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            if 'cap' in locals() and cap is not None:
                cap.release()
        except:
            pass
        return jsonify({'status': 'error', 'message': f"Analysis failed: {str(e)}"}), 500

@app.route('/speak', methods=['POST'])
def speak():
    """
    Returns the detected label for the requested source so the BROWSER can speak it.
    
    Speech is handled entirely by the browser's Web Speech API (window.speechSynthesis).
    This approach works on all cloud servers like Render because no audio hardware
    or system audio libraries are needed on the server side.
    
    The browser sends: { "source": "live" | "video" | "image" }
    The server responds with: { "status": "success", "label": "Stop Sign" }
    Then the browser calls: speechSynthesis.speak(new SpeechSynthesisUtterance(label))
    """
    # Read which page is calling (live, video, or image)
    data = request.get_json(silent=True) or {}
    source = data.get('source', 'live')

    # Pick the correct label for the page that made the request
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

    # Return the label — the browser's speechSynthesis will do the actual speaking
    return jsonify({
        'status': 'success',
        'label': label_to_speak,
        'message': f'Speak: {label_to_speak}'
    })


@app.route('/tts_test', methods=['POST'])
def tts_test():
    """
    Returns a test phrase for the browser to speak.
    Useful for verifying that the browser's Web Speech API is working.
    """
    return jsonify({
        'status': 'success',
        'label': 'This is a test of the road sign voice alert system.',
        'message': 'Browser will speak the test phrase'
    })


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

# Load the trained model into memory so it works with Gunicorn/production
print("Initializing Road Sign Detection Application...")
if not initialize_model():
    print("WARNING: Running without model. Please train the model first.")

if __name__ == '__main__':
    # NOTE: No TTS initialization needed — browser handles speech via Web Speech API

    print("\nStarting Flask server...")
    print("Open http://localhost:5000 in your browser")

    # Start the web server
    # threaded=True allows multiple browser tabs to connect at once
    # host='0.0.0.0' makes it accessible from other devices on the same network
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
