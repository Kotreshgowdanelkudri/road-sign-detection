# 🚦 Road Sign Voice Alert Web Application

A production-ready web application for real-time traffic sign detection and recognition using deep learning. Supports webcam feeds, uploaded images, and video files with voice alerts.

## ✨ Features

- **Real-time Detection**: Process live webcam feed with real-time traffic sign detection
- **Image Upload**: Upload and analyze individual images
- **Video Processing**: Upload and process recorded traffic videos (including YouTube downloads)
- **ROI Localization**: Advanced color-based segmentation and contour detection
- **Voice Alerts**: Manual voice alerts for detected signs using offline TTS
- **Modern UI**: Responsive web interface with glassmorphism design
- **High Accuracy**: CNN model with batch normalization and data augmentation
- **Production Ready**: Thread-safe, optimized for performance

## 📋 Requirements

- Python 3.8+
- Webcam (for live detection)
- Labeled traffic sign dataset (organized in folders)

## 🚀 Installation

### 1. Clone or Download the Project

```bash
cd road-sign-app
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📁 Dataset Setup

### Option A: Folder-Based Dataset (Recommended)

Organize your dataset with numeric folder names:

```
dataset/
├── 0/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 1/
│   ├── image1.jpg
│   └── ...
├── 2/
└── ...
```

### Option B: Custom Label Mapping

Edit `label_mapping.json` to map numeric class IDs to sign names:

```json
{
  "0": "Stop",
  "1": "Speed Limit 40",
  "2": "Yield",
  ...
}
```

## 🎯 Training the Model

### 1. Prepare Dataset

Place your labeled dataset in the `dataset/` folder.

### 2. Run Training Script

```bash
python train_model.py
```

This will:
- Load and preprocess images
- Split into train/validation/test sets
- Train CNN model with data augmentation
- Save best model to `model/model.h5`
- Generate confusion matrix and accuracy report

**Training Parameters:**
- Image size: 64x64
- Batch size: 32
- Epochs: 50 (with early stopping)
- Optimizer: Adam (lr=0.001)
- Data augmentation: rotation, shift, zoom, brightness

## 🌐 Running the Application

### 1. Start Flask Server

```bash
python app.py
```

### 2. Open Browser

Navigate to: `http://localhost:5000`

## 📖 Usage Guide

### Webcam Detection

1. Click **"Start Detection"** button
2. Allow camera access if prompted
3. Show traffic signs to the camera
4. Detected signs will be highlighted with bounding boxes
5. Click **"Speak"** to hear the last detected sign

### Image Upload

1. Click on **"Upload Image"** area
2. Select an image file (PNG, JPG, JPEG, BMP)
3. Detection results will appear automatically
4. Click **"Speak"** to hear the detected sign

### Video Upload

1. Click on **"Upload Video"** area
2. Select a video file (MP4, AVI, MOV, MKV)
3. Click **"Start Detection"** to process the video
4. Detections will appear in real-time
5. Click **"Speak"** to hear detected signs

## 🔧 Configuration

Edit `config.py` to customize:

```python
IMAGE_SIZE = 64              # Model input size
CONFIDENCE_THRESHOLD = 0.50  # Strong detection threshold
MIN_DISPLAY_CONFIDENCE = 0.25  # Show best guess for difficult images
FRAME_SKIP = 2               # Process every Nth frame
BATCH_SIZE = 32              # Training batch size
EPOCHS = 50                  # Training epochs
```

**Environment Variables:**
- `DATASET_PATH` – Override dataset path for training (default: `dataset/Train`)
- `MAX_UPLOAD_SIZE` – Max upload size in bytes (default: 200MB)

## 🏗️ Project Structure

```
road-sign-app/
│
├── app.py                    # Flask application
├── train_model.py            # Model training script
├── config.py                 # Configuration settings
├── label_mapping.json        # Class ID to name mapping
├── requirements.txt          # Python dependencies
│
├── model/
│   └── model.h5             # Trained model (generated)
│
├── dataset/                 # Your labeled dataset
│   ├── 0/
│   ├── 1/
│   └── ...
│
├── utils/
│   ├── label_mapping.py     # Label utilities
│   ├── preprocessing.py     # Image preprocessing
│   └── roi_detection.py     # ROI detection
│
├── templates/
│   └── index.html           # Frontend HTML
│
├── static/
│   └── style.css            # Frontend CSS
│
└── uploads/                 # Uploaded files (generated)
```

## 🧪 Testing

### Test with Webcam

1. Start the application
2. Use printed traffic signs or display signs on screen
3. Verify bounding boxes and labels appear correctly

### Test with Images

1. Download sample traffic sign images
2. Upload through the web interface
3. Verify detection accuracy

### Test with Videos

1. Download traffic videos from YouTube
2. Upload through the web interface
3. Verify real-time processing

## 🚀 Deployment

### Heroku Deployment

1. Create `Procfile`:
```
web: gunicorn app:app
```

2. Add `gunicorn` to `requirements.txt`

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### AWS EC2 Deployment

1. Launch EC2 instance (Ubuntu)
2. Install dependencies:
```bash
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt
```

3. Run with gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

4. Set up nginx as reverse proxy

## 🐛 Troubleshooting

### Model Not Found

**Error**: `Model not found at model/model.h5`

**Solution**: Run `python train_model.py` to train the model first

### No Detections

**Possible causes**:
- Low confidence threshold (adjust in `config.py`)
- Poor lighting conditions
- Signs too small or far from camera
- Color-based detection not working (try different lighting)

### Camera Not Working

**Solution**: 
- Check camera permissions
- Ensure no other application is using the camera
- Try different camera index in `app.py` (change `source=0` to `source=1`)

### TTS Not Working

**Solution**:
- Ensure `pyttsx3` is installed correctly
- On Linux, install: `sudo apt install espeak`
- On Windows, ensure SAPI5 is available

## 📊 Model Performance

Expected performance metrics:
- **Validation Accuracy**: 85-95%
- **Test Accuracy**: 85-95%
- **Inference Speed**: 15-30 FPS (depending on hardware)
- **Confidence Threshold**: 0.80 (80%)

## 🔒 Security Notes

- Uploaded files are stored in `uploads/` folder
- Consider implementing file size limits
- Validate file types before processing
- Use HTTPS in production
- Implement user authentication for production use

## 📝 License

This project is for educational and internship purposes.

## 👨‍💻 Author

Built as a production-ready internship project demonstrating:
- Computer Vision
- Deep Learning (CNN)
- Real-time Video Processing
- Full-stack Web Development
- ROI Localization
- Voice Integration

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning framework
- OpenCV for computer vision
- Flask for web framework
- pyttsx3 for text-to-speech

---

**Note**: This application requires a trained model. Make sure to run `train_model.py` with your dataset before starting the web application.
