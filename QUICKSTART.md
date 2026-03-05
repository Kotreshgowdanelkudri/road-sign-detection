# 🚀 Quick Start Guide

## Prerequisites

1. **Python 3.8+** installed
2. **Webcam** (for live detection)
3. **Labeled dataset** organized in folders

## Step-by-Step Setup

### 1️⃣ Navigate to Project

```bash
cd e:\kk\road-sign-app
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Prepare Your Dataset

**Option A: You have a folder-based dataset**

Copy your dataset to the `dataset/` folder:

```
dataset/
├── 0/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 1/
├── 2/
└── ...
```

**Option B: You need to organize your dataset**

1. Create the dataset folder structure
2. Organize images by class (0, 1, 2, etc.)
3. Each folder should contain images of one traffic sign type

### 5️⃣ Check Dataset Structure

```bash
python check_dataset.py
```

This will verify your dataset is properly organized.

### 6️⃣ Configure Label Mapping (Optional)

Edit `label_mapping.json` to customize traffic sign names:

```json
{
  "0": "Stop",
  "1": "Speed Limit 40",
  "2": "Yield",
  ...
}
```

### 7️⃣ Train the Model

```bash
python train_model.py
```

**Expected output:**
- Training progress with accuracy/loss
- Model saved to `model/model.h5`
- Confusion matrix: `confusion_matrix.png`
- Training history: `training_history.png`

**Training time:** 30 minutes to 2 hours (depending on dataset size and hardware)

### 8️⃣ Run the Application

```bash
python app.py
```

### 9️⃣ Open Browser

Navigate to: **http://localhost:5000**

## 🎯 Using the Application

### Webcam Detection

1. Click **"Start Detection"**
2. Allow camera access
3. Show traffic signs to camera
4. Click **"Speak"** to hear detected sign

### Image Upload

1. Click **"Upload Image"** area
2. Select an image
3. View detection result
4. Click **"Speak"** to hear result

### Video Upload

1. Click **"Upload Video"** area
2. Select a video file
3. Click **"Start Detection"**
4. View real-time detections

## 🐛 Troubleshooting

### "Model not found"

**Solution:** Run `python train_model.py` first

### "Dataset not found"

**Solution:** 
1. Create `dataset/` folder
2. Organize images into class folders (0/, 1/, 2/, etc.)
3. Run `python check_dataset.py` to verify

### "No detections appearing"

**Possible fixes:**
1. Adjust `CONFIDENCE_THRESHOLD` in `config.py` (try 0.60)
2. Improve lighting conditions
3. Move sign closer to camera
4. Ensure sign has red or blue colors

### Camera not working

**Solution:**
1. Close other applications using camera
2. Check camera permissions
3. Try different camera index (edit `app.py`, change `source=0` to `source=1`)

## 📊 Expected Performance

- **Training Accuracy:** 85-95%
- **Inference Speed:** 15-30 FPS
- **Confidence Threshold:** 0.80 (80%)

## 🎓 Next Steps

1. ✅ Train model with your dataset
2. ✅ Test with webcam
3. ✅ Test with uploaded images
4. ✅ Test with uploaded videos
5. ✅ Adjust confidence threshold if needed
6. ✅ Customize label mapping
7. ✅ Deploy to production (Heroku/AWS)

## 📞 Need Help?

Check the full `README.md` for detailed documentation.

---

**Ready to start? Run these commands:**

```bash
cd e:\kk\road-sign-app
venv\Scripts\activate
python check_dataset.py
python train_model.py
python app.py
```
