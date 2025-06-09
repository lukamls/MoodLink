# Final project for ECE 284

# MoodLink Emotion Detection Pipeline

A simple, real-time demo that combines MediaPipe FaceMesh with a MobileNetV3 emotion classifier exported to ONNX. After detecting your facial landmarks via webcam, it aligns and crops each face, then feeds that patch into the ONNX model you choose—finally drawing both the aligned thumbnail and your predicted emotion back on the live video feed.

---

## 🔍 Features

- **Face alignment** using five key landmarks (eyes, nose, mouth)  
- **Warp & crop** to a 256×256 “aligned” face  
- **ONNX-powered emotion classification** (MobileNetV3)  
- **Choice of three pre-trained models**:  
  - `mobilenetv3_fer13.onnx`  
  - `mobilenetv3_rafdb.onnx`  
  - `mobilenetv3_rafdb_preprocessed.onnx`  
- **Live video overlay** of both the aligned crop and detected emotion  
- **Cross-platform** (Windows, macOS, Linux)

---

## 🛠 Prerequisites

- **Python 3.7+**  
- A working **webcam**  
- The three ONNX model files in your project folder  
- (Optional but recommended) A virtual environment

---

## 📥 Installation

1. Clone or download this repo:  
   ```bash
   git clone https://github.com/yourusername/MoodLink.git
   cd MoodLink

## Create Venv

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

## Install Required Packages

pip install numpy opencv-python mediapipe onnxruntime