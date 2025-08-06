# MoodLink
Mood Link is a real-time emotion recognition system. The goal was to develop a lightweight system recognizing emotions based on camera input. It uses MediaPipe FaceMesh for landmark extraction followed by pre-processing and MobileNetV3-Large to map the pre-processed image to an emotion. MobileNetV3-Large is can be chosen during runtime in 3 configurations: trained with FER-13 dataset, trained with RAF-DB dataset, and trained with a pre-processed version of the RAF-DB dataset.
This was the final project for the class "ECE284 Mobile Health Design" at UC San Diego.

# Code Base Explaination
Code base explaination: The emotion detection pipeline is in the root directory, along with all three exported .onnx models of the trained versions of MobileNetV3-Large. A detailed description of the pipeline's code is in the comments of emotion_detection_pipeline.py.
The directory Training_Scripts contains the training python scripts used for training MobileNetV3 and obtaining the .onnx files in the root directory. 

# Emotion Detection Pipeline

A simple, real-time demo that combines MediaPipe FaceMesh with a MobileNetV3 emotion classifier exported to ONNX. After detecting your facial landmarks via webcam, it aligns and crops each face, then feeds that patch into the ONNX model you choose—finally drawing both the aligned thumbnail and your predicted emotion back on the live video feed.

---

## Features

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

## Prerequisites

- **Python 3.7+**  
- A working **webcam**  
- The three ONNX model files in your project folder  
- (Optional but recommended) A virtual environment

---

## Installation

### Clone the repository

```bash
# Clone the MoodLink repo and enter the directory
git clone https://github.com/yourusername/MoodLink.git
cd MoodLink


# Create Venv

python -m venv venv
## Windows
venv\Scripts\activate
## macOS/Linux
source venv/bin/activate

# Install Required Packages

pip install numpy opencv-python mediapipe onnxruntime