

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

# alignment constants
OUT_SIZE = 256
REF_LANDMARKS = np.array([
    [0.35, 0.35],   # left-eye centre
    [0.65, 0.35],   # right-eye centre
    [0.50, 0.55],   # nose tip
    [0.38, 0.72],   # left mouth
    [0.62, 0.72]    # right mouth
], dtype=np.float32) * OUT_SIZE

# emotion model initialize
# prompt user to select which ONNX model to load
AVAILABLE_MODELS = [
    "mobilenetv3_fer13.onnx",
    "mobilenetv3_rafdb.onnx",
    "mobilenetv3_rafdb_preprocessed.onnx"
]
print("Available emotion classification models:")
for idx, name in enumerate(AVAILABLE_MODELS, start=1):
    print(f"  {idx}: {name}")
choice = input("Select model [1-3]: ").strip()
try:
    MODEL_PATH = AVAILABLE_MODELS[int(choice) - 1]
except (ValueError, IndexError):
    print(f"Invalid choice '{choice}', defaulting to {AVAILABLE_MODELS[-1]}")
    MODEL_PATH = AVAILABLE_MODELS[-1]

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# reorder labels as per training
EMOTION_LABELS = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise",
    "Neutral"
]

# normalization to image net for pretrained MobileNetV3-L
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# MediaPipe FaceMesh init
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# open camera
cap = cv2.VideoCapture(0)

# desired landmark positions
IDX_LEFT_EYE  = [33, 133]
IDX_RIGHT_EYE = [362, 263]
IDX_NOSE      = 1
IDX_MOUTH_L   = 61
IDX_MOUTH_R   = 291

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        def mp_pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

        left_eye  = (mp_pt(*IDX_LEFT_EYE[:1]) + mp_pt(*IDX_LEFT_EYE[1:])) / 2
        right_eye = (mp_pt(*IDX_RIGHT_EYE[:1]) + mp_pt(*IDX_RIGHT_EYE[1:])) / 2
        nose_tip  = mp_pt(IDX_NOSE)
        mouth_l   = mp_pt(IDX_MOUTH_L)
        mouth_r   = mp_pt(IDX_MOUTH_R)

        src = np.stack([left_eye, right_eye, nose_tip, mouth_l, mouth_r])

        # Pre-Processing: Similarity Transform + Warping
        M, _ = cv2.estimateAffinePartial2D(src, REF_LANDMARKS, method=cv2.LMEDS)
        aligned = cv2.warpAffine(frame, M, (OUT_SIZE, OUT_SIZE))

        ######### emotion detection 
        #  BGR to RGB & scale to [0,1]
        # Resize the aligned face to match the ONNX model's 224×224 input
        face_for_model = cv2.resize(aligned, (224, 224))
        x = cv2.cvtColor(face_for_model, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        #  normalization
        x = (x - MEAN) / STD
        #  HWC to CHW & add batch dim
        x = np.transpose(x, (2, 0, 1))[None, ...]

        logits = session.run([output_name], {input_name: x})[0][0]
        #   pick top w softmax
        probs = np.exp(logits) / np.sum(np.exp(logits))
        emotion = EMOTION_LABELS[np.argmax(probs)]

        # Draw on frame:
        #aligned thumbnail
        thumb = cv2.resize(aligned, (128, 128))
        frame[5:133, w-133:w-5] = thumb

        # label and keypoints
        cv2.putText(
            frame, f"Emotion: {emotion}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        for p in src.astype(int):
            cv2.circle(frame, tuple(p), 2, (0, 255, 0), -1)

    cv2.imshow("Alignment + Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
