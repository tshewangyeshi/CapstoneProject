import serial
import struct
import cv2
import numpy as np
import time
from collections import deque

from ultralytics import YOLO
import onnxruntime as ort

# ======================================================
# CONFIGURATION
# ======================================================
SERIAL_PORT = "COM3"
BAUD_RATE = 2000000

YOLO_MODEL = "best.pt"
RESNET_ONNX = "resnet50_waste_classifier.onnx"
VIT_ONNX = "vit_waste_classifier.onnx"

CLASS_NAMES = [
    "Biodegradable",
    "Hazardous",
    "Non-Biodegradable"
]

# ---- Scene complexity thresholds ----
LOW_OBJECT_THRESHOLD = 2      # <=2 objects → ViT, >2 → ResNet

BOX_CHANGE_THRESH = 40
YOLO_SKIP = 5
CLASSIFY_COOLDOWN = 1.5

WINDOW_W, WINDOW_H = 960, 720

# ======================================================
# SERIAL
# ======================================================
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# ======================================================
# MODELS (GPU)
# ======================================================
yolo = YOLO(YOLO_MODEL)
yolo.to("cuda")

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
resnet = ort.InferenceSession(RESNET_ONNX, providers=providers)
vit = ort.InferenceSession(VIT_ONNX, providers=providers)

resnet_input = resnet.get_inputs()[0].name
vit_input = vit.get_inputs()[0].name

print("All models loaded")

# ======================================================
# HELPERS
# ======================================================
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, axis=0)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def largest_box(boxes):
    return max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

def box_changed(b1, b2):
    return np.linalg.norm(np.array(b1) - np.array(b2)) > BOX_CHANGE_THRESH

# ======================================================
# STATE VARIABLES
# ======================================================
frame_count = 0
last_boxes = []

locked_box = None
classified = False
final_label = ""
final_model = ""
final_confidence = 0.0

last_classify_time = 0

# --- FPS ---
fps_times = deque(maxlen=30)

# --- Switching statistics ---
resnet_count = 0
vit_count = 0

# ======================================================
# WINDOW
# ======================================================
cv2.namedWindow("Adaptive Waste Classification", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Adaptive Waste Classification", WINDOW_W, WINDOW_H)

# ======================================================
# MAIN LOOP
# ======================================================
while True:
    start_time = time.time()

    # ---------- Read frame ----------
    if ser.read(2) != b'\xff\xd8':
        continue

    size = struct.unpack('<I', ser.read(4))[0]
    jpg = ser.read(size)

    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        continue

    display = frame.copy()
    frame_count += 1
    current_time = time.time()

    # ==================================================
    # YOLO DETECTION (FRAME SKIP)
    # ==================================================
    if frame_count % YOLO_SKIP == 0:
        results = yolo.predict(
            frame,
            imgsz=640,
            conf=0.3,
            device="cuda",
            verbose=False
        )[0]

        if results.boxes is not None and len(results.boxes) > 0:
            last_boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        else:
            last_boxes = []

    boxes = last_boxes
    num_objects = len(boxes)

    # ==================================================
    # RESET IF NO OBJECT
    # ==================================================
    if num_objects == 0:
        locked_box = None
        classified = False

    # ==================================================
    # OBJECT HANDLING
    # ==================================================
    if num_objects > 0:
        current_box = largest_box(boxes)

        if locked_box is None or box_changed(locked_box, current_box):
            locked_box = current_box
            classified = False

        # ---------- CLASSIFY (SCENE-COMPLEXITY BASED) ----------
        if (not classified) and (current_time - last_classify_time > CLASSIFY_COOLDOWN):
            x1, y1, x2, y2 = locked_box
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                inp = preprocess(roi)

                # -------- MODEL SELECTION --------
                if num_objects <= LOW_OBJECT_THRESHOLD:
                    # Low complexity → ViT
                    logits = vit.run(None, {vit_input: inp})[0][0]
                    probs = softmax(logits)
                    idx = int(np.argmax(probs))

                    final_label = CLASS_NAMES[idx]
                    final_confidence = float(probs[idx])
                    final_model = "ViT"
                    vit_count += 1
                else:
                    # High complexity → ResNet
                    logits = resnet.run(None, {resnet_input: inp})[0][0]
                    probs = softmax(logits)
                    idx = int(np.argmax(probs))

                    final_label = CLASS_NAMES[idx]
                    final_confidence = float(probs[idx])
                    final_model = "ResNet50"
                    resnet_count += 1

                classified = True
                last_classify_time = current_time

        # ---------- DRAW ----------
        x1, y1, x2, y2 = locked_box
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(display, f"Objects: {num_objects}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2)

        cv2.putText(display, f"Class: {final_label}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 0, 0), 3)

        cv2.putText(display, f"Model: {final_model}",
                    (30, 125), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)

        cv2.putText(display, f"Confidence: {final_confidence:.2f}",
                    (30, 165), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 128, 255), 2)

    # ==================================================
    # FPS
    # ==================================================
    fps_times.append(time.time() - start_time)
    avg_fps = 1.0 / (sum(fps_times) / len(fps_times))

    cv2.putText(display, f"FPS: {avg_fps:.1f}",
                (WINDOW_W - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 2)

    # ==================================================
    # SWITCHING STATISTICS
    # ==================================================
    total = resnet_count + vit_count
    if total > 0:
        r_pct = 100 * resnet_count / total
        v_pct = 100 * vit_count / total
    else:
        r_pct = v_pct = 0

    cv2.putText(display, f"ResNet: {resnet_count} ({r_pct:.0f}%)",
                (WINDOW_W - 350, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 0), 2)

    cv2.putText(display, f"ViT: {vit_count} ({v_pct:.0f}%)",
                (WINDOW_W - 350, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 0), 2)

    # ==================================================
    # SHOW
    # ==================================================
    cv2.imshow("Adaptive Waste Classification", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ======================================================
# CLEANUP
# ======================================================
ser.close()
cv2.destroyAllWindows()
