import serial
import struct
import cv2
import numpy as np
import onnxruntime as ort
import time

# =========================
# CONFIGURATION
# =========================
SERIAL_PORT = "COM3"
BAUD_RATE = 2000000

RESNET_ONNX = "resnet50_waste_classifier.onnx"
VIT_ONNX = "vit_waste_classifier.onnx"

CONF_THRESHOLD = 0.75   # below this → switch to ViT

CLASS_NAMES = [
    "biodegradable",
    "hazardous",
    "non_biodegradable"
]

# =========================
# SERIAL INITIALIZATION
# =========================
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# =========================
# ONNX RUNTIME INITIALIZATION
# =========================
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

resnet_session = ort.InferenceSession(RESNET_ONNX, providers=providers)
vit_session = ort.InferenceSession(VIT_ONNX, providers=providers)

resnet_input_name = resnet_session.get_inputs()[0].name
vit_input_name = vit_session.get_inputs()[0].name

print("Models loaded successfully")

# =========================
# PREPROCESSING FUNCTION
# =========================
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    frame = (frame - mean) / std
    frame = frame.transpose(2, 0, 1)          # HWC → CHW
    frame = np.expand_dims(frame, axis=0)     # (1,3,224,224)

    return frame

# =========================
# SOFTMAX
# =========================
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

# =========================
# MAIN LOOP
# =========================
while True:
    # Wait for JPEG start marker
    if ser.read(2) != b'\xff\xd8':
        continue

    size_bytes = ser.read(4)
    size = struct.unpack('<I', size_bytes)[0]

    jpg = ser.read(size)
    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        continue

    # =========================
    # PREPROCESS
    # =========================
    input_tensor = preprocess(img)

    # =========================
    # RESNET INFERENCE (FAST)
    # =========================
    start = time.time()
    resnet_logits = resnet_session.run(
        None,
        {resnet_input_name: input_tensor}
    )[0]

    resnet_probs = softmax(resnet_logits[0])
    resnet_pred = int(np.argmax(resnet_probs))
    resnet_conf = float(resnet_probs[resnet_pred])

    used_model = "ResNet50"

    # =========================
    # SWITCH TO ViT IF NEEDED
    # =========================
    if resnet_conf < CONF_THRESHOLD:
        vit_logits = vit_session.run(
            None,
            {vit_input_name: input_tensor}
        )[0]

        vit_probs = softmax(vit_logits[0])
        vit_pred = int(np.argmax(vit_probs))
        vit_conf = float(vit_probs[vit_pred])

        pred_class = vit_pred
        confidence = vit_conf
        used_model = "ViT"
    else:
        pred_class = resnet_pred
        confidence = resnet_conf

    latency_ms = (time.time() - start) * 1000

    # =========================
    # DISPLAY RESULTS
    # =========================
    label = f"{CLASS_NAMES[pred_class]} | {used_model} | {confidence:.2f}"
    latency_label = f"{latency_ms:.1f} ms"

    cv2.putText(img, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(img, latency_label, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("ESP32-S3 Adaptive AI Inference", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP
# =========================
ser.close()
cv2.destroyAllWindows()
