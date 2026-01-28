import serial
import struct
import cv2
import numpy as np

ser = serial.Serial('COM3', 2000000, timeout=1)

while True:
    # wait for JPEG marker
    if ser.read(2) != b'\xff\xd8':
        continue

    size_bytes = ser.read(4)
    size = struct.unpack('<I', size_bytes)[0]

    jpg = ser.read(size)
    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is not None:
        cv2.imshow("ESP32 USB Camera", img)

    if cv2.waitKey(1) == 27:
        break
