import serial
import numpy as np
import tensorflow as tf
import json
import time

# -----------------------------
# CONFIG
# -----------------------------
SERIAL_PORT = "COM3"  # Change to your ESP32 port
BAUDRATE = 115200
TIMESTEPS = 200  # your model input length
# -----------------------------

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load normalization stats and classes
with open("label_map.json") as f:
    label_map = json.load(f)
mean, std = np.array(label_map["mean"]), np.array(label_map["std"])
classes = label_map["classes"]

# Open serial port
ser = serial.Serial(SERIAL_PORT, BAUDRATE)
time.sleep(2)  # wait for ESP32 reset
print("Connected to ESP32 on", SERIAL_PORT)

def record_gesture():
    """Send REC command and read TIMESTEPS samples"""
    buffer = []

    # send command to ESP32
    ser.write(b"REC\n")
    print("Recording gesture... move your hand now!")

    while True:
        line = ser.readline().decode().strip()
        if not line:
            continue
        if line == "START":
            continue
        if line == "END":
            break
        # parse gx, gy, gz
        parts = line.split(",")
        try:
            gx, gy, gz = map(float, parts[-3:])
            buffer.append([gx, gy, gz])
        except:
            continue

    if len(buffer) != TIMESTEPS:
        print(f"Warning: expected {TIMESTEPS} samples, got {len(buffer)}")
        return None

    return np.array(buffer, dtype=np.float32)

def predict(buffer):
    """Normalize, quantize, and run inference"""
    x = (buffer - mean) / std
    x = x.reshape(1, TIMESTEPS, 3)
    scale, zero_point = input_details[0]['quantization']
    x_q = x / scale + zero_point
    x_q = np.clip(np.round(x_q), -128, 127).astype(np.int8)
    interpreter.set_tensor(input_details[0]['index'], x_q)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred_class = classes[np.argmax(output)]
    return pred_class

print("Press Enter to record a gesture. Ctrl+C to exit.")

try:
    while True:
        input(">> ")  # wait for Enter
        buf = record_gesture()
        if buf is not None:
            pred = predict(buf)
            print("Prediction:", pred)
        else:
            print("Gesture recording failed. Try again.")
except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close()
