# Gesture recognition with edge AI in ESP32 microcontroller

**Real‑time gesture recognition** on the **Waveshare ESP32‑S3 Touch LCD 2.8"** using **Edge AI**.  
This project covers the full workflow — from collecting gyroscope data, to training a lightweight neural network, to deploying it for on‑device inference — all running locally on the ESP32 with a **touchscreen LCD** interface.

---

## 📦 Hardware

- [Waveshare ESP32‑S3 Touch LCD 2.8"](https://www.waveshare.com/wiki/ESP32-S3-Touch-LCD-2)  
- Built‑in IMU (gyroscope)  
- Capacitive touchscreen for UI interaction and control

---

## 🛠 Software Setup

This project has been implemented and tested in the environment provided by Arduino IDE.

1. **Add ESP32 Board Manager URL**  
   Add the following URL in `File > Preferences > Settings > Additional boards manager URLs`:
   ```
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```
2. **Install Board Package**  
   - Search for `esp32` by Espressif in Boards Manager and install.

3. **Install Required Libraries**
   | Library | Purpose | Version |
   |---------|---------|---------|
   | lvgl | Graphical UI | v8.4.0 |
   | GFX_Library_for_Arduino | LCD driver | v1.5.0 |
   | FastIMU | IMU driver | v1.2.6 |
   | bsp_cst816 | Touch driver | — |

   **Offline Install:**  
   Download from:  
   [ESP32-S3-Touch-LCD-2 Demo ZIP](https://files.waveshare.com/wiki/ESP32-S3-Touch-LCD-2/ESP32-S3-Touch-LCD-2-Demo.zip)  
   Copy the `libraries` folder contents into your Arduino `Documents/Arduino/libraries` directory.

4. **Board Settings**
   ```
   Tools > Board > ESP32S3 Dev Module
   Tools > USB CDC On Boot > Enabled
   Tools > Flash Size > 16MB (128Mb)
   Tools > Partition Scheme > 16M Flash (3MB APP/9.9MB FATFS)
   Tools > PSRAM > OPI PSRAM
   Tools > Upload Mode > USB-OTG CDC (TinyUSB)
   ```

---

## 📋 Project Workflow

### **1. Data Collection**
- Arduino sketch captures **gyroscope data** in **1–2 second intervals**.
- Perform **30 repetitions** for each of **two distinct gestures**.
- Save readings in **CSV format** for training.

### **2. Model Training**
- Train a **small MLP** in TensorFlow:
  - 1 fully connected layer (<100 neurons)
  - 8‑bit quantization for deployment
- Export as **TensorFlow Lite Micro** model.

### **3. Deployment**
- Arduino sketch:
  - Captures a **1‑second window** of gyroscope data on demand.
  - Runs inference locally using the TFLite Micro model.
  - Classifies the gesture and displays the result on the **touchscreen LCD**.

---

## 📂 Repository Structure

```
.
├── 01_record_gyro  # Arduino sketches for recording raw gyro/IMU data
│   └── 01_record_gyro.ino
├── 02_capture_data  # Python script + CSV datasets for gesture capture
│   ├── capture.py
│   └── data
│       ├── down.csv
│       ├── front.csv
│       ├── left.csv
│       ├── right.csv
│       ├── still.csv
│       └── up.csv
├── 03_visualize_check_data  # Check and visualize collected data
│   ├── check_csv.py
│   ├── visualize_gestures_metrics.py
│   └── visualize_metrics.py
├── 04_model_training  # Training, conversion, and evaluation of ML models
│   ├── model
│   │   ├── label_map.json
│   │   ├── model.tflite
│   │   └── model_quant.tflite
│   ├── tflite_to_c.py
│   └── train_mlp.py
├── 05_inference_gyro    # Arduino sketch for running inference on PC/ESP32
│   ├── 05_inference_gyro.ino
│   ├── model.h
│   └── model_params.h
├── 06_inference_gyro_game
│   ├── 06_inference_gyro_game.ino
│   ├── model.h
│   └── model_params.h
├── LICENSE
├── README.md
└── demo
    └── esp32_gesture_recognition.pdf
```

---

## 🚀 Getting Started

1. **Clone this repo**  
   ```bash
   git clone https://github.com/mat-mgm/esp32_gyroscope_gesture_recognition.git
   ```
2. **Follow the Software Setup** steps above.
3. **Run `data_collection` sketch** to gather your dataset.
4. **Train the model** using scripts in `model_training/`.
5. **Deploy the model** with the `inference` sketch.

---

## 📌 Notes
- All inference is done **on-device** — no internet connection required.
- Designed for **low-latency, low-power** gesture recognition.
- Touchscreen interface can be extended for gesture selection, calibration, or visualization.

---

## 📜 License
This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.

---
