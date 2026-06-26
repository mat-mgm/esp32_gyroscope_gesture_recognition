# Gesture recognition with edge AI in ESP32 microcontroller

**Real-time gesture recognition** on the **Waveshare ESP32-S3 Touch LCD 2.8"** using **Edge AI**.  
This project covers the full workflow — from collecting gyroscope data, to training a lightweight neural network, to deploying it for on-device inference — all running locally on the ESP32 with a **touchscreen LCD** interface.

---

## Hardware

- [Waveshare ESP32-S3 Touch LCD 2.8"](https://www.waveshare.com/wiki/ESP32-S3-Touch-LCD-2)  
- Built-in IMU (gyroscope)  
- Capacitive touchscreen for UI interaction and control

---

## Software Setup

This project has been implemented and tested in the environment provided by Arduino IDE.

1. **Add ESP32 Board Manager URL**  
   Add the following URL in `File > Preferences > Settings > Additional boards manager URLs`:
   ```
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```
2. **Install Board Package**  
   Search for `esp32` by Espressif in Boards Manager and install.

3. **Install Required Libraries**
   | Library | Purpose | Version |
   |---------|---------|---------|
   | lvgl | Graphical UI | v8.4.0 |
   | GFX_Library_for_Arduino | LCD driver | v1.5.0 |
   | FastIMU | IMU driver | v1.2.6 |
   | bsp_cst816 | Touch driver | — |

   **Offline Install:**  
   Download from [ESP32-S3-Touch-LCD-2 Demo ZIP](https://files.waveshare.com/wiki/ESP32-S3-Touch-LCD-2/ESP32-S3-Touch-LCD-2-Demo.zip) and copy the `libraries` folder contents into your Arduino `Documents/Arduino/libraries` directory.

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

## Project Workflow

### 1. Data Collection
- Arduino sketch captures **gyroscope data** in 1-2 second intervals.
- Perform **30 repetitions** for each gesture.
- Save readings in CSV format for training.

### 2. Model Training
- Train a small MLP in TensorFlow: 1 fully connected layer (<100 neurons), 8-bit quantization.
- Export as **TensorFlow Lite Micro** model.

### 3. Deployment
- Arduino sketch captures a 1-second window of gyroscope data on demand, runs inference locally using the TFLite Micro model, and displays the result on the touchscreen LCD.

---

## Repository Structure

```
.
├── 01_record_gyro           # Arduino sketch for recording raw gyro/IMU data
├── 02_capture_data          # Python capture script and CSV gesture datasets
│   └── data/                # down, front, left, right, still, up
├── 03_visualize_check_data  # Scripts to inspect and visualize collected data
├── 04_model_training        # Training script, TFLite conversion, exported models
│   └── model/               # model.tflite, model_quant.tflite, label_map.json
├── 05_inference_gyro        # Arduino inference sketch (serial output)
├── 06_inference_gyro_game   # Arduino inference sketch with LCD memory game
└── demo/                    # Presentation PDF
```

---

## Getting Started

1. **Clone this repo**
   ```bash
   git clone https://github.com/mat-mgm/esp32_gyroscope_gesture_recognition.git
   ```
2. Follow the Software Setup steps above.
3. Flash `01_record_gyro` to collect gesture data.
4. Run `train_mlp.py` in `04_model_training/` to train and export the model.
5. Flash `05_inference_gyro` or `06_inference_gyro_game` to run on-device inference.

---

## Notes

- All inference runs on-device — no internet connection required.
- Designed for low-latency, low-power gesture recognition.
- The quantized INT8 model (`model_quant.tflite`) is included so inference can be tested without retraining.

---

## License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.
