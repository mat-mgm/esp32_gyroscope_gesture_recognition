/**
 * inference_gyro.ino
 * ESP32 TinyML inference with 6-channel IMU (gx,gy,gz,ax,ay,az)
 */

#include <Wire.h>
#include "FastIMU.h"

// Your converted model
#include "model.h"

// Runtime for ESP32 + EloquentTinyML wrapper
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// CONFIG
#define IMU_ADDRESS 0x6B
#define SAMPLE_HZ 100
#define SAMPLE_MS (1000 / SAMPLE_HZ)
#define TIMESTEPS 200
#define CHANNELS 6
#define N_INPUTS (TIMESTEPS * CHANNELS)
#define N_OUTPUTS 6
#define ARENA_SIZE 20*1024
#define TF_NUM_OPS 20

// EloquentTinyML object
Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

// IMU
QMI8658 IMU;
calData calib = {0};
float buffer[N_INPUTS];

// Class labels
const char* gestureLabels[N_OUTPUTS] = {
    "up",
    "down",
    "left",
    "right",
    "front",
    "still"
};

// Normalization values from training
float mean[CHANNELS] = {-0.4765488803386688, 0.5123229026794434, -0.21973125636577606, -0.01396484486758709, 0.09708882868289948, -0.49677619338035583}; // actual from label_map.json
float std_dev[CHANNELS]  = {16.046907424926758, 15.53460693359375, 16.19628143310547, 0.17297233641147614, 0.2079198658466339, 0.5193648338317871};   // actual

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("ESP32 TinyML inference demo");

    // Init IMU
    Wire.begin(48, 47);
    Wire.setClock(400000);
    if (IMU.init(calib, IMU_ADDRESS) != 0) {
        Serial.println("IMU INIT FAILED");
        while (1) delay(1000);
    }

    // Init TinyML model
    tf.setNumInputs(N_INPUTS);
    tf.setNumOutputs(N_OUTPUTS);
    tf.resolver.AddFullyConnected();
    tf.resolver.AddSoftmax();
    tf.resolver.AddReshape();

    while (!tf.begin(model).isOk()) {
        Serial.println(tf.exception.toString());
        delay(1000);
    }
    Serial.println("Model loaded! Press Enter to record a gesture.");
}

void loop() {
    // Trigger a recording round when user presses Enter
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd.length() > 0) {
            recordAndPredict();
        }
    }
}

void recordAndPredict() {
    Serial.println("Recording gesture...");
    unsigned long start = millis();
    int idx = 0;

    // Fill buffer with raw data
    while (idx < N_INPUTS && (millis() - start) < TIMESTEPS * SAMPLE_MS) {
        IMU.update();
        GyroData g;
        AccelData a;
        IMU.getGyro(&g);
        IMU.getAccel(&a);

        buffer[idx++] = g.gyroX;
        buffer[idx++] = g.gyroY;
        buffer[idx++] = g.gyroZ;
        buffer[idx++] = a.accelX;
        buffer[idx++] = a.accelY;
        buffer[idx++] = a.accelZ;

        delay(SAMPLE_MS);
    }

    // Normalize
    for (int t = 0; t < TIMESTEPS; t++) {
        for (int c = 0; c < CHANNELS; c++) {
            buffer[t * CHANNELS + c] = (buffer[t * CHANNELS + c] - mean[c]) / std_dev[c];
        }
    }

    // Predict
    if (!tf.predict(buffer).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }

    int prediction = tf.classification;
    Serial.print("Prediction: ");
    Serial.print(prediction);
    Serial.print(" -> ");
    Serial.println(gestureLabels[prediction]);
}
