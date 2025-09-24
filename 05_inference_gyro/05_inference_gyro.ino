/**
 * inference_gyro_pc.ino
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
#define N_OUTPUTS 4
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
    "right"
};

// Normalization values from training
float mean[CHANNELS] = {-2.1658389568328857, 0.7307839393615723, 0.07644910365343094, -0.024928510189056396, 0.21519939601421356, -0.9713585376739502}; // actual from label_map.json
float std_dev[CHANNELS]  = {25.900360107421875, 24.879959106445312, 27.77985382080078, 0.2450098842382431, 0.32145294547080994, 0.2965543270111084};   // actual

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("ESP32 TinyML 6-channel inference demo");

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
