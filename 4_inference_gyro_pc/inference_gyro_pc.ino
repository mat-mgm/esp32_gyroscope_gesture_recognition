/**
 * inference_gyro_pc.ino
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
#define N_INPUTS (TIMESTEPS * 3)
#define N_OUTPUTS 4
#define ARENA_SIZE 20*1024
#define TF_NUM_OPS 20

// EloquentTinyML object
Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

// Gyro
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
    Serial.println("Model loaded! Press Enter in Serial Monitor to record a gesture.");
}

void loop() {
    // Trigger a recording round when user presses Enter from Serial Monitor
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

    while (millis() - start < TIMESTEPS * SAMPLE_MS && idx < N_INPUTS) {
        IMU.update();
        GyroData g;
        IMU.getGyro(&g);

        buffer[idx++] = g.gyroX;
        buffer[idx++] = g.gyroY;
        buffer[idx++] = g.gyroZ;

        delay(SAMPLE_MS);
    }

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
