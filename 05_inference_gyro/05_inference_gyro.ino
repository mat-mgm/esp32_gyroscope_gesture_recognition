/**
 * inference_gyro.ino
 * ESP32 TinyML inference
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
#define SAMPLE_HZ 50
#define SAMPLE_MS (1000 / SAMPLE_HZ)
#define TIMESTEPS 100
#define CHANNELS 6
#define N_INPUTS (TIMESTEPS * CHANNELS)
#define N_OUTPUTS 7
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
    "down",
    "front",
    "left",
    "right",
    "still",
    "undefined",
    "up"
};

// Normalization values from training
float mean_vals[CHANNELS] = {
    -0.9516523480415344,
    1.0130267143249512,
    -0.44401687383651733,
    -0.027566734701395035,
    0.19213300943374634,
    -0.9869133830070496
}; // actual from label_map.json
float std_vals[CHANNELS]  = {
    22.676923751831055,
    21.948461532592773,
    22.89988136291504,
    0.24375328421592712,
    0.26008835434913635,
    0.22889447212219238
};   // actual

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
    int idx = 0;

    for (int i = 0; i < TIMESTEPS; ++i) {
        unsigned long start = millis();
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

        long wait = SAMPLE_MS - (long)(millis() - start);
        if (wait > 0) delay(wait);
    }

    // Normalize per-channel (same as training)
    for (int t = 0; t < TIMESTEPS; t++) {
        for (int c = 0; c < CHANNELS; c++) {
            int pos = t * CHANNELS + c;
            buffer[pos] = (buffer[pos] - mean_vals[c]) / std_vals[c];
        }
    }

    // Optional: print buffer stats for debugging
    float minv = buffer[0], maxv = buffer[0], sum = 0;
    for (int i = 0; i < N_INPUTS; i++) {
        if (buffer[i] < minv) minv = buffer[i];
        if (buffer[i] > maxv) maxv = buffer[i];
        sum += buffer[i];
    }
    Serial.print("buff_min: "); Serial.print(minv);
    Serial.print(" max: "); Serial.print(maxv);
    Serial.print(" mean: "); Serial.println(sum / N_INPUTS);

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
