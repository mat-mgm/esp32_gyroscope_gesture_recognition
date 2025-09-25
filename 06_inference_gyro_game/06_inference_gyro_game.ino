/**
 * inference_gyro_game.ino
 * ESP32 TinyML Gesture Recognition + Memory Game
 */

#include <Wire.h>
#include "FastIMU.h"
#include "model.h"
#include "model_params.h"
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>
#include <Arduino_GFX_Library.h>

// CONFIG
#define IMU_ADDRESS 0x6B
#define SAMPLE_HZ 50
#define SAMPLE_MS (1000 / SAMPLE_HZ)
#define N_INPUTS (TIMESTEPS * N_FEATURES)
#define ARENA_SIZE 20*1024
#define TF_NUM_OPS 20
#define MAX_SEQUENCE 50

// EloquentTinyML object
Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

// IMU
QMI8658 IMU;
calData calib = {0};
float buffer[N_INPUTS];

// GRAPHICS
// Pin definitions for Waveshare ESP32-S3 Touch LCD 2.8"
#define LCD_SCLK 39
#define LCD_MOSI 38
#define LCD_MISO 40
#define LCD_DC   42
#define LCD_RST  -1
#define LCD_CS   45
#define LCD_BL   1
// LCD size
#define LCD_WIDTH  240
#define LCD_HEIGHT 320
// Setup Arduino_GFX bus and display
Arduino_DataBus *bus = new Arduino_ESP32SPI(LCD_DC, LCD_CS, LCD_SCLK, LCD_MOSI, LCD_MISO);
Arduino_GFX *gfx = new Arduino_ST7789(bus, LCD_RST, 0 /* rotation */, true /* IPS */, LCD_WIDTH, LCD_HEIGHT);
// Define 8 colors (RGB565 format)
uint16_t colors[9] = {
  MAGENTA,  // 0, down
  WHITE,    // 1, front
  BLUE,     // 2, left
  YELLOW,   // 3, right
  CYAN,     // 4, still
  BLACK,    // 5, undefined
  gfx->color565(255, 165, 0),  // 6, up (orange)
  RED,      // 7
  GREEN,    // 8
};


// Game sequence
int sequence[MAX_SEQUENCE];
int sequence_len = 0;

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("ESP32 Gesture Recognition & Memory Game");
    Serial.println("Choose mode: 1 = Single gesture, 2 = Memory game");

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
    Serial.println("Model loaded! Enter 1 or 2:");

    // Setup screen
    gfx->begin();

    // Turn on backlight
    pinMode(LCD_BL, OUTPUT);
    digitalWrite(LCD_BL, HIGH);
}

void loop() {
    delay(2000);
    gfx->fillScreen(colors[5]); // clear screen
    if (Serial.available()) {
        String mode = Serial.readStringUntil('\n');
        mode.trim();
        if (mode == "1") {
            Serial.println("Single gesture recognition mode.");
            Serial.println("Press Enter to record gesture:");
            waitForEnter();
            int g = recordAndPredict();
            Serial.print("Detected gesture: "); Serial.println(gestureLabels[g]);
            gfx->fillScreen(colors[g]);
            Serial.println("\nEnter 1 = Single gesture, 2 = Memory game:");
        }
        else if (mode == "2") {
            Serial.println("Memory game mode!");
            playMemoryGame();
            Serial.println("\nEnter 1 = Single gesture, 2 = Memory game:");
        }
        else {
            Serial.println("\nEnter 1 = Single gesture, 2 = Memory game:");
        }
    }
}

void waitForEnter() {
    while (Serial.available()) Serial.read(); // flush
    while (!Serial.available()) delay(10);
    Serial.readStringUntil('\n');
}

int recordAndPredict() {
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

    // Normalize
    for (int t = 0; t < TIMESTEPS; t++) {
        for (int c = 0; c < N_FEATURES; c++) {
            int pos = t * N_FEATURES + c;
            buffer[pos] = (buffer[pos] - mean_vals[c]) / std_vals[c];
        }
    }

    if (!tf.predict(buffer).isOk()) {
        Serial.println("Prediction error!");
        return -1;
    }
    return tf.classification;
}

void playMemoryGame() {
    sequence_len = 0;
    int round = 1;
    bool game_over = false;

    while (!game_over && sequence_len < MAX_SEQUENCE) {
        Serial.print("Round "); Serial.println(round);
        Serial.println("Press Enter for player to start...");
        waitForEnter();

        // Player repeats previous gestures
        for (int i = 0; i < sequence_len; i++) {
            int g = recordAndPredict();
            Serial.print("Expected: "); Serial.print(gestureLabels[sequence[i]]);
            Serial.print(" | Detected: "); Serial.println(gestureLabels[g]);
            if (g != sequence[i]) {
                Serial.println("Incorrect! Game over.");
                game_over = true;
                gfx->fillScreen(colors[7]);
                return;
            }
            else {
                gfx->fillScreen(colors[8]);
            }
        }

        // Player adds new gesture
        Serial.println("Add a new gesture:");
        waitForEnter();
        int new_g = recordAndPredict();
        gfx->fillScreen(colors[new_g]);
        sequence[sequence_len++] = new_g;
        Serial.print("Sequence length: "); Serial.println(sequence_len);

        round++;
    }

    Serial.println("Congratulations! You completed all rounds.");
}
