// record_gyro.ino -- sends CSV to Serial after receiving "REC\n"
#include <Wire.h>
#include "FastIMU.h"

#define IMU_ADDRESS 0x6B  

QMI8658 IMU;
calData calib = {0}; 

const int SAMPLE_HZ = 50;
const int SAMPLE_MS = 1000 / SAMPLE_HZ;

void setup() {
  Wire.begin(48, 47);
  Wire.setClock(400000);
  Serial.begin(115200);
  delay(200);

  int err = IMU.init(calib, IMU_ADDRESS);
  if (err != 0) {
    Serial.println("IMU_INIT_FAILED");
    while(1) delay(1000);
  }

  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "REC") {
      recordWindow(2000); // 2 seconds
    } else if (cmd.startsWith("REC,")) {
      int dur = cmd.substring(4).toInt();
      if (dur > 0) recordWindow(dur);
    }
  }
}

void recordWindow(int duration_ms) {
  Serial.println("START");
  unsigned long start = millis();
  unsigned long nextSample = start;
  while (millis() - start < (unsigned long)duration_ms) {
    IMU.update();

    GyroData g;
    AccelData a;
    IMU.getGyro(&g);
    IMU.getAccel(&a);

    unsigned long t = millis() - start;

    Serial.print(t);
    Serial.print(',');
    Serial.print(g.gyroX, 3);
    Serial.print(',');
    Serial.print(g.gyroY, 3);
    Serial.print(',');
    Serial.print(g.gyroZ, 3);
    Serial.print(',');
    Serial.print(a.accelX, 3);
    Serial.print(',');
    Serial.print(a.accelY, 3);
    Serial.print(',');
    Serial.println(a.accelZ, 3);

    nextSample += SAMPLE_MS;
    long wait = (long)nextSample - (long)millis();
    if (wait > 0) delay(wait);
  }
  Serial.println("END");
}
