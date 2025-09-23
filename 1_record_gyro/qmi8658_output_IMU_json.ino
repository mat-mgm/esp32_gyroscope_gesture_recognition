#include "FastIMU.h"
#include <Wire.h>

#define IMU_ADDRESS 0x6B
#define PERFORM_CALIBRATION
QMI8658 IMU;

calData calib = { 0 };
AccelData accelData;
GyroData gyroData;
MagData magData;

void setup() {
  Wire.begin(48, 47);
  Wire.setClock(400000);
  Serial.begin(115200);
  while (!Serial) {}

  int err = IMU.init(calib, IMU_ADDRESS);
  if (err != 0) {
    Serial.print("Error initializing IMU: ");
    Serial.println(err);
    while (true) {}
  }

#ifdef PERFORM_CALIBRATION
  if (IMU.hasMagnetometer()) {
    delay(1000);
    IMU.calibrateMag(&calib);
  } else {
    delay(5000);
  }
  delay(5000);
  IMU.calibrateAccelGyro(&calib);
  IMU.init(calib, IMU_ADDRESS);
#endif
}

void loop() {
  IMU.update();
  IMU.getAccel(&accelData);
  IMU.getGyro(&gyroData);

  Serial.print("{\"accelX\":"); Serial.print(accelData.accelX);
  Serial.print(",\"accelY\":"); Serial.print(accelData.accelY);
  Serial.print(",\"accelZ\":"); Serial.print(accelData.accelZ);
  Serial.print(",\"gyroX\":");  Serial.print(gyroData.gyroX);
  Serial.print(",\"gyroY\":");  Serial.print(gyroData.gyroY);
  Serial.print(",\"gyroZ\":");  Serial.print(gyroData.gyroZ);

  if (IMU.hasMagnetometer()) {
    IMU.getMag(&magData);
    Serial.print(",\"magX\":"); Serial.print(magData.magX);
    Serial.print(",\"magY\":"); Serial.print(magData.magY);
    Serial.print(",\"magZ\":"); Serial.print(magData.magZ);
  }
  if (IMU.hasTemperature()) {
    Serial.print(",\"temperature\":"); Serial.print(IMU.getTemp());
  }

  Serial.println("}");
  delay(50);
}
