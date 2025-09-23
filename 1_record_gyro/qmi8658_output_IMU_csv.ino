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

  // Print CSV header
  Serial.print("accelX,accelY,accelZ,gyroX,gyroY,gyroZ");
  if (IMU.hasMagnetometer()) {
    Serial.print(",magX,magY,magZ");
  }
  if (IMU.hasTemperature()) {
    Serial.print(",temperature");
  }
  Serial.println();
}

void loop() {
  IMU.update();
  IMU.getAccel(&accelData);
  IMU.getGyro(&gyroData);

  Serial.print(accelData.accelX); Serial.print(",");
  Serial.print(accelData.accelY); Serial.print(",");
  Serial.print(accelData.accelZ); Serial.print(",");
  Serial.print(gyroData.gyroX);   Serial.print(",");
  Serial.print(gyroData.gyroY);   Serial.print(",");
  Serial.print(gyroData.gyroZ);

  if (IMU.hasMagnetometer()) {
    IMU.getMag(&magData);
    Serial.print(","); Serial.print(magData.magX);
    Serial.print(","); Serial.print(magData.magY);
    Serial.print(","); Serial.print(magData.magZ);
  }
  if (IMU.hasTemperature()) {
    Serial.print(","); Serial.print(IMU.getTemp());
  }
  Serial.println();
  delay(50);
}