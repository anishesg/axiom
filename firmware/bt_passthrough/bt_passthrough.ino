#include <SoftwareSerial.h>
SoftwareSerial btSerial(10, 11);

void setup() {
  Serial.begin(9600);
  btSerial.begin(9600);
  Serial.println("BT PASSTHROUGH READY");
}

void loop() {
  while (Serial.available()) btSerial.write(Serial.read());
  while (btSerial.available()) Serial.write(btSerial.read());
}
