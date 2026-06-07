/*
 * robot_car.ino - 4-motor differential drive + servo + ultrasonic
 * Accepts commands via USB Serial AND BLE (HM-10 on SoftwareSerial D10/D11)
 */

#include <SoftwareSerial.h>

SoftwareSerial btSerial(11, 10); // RX=D11 (HM-10 TX), TX=D10 (HM-10 RX)

const uint8_t MOTOR_PIN[4] = {3, 5, 6, 9};
const uint8_t LEFT_IDX[2]  = {2, 3};   // D6=rear-left, D9=front-left
const uint8_t RIGHT_IDX[2] = {0, 1};   // D3=rear-right, D5=front-right

const uint8_t SERVO_PIN = 7;
const uint8_t TRIG_PIN  = 12;
const uint8_t ECHO_PIN  = 13;

uint8_t motorSpeed[4] = {0, 0, 0, 0};
uint8_t cruiseSpeed   = 200;
uint8_t servoAngle    = 90;

void rampMotor(uint8_t idx, uint8_t target) {
  uint8_t cur = motorSpeed[idx];
  if (target > cur) {
    for (uint8_t v = cur; v < target; v += 5) {
      analogWrite(MOTOR_PIN[idx], v);
      delay(3);
    }
  }
  motorSpeed[idx] = target;
  analogWrite(MOTOR_PIN[idx], target);
}

void setMotor(uint8_t idx, uint8_t spd) {
  rampMotor(idx, spd);
}

void setLeft(uint8_t spd)  { for (int i = 0; i < 2; i++) setMotor(LEFT_IDX[i],  spd); }
void setRight(uint8_t spd) { for (int i = 0; i < 2; i++) setMotor(RIGHT_IDX[i], spd); }
void stopAll()             { for (int i = 0; i < 4; i++) { motorSpeed[i] = 0; analogWrite(MOTOR_PIN[i], 0); } }

void servoPulse(uint8_t angle) {
  uint16_t pw = map(constrain(angle, 0, 180), 0, 180, 544, 2400);
  for (uint8_t i = 0; i < 30; i++) {
    digitalWrite(SERVO_PIN, HIGH);
    delayMicroseconds(pw);
    digitalWrite(SERVO_PIN, LOW);
    delayMicroseconds(5000);
    delayMicroseconds(5000);
    delayMicroseconds(5000);
    delayMicroseconds(5000 - pw);
  }
}

void servoSet(uint8_t angle) {
  servoAngle = constrain(angle, 0, 180);
  servoPulse(servoAngle);
}

long readUltrasonicCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return -1;
  return duration / 58;
}

void handleCmd(const String &raw, Stream &io) {
  String cmd = raw;
  cmd.trim();
  if (!cmd.length()) return;
  char c = cmd.charAt(0);

  if (c == '?') { io.println("READY"); return; }

  if (c == 'S') { stopAll(); io.println("STOP"); return; }

  if (c == 'F') {
    uint8_t spd = cmd.length() > 1 ? cmd.substring(1).toInt() : cruiseSpeed;
    setLeft(spd); setRight(spd);
    io.print("FWD "); io.println(spd); return;
  }

  if (c == 'L') {
    uint8_t spd = cmd.length() > 1 ? cmd.substring(1).toInt() : cruiseSpeed;
    setLeft(0); setRight(spd);
    io.print("LEFT "); io.println(spd); return;
  }

  if (c == 'R') {
    uint8_t spd = cmd.length() > 1 ? cmd.substring(1).toInt() : cruiseSpeed;
    setLeft(spd); setRight(0);
    io.print("RIGHT "); io.println(spd); return;
  }

  if (c == 'T') {
    int comma = cmd.indexOf(',');
    if (comma > 0) {
      uint8_t l = cmd.substring(1, comma).toInt();
      uint8_t r = cmd.substring(comma + 1).toInt();
      setLeft(l); setRight(r);
      io.print("TANK L="); io.print(l); io.print(" R="); io.println(r);
    }
    return;
  }

  if (c == 'M') {
    int eq = cmd.indexOf('=');
    if (eq > 0) {
      uint8_t idx = cmd.substring(1, eq).toInt();
      uint8_t val = cmd.substring(eq + 1).toInt();
      if (idx < 4) { setMotor(idx, val); io.print("M"); io.print(idx); io.print("="); io.println(val); }
    }
    return;
  }

  if (c == 'V') {
    cruiseSpeed = cmd.substring(1).toInt();
    io.print("CRUISE="); io.println(cruiseSpeed); return;
  }

  if (c == 'I') {
    io.println("IDENTIFY: pulsing each motor 1s");
    stopAll();
    for (uint8_t i = 0; i < 4; i++) {
      io.print("Motor "); io.print(i); io.print(" D"); io.print(MOTOR_PIN[i]); io.println(" ON");
      setMotor(i, 220); delay(1000);
      setMotor(i, 0);   delay(400);
    }
    io.println("IDENTIFY done"); return;
  }

  if (c == 'P') {
    int eq = cmd.indexOf('=');
    if (eq > 0) {
      uint8_t pin = cmd.substring(1, eq).toInt();
      uint8_t val = cmd.substring(eq + 1).toInt();
      pinMode(pin, OUTPUT);
      if (val < 2) digitalWrite(pin, val); else analogWrite(pin, val);
      io.print("PIN "); io.print(pin); io.print("="); io.println(val);
    }
    return;
  }

  if (c == 'A') {
    uint8_t ang = cmd.length() > 1 ? cmd.substring(1).toInt() : 90;
    servoSet(ang);
    io.print("SERVO "); io.println(ang); return;
  }

  if (c == 'U') {
    long cm = readUltrasonicCm();
    if (cm < 0) { io.println("DIST -1"); }
    else { io.print("DIST "); io.println(cm); }
    return;
  }

  if (c == 'H') {
    io.println("SWEEP start");
    for (uint8_t a = 0; a <= 180; a += 10) {
      servoSet(a);
      io.print("SERVO "); io.println(a);
      delay(200);
    }
    for (int a = 180; a >= 0; a -= 10) {
      servoSet((uint8_t)a);
      io.print("SERVO "); io.println(a);
      delay(200);
    }
    io.println("SWEEP done"); return;
  }

  io.print("ERR: "); io.println(cmd);
}

void setup() {
  // Drive motor pins LOW via direct register writes before anything else.
  // D3=PD3, D5=PD5, D6=PD6, D9=PB1
  DDRD  |=  (1<<3)|(1<<5)|(1<<6);   // set as output
  PORTD &= ~((1<<3)|(1<<5)|(1<<6)); // drive LOW
  DDRB  |=  (1<<1);
  PORTB &= ~(1<<1);

  Serial.begin(9600);
  btSerial.begin(9600);
  for (int i = 0; i < 4; i++) {
    pinMode(MOTOR_PIN[i], OUTPUT);
    analogWrite(MOTOR_PIN[i], 0);
  }
  pinMode(SERVO_PIN, OUTPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  servoSet(90);
  Serial.println("ROBOT READY");
  btSerial.println("ROBOT READY");
}

String buf = "";
String btBuf = "";
void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') { handleCmd(buf, Serial); buf = ""; }
    else buf += c;
  }
  while (btSerial.available()) {
    char c = btSerial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      handleCmd(btBuf, btSerial);
      btBuf = "";
    }
    else btBuf += c;
  }
}
