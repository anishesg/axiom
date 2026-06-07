#include <Servo.h>

#define SERVO_PIN 6
#define TRIG_PIN 10
#define ECHO_PIN 9

#define SWEEP_MIN 0
#define SWEEP_MAX 180
#define SWEEP_STEP 2
#define SETTLE_MS 25
#define MAX_DISTANCE_CM 300

Servo scanner;

int pos = SWEEP_MIN;
int dir = 1;

// Motor pins (L298N or similar driver) — define when hardware is connected
// #define ENA 3
// #define IN1 4
// #define IN2 7
// #define IN3 8
// #define IN4 12
// #define ENB 11

float readDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(ECHO_PIN, HIGH, 25000);
  if (duration == 0) return -1.0;

  float cm = duration * 0.0343 / 2.0;
  if (cm > MAX_DISTANCE_CM) return -1.0;
  return cm;
}

void setup() {
  Serial.begin(9600);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  scanner.attach(SERVO_PIN);
  scanner.write(SWEEP_MIN);
  delay(500);

  Serial.println("AXIOM_ROVER_READY");
}

void loop() {
  // Check for incoming motor commands from the server
  if (Serial.available()) {
    handleCommand();
  }

  scanner.write(pos);
  delay(SETTLE_MS);

  // Take 2 readings and use the shorter (more reliable) one
  float d1 = readDistance();
  float d2 = readDistance();
  float d;
  if (d1 < 0 && d2 < 0) d = -1.0;
  else if (d1 < 0) d = d2;
  else if (d2 < 0) d = d1;
  else d = min(d1, d2);

  // Protocol: R,<angle>,<distance_cm>\n
  // R = radar reading. -1 means no echo / out of range
  Serial.print("R,");
  Serial.print(pos);
  Serial.print(",");
  Serial.println(d, 1);

  // Advance sweep
  pos += dir * SWEEP_STEP;
  if (pos >= SWEEP_MAX) {
    pos = SWEEP_MAX;
    dir = -1;
  } else if (pos <= SWEEP_MIN) {
    pos = SWEEP_MIN;
    dir = 1;
  }
}

void handleCommand() {
  static char buf[32];
  static uint8_t idx = 0;

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (idx > 0) {
        buf[idx] = '\0';
        processCommand(buf);
        idx = 0;
      }
    } else if (idx < sizeof(buf) - 1) {
      buf[idx++] = c;
    }
  }
}

void processCommand(const char* cmd) {
  // M,<left_speed>,<right_speed>
  //   speed: -255 to 255 (negative = reverse)
  // S,<angle>
  //   point the scanner at a specific angle (overrides sweep temporarily)
  // P
  //   ping — responds with PONG for connection test
  // H
  //   halt all motors immediately

  if (cmd[0] == 'P') {
    Serial.println("PONG");
    return;
  }

  if (cmd[0] == 'H') {
    // stopMotors();
    Serial.println("A,HALT");
    return;
  }

  if (cmd[0] == 'S' && cmd[1] == ',') {
    int angle = atoi(cmd + 2);
    angle = constrain(angle, SWEEP_MIN, SWEEP_MAX);
    pos = angle;
    dir = 0; // pause sweep
    scanner.write(pos);
    Serial.print("A,SCAN,");
    Serial.println(pos);
    return;
  }

  if (cmd[0] == 'R') {
    // Resume automatic sweep
    dir = 1;
    Serial.println("A,RESUME");
    return;
  }

  if (cmd[0] == 'M' && cmd[1] == ',') {
    // Parse M,<left>,<right>
    char copy[32];
    strncpy(copy, cmd + 2, sizeof(copy) - 1);
    copy[sizeof(copy) - 1] = '\0';
    char* comma = strchr(copy, ',');
    if (comma) {
      *comma = '\0';
      int left = atoi(copy);
      int right = atoi(comma + 1);
      left = constrain(left, -255, 255);
      right = constrain(right, -255, 255);
      setMotors(left, right);
      Serial.print("A,MOT,");
      Serial.print(left);
      Serial.print(",");
      Serial.println(right);
    }
    return;
  }
}

void setMotors(int left, int right) {
  // Uncomment when motor driver is wired up:
  //
  // // Left motor
  // if (left >= 0) {
  //   digitalWrite(IN1, HIGH);
  //   digitalWrite(IN2, LOW);
  //   analogWrite(ENA, left);
  // } else {
  //   digitalWrite(IN1, LOW);
  //   digitalWrite(IN2, HIGH);
  //   analogWrite(ENA, -left);
  // }
  //
  // // Right motor
  // if (right >= 0) {
  //   digitalWrite(IN3, HIGH);
  //   digitalWrite(IN4, LOW);
  //   analogWrite(ENB, right);
  // } else {
  //   digitalWrite(IN3, LOW);
  //   digitalWrite(IN4, HIGH);
  //   analogWrite(ENB, -right);
  // }
  (void)left;
  (void)right;
}

// void stopMotors() {
//   analogWrite(ENA, 0);
//   analogWrite(ENB, 0);
//   digitalWrite(IN1, LOW);
//   digitalWrite(IN2, LOW);
//   digitalWrite(IN3, LOW);
//   digitalWrite(IN4, LOW);
// }
