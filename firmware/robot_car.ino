/*
 * robot_car.ino — 4-motor differential drive
 *
 * Circuit: PN2222 low-side switch per motor (unidirectional)
 *   Arduino PWM → 1kΩ → PN2222 Base
 *   Emitter → GND  |  Collector → Motor(-)  |  Motor(+) → Vcc
 *
 * ── Pin map (update LEFT_*/RIGHT_* if motors are swapped) ───────────
 *   D3  = Motor 0   D5  = Motor 1   (LEFT  side by default)
 *   D6  = Motor 2   D9  = Motor 3   (RIGHT side by default)
 *
 * ── Serial commands (9600 baud, newline-terminated) ─────────────────
 *   ?              ping → READY
 *   F<spd>         forward,  spd 0-255
 *   S              stop all
 *   L<spd>         pivot left  (right motors on, left off)
 *   R<spd>         pivot right (left motors on, right off)
 *   T<l>,<r>       tank: set left speed, right speed individually
 *   M<n>=<v>       set single motor n (0-3), v (0-255)
 *   I              identify: pulse each motor 0-3 in sequence
 *   V<spd>         set cruise speed (stored, used by F/L/R)
 */

#include <Arduino.h>

// ── Motor pin assignment — EDIT THESE if wiring differs ─────────────
const uint8_t MOTOR_PIN[4] = { 3, 5, 6, 9 };
// Left side motors (indices into MOTOR_PIN[])
const uint8_t LEFT_IDX[2]  = { 0, 1 };   // D3, D5
// Right side motors
const uint8_t RIGHT_IDX[2] = { 2, 3 };   // D6, D9

// ── State ────────────────────────────────────────────────────────────
uint8_t motorSpeed[4]  = {0, 0, 0, 0};
uint8_t cruiseSpeed    = 200;

// ── Motor helpers ────────────────────────────────────────────────────
void setMotor(uint8_t idx, uint8_t spd) {
  motorSpeed[idx] = spd;
  analogWrite(MOTOR_PIN[idx], spd);
}

void setLeft(uint8_t spd)  { for (uint8_t i : LEFT_IDX)  setMotor(i, spd); }
void setRight(uint8_t spd) { for (uint8_t i : RIGHT_IDX) setMotor(i, spd); }
void stopAll()             { for (uint8_t i = 0; i < 4; i++) setMotor(i, 0); }

void printState(Stream &io) {
  io.print(F("M0="));  io.print(motorSpeed[0]);
  io.print(F(" M1=")); io.print(motorSpeed[1]);
  io.print(F(" M2=")); io.print(motorSpeed[2]);
  io.print(F(" M3=")); io.println(motorSpeed[3]);
}

// ── Command handler ───────────────────────────────────────────────────
void handleCmd(const String &raw, Stream &io) {
  String cmd = raw;
  cmd.trim();
  if (!cmd.length()) return;

  char c = cmd.charAt(0);

  if (c == '?') {
    io.println(F("READY"));
    return;
  }

  if (c == 'S' || cmd == "STOP") {
    stopAll();
    io.println(F("STOP"));
    return;
  }

  if (c == 'F') {
    uint8_t spd = cmd.length() > 1 ? cmd.substring(1).toInt() : cruiseSpeed;
    setLeft(spd); setRight(spd);
    io.print(F("FWD ")); io.println(spd);
    return;
  }

  if (c == 'L') {
    uint8_t spd = cmd.length() > 1 ? cmd.substring(1).toInt() : cruiseSpeed;
    setLeft(0); setRight(spd);
    io.print(F("LEFT ")); io.println(spd);
    return;
  }

  if (c == 'R') {
    uint8_t spd = cmd.length() > 1 ? cmd.substring(1).toInt() : cruiseSpeed;
    setLeft(spd); setRight(0);
    io.print(F("RIGHT ")); io.println(spd);
    return;
  }

  // T<left>,<right> — tank / skid steer
  if (c == 'T') {
    int comma = cmd.indexOf(',');
    if (comma > 0) {
      uint8_t l = cmd.substring(1, comma).toInt();
      uint8_t r = cmd.substring(comma + 1).toInt();
      setLeft(l); setRight(r);
      io.print(F("TANK L=")); io.print(l);
      io.print(F(" R=")); io.println(r);
    }
    return;
  }

  // M<n>=<v> — individual motor
  if (c == 'M') {
    int eq = cmd.indexOf('=');
    if (eq > 0) {
      uint8_t idx = cmd.substring(1, eq).toInt();
      uint8_t val = cmd.substring(eq + 1).toInt();
      if (idx < 4) {
        setMotor(idx, val);
        io.print(F("MOTOR")); io.print(idx); io.print('='); io.println(val);
      }
    }
    return;
  }

  // V<spd> — set cruise speed
  if (c == 'V') {
    cruiseSpeed = cmd.substring(1).toInt();
    io.print(F("CRUISE=")); io.println(cruiseSpeed);
    return;
  }

  // I — identify each motor sequentially
  if (c == 'I') {
    io.println(F("IDENTIFY: pulsing each motor 1s"));
    stopAll();
    for (uint8_t i = 0; i < 4; i++) {
      io.print(F("Motor ")); io.print(i);
      io.print(F(" (D")); io.print(MOTOR_PIN[i]); io.println(F(") ON"));
      setMotor(i, 220);
      delay(1000);
      setMotor(i, 0);
      delay(400);
    }
    io.println(F("IDENTIFY done"));
    return;
  }

  // P<pin>=<val> — raw pin (probe/debug)
  if (c == 'P') {
    int eq = cmd.indexOf('=');
    if (eq > 0) {
      uint8_t pin = cmd.substring(1, eq).toInt();
      uint8_t val = cmd.substring(eq + 1).toInt();
      pinMode(pin, OUTPUT);
      if (val < 2) digitalWrite(pin, val);
      else         analogWrite(pin, val);
      io.print(F("PIN ")); io.print(pin); io.print('='); io.println(val);
    }
    return;
  }

  io.print(F("ERR unknown: ")); io.println(cmd);
}

// ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(9600);
  for (uint8_t i = 0; i < 4; i++) {
    pinMode(MOTOR_PIN[i], OUTPUT);
    analogWrite(MOTOR_PIN[i], 0);
  }
  Serial.println(F("ROBOT READY"));
  Serial.println(F("? F S L R T<l,r> M<n>=<v> V<spd> I"));
}

String buf = "";
void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      handleCmd(buf, Serial);
      buf = "";
    } else {
      buf += c;
    }
  }
}
