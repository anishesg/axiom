/*
 * circuit_probe.ino — granular transistor circuit diagnostic
 *
 * VOLTMETER: connect any Arduino analog pin (A0-A5) to a circuit
 *            node with a jumper wire. The sketch reports voltage live.
 *
 * Commands (9600 baud):
 *   V<n>          read voltage on analog pin n (0-5) — 10 samples, averaged
 *   VL<n>         live voltage on pin n — streams every 200ms until any key
 *   P<pin>=<val>  drive digital pin (0-19) to val (0=LOW,1=HIGH,2-255=PWM)
 *   STOP          all outputs LOW
 *   SCAN          drive each of D3,D5,D6,D9 HIGH for 2s and report
 *   CONT <a> <b>  continuity: drive pin a HIGH, read pin b (digital cross-check)
 *   VMAP          full voltage map: for each motor pin, drive it and read A0-A5
 */

void allOff() {
  for (uint8_t p = 2; p <= 19; p++) {
    pinMode(p, OUTPUT);
    analogWrite(p, 0);
    digitalWrite(p, LOW);
  }
}

float readVolts(uint8_t apin) {
  long sum = 0;
  for (int i = 0; i < 16; i++) { sum += analogRead(apin); delayMicroseconds(500); }
  return (sum / 16.0f) * (5.0f / 1023.0f);
}

void printVolts(uint8_t apin, float v) {
  Serial.print(F("A")); Serial.print(apin);
  Serial.print(F(" = ")); Serial.print(v, 3);
  Serial.print(F("V  ("));
  if      (v > 4.5)  Serial.print(F("~5V HIGH"));
  else if (v > 3.0)  Serial.print(F("pulled HIGH"));
  else if (v > 1.5)  Serial.print(F("mid/floating"));
  else if (v > 0.4)  Serial.print(F("low/partial"));
  else               Serial.print(F("~0V LOW/GND"));
  Serial.println(F(")"));
}

// ── command handlers ─────────────────────────────────────────────────

void cmdV(uint8_t apin) {
  pinMode(A0 + apin, INPUT);
  float v = readVolts(apin);
  printVolts(apin, v);
}

void cmdVLive(uint8_t apin, Stream &io) {
  io.print(F("Live A")); io.print(apin); io.println(F(" — send any char to stop"));
  pinMode(A0 + apin, INPUT);
  while (!io.available()) {
    float v = readVolts(apin);
    io.print(F("  ")); printVolts(apin, v);
    delay(200);
  }
  while (io.available()) io.read();
}

void cmdPin(uint8_t pin, uint8_t val) {
  pinMode(pin, OUTPUT);
  if (val == 0)      digitalWrite(pin, LOW);
  else if (val == 1) digitalWrite(pin, HIGH);
  else               analogWrite(pin, val);
  Serial.print(F("SET D")); Serial.print(pin);
  Serial.print(F(" = ")); Serial.println(val);
}

void cmdScan() {
  const uint8_t MPINS[] = {3, 5, 6, 9};
  Serial.println(F("SCAN: driving each motor pin HIGH for 2s"));
  Serial.println(F("Watch for wheel movement. Voltage on A0 reported."));
  for (uint8_t i = 0; i < 4; i++) {
    uint8_t p = MPINS[i];
    allOff();
    pinMode(p, OUTPUT);
    analogWrite(p, 255);
    Serial.print(F("\n>>> D")); Serial.print(p); Serial.println(F(" = PWM 255"));

    // Read A0-A3 to see if any node changes
    for (uint8_t a = 0; a < 4; a++) {
      pinMode(A0 + a, INPUT);
      float v = readVolts(a);
      Serial.print(F("  ")); printVolts(a, v);
    }
    delay(2000);
    allOff();
    delay(500);
  }
  Serial.println(F("SCAN done"));
}

void cmdCont(uint8_t drvPin, uint8_t readPin) {
  // Drive drvPin HIGH, read readPin as digital — tests if they share a net
  allOff();
  pinMode(drvPin, OUTPUT);
  digitalWrite(drvPin, HIGH);
  delayMicroseconds(500);
  pinMode(readPin, INPUT);
  int v = digitalRead(readPin);
  Serial.print(F("Drive D")); Serial.print(drvPin);
  Serial.print(F(" HIGH, read D")); Serial.print(readPin);
  Serial.print(F(" = ")); Serial.println(v ? F("HIGH (connected)") : F("LOW (not connected)"));
  allOff();
}

void cmdVMap() {
  const uint8_t MPINS[] = {3, 5, 6, 9};
  Serial.println(F("VMAP: for each motor pin, measure A0-A5 baseline vs driven"));
  Serial.println(F("Connect jumper from A0 to different circuit nodes to probe."));
  Serial.println();

  // Baseline (all off)
  allOff();
  delay(100);
  float base[6];
  for (uint8_t a = 0; a < 6; a++) base[a] = readVolts(a);
  Serial.println(F("=== BASELINE (all pins LOW) ==="));
  for (uint8_t a = 0; a < 6; a++) { Serial.print(F("  ")); printVolts(a, base[a]); }

  for (uint8_t i = 0; i < 4; i++) {
    uint8_t p = MPINS[i];
    allOff(); delay(50);
    pinMode(p, OUTPUT);
    analogWrite(p, 255);
    delay(100);

    Serial.print(F("\n=== D")); Serial.print(p); Serial.println(F(" = 255 ==="));
    for (uint8_t a = 0; a < 6; a++) {
      float v = readVolts(a);
      float delta = v - base[a];
      Serial.print(F("  ")); printVolts(a, v);
      if (abs(delta) > 0.1) {
        Serial.print(F("    *** CHANGED by ")); Serial.print(delta, 3); Serial.println(F("V ***"));
      }
    }
    delay(500);
  }
  allOff();
  Serial.println(F("\nVMAP done"));
}

// ── command parser ────────────────────────────────────────────────────

void handleCmd(const String &cmd) {
  if (cmd == "?")    { Serial.println(F("READY")); return; }
  if (cmd == "STOP") { allOff(); Serial.println(F("STOP")); return; }
  if (cmd == "SCAN") { cmdScan(); return; }
  if (cmd == "VMAP") { cmdVMap(); return; }

  if (cmd.startsWith(F("VL"))) {
    uint8_t a = cmd.substring(2).toInt();
    cmdVLive(a, Serial); return;
  }
  if (cmd.startsWith(F("V")) && cmd.length() == 2) {
    uint8_t a = cmd.charAt(1) - '0';
    if (a < 6) { cmdV(a); return; }
  }
  if (cmd.startsWith(F("P"))) {
    int eq = cmd.indexOf('=');
    if (eq > 1) {
      uint8_t pin = cmd.substring(1, eq).toInt();
      uint8_t val = cmd.substring(eq + 1).toInt();
      cmdPin(pin, val); return;
    }
  }
  if (cmd.startsWith(F("CONT "))) {
    int sp = cmd.indexOf(' ', 5);
    uint8_t a = cmd.substring(5, sp < 0 ? cmd.length() : sp).toInt();
    uint8_t b = sp < 0 ? 0 : cmd.substring(sp + 1).toInt();
    cmdCont(a, b); return;
  }

  Serial.print(F("ERR: ")); Serial.println(cmd);
  Serial.println(F("Cmds: V0-V5  VL0-VL5  P<pin>=<val>  SCAN  VMAP  CONT a b  STOP"));
}

// ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(9600);
  allOff();
  Serial.println(F("=== CIRCUIT PROBE READY ==="));
  Serial.println(F("SCAN=motor sweep | VMAP=voltage map | V0=read A0 | VL0=live A0"));
  Serial.println(F("P3=255=drive D3 full | CONT 3 7=continuity D3->D7 | STOP"));
}

String buf = "";
void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') { buf.trim(); if (buf.length()) handleCmd(buf); buf = ""; }
    else buf += c;
  }
}
