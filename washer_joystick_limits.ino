/* WasherVision: Arduino Uno + CNC Shield V3
   Manual joystick, NC endstops, serial diagnostics.
   NO AUTOMATIC MOTION. No homing or calibrated position.
   Verify shield pinout and axis directions before motor power.
*/
#include <Arduino.h>

const uint8_t STEP_X=2, STEP_Y=3, STEP_Z=4;
const uint8_t DIR_X=5, DIR_Y=6, DIR_Z=7;
const uint8_t ENABLE_PIN=8;
const uint8_t LIMIT_X=9, LIMIT_Y=10, LIMIT_Z=11;
const uint8_t JOY_X=A0, JOY_Y=A1, JOY_BUTTON=A2;

const int JOY_CENTER=512;
const int JOY_LOW=470, JOY_HIGH=550;
const unsigned long MIN_HALF_PERIOD_US=500;
const unsigned long MAX_HALF_PERIOD_US=4000;
const unsigned long BUTTON_DEBOUNCE_MS=40;

bool zMode=false;
bool autoLockout=false;
bool stopLatched=false;
bool joystickArmed=false;
int lastButtonReading=HIGH, stableButton=HIGH;
unsigned long lastButtonChange=0;
unsigned long lastStepAt=0;
int8_t lastAxis=-1;
char inputLine[48];
uint8_t inputLength=0;

bool limitOpen(uint8_t pin) { return digitalRead(pin)==HIGH; }

void reportStatus() {
  Serial.print(F("STATUS mode="));
  Serial.print(autoLockout ? F("AUTO_LOCKOUT") : F("MANUAL"));
  Serial.print(F(" axis="));
  Serial.print(zMode ? F("Z") : F("XY"));
  Serial.print(F(" armed="));
  Serial.print(joystickArmed ? 1 : 0);
  Serial.print(F(" stop="));
  Serial.print(stopLatched ? 1 : 0);
  Serial.print(F(" X="));
  Serial.print(limitOpen(LIMIT_X) ? F("OPEN") : F("CLOSED"));
  Serial.print(F(" Y="));
  Serial.print(limitOpen(LIMIT_Y) ? F("OPEN") : F("CLOSED"));
  Serial.print(F(" Z="));
  Serial.println(limitOpen(LIMIT_Z) ? F("OPEN") : F("CLOSED"));
}

bool joystickNeutral() {
  int x=analogRead(JOY_X), y=analogRead(JOY_Y);
  return x>=JOY_LOW && x<=JOY_HIGH && y>=JOY_LOW && y<=JOY_HIGH;
}

void handleCommand(const char *cmd) {
  if (!strcmp(cmd,"PING")) Serial.println(F("PONG WASHERVISION_BRIDGE_V1"));
  else if (!strcmp(cmd,"STATUS")) reportStatus();
  else if (!strcmp(cmd,"MODE AUTO")) {
    autoLockout=true;
    Serial.println(F("OK AUTO_LOCKOUT NO_MOTION"));
  } else if (!strcmp(cmd,"MODE MANUAL")) {
    if (!joystickNeutral()) Serial.println(F("ERR CENTER_JOYSTICK_FIRST"));
    else { autoLockout=false; Serial.println(F("OK MANUAL")); }
  } else if (!strcmp(cmd,"STOP")) {
    stopLatched=true;
    Serial.println(F("OK STOP_LATCHED"));
  } else if (!strcmp(cmd,"RESETSTOP")) {
    if (!joystickNeutral()) Serial.println(F("ERR CENTER_JOYSTICK_FIRST"));
    else { stopLatched=false; Serial.println(F("OK STOP_CLEARED")); }
  } else Serial.println(F("ERR UNKNOWN_COMMAND"));
}

void readSerial() {
  while (Serial.available()) {
    char c=(char)Serial.read();
    if (c=='\n' || c=='\r') {
      if (inputLength) {
        inputLine[inputLength]='\0';
        handleCommand(inputLine);
        inputLength=0;
      }
    } else if (inputLength < sizeof(inputLine)-1) {
      inputLine[inputLength++]=c;
    } else {
      inputLength=0;
      Serial.println(F("ERR LINE_TOO_LONG"));
    }
  }
}

void readButton() {
  int reading=digitalRead(JOY_BUTTON);
  if (reading!=lastButtonReading) lastButtonChange=millis();
  if (millis()-lastButtonChange >= BUTTON_DEBOUNCE_MS && reading!=stableButton) {
    stableButton=reading;
    if (stableButton==LOW && !autoLockout && !stopLatched && joystickNeutral()) {
      zMode=!zMode;
      Serial.println(zMode ? F("Mode: Z-Axis Active") : F("Mode: X/Y Axis Control Active"));
    }
  }
  lastButtonReading=reading;
}

void setup() {
  pinMode(STEP_X,OUTPUT); pinMode(STEP_Y,OUTPUT); pinMode(STEP_Z,OUTPUT);
  pinMode(DIR_X,OUTPUT); pinMode(DIR_Y,OUTPUT); pinMode(DIR_Z,OUTPUT);
  digitalWrite(STEP_X,LOW); digitalWrite(STEP_Y,LOW); digitalWrite(STEP_Z,LOW);
  pinMode(ENABLE_PIN,OUTPUT);
  digitalWrite(ENABLE_PIN,LOW); // keep holding torque (especially Z)
  pinMode(LIMIT_X,INPUT_PULLUP);
  pinMode(LIMIT_Y,INPUT_PULLUP);
  pinMode(LIMIT_Z,INPUT_PULLUP);
  pinMode(JOY_BUTTON,INPUT_PULLUP);
  Serial.begin(9600);
  Serial.println(F("PONG WASHERVISION_BRIDGE_V1"));
  Serial.println(F("READY MANUAL_LIMITS NO_AUTO_MOTION"));
  reportStatus();
}

void loop() {
  readSerial();
  readButton();
  if (!joystickArmed && joystickNeutral()) joystickArmed=true;
  if (stopLatched || autoLockout || !joystickArmed) {
    lastAxis=-1;
    // Keep drivers enabled to preserve Z holding torque.
    digitalWrite(ENABLE_PIN,LOW);
    return;
  }
  int x=analogRead(JOY_X), y=analogRead(JOY_Y);
  int8_t axis=-1;
  bool towardLimit=false;
  int magnitude=0;
  if (zMode) {
    if (y>JOY_HIGH) { axis=2; towardLimit=true; magnitude=y-JOY_HIGH; }
    else if (y<JOY_LOW) { axis=2; towardLimit=false; magnitude=JOY_LOW-y; }
  } else if (abs(y-JOY_CENTER)>=abs(x-JOY_CENTER)) {
    if (y>JOY_HIGH) { axis=1; towardLimit=true; magnitude=y-JOY_HIGH; }
    else if (y<JOY_LOW) { axis=1; towardLimit=false; magnitude=JOY_LOW-y; }
  } else {
    if (x>JOY_HIGH) { axis=0; towardLimit=true; magnitude=x-JOY_HIGH; }
    else if (x<JOY_LOW) { axis=0; towardLimit=false; magnitude=JOY_LOW-x; }
  }
  if (axis<0) { lastAxis=-1; return; }
  // Original joystick mapping: X right HIGH, Y backward LOW, Z down HIGH.
  // Y backward is LOW on the DIR pin, unlike X right and Z down.
  bool directionHigh=(axis==1) ? !towardLimit : towardLimit;
  uint8_t limitPin=axis==0?LIMIT_X:(axis==1?LIMIT_Y:LIMIT_Z);
  if (towardLimit && limitOpen(limitPin)) { lastAxis=-1; return; }
  digitalWrite(ENABLE_PIN,LOW);
  unsigned long halfPeriod=map(constrain(magnitude,0,473),0,473,MAX_HALF_PERIOD_US,MIN_HALF_PERIOD_US);
  uint8_t stepPin=axis==0?STEP_X:(axis==1?STEP_Y:STEP_Z);
  uint8_t dirPin=axis==0?DIR_X:(axis==1?DIR_Y:DIR_Z);
  unsigned long now=micros();
  if (axis!=lastAxis) { lastAxis=axis; lastStepAt=now; return; }
  if ((unsigned long)(now-lastStepAt) < halfPeriod*2UL) return;
  digitalWrite(dirPin,directionHigh?HIGH:LOW);
  delayMicroseconds(5);
  if (towardLimit && limitOpen(limitPin)) { lastAxis=-1; return; }
  digitalWrite(stepPin,HIGH);
  delayMicroseconds(5);
  digitalWrite(stepPin,LOW);
  lastStepAt=micros();
}
