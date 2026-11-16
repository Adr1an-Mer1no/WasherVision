/* WasherVision Phase 2: joystick + NC limits + relative step counters.
   Arduino Uno / CNC Shield V3. Not GRBL. No automatic movement.
   X left limit D9; Y backward D10; Z bottom D11.
   Joystick RIGHT -> X right; UP -> Y backward; UP in Z mode -> Z down.
   WARNING: Counters are relative, not homed machine coordinates.
*/
const byte SX=2, DX=5, SY=3, DY=6, SZ=4, DZ=7, EN=8;
const byte LX=9, LY=10, LZ=11;
const byte JX=A0, JY=A1, JB=A2;
long stepsX=0, stepsY=0, stepsZ=0;
bool zMode=false, stopped=false, autoLock=false;
bool buttonPrev=HIGH;
unsigned long buttonChanged=0, lastStepAt=0;
char command[48]; byte commandLen=0;

bool openLimit(byte pin) { return digitalRead(pin)==HIGH; } // NC: HIGH = pressed or wire broken
void report() {
  Serial.print(F("STATUS mode=")); Serial.print(autoLock?F("AUTO_LOCK"):F("MANUAL"));
  Serial.print(F(" axis=")); Serial.print(zMode?F("Z"):F("XY"));
  Serial.print(F(" stop=")); Serial.print(stopped?1:0);
  Serial.print(F(" X=")); Serial.print(openLimit(LX)?F("OPEN"):F("CLOSED"));
  Serial.print(F(" Y=")); Serial.print(openLimit(LY)?F("OPEN"):F("CLOSED"));
  Serial.print(F(" Z=")); Serial.print(openLimit(LZ)?F("OPEN"):F("CLOSED"));
  Serial.print(F(" stepsX=")); Serial.print(stepsX);
  Serial.print(F(" stepsY=")); Serial.print(stepsY);
  Serial.print(F(" stepsZ=")); Serial.println(stepsZ);
}
void handleCommand(const char* c) {
  if (!strcmp(c,"PING")) Serial.println(F("PONG WASHERVISION_BRIDGE_V1"));
  else if (!strcmp(c,"STATUS") || !strcmp(c,"COUNTS")) report();
  else if (!strcmp(c,"STOP")) { stopped=true; Serial.println(F("OK STOP")); }
  else if (!strcmp(c,"RESETSTOP")) { stopped=false; Serial.println(F("OK RESETSTOP")); }
  else if (!strcmp(c,"MODE AUTO")) { autoLock=true; Serial.println(F("OK AUTO_LOCK - NO MOTION")); }
  else if (!strcmp(c,"MODE MANUAL")) { autoLock=false; Serial.println(F("OK MANUAL")); }
  else if (!strcmp(c,"ZEROCOUNTS")) {
    if (stopped && analogRead(JX)>=470 && analogRead(JX)<=550 && analogRead(JY)>=470 && analogRead(JY)<=550) {
      stepsX=stepsY=stepsZ=0; Serial.println(F("OK RELATIVE COUNTS ZEROED - NOT HOMED"));
    } else Serial.println(F("ERR STOP AND CENTER JOYSTICK FIRST"));
  }
  else Serial.println(F("ERR UNKNOWN COMMAND"));
}
void readSerial() {
  while (Serial.available()) {
    char c=Serial.read();
    if (c=='\n' || c=='\r') {
      if (commandLen) { command[commandLen]=0; handleCommand(command); commandLen=0; }
    } else if (commandLen<sizeof(command)-1) command[commandLen++]=c;
    else { commandLen=0; Serial.println(F("ERR COMMAND TOO LONG")); }
  }
}
void pulse(byte stepPin, byte dirPin, bool dirHigh, long &counter, int sign, unsigned long periodUs) {
  // Re-check switch immediately before each pulse in the caller.
  unsigned long now=micros();
  if ((unsigned long)(now-lastStepAt)<periodUs) return;
  lastStepAt=now;
  digitalWrite(dirPin,dirHigh?HIGH:LOW);
  delayMicroseconds(5);
  digitalWrite(stepPin,HIGH); delayMicroseconds(5); digitalWrite(stepPin,LOW);
  counter+=sign; // commanded pulses, not encoder-verified motion
}
void setup() {
  pinMode(SX,OUTPUT); pinMode(DX,OUTPUT);
  pinMode(SY,OUTPUT); pinMode(DY,OUTPUT);
  pinMode(SZ,OUTPUT); pinMode(DZ,OUTPUT);
  pinMode(EN,OUTPUT); digitalWrite(EN,HIGH); // disabled while configuring
  pinMode(LX,INPUT_PULLUP); pinMode(LY,INPUT_PULLUP); pinMode(LZ,INPUT_PULLUP);
  pinMode(JB,INPUT_PULLUP);
  Serial.begin(9600);
  digitalWrite(EN,LOW); // drivers hold position; disconnect motor supply during first test
  Serial.println(F("WASHERVISION PHASE2 READY - MANUAL ONLY"));
  report();
}
void loop() {
  readSerial();
  bool button=digitalRead(JB);
  if (button!=buttonPrev && millis()-buttonChanged>40) {
    buttonChanged=millis();
    if (button==LOW) { zMode=!zMode; report(); }
    buttonPrev=button;
  }
  if (stopped || autoLock) return;
  int x=analogRead(JX), y=analogRead(JY);
  int dx=abs(x-512), dy=abs(y-512);
  if (zMode) {
    if (y>550 && !openLimit(LZ)) pulse(SZ,DZ,HIGH,stepsZ,-1,2500); // DOWN blocked at bottom
    else if (y<470) pulse(SZ,DZ,LOW,stepsZ,+1,2500); // UP away
  } else if (dy>=dx) {
    if (y>550 && !openLimit(LY)) pulse(SY,DY,LOW,stepsY,-1,2500); // BACK blocked
    else if (y<470) pulse(SY,DY,HIGH,stepsY,+1,2500); // FORWARD away
  } else {
    if (x>550) pulse(SX,DX,HIGH,stepsX,+1,2500); // RIGHT away
    else if (x<470 && !openLimit(LX)) pulse(SX,DX,LOW,stepsX,-1,2500); // LEFT blocked
  }
}
