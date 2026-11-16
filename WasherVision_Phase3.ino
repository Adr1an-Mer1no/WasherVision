/* WasherVision Phase 3 - experimental bounded RELATIVE X/Y motion.
   Arduino Uno + CNC Shield V3, CUSTOM firmware (NOT GRBL).
   Based on Phase 2 joystick, NC limits, serial counters.
   CAUTION: No homing, no soft travel boundaries, no encoder feedback.
   AUTO commands are restricted to 10 mm per axis per command, not a
   substitute for verified end-of-travel protection or emergency stop.
   Confirm DIR signs and switch behavior before energizing motors.
*/
#include <Arduino.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

const byte SX=2, DX=5, SY=3, DY=6, SZ=4, DZ=7, EN=8;
const byte LX=9, LY=10, LZ=11;
const byte JX=A0, JY=A1, JB=A2;
const float X_STEPS_PER_MM=5.0f, Y_STEPS_PER_MM=5.0f, Z_STEPS_PER_MM=25.0f;
const float MAX_RELATIVE_MM=10.0f;
const unsigned long STEP_INTERVAL_US=5000UL;
const unsigned long AUTO_TIMEOUT_MS=15000UL;

long stepsX=0,stepsY=0,stepsZ=0;
bool zMode=false,stopped=true,autoMode=false;
bool buttonPrev=HIGH;
unsigned long buttonChanged=0,lastManualStepAt=0,lastAutoStepAt=0,autoStart=0;
char command[64]; byte commandLen=0;
long remainingX=0,remainingY=0;
int autoSignX=0,autoSignY=0;
bool moving=false;

bool openLimit(byte pin) { return digitalRead(pin)==HIGH; } // NC: HIGH means pressed OR wire broken
bool joystickCentered() {
  int x=analogRead(JX),y=analogRead(JY);
  return x>=470 && x<=550 && y>=470 && y<=550;
}
void cancelMove() { moving=false; remainingX=remainingY=0; }
void report() {
  Serial.print(F("STATUS mode=")); Serial.print(autoMode?F("AUTO"):F("MANUAL"));
  Serial.print(F(" axis=")); Serial.print(zMode?F("Z"):F("XY"));
  Serial.print(F(" stop=")); Serial.print(stopped?1:0);
  Serial.print(F(" moving=")); Serial.print(moving?1:0);
  Serial.print(F(" X=")); Serial.print(openLimit(LX)?F("OPEN"):F("CLOSED"));
  Serial.print(F(" Y=")); Serial.print(openLimit(LY)?F("OPEN"):F("CLOSED"));
  Serial.print(F(" Z=")); Serial.print(openLimit(LZ)?F("OPEN"):F("CLOSED"));
  Serial.print(F(" stepsX=")); Serial.print(stepsX);
  Serial.print(F(" stepsY=")); Serial.print(stepsY);
  Serial.print(F(" stepsZ=")); Serial.println(stepsZ);
}
void emitPulse(byte stepPin,byte dirPin,bool dirHigh,long &counter,int sign) {
  digitalWrite(dirPin,dirHigh?HIGH:LOW);
  delayMicroseconds(5);
  digitalWrite(stepPin,HIGH); delayMicroseconds(5); digitalWrite(stepPin,LOW);
  counter+=sign;
}
// Strict syntax: MOVE X<signed-mm> Y<signed-mm>, either axis optional.
// Examples: MOVE X5, MOVE Y-2, MOVE X5 Y-3.
bool parseMove(const char *line,float &x,float &y) {
  if (strncmp(line,"MOVE ",5)!=0) return false;
  x=0; y=0; bool seenX=false,seenY=false;
  const char *p=line+5;
  while (*p) {
    while (*p==' ') ++p;
    if (!*p) break;
    char axis=*p++;
    if ((axis!='X' && axis!='Y') || (axis=='X'&&seenX) || (axis=='Y'&&seenY)) return false;
    char *end;
    double val=strtod(p,&end);
    if (end==p || !isfinite(val) || fabs(val)>MAX_RELATIVE_MM) return false;
    if (*end!=' ' && *end!='\0') return false;
    if (axis=='X') { x=(float)val; seenX=true; } else { y=(float)val; seenY=true; }
    p=end;
  }
  return seenX||seenY;
}
void handleCommand(const char *c) {
  if (!strcmp(c,"PING")) Serial.println(F("PONG WASHERVISION_BRIDGE_V1"));
  else if (!strcmp(c,"STATUS") || !strcmp(c,"COUNTS")) report();
  else if (!strcmp(c,"STOP")) {
    stopped=true; cancelMove(); Serial.println(F("OK STOP"));
  } else if (!strcmp(c,"RESETSTOP")) {
    if (moving || !joystickCentered()) Serial.println(F("ERR CENTER JOYSTICK"));
    else { stopped=false; Serial.println(F("OK RESETSTOP")); }
  } else if (!strcmp(c,"MODE AUTO")) {
    if (!stopped || moving || !joystickCentered()) Serial.println(F("ERR STOP AND CENTER JOYSTICK FIRST"));
    else { autoMode=true; Serial.println(F("OK AUTO - RELATIVE X/Y ONLY")); }
  } else if (!strcmp(c,"MODE MANUAL")) {
    cancelMove(); stopped=true; autoMode=false;
    Serial.println(F("OK MANUAL STOPPED - RESETSTOP TO JOG"));
  } else if (!strcmp(c,"ZEROCOUNTS")) {
    if (stopped && !moving && joystickCentered()) {
      stepsX=stepsY=stepsZ=0;
      Serial.println(F("OK RELATIVE COUNTS ZEROED - NOT HOMED"));
    } else Serial.println(F("ERR STOP AND CENTER JOYSTICK FIRST"));
  } else if (!strncmp(c,"MOVE ",5)) {
    float mmX,mmY;
    if (!parseMove(c,mmX,mmY)) { Serial.println(F("ERR MOVE SYNTAX OR RANGE - MAX 10MM PER AXIS")); return; }
    if (!autoMode || stopped || moving || !joystickCentered()) {
      Serial.println(F("ERR REQUIRE AUTO MODE, RESETSTOP, IDLE, CENTERED JOYSTICK")); return;
    }
    long x=lroundf(mmX*X_STEPS_PER_MM),y=lroundf(mmY*Y_STEPS_PER_MM);
    if ((x<0 && openLimit(LX)) || (y<0 && openLimit(LY))) {
      Serial.println(F("ERR LIMIT OPEN IN REQUESTED DIRECTION")); return;
    }
    remainingX=labs(x); remainingY=labs(y);
    autoSignX=(x>0)?1:((x<0)?-1:0);
    autoSignY=(y>0)?1:((y<0)?-1:0);
    moving=(remainingX||remainingY);
    autoStart=millis(); lastAutoStepAt=micros();
    Serial.print(F("OK MOVE START Xsteps=")); Serial.print(x);
    Serial.print(F(" Ysteps=")); Serial.println(y);
    if (!moving) Serial.println(F("OK MOVE DONE"));
  } else Serial.println(F("ERR UNKNOWN COMMAND"));
}
void readSerial() {
  while (Serial.available()) {
    char c=(char)Serial.read();
    if (c=='\n'||c=='\r') {
      if (commandLen) { command[commandLen]=0; handleCommand(command); commandLen=0; }
    } else if (commandLen<sizeof(command)-1) command[commandLen++]=c;
    else { commandLen=0; stopped=true; cancelMove(); Serial.println(F("ERR COMMAND TOO LONG - STOPPED")); }
  }
}
void autoTick() {
  if (!moving || stopped || !autoMode) return;
  if (!joystickCentered()) { stopped=true; cancelMove(); Serial.println(F("ERR JOYSTICK MOVED - STOPPED")); return; }
  if (millis()-autoStart>AUTO_TIMEOUT_MS) { stopped=true; cancelMove(); Serial.println(F("ERR AUTO TIMEOUT - STOPPED")); return; }
  if ((remainingX && autoSignX<0 && openLimit(LX)) || (remainingY && autoSignY<0 && openLimit(LY))) {
    stopped=true; cancelMove(); Serial.println(F("ERR LIMIT TRIGGERED - STOPPED")); return;
  }
  unsigned long now=micros();
  if ((unsigned long)(now-lastAutoStepAt)<STEP_INTERVAL_US) return;
  lastAutoStepAt=now;
  // Sequential X then Y, one pulse per tick; conservative initial testing.
  if (remainingX) {
    emitPulse(SX,DX,autoSignX>0,stepsX,autoSignX); --remainingX;
  } else if (remainingY) {
    emitPulse(SY,DY,autoSignY>0,stepsY,autoSignY); --remainingY;
  }
  if (!remainingX && !remainingY) { moving=false; Serial.println(F("OK MOVE DONE")); report(); }
}
void manualTick() {
  if (stopped || autoMode) return;
  int x=analogRead(JX),y=analogRead(JY);
  int dx=abs(x-512),dy=abs(y-512);
  if ((unsigned long)(micros()-lastManualStepAt)<STEP_INTERVAL_US) return;
  if (zMode) {
    if (y>550 && !openLimit(LZ)) { emitPulse(SZ,DZ,HIGH,stepsZ,-1); lastManualStepAt=micros(); }
    else if (y<470) { emitPulse(SZ,DZ,LOW,stepsZ,+1); lastManualStepAt=micros(); }
  } else if (dy>=dx) {
    if (y>550 && !openLimit(LY)) { emitPulse(SY,DY,LOW,stepsY,-1); lastManualStepAt=micros(); }
    else if (y<470) { emitPulse(SY,DY,HIGH,stepsY,+1); lastManualStepAt=micros(); }
  } else {
    if (x>550) { emitPulse(SX,DX,HIGH,stepsX,+1); lastManualStepAt=micros(); }
    else if (x<470 && !openLimit(LX)) { emitPulse(SX,DX,LOW,stepsX,-1); lastManualStepAt=micros(); }
  }
}
void setup() {
  pinMode(SX,OUTPUT);pinMode(DX,OUTPUT);
  pinMode(SY,OUTPUT);pinMode(DY,OUTPUT);
  pinMode(SZ,OUTPUT);pinMode(DZ,OUTPUT);
  pinMode(EN,OUTPUT);digitalWrite(EN,HIGH);
  pinMode(LX,INPUT_PULLUP);pinMode(LY,INPUT_PULLUP);pinMode(LZ,INPUT_PULLUP);
  pinMode(JB,INPUT_PULLUP);
  Serial.begin(9600);
  digitalWrite(EN,LOW);
  Serial.println(F("WASHERVISION PHASE3 READY - STOPPED; NO HOMING"));
  report();
}
void loop() {
  readSerial();
  bool button=digitalRead(JB);
  if (button!=buttonPrev && millis()-buttonChanged>40) {
    buttonChanged=millis();
    if (button==LOW && !autoMode && !moving) { zMode=!zMode; report(); }
    buttonPrev=button;
  }
  autoTick();
  manualTick();
}
