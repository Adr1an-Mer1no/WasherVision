/* WasherVision Phase 4 - 1/16 microstep configuration (UNVERIFIED: bench test first).
   Custom Arduino Uno + CNC Shield V3 firmware, NOT GRBL.
   Based on user's working Phase 3 sketch. TEST WITH MOTORS UNPOWERED FIRST.
   NC switches: LOW=healthy/not pressed, HIGH=pressed OR disconnected.
   X home LEFT (-X), Y home BACKWARD (-Y). Z automatic motion disabled. Bottom Z limit inhibits downward joystick travel.
   Requires external accessible motor-power emergency cutoff. No encoders.
*/
#include <Arduino.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

const byte SX=2, DX=5, SY=3, DY=6, SZ=4, DZ=7, EN=8;
const byte LX=9, LY=10, LZ=11;
const byte JX=A0, JY=A1, JB=A2;
const float X_STEPS_PER_MM=5.0f, Y_STEPS_PER_MM=5.0f, Z_STEPS_PER_MM=25.0f; // measured effective pulses/mm; hardware microstep mode unverified
// Measured total travels: X 200.025 mm; Y 241.3 mm.
// HOME establishes 0 roughly 5 mm past the switch release point.
// Conservative maxima leave ~5 mm at the far ends, subject to verification.
const long X_MAX_STEPS=950L; // 190 mm
const long Y_MAX_STEPS=1155L; // 231 mm
const long MAX_COMMAND_STEPS=50L; // 10 mm per axis, including GOTO
// Same pulse intervals as the previously smooth 1/16 test, but 5 pulses/mm.
// CAUTION: short moves can still jerk/slip; first test without washers.
const unsigned long XY_START_INTERVAL_US=3000UL;
const unsigned long XY_CRUISE_INTERVAL_US=1800UL;
const unsigned long XY_MANUAL_INTERVAL_US=2500UL;
const unsigned long Z_MANUAL_INTERVAL_US=5000UL; // unchanged Z behavior
const long XY_RAMP_STEPS=8L; // short symmetric ramp for 5mm (25-pulse) moves
const unsigned long HOME_INTERVAL_US=10000UL; // conservative homing; unchanged effective distance
const unsigned long HOME_TIMEOUT_MS=25000UL;
const unsigned long AUTO_TIMEOUT_MS=20000UL;
const long HOME_X_SEEK_MAX=1100L, HOME_Y_SEEK_MAX=1350L;
const long HOME_BACKOFF_STEPS=25L; // 5 mm AFTER switch release
const long HOME_RELEASE_MAX=100L; // at most 20 mm to release switch

long stepsX=0,stepsY=0,stepsZ=0;
long posX=0,posY=0;
bool homedX=false,homedY=false;
bool zMode=false,stopped=true,autoMode=false;
bool buttonPrev=HIGH;
unsigned long buttonChanged=0,lastManualStepAt=0,lastAutoStepAt=0,autoStart=0;
char command[64]; byte commandLen=0;
long remainingX=0,remainingY=0;
long moveTotalSteps=0,moveCompletedSteps=0;
int autoSignX=0,autoSignY=0;
bool moving=false;

enum HomeState { HOME_IDLE, HOME_SEEK, HOME_RELEASE, HOME_CLEAR };
HomeState homeState=HOME_IDLE;
byte homeAxis=0; // 0=X, 1=Y
bool homeBoth=false;
long homePulses=0,clearPulses=0;
unsigned long homeStart=0,lastHomeStep=0;

bool openLimit(byte pin) { return digitalRead(pin)==HIGH; }
byte homeLimit() { return homeAxis==0?LX:LY; }
long homeSeekMax() { return homeAxis==0?HOME_X_SEEK_MAX:HOME_Y_SEEK_MAX; }
bool joystickCentered() {
  int x=analogRead(JX), y=analogRead(JY);
  return x>=470 && x<=550 && y>=470 && y<=550;
}
void cancelMove() { moving=false; remainingX=remainingY=0; moveTotalSteps=moveCompletedSteps=0; }
void cancelHome() { homeState=HOME_IDLE; homeBoth=false; }
void invalidateHome() { homedX=false; homedY=false; }
void failStop(const __FlashStringHelper *msg) {
  stopped=true; cancelMove(); cancelHome(); invalidateHome(); Serial.println(msg);
}
void report() {
  Serial.print(F("STATUS mode=")); Serial.print(autoMode?F("AUTO"):F("MANUAL"));
  Serial.print(F(" axis=")); Serial.print(zMode?F("Z"):F("XY"));
  Serial.print(F(" stop=")); Serial.print(stopped?1:0);
  Serial.print(F(" moving=")); Serial.print(moving?1:0);
  Serial.print(F(" homing=")); Serial.print(homeState!=HOME_IDLE?1:0);
  Serial.print(F(" X=")); Serial.print(openLimit(LX)?F("OPEN"):F("CLOSED"));
  Serial.print(F(" Y=")); Serial.print(openLimit(LY)?F("OPEN"):F("CLOSED"));
  Serial.print(F(" Z=")); Serial.print(openLimit(LZ)?F("OPEN"):F("CLOSED"));
  Serial.print(F(" homeX=")); Serial.print(homedX?1:0);
  Serial.print(F(" homeY=")); Serial.print(homedY?1:0);
  Serial.print(F(" posXsteps=")); Serial.print(posX);
  Serial.print(F(" posYsteps=")); Serial.print(posY);
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
void pulseAxis(byte axis,int sign) {
  if (axis==0) { emitPulse(SX,DX,sign>0,stepsX,sign); posX+=sign; }
  else { emitPulse(SY,DY,sign>0,stepsY,sign); posY+=sign; }
}
// Parse either "MOVE X<mm> Y<mm>" (relative) or "GOTO X<mm> Y<mm>" (absolute).
// Values are finite, and duplicate/unknown axes are rejected.
bool parseXY(const char *line,const char *prefix,float &x,float &y,bool &seenX,bool &seenY) {
  size_t n=strlen(prefix);
  if (strncmp(line,prefix,n)!=0) return false;
  x=0;y=0;seenX=false;seenY=false;
  const char *p=line+n;
  while (*p) {
    while (*p==' ') ++p;
    if (!*p) break;
    char axis=*p++;
    if ((axis!='X' && axis!='Y') || (axis=='X'&&seenX) || (axis=='Y'&&seenY)) return false;
    char *end;
    double val=strtod(p,&end);
    if (end==p || !isfinite(val) || fabs(val)>1000.0) return false;
    if (*end!=' ' && *end!='\0') return false;
    if (axis=='X') {x=(float)val;seenX=true;} else {y=(float)val;seenY=true;}
    p=end;
  }
  return seenX||seenY;
}
void startHome(byte axis,bool both) {
  if (openLimit(axis==0?LX:LY)) { Serial.println(F("ERR HOME SWITCH OPEN - CHECK WIRING OR MOVE AWAY MANUALLY")); return; }
  invalidateHome();
  homeAxis=axis;homeBoth=both;homeState=HOME_SEEK;
  homePulses=clearPulses=0;homeStart=millis();lastHomeStep=micros();
  Serial.println(axis==0?F("OK HOME X SEEK LEFT"):F("OK HOME Y SEEK BACKWARD"));
}
void startMove(long dx,long dy) {
  if (labs(dx)>MAX_COMMAND_STEPS || labs(dy)>MAX_COMMAND_STEPS) {
    Serial.println(F("ERR MAX 10MM PER AXIS PER COMMAND")); return;
  }
  long targetX=posX+dx,targetY=posY+dy;
  if (targetX<0 || targetX>X_MAX_STEPS || targetY<0 || targetY>Y_MAX_STEPS) {
    Serial.println(F("ERR SOFTWARE TRAVEL BOUNDARY")); return;
  }
  if ((dx<0&&openLimit(LX)) || (dy<0&&openLimit(LY))) {
    Serial.println(F("ERR LIMIT OPEN IN REQUESTED DIRECTION")); return;
  }
  remainingX=labs(dx);remainingY=labs(dy);
  autoSignX=(dx>0)?1:((dx<0)?-1:0);
  autoSignY=(dy>0)?1:((dy<0)?-1:0);
  moving=remainingX||remainingY;
  moveTotalSteps=remainingX+remainingY;moveCompletedSteps=0;
  autoStart=millis();lastAutoStepAt=micros();
  Serial.print(F("OK MOVE START Xsteps="));Serial.print(dx);
  Serial.print(F(" Ysteps="));Serial.println(dy);
  if (!moving) Serial.println(F("OK MOVE DONE"));
}
void handleCommand(const char *c) {
  if (!strcmp(c,"PING")) Serial.println(F("PONG WASHERVISION_BRIDGE_V1"));
  else if (!strcmp(c,"STATUS") || !strcmp(c,"COUNTS") || !strcmp(c,"POS")) report();
  else if (!strcmp(c,"STOP")) {
    stopped=true;cancelMove();cancelHome();Serial.println(F("OK STOP"));
  } else if (!strcmp(c,"RESETSTOP")) {
    if (moving || homeState!=HOME_IDLE || !joystickCentered()) Serial.println(F("ERR CENTER JOYSTICK AND WAIT"));
    else {stopped=false;Serial.println(F("OK RESETSTOP"));}
  } else if (!strcmp(c,"MODE AUTO")) {
    if (!stopped || moving || homeState!=HOME_IDLE || !joystickCentered()) Serial.println(F("ERR STOP AND CENTER JOYSTICK FIRST"));
    else {autoMode=true;Serial.println(F("OK AUTO - HOME BEFORE MOVE"));}
  } else if (!strcmp(c,"MODE MANUAL")) {
    cancelMove();cancelHome();stopped=true;autoMode=false;invalidateHome();
    Serial.println(F("OK MANUAL STOPPED - RESETSTOP TO JOG; HOME INVALID"));
  } else if (!strcmp(c,"ZEROCOUNTS")) {
    if (stopped && !moving && homeState==HOME_IDLE && joystickCentered()) {
      stepsX=stepsY=stepsZ=0;invalidateHome();
      Serial.println(F("OK COUNTS ZEROED - HOME INVALID"));
    } else Serial.println(F("ERR STOP AND CENTER JOYSTICK FIRST"));
  } else if (!strcmp(c,"HOME X") || !strcmp(c,"HOME Y") || !strcmp(c,"HOME XY")) {
    if (!autoMode || stopped || moving || homeState!=HOME_IDLE || !joystickCentered()) {
      Serial.println(F("ERR REQUIRE AUTO, RESETSTOP, IDLE, CENTERED JOYSTICK")); return;
    }
    if (openLimit(LX) || openLimit(LY)) {
      Serial.println(F("ERR X/Y SWITCH OPEN - CHECK WIRING AND CLEAR SWITCHES MANUALLY"));return;
    }
    startHome(!strcmp(c,"HOME Y")?1:0,!strcmp(c,"HOME XY"));
  } else if (!strncmp(c,"MOVE ",5) || !strncmp(c,"GOTO ",5)) {
    bool absolute=c[0]=='G';float x,y;bool sx,sy;
    if (!parseXY(c,absolute?"GOTO ":"MOVE ",x,y,sx,sy)) {
      Serial.println(F("ERR XY SYNTAX"));return;
    }
    if (!autoMode || stopped || moving || homeState!=HOME_IDLE || !joystickCentered() || !homedX || !homedY) {
      Serial.println(F("ERR REQUIRE BOTH AXES HOMED, AUTO, RESETSTOP, IDLE"));return;
    }
    long dx=absolute?(sx?lroundf(x*X_STEPS_PER_MM)-posX:0):(sx?lroundf(x*X_STEPS_PER_MM):0);
    long dy=absolute?(sy?lroundf(y*Y_STEPS_PER_MM)-posY:0):(sy?lroundf(y*Y_STEPS_PER_MM):0);
    startMove(dx,dy);
  } else Serial.println(F("ERR UNKNOWN COMMAND"));
}
void readSerial() {
  while (Serial.available()) {
    char c=(char)Serial.read();
    if (c=='\n'||c=='\r') {
      if (commandLen) {command[commandLen]=0;handleCommand(command);commandLen=0;}
    } else if (commandLen<sizeof(command)-1) command[commandLen++]=c;
    else {commandLen=0;failStop(F("ERR COMMAND TOO LONG - STOPPED"));}
  }
}
void homeTick() {
  if (homeState==HOME_IDLE || stopped || !autoMode) return;
  if (!joystickCentered()) {failStop(F("ERR JOYSTICK MOVED DURING HOME - STOPPED"));return;}
  if (millis()-homeStart>HOME_TIMEOUT_MS) {failStop(F("ERR HOME TIMEOUT - STOPPED"));return;}
  if ((unsigned long)(micros()-lastHomeStep)<HOME_INTERVAL_US) return;
  lastHomeStep=micros();
  if (homeState==HOME_SEEK) {
    if (openLimit(homeLimit())) {
      homeState=HOME_RELEASE;homePulses=0;
      Serial.println(F("OK HOME SWITCH FOUND - RELEASING"));return;
    }
    if (homePulses>=homeSeekMax()) {failStop(F("ERR HOME SEEK DISTANCE EXCEEDED - STOPPED"));return;}
    pulseAxis(homeAxis,-1);++homePulses;
  } else if (homeState==HOME_RELEASE) {
    if (!openLimit(homeLimit())) {
      homeState=HOME_CLEAR;clearPulses=0;
      Serial.println(F("OK HOME SWITCH RELEASED - MOVING 5MM CLEAR"));return;
    }
    if (homePulses>=HOME_RELEASE_MAX) {failStop(F("ERR HOME SWITCH DID NOT RELEASE - STOPPED"));return;}
    pulseAxis(homeAxis,+1);++homePulses;
  } else if (homeState==HOME_CLEAR) {
    if (openLimit(homeLimit())) {failStop(F("ERR HOME SWITCH REOPENED - STOPPED"));return;}
    if (clearPulses<HOME_BACKOFF_STEPS) {pulseAxis(homeAxis,+1);++clearPulses;return;}
    if (homeAxis==0) {posX=0;homedX=true;Serial.println(F("OK HOME X DONE"));}
    else {posY=0;homedY=true;Serial.println(F("OK HOME Y DONE"));}
    if (homeBoth && homeAxis==0) {
      homeAxis=1;homeState=HOME_SEEK;homePulses=0;clearPulses=0;
      homeStart=millis();lastHomeStep=micros();
      Serial.println(F("OK HOME Y SEEK BACKWARD"));
    } else {cancelHome();report();}
  }
}
unsigned long xyStepInterval() {
  // Symmetric linear ramp, based on completed and remaining steps.
  long left=moveTotalSteps-moveCompletedSteps;
  long phase=moveCompletedSteps;
  if (left<phase) phase=left;
  if (phase>XY_RAMP_STEPS) phase=XY_RAMP_STEPS;
  return XY_START_INTERVAL_US-
      ((XY_START_INTERVAL_US-XY_CRUISE_INTERVAL_US)*(unsigned long)phase)/XY_RAMP_STEPS;
}
void autoTick() {
  if (!moving || stopped || !autoMode) return;
  if (!joystickCentered()) {failStop(F("ERR JOYSTICK MOVED - STOPPED"));return;}
  if (millis()-autoStart>AUTO_TIMEOUT_MS) {failStop(F("ERR AUTO TIMEOUT - STOPPED"));return;}
  if ((remainingX && autoSignX<0 && openLimit(LX)) || (remainingY && autoSignY<0 && openLimit(LY))) {
    failStop(F("ERR LIMIT TRIGGERED - STOPPED"));return;
  }
  if ((unsigned long)(micros()-lastAutoStepAt)<xyStepInterval()) return;
  lastAutoStepAt=micros();
  if (remainingX) {pulseAxis(0,autoSignX);--remainingX;}
  else if (remainingY) {pulseAxis(1,autoSignY);--remainingY;}
  ++moveCompletedSteps;
  if (!remainingX&&!remainingY) {moving=false;Serial.println(F("OK MOVE DONE"));report();}
}
void manualTick() {
  if (stopped || autoMode) return;
  int x=analogRead(JX),y=analogRead(JY);
  int dx=abs(x-512),dy=abs(y-512);
  if ((unsigned long)(micros()-lastManualStepAt)<(zMode?Z_MANUAL_INTERVAL_US:XY_MANUAL_INTERVAL_US)) return;
  if (zMode) {
    // Observed on this machine: y>550 raises Z (away from bottom limit),
    // y<470 lowers Z (toward bottom limit).  NC switch HIGH = pressed/open wire.
    // Only inhibit downward travel when the bottom limit is OPEN.
    // This is directional interlock, NOT a verified collision-safe stopping distance.
    if (y>550) {emitPulse(SZ,DZ,HIGH,stepsZ,-1);lastManualStepAt=micros();}
    else if (y<470 && !openLimit(LZ)) {emitPulse(SZ,DZ,LOW,stepsZ,+1);lastManualStepAt=micros();}
  } else if (dy>=dx) {
    if (y>550 && !openLimit(LY)) {pulseAxis(1,-1);lastManualStepAt=micros();}
    else if (y<470) {pulseAxis(1,+1);lastManualStepAt=micros();}
  } else {
    if (x>550) {pulseAxis(0,+1);lastManualStepAt=micros();}
    else if (x<470 && !openLimit(LX)) {pulseAxis(0,-1);lastManualStepAt=micros();}
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
  Serial.println(F("WASHERVISION PHASE7 Z LOWER LIMIT FIX READY - STOPPED; NOT HOMED"));
  report();
}
void loop() {
  readSerial();
  bool button=digitalRead(JB);
  if (button!=buttonPrev && millis()-buttonChanged>40) {
    buttonChanged=millis();
    if (button==LOW && !autoMode && !moving && homeState==HOME_IDLE) {zMode=!zMode;report();}
    buttonPrev=button;
  }
  homeTick();autoTick();manualTick();
}
