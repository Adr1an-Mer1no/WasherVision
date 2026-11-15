// Arduino Uno + CNC Shield V3: manual joystick plus serial diagnostic interface.
// NO SERIAL MOTION COMMANDS. AUTO mode disables joystick movement until explicitly restored.
// Physical emergency stop, travel limits, and homing are required before automated motion.
const byte stepPins[3] = {2,3,4};
const byte dirPins[3] = {5,6,7};
const byte enPin=8, joyX=A0, joyY=A1, joyButton=A2;
bool zMode=false, autoMode=false, stopped=false;
bool lastButton=HIGH;
unsigned long lastDebounce=0;
char buffer[48]; byte count=0;
void drivers(bool enabled){ digitalWrite(enPin, enabled ? LOW : HIGH); }
void replyStatus(){
 Serial.print(F("STATUS mode=")); Serial.print(stopped ? F("STOPPED") : (autoMode ? F("AUTO_LOCKED") : F("MANUAL")));
 Serial.print(F(" axis=")); Serial.println(zMode ? F("Z") : F("XY"));
}
void command(char* s){
 if(!strcmp(s,"PING")){ Serial.println(F("PONG WASHERVISION_BRIDGE_V1")); }
 else if(!strcmp(s,"STATUS")){ replyStatus(); }
 else if(!strcmp(s,"MODE AUTO")){
   autoMode=true; drivers(false); Serial.println(F("OK AUTO_LOCKED NO_MOTION"));
 } else if(!strcmp(s,"MODE MANUAL")){
   if(stopped){Serial.println(F("ERR STOPPED RESET_REQUIRED"));}
   else {autoMode=false; drivers(true); Serial.println(F("OK MANUAL"));}
 } else if(!strcmp(s,"STOP")){
   stopped=true; drivers(false); Serial.println(F("OK STOPPED"));
 } else if(!strcmp(s,"RESET")){
   // Only reset when a human has checked clearance and external E-stop is released.
   stopped=false; autoMode=false; drivers(true); Serial.println(F("OK MANUAL RESET"));
 } else {Serial.println(F("ERR UNKNOWN_OR_MOTION_DISABLED"));}
}
void readSerial(){
 while(Serial.available()){
   char c=(char)Serial.read();
   if(c=='\n' || c=='\r'){
     if(count){buffer[count]='\0'; command(buffer); count=0;}
   } else if(count<sizeof(buffer)-1){buffer[count++]=c;}
   else {count=0; Serial.println(F("ERR LINE_TOO_LONG"));}
 }
}
void pulse(byte axis, bool direction, unsigned int delayUs){
 digitalWrite(dirPins[axis],direction?HIGH:LOW);
 digitalWrite(stepPins[axis],HIGH); delayMicroseconds(delayUs);
 digitalWrite(stepPins[axis],LOW); delayMicroseconds(delayUs);
}
void setup(){
 for(byte i=0;i<3;i++){pinMode(stepPins[i],OUTPUT);pinMode(dirPins[i],OUTPUT);}
 pinMode(enPin,OUTPUT); pinMode(joyButton,INPUT_PULLUP);
 drivers(false); Serial.begin(9600); Serial.println(F("WASHERVISION_BRIDGE_V1 READY"));
 // Initial mode is manual; ensure machine is clear before powering motors.
 drivers(true);
}
void loop(){
 readSerial();
 if(stopped || autoMode) return;
 bool button=digitalRead(joyButton);
 if(button==LOW && lastButton==HIGH && millis()-lastDebounce>250){
   zMode=!zMode; lastDebounce=millis(); replyStatus();
 }
 lastButton=button;
 int x=analogRead(joyX), y=analogRead(joyY);
 if(zMode){
   if(y>550) pulse(2,true,(unsigned int)map(y,550,1023,4000,500));
   else if(y<470) pulse(2,false,(unsigned int)map(y,470,0,4000,500));
 }else{
   if(abs(y-512)>=abs(x-512)){
     if(y>550) pulse(1,false,(unsigned int)map(y,550,1023,4000,500));
     else if(y<470) pulse(1,true,(unsigned int)map(y,470,0,4000,500));
   }else{
     if(x>550) pulse(0,true,(unsigned int)map(x,550,1023,4000,500));
     else if(x<470) pulse(0,false,(unsigned int)map(x,470,0,4000,500));
   }
 }
}
