#include <Servo.h>
Servo servo1; // Axial Servo
Servo servo2; // Pivot Servo
int input;
String inByte;
int servo_pos1;
int servo_pos2;
int wait = 3000; // 3 seconds 
int wait2 = 1000; // 1 second

void setup() {
  // put your setup code here, to run once:
  servo1.attach(9);
  servo2.attach(10);
  servo_pos1 = 90;
  servo_pos2 = 90;
  servo1.write(servo_pos1);
  servo2.write(servo_pos2);
  Serial.begin(9600);

}

void loop() {
  // put your main code here, to run repeatedly:

}

void serialEvent() {
  if(Serial.available()) {
    inByte = Serial.readStringUntil('\n');
    input = inByte.toInt();
    
    if (input == 0) {
      sort_to_paper();
      Serial.println("Recieved command...sorting to Paper");
    } else if (input == 1) {
      sort_to_plastic();
      reposition();
      Serial.println("Sorting to Plastic");      
    } else if (input == 2) {
      sort_to_glass();
      Serial.println("Sorting to Glass");      
    } else {
      sort_to_trash();
      Serial.println("Sending to trash");      
    }   
  }
}

void test_serial() {
  Serial.println("Command Recieved");
}


// Reorients carriage to default position after servo control 
void reposition() {
  // First returns carriage to normal postion
  if(servo_pos2 != 90) {
    servo_pos2 = 90;
    servo2.write(servo_pos2);
    // If carriage was rotated acially, returns to normal position
    if(servo_pos1 != 90) {
      servo_pos1 = 90;
      servo1.write(servo_pos1);
    }
  }
}

void sort_to_trash() {
  // Axial servo does not move
  // Pivots to left
  servo_pos2 = 20;
  servo2.write(servo_pos2);
  delay(wait);
  reposition();
}

void sort_to_paper() {
  // Axial rotates right
  // Pivots left
  servo_pos1 = 180;
  servo_pos2 = 20;

  servo1.write(servo_pos1);
  delay(wait2);
  servo2.write(servo_pos2);
  delay(wait);
  reposition();
}

void sort_to_plastic() {
  // Axial servo does not move
  // Pivots to right
  servo_pos2 = 160;
  servo2.write(servo_pos2);
  delay(wait);
  reposition();
};

void sort_to_glass() {
  // Axial rotates right
  // Pivots left
  servo_pos1 = 180;
  servo_pos2 = 160;

  servo1.write(servo_pos1);
  delay(wait2);
  servo2.write(servo_pos2);
  delay(wait);
  reposition();
}
