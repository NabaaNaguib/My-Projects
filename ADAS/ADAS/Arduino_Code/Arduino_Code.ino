#include "MapFloat.h"

void setup() {
  pinMode(10, OUTPUT);
  pinMode(9, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  byte buffer[8];
  int steering_angle;
  float object_distance;


  // Wait for data to be available on the serial port
  while (Serial.available() < 8) {}

  // Read the byte string into the buffer
  Serial.readBytes(buffer, 8);

  // Copy the bytes into the appropriate data types
  memcpy(&steering_angle, &buffer[0], sizeof(steering_angle));
  memcpy(&object_distance, &buffer[4], sizeof(object_distance));
  steering_angle = map(steering_angle,0,90,0,255);
  object_distance = mapFloat(object_distance,0.0,2.0,0.0,255.0);
  analogWrite(10, steering_angle);
  analogWrite(9, object_distance);

}
  // Print the values to the serial monitor
  //Serial.print("Steering Angle: ");
  //Serial.println(steering_angle);
  //Serial.print("Object Distance: ");
  //Serial.println(object_distance);

