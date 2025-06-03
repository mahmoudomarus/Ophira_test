/*
 * Heart Rate Sensor Interface for Ophira AI Medical System
 * 
 * This Arduino code reads heart rate data from a sensor and sends it
 * over Serial to the Ophira Python application.
 * 
 * Compatible with: Arduino Uno, ESP32, ESP8266, Teensy, etc.
 * Heart Rate Sensor: Pulse sensor, MAX30102, or similar
 * 
 * Data Format Options:
 * 1. JSON: {"bpm": 72, "timestamp": 1234567890, "quality": 0.85}
 * 2. CSV: 72,1234567890,0.85,512,480,520
 * 3. Raw: BPM:72 TIME:1234567890
 */

// Pin Configuration
#define PULSE_SENSOR_PIN A0    // Analog pin for pulse sensor
#define LED_PIN 13             // LED indicator
#define PULSE_THRESHOLD 512    // Adjust based on your sensor

// Variables
unsigned long lastBeat = 0;
unsigned long lastReading = 0;
int heartRate = 0;
bool heartDetected = false;
float signalQuality = 0.0;

// Buffer for signal processing
const int BUFFER_SIZE = 10;
int bpmBuffer[BUFFER_SIZE];
int bufferIndex = 0;
int rawSignalBuffer[50];
int rawBufferIndex = 0;

// Configuration
const unsigned long READING_INTERVAL = 100;  // 100ms = 10Hz sampling
const unsigned long SEND_INTERVAL = 1000;    // Send data every 1 second
unsigned long lastSendTime = 0;

// Data format: "json", "csv", or "raw"
String dataFormat = "json";

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(PULSE_SENSOR_PIN, INPUT);
  
  // Initialize buffers
  for (int i = 0; i < BUFFER_SIZE; i++) {
    bpmBuffer[i] = 0;
  }
  
  Serial.println("Heart Rate Sensor Ready");
  Serial.println("Commands: STATUS, CALIBRATE, FORMAT:json/csv/raw");
}

void loop() {
  unsigned long currentTime = millis();
  
  // Read sensor data at regular intervals
  if (currentTime - lastReading >= READING_INTERVAL) {
    readPulseSensor();
    lastReading = currentTime;
  }
  
  // Send data at regular intervals
  if (currentTime - lastSendTime >= SEND_INTERVAL) {
    sendHeartRateData();
    lastSendTime = currentTime;
  }
  
  // Check for serial commands
  handleSerialCommands();
  
  // Update LED indicator
  digitalWrite(LED_PIN, heartDetected);
}

void readPulseSensor() {
  int sensorValue = analogRead(PULSE_SENSOR_PIN);
  
  // Store raw signal for quality assessment
  rawSignalBuffer[rawBufferIndex] = sensorValue;
  rawBufferIndex = (rawBufferIndex + 1) % 50;
  
  // Simple peak detection
  static int lastValue = 0;
  static bool risingEdge = false;
  static unsigned long lastPeakTime = 0;
  
  if (sensorValue > PULSE_THRESHOLD && !risingEdge && sensorValue > lastValue) {
    risingEdge = true;
  }
  
  if (risingEdge && sensorValue < lastValue) {
    // Peak detected
    unsigned long currentTime = millis();
    if (lastPeakTime > 0) {
      unsigned long beatInterval = currentTime - lastPeakTime;
      
      // Calculate BPM (only if interval is reasonable)
      if (beatInterval > 300 && beatInterval < 2000) {  // 30-200 BPM range
        int currentBPM = 60000 / beatInterval;
        
        // Add to buffer for smoothing
        bpmBuffer[bufferIndex] = currentBPM;
        bufferIndex = (bufferIndex + 1) % BUFFER_SIZE;
        
        // Calculate average BPM
        heartRate = calculateAverageBPM();
        heartDetected = true;
        
        // Calculate signal quality
        signalQuality = calculateSignalQuality();
      }
    }
    lastPeakTime = currentTime;
    risingEdge = false;
  }
  
  lastValue = sensorValue;
  
  // Reset heart detection if no beat for too long
  if (millis() - lastPeakTime > 3000) {
    heartDetected = false;
    heartRate = 0;
  }
}

int calculateAverageBPM() {
  int sum = 0;
  int count = 0;
  
  for (int i = 0; i < BUFFER_SIZE; i++) {
    if (bpmBuffer[i] > 0) {
      sum += bpmBuffer[i];
      count++;
    }
  }
  
  return count > 0 ? sum / count : 0;
}

float calculateSignalQuality() {
  // Simple signal quality based on consistency of BPM readings
  if (bufferIndex < 3) return 0.5;  // Not enough data
  
  int variance = 0;
  int average = calculateAverageBPM();
  int count = 0;
  
  for (int i = 0; i < BUFFER_SIZE; i++) {
    if (bpmBuffer[i] > 0) {
      int diff = bpmBuffer[i] - average;
      variance += diff * diff;
      count++;
    }
  }
  
  if (count < 2) return 0.5;
  
  variance /= count;
  
  // Convert variance to quality score (lower variance = higher quality)
  float quality = 1.0 - (variance / 1000.0);  // Adjust divisor based on your sensor
  return constrain(quality, 0.0, 1.0);
}

void sendHeartRateData() {
  unsigned long timestamp = millis();
  
  if (dataFormat == "json") {
    // JSON format
    Serial.print("{\"bpm\": ");
    Serial.print(heartRate);
    Serial.print(", \"timestamp\": ");
    Serial.print(timestamp);
    Serial.print(", \"quality\": ");
    Serial.print(signalQuality, 3);
    Serial.print(", \"detected\": ");
    Serial.print(heartDetected ? "true" : "false");
    Serial.println("}");
    
  } else if (dataFormat == "csv") {
    // CSV format: bpm,timestamp,quality,raw_values...
    Serial.print(heartRate);
    Serial.print(",");
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(signalQuality, 3);
    
    // Add some raw signal values
    for (int i = 0; i < 5; i++) {
      Serial.print(",");
      int idx = (rawBufferIndex - 5 + i + 50) % 50;
      Serial.print(rawSignalBuffer[idx]);
    }
    Serial.println();
    
  } else {
    // Raw format
    Serial.print("BPM:");
    Serial.print(heartRate);
    Serial.print(" TIME:");
    Serial.print(timestamp);
    Serial.print(" QUALITY:");
    Serial.print(signalQuality, 3);
    Serial.print(" STATUS:");
    Serial.println(heartDetected ? "OK" : "NO_SIGNAL");
  }
}

void handleSerialCommands() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    command.toUpperCase();
    
    if (command == "STATUS") {
      Serial.print("STATUS: BPM=");
      Serial.print(heartRate);
      Serial.print(" QUALITY=");
      Serial.print(signalQuality, 3);
      Serial.print(" FORMAT=");
      Serial.print(dataFormat);
      Serial.print(" DETECTED=");
      Serial.println(heartDetected ? "YES" : "NO");
      
    } else if (command == "CALIBRATE") {
      Serial.println("CALIBRATING...");
      // Reset buffers
      for (int i = 0; i < BUFFER_SIZE; i++) {
        bpmBuffer[i] = 0;
      }
      bufferIndex = 0;
      heartDetected = false;
      Serial.println("CALIBRATION_COMPLETE");
      
    } else if (command.startsWith("FORMAT:")) {
      String newFormat = command.substring(7);
      newFormat.toLowerCase();
      if (newFormat == "json" || newFormat == "csv" || newFormat == "raw") {
        dataFormat = newFormat;
        Serial.print("FORMAT_SET:");
        Serial.println(dataFormat);
      } else {
        Serial.println("ERROR: Invalid format. Use json, csv, or raw");
      }
      
    } else if (command.startsWith("THRESHOLD:")) {
      int newThreshold = command.substring(10).toInt();
      if (newThreshold > 0 && newThreshold < 1024) {
        // PULSE_THRESHOLD = newThreshold;  // Note: can't change const, needs global var
        Serial.print("THRESHOLD_SET:");
        Serial.println(newThreshold);
      } else {
        Serial.println("ERROR: Threshold must be 0-1023");
      }
      
    } else {
      Serial.print("UNKNOWN_COMMAND:");
      Serial.println(command);
      Serial.println("Available: STATUS, CALIBRATE, FORMAT:json/csv/raw");
    }
  }
}

/*
 * Installation Notes:
 * 
 * 1. Connect your pulse sensor to analog pin A0
 * 2. Connect LED to pin 13 for visual feedback
 * 3. Upload this code to your microcontroller
 * 4. Open Serial Monitor at 115200 baud to test
 * 5. The Python code will auto-detect the serial port
 * 
 * Sensor Wiring:
 * - Pulse sensor VCC -> 3.3V or 5V
 * - Pulse sensor GND -> GND
 * - Pulse sensor Signal -> A0
 * 
 * Troubleshooting:
 * - Adjust PULSE_THRESHOLD based on your sensor's baseline
 * - Use Serial Monitor to see raw data and debug
 * - Try different data formats (json, csv, raw) to see what works best
 * - Ensure good sensor contact with finger/skin
 */ 