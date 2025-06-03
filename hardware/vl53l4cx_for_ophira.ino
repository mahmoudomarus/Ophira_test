/**
 * VL53L4CX ToF Sensor for Ophira AI Medical System
 * 
 * This Arduino sketch interfaces with the Adafruit VL53L4CX Time of Flight
 * sensor and outputs distance measurements in a format compatible with
 * the Ophira AI Python sensor classes.
 * 
 * Hardware Connections:
 * - VCC -> 3.3V (NOT 5V!)
 * - GND -> GND
 * - SCL -> A5 (or SCL pin)
 * - SDA -> A4 (or SDA pin)
 * - XSHUT -> D2 (optional, for reset control)
 * - GPIO1 -> D3 (optional, for interrupt)
 * 
 * Required Libraries:
 * - Adafruit_VL53L4CX (install via Library Manager)
 * - Wire (built-in)
 */

#include <Arduino.h>
#include <Wire.h>
#include <vl53l4cx_class.h>

// Pin definitions
#define XSHUT_PIN 2      // Shutdown pin (optional)
#define GPIO1_PIN 3      // Interrupt pin (optional)
#define LED_PIN LED_BUILTIN

// I2C and sensor setup
#define DEV_I2C Wire
VL53L4CX sensor(&DEV_I2C, XSHUT_PIN);

// Measurement tracking
unsigned long measurementCount = 0;
unsigned long lastMeasurementTime = 0;
const unsigned long MEASUREMENT_INTERVAL = 100; // ms between measurements

// Status tracking
bool sensorInitialized = false;
bool ledState = false;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial) {
    delay(10); // Wait for serial port to connect
  }
  
  // Initialize pins
  pinMode(LED_PIN, OUTPUT);
  pinMode(XSHUT_PIN, OUTPUT);
  pinMode(GPIO1_PIN, INPUT);
  
  // Initialize I2C
  DEV_I2C.begin();
  
  Serial.println("Ophira AI - VL53L4CX ToF Sensor Starting...");
  Serial.println("Initializing sensor...");
  
  // Initialize sensor
  if (initializeSensor()) {
    Serial.println("Sensor initialized successfully!");
    sensorInitialized = true;
    
    // Start continuous measurements
    if (startMeasurements()) {
      Serial.println("Measurements started!");
      Serial.println("Format: VL53L4CX Satellite: Count=X, #Objs=Y status=Z, D=Amm, Signal=B Mcps, Ambient=C Mcps");
    } else {
      Serial.println("Failed to start measurements!");
      sensorInitialized = false;
    }
  } else {
    Serial.println("Failed to initialize sensor!");
    Serial.println("Check wiring and power supply (3.3V only!)");
  }
  
  // Initial LED state
  digitalWrite(LED_PIN, LOW);
}

bool initializeSensor() {
  // Reset sensor
  digitalWrite(XSHUT_PIN, LOW);
  delay(10);
  digitalWrite(XSHUT_PIN, HIGH);
  delay(10);
  
  // Initialize sensor
  if (sensor.begin() != 0) {
    Serial.println("Error: Could not initialize sensor!");
    return false;
  }
  
  // Turn off sensor
  if (sensor.VL53L4CX_Off() != 0) {
    Serial.println("Error: Could not turn off sensor!");
    return false;
  }
  
  // Initialize sensor with default address
  if (sensor.InitSensor(0x29) != 0) {
    Serial.println("Error: Could not init sensor!");
    return false;
  }
  
  return true;
}

bool startMeasurements() {
  // Start measurements
  if (sensor.VL53L4CX_StartMeasurement() != 0) {
    Serial.println("Error: Could not start measurements!");
    return false;
  }
  
  return true;
}

void loop() {
  if (!sensorInitialized) {
    // Blink LED to indicate error
    digitalWrite(LED_PIN, millis() % 500 < 250);
    delay(100);
    return;
  }
  
  // Check if it's time for a new measurement
  unsigned long currentTime = millis();
  if (currentTime - lastMeasurementTime >= MEASUREMENT_INTERVAL) {
    takeMeasurement();
    lastMeasurementTime = currentTime;
  }
  
  delay(10); // Small delay to prevent overwhelming the system
}

void takeMeasurement() {
  VL53L4CX_MultiRangingData_t multiRangingData;
  VL53L4CX_MultiRangingData_t *pMultiRangingData = &multiRangingData;
  uint8_t newDataReady = 0;
  int status;
  
  // Check if new data is ready
  do {
    status = sensor.VL53L4CX_GetMeasurementDataReady(&newDataReady);
    if (status != 0) {
      Serial.println("Error: Could not check if data is ready!");
      return;
    }
  } while (!newDataReady);
  
  // Turn on LED during measurement
  digitalWrite(LED_PIN, HIGH);
  ledState = true;
  
  // Get measurement data
  if (status == 0 && newDataReady != 0) {
    status = sensor.VL53L4CX_GetMultiRangingData(pMultiRangingData);
    
    if (status == 0) {
      measurementCount++;
      int numberOfObjects = pMultiRangingData->NumberOfObjectsFound;
      
      // Output measurement data in the format expected by Python code
      Serial.print("VL53L4CX Satellite: Count=");
      Serial.print(measurementCount);
      Serial.print(", #Objs=");
      Serial.print(numberOfObjects);
      Serial.print(" ");
      
      // Output data for each detected object
      for (int j = 0; j < numberOfObjects; j++) {
        if (j != 0) {
          Serial.print(" | "); // Separator for multiple objects
        }
        
        int rangeStatus = pMultiRangingData->RangeData[j].RangeStatus;
        int distanceMm = pMultiRangingData->RangeData[j].RangeMilliMeter;
        float signalRate = (float)pMultiRangingData->RangeData[j].SignalRateRtnMegaCps / 65536.0;
        float ambientRate = (float)pMultiRangingData->RangeData[j].AmbientRateRtnMegaCps / 65536.0;
        
        Serial.print("status=");
        Serial.print(rangeStatus);
        Serial.print(", D=");
        Serial.print(distanceMm);
        Serial.print("mm, Signal=");
        Serial.print(signalRate, 3); // 3 decimal places
        Serial.print(" Mcps, Ambient=");
        Serial.print(ambientRate, 3); // 3 decimal places
        Serial.print(" Mcps");
      }
      Serial.println(""); // End line
      
      // Clear interrupt and start next measurement
      status = sensor.VL53L4CX_ClearInterruptAndStartMeasurement();
      if (status != 0) {
        Serial.println("Error: Could not clear interrupt and restart!");
      }
    } else {
      Serial.println("Error: Could not get measurement data!");
    }
  }
  
  // Turn off LED
  digitalWrite(LED_PIN, LOW);
  ledState = false;
}

// Helper function to print sensor info (called once at startup)
void printSensorInfo() {
  Serial.println("=== VL53L4CX Sensor Information ===");
  Serial.println("Manufacturer: STMicroelectronics");
  Serial.println("Range: 1mm to 6000mm");
  Serial.println("Accuracy: ±3mm (typical)");
  Serial.println("FOV: 18° x 27°");
  Serial.println("Update Rate: Up to 60Hz");
  Serial.println("===================================");
}

// Error handling function
void handleError(const char* errorMsg) {
  Serial.print("ERROR: ");
  Serial.println(errorMsg);
  
  // Blink LED rapidly to indicate error
  for (int i = 0; i < 10; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }
}

// Diagnostic function (can be called via serial commands if needed)
void runDiagnostics() {
  Serial.println("=== Running Diagnostics ===");
  
  // Check I2C communication
  Wire.beginTransmission(0x29); // Default VL53L4CX address
  int error = Wire.endTransmission();
  
  if (error == 0) {
    Serial.println("✓ I2C communication OK");
  } else {
    Serial.print("✗ I2C error code: ");
    Serial.println(error);
  }
  
  // Check sensor status
  if (sensorInitialized) {
    Serial.println("✓ Sensor initialized");
  } else {
    Serial.println("✗ Sensor not initialized");
  }
  
  Serial.println("=== Diagnostics Complete ===");
} 