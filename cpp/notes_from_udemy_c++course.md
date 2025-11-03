## **SECTION 2-8: FUNDAMENTALS**

### **General Concept:**
Programming is giving instructions to a computer. You need to store data (variables), perform calculations (operators), and the computer needs to understand what type of data you're working with (data types).

### **Detailed Breakdown:**

#### **1. Data Types** (Why they matter in robotics)
```cpp
int motorSpeed = 100;        // Whole numbers (-2,147,483,648 to 2,147,483,647)
float sensorValue = 3.14;    // Decimal numbers (6-7 digits precision)
double distance = 3.14159;   // More precise decimals (15-16 digits)
char command = 'F';          // Single character (Forward, Backward, etc.)
bool isObstacle = true;      // true or false (sensor detected something?)
```

**Robotics application:**
- `int` → Motor speeds, encoder counts, loop counters
- `float/double` → Sensor readings (temperature, distance, angles)
- `bool` → Sensor states (is button pressed? is line detected?)
- `char` → Command codes

#### **2. Variables** (Storage containers)
```cpp
// Declaration and initialization
int wheelDiameter = 10;  // cm
float batteryVoltage = 12.5;  // volts

// You can change values
batteryVoltage = 11.8;  // battery draining

// Naming rules:
int motor_speed;  // Good: descriptive
int ms;          // Bad: unclear
int 2fast;       // Error: can't start with number
```

**Think of variables as labeled boxes** where you store data temporarily.

#### **3. Operators & Expressions**

**Arithmetic Operators:**
```cpp
int a = 10, b = 3;

int sum = a + b;        // 13
int difference = a - b;  // 7
int product = a * b;     // 30
int quotient = a / b;    // 3 (integer division!)
int remainder = a % b;   // 1 (modulo - very useful!)

float precise = 10.0 / 3.0;  // 3.333... (float division)
```

**Robotics example:**
```cpp
// Calculate motor PWM value (0-255) from percentage (0-100)
int percentage = 75;
int pwmValue = (percentage * 255) / 100;  // 191

// Check if encoder count is even or odd
int encoderCount = 1523;
if (encoderCount % 2 == 0) {
    // even
} else {
    // odd
}
```

**Compound Assignment:**
```cpp
int speed = 100;
speed += 10;   // speed = speed + 10  → 110
speed -= 5;    // speed = speed - 5   → 105
speed *= 2;    // speed = speed * 2   → 210
speed /= 3;    // speed = speed / 3   → 70
```

**Increment/Decrement:**
```cpp
int count = 5;
count++;  // count = 6 (post-increment)
++count;  // count = 7 (pre-increment)
count--;  // count = 6 (decrement)

// Difference:
int a = 5;
int b = a++;  // b = 5, a = 6 (use then increment)
int c = ++a;  // c = 7, a = 7 (increment then use)
```

**Robotics use:**
```cpp
for (int i = 0; i < 10; i++) {  // Loop counter
    readSensor();
}
```

**Operator Precedence:**
```cpp
int result = 5 + 3 * 2;  // 11, not 16! (* before +)
int result = (5 + 3) * 2;  // 16 (parentheses first)

// In robotics calculations:
float velocity = distance / time + offset;  // Wrong!
float velocity = (distance / time) + offset;  // Clear!
```

---

## **SECTION 9: CONDITIONAL STATEMENTS**

### **General Concept:**
Your robot needs to make decisions. "If the sensor detects an obstacle, then stop. Otherwise, keep moving."

### **Detailed Breakdown:**

#### **1. if Statement**
```cpp
int distance = 15;  // cm

if (distance < 20) {
    stopMotors();
    // This code runs only if condition is true
}
```

#### **2. if-else**
```cpp
int batteryLevel = 15;  // percentage

if (batteryLevel < 20) {
    goToChargingStation();
} else {
    continueTask();
}
```

#### **3. Relational Operators**
```cpp
int a = 10, b = 5;

a == b  // Equal to? false
a != b  // Not equal? true
a > b   // Greater than? true
a < b   // Less than? false
a >= b  // Greater or equal? true
a <= b  // Less or equal? false
```

#### **4. Logical Operators** (Combining conditions)
```cpp
bool frontClear = true;
bool batteryOk = true;

// AND (&&) - Both must be true
if (frontClear && batteryOk) {
    moveForward();
}

// OR (||) - At least one must be true
if (obstacleLeft || obstacleRight) {
    stop();
}

// NOT (!) - Reverse the condition
if (!emergencyStop) {
    operate();
}
```

**Robotics example:**
```cpp
int leftSensor = 850;   // Line sensor value
int rightSensor = 200;
int threshold = 500;

if (leftSensor > threshold && rightSensor > threshold) {
    // Both sensors on black line
    moveForward();
} 
else if (leftSensor > threshold && rightSensor < threshold) {
    // Left on black, right on white → Turn right
    turnRight();
}
else if (leftSensor < threshold && rightSensor > threshold) {
    // Left on white, right on black → Turn left
    turnLeft();
}
else {
    // Both on white - lost the line!
    stop();
}
```

#### **5. Nested if**
```cpp
if (sensorDetected) {
    if (objectType == "obstacle") {
        if (distance < 10) {
            emergencyStop();
        } else {
            slowDown();
        }
    }
}
```

#### **6. switch-case** (Multiple specific values)
```cpp
char command = 'F';

switch(command) {
    case 'F':
        moveForward();
        break;
    case 'B':
        moveBackward();
        break;
    case 'L':
        turnLeft();
        break;
    case 'R':
        turnRight();
        break;
    case 'S':
        stop();
        break;
    default:
        // Invalid command
        errorBeep();
}
```

---

## **SECTION 10: LOOPS**

### **General Concept:**
Repeat actions. Robots do repetitive tasks: read sensors constantly, update motor speeds, check battery level every second.

### **Detailed Breakdown:**

#### **1. for Loop** (Known number of repetitions)
```cpp
// Syntax: for (initialization; condition; increment)
for (int i = 0; i < 10; i++) {
    readSensor();
    // Runs exactly 10 times
}

// Count backwards
for (int i = 10; i > 0; i--) {
    Serial.println(i);  // Countdown
}

// Step by different amounts
for (int speed = 0; speed <= 255; speed += 10) {
    setMotorSpeed(speed);  // Gradually increase speed
    delay(100);
}
```

**Robotics example:**
```cpp
// Sample sensor 100 times and average
int total = 0;
for (int i = 0; i < 100; i++) {
    total += readDistanceSensor();
}
int average = total / 100;
```

#### **2. while Loop** (Condition-based)
```cpp
while (condition) {
    // Runs as long as condition is true
}

// Example:
while (obstacleDetected()) {
    moveBackward();
    delay(100);
}
// Once path is clear, exit loop
```

**Robotics example:**
```cpp
// Wait until button is pressed
while (!buttonPressed()) {
    // Do nothing, just wait
    delay(10);
}
startProgram();

// Keep robot on line
while (batteryLevel > 10) {
    followLine();
    checkBattery();
}
```

#### **3. do-while Loop** (Execute at least once)
```cpp
do {
    action();
} while (condition);

// Example:
do {
    tryConnectWiFi();
} while (!connected && attempts < 5);
```

#### **4. Loop Control**
```cpp
// break - Exit loop immediately
for (int i = 0; i < 100; i++) {
    if (emergencyButton) {
        break;  // Stop loop right now!
    }
    moveForward();
}

// continue - Skip to next iteration
for (int i = 0; i < 10; i++) {
    if (i == 5) {
        continue;  // Skip when i = 5
    }
    processSensor(i);
}
```

**Real robotics pattern:**
```cpp
// Main control loop (runs forever in embedded systems)
while (true) {
    readSensors();
    makeDecision();
    updateMotors();
    delay(20);  // 50Hz update rate
}
```

---

## **SECTION 11: ARRAYS**

### **General Concept:**
Store multiple values of the same type in one variable. Like a row of mailboxes, each with a number.

### **Detailed Breakdown:**

#### **1. Array Declaration**
```cpp
// Declare array of 5 integers
int sensorReadings[5];

// Declare and initialize
int motorSpeeds[4] = {100, 120, 100, 120};  // 4 motors

// Auto-size (compiler counts elements)
float temperatures[] = {25.5, 26.0, 25.8};  // Size = 3
```

#### **2. Accessing Elements** (Index starts at 0!)
```cpp
int data[5] = {10, 20, 30, 40, 50};

int first = data[0];   // 10
int second = data[1];  // 20
int last = data[4];    // 50

// Modify values
data[2] = 99;  // Now: {10, 20, 99, 40, 50}
```

**Common mistake:**
```cpp
int arr[5];
int x = arr[5];  // ERROR! Valid indices: 0-4, not 5
```

#### **3. Looping Through Arrays**
```cpp
int sensors[4] = {100, 200, 150, 180};

// Traditional for loop
for (int i = 0; i < 4; i++) {
    Serial.println(sensors[i]);
}

// For-each loop (C++11) - cleaner!
for (int value : sensors) {
    Serial.println(value);
}
```

**Robotics example:**
```cpp
// Store last 10 distance readings
float distanceHistory[10];
int index = 0;

// In your main loop:
distanceHistory[index] = readDistance();
index = (index + 1) % 10;  // Wrap around: 0,1,2...9,0,1...

// Calculate moving average
float sum = 0;
for (float d : distanceHistory) {
    sum += d;
}
float average = sum / 10;
```

#### **4. Finding Max/Min**
```cpp
int readings[5] = {45, 78, 23, 90, 56};

int maxValue = readings[0];
for (int i = 1; i < 5; i++) {
    if (readings[i] > maxValue) {
        maxValue = readings[i];
    }
}
// maxValue = 90
```

#### **5. 2D Arrays** (Grid/Matrix)
```cpp
// 3 rows, 4 columns
int grid[3][4] = {
    {1, 2, 3, 4},
    {5, 6, 7, 8},
    {9, 10, 11, 12}
};

int value = grid[1][2];  // Row 1, Col 2 = 7

// Loop through 2D array
for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 4; col++) {
        Serial.print(grid[row][col]);
    }
}
```

**Robotics use - Map representation:**
```cpp
// Simple occupancy grid (0=free, 1=obstacle)
int map[10][10] = {0};  // Initialize all to 0
map[5][5] = 1;  // Obstacle at position (5,5)
```

---

## **SECTION 12: POINTERS** ⚠️ **MOST CRITICAL!**

### **General Concept:**
A pointer is a variable that stores a memory address. Think of it as a "reference" or "link" to where data actually lives in memory.

### **Why Pointers Matter in Robotics:**
- Efficient memory usage (embedded systems have limited RAM)
- Pass large data structures without copying
- Dynamic memory allocation
- Direct hardware access (registers, GPIO pins)
- Essential for ROS2 and real-time systems

### **Detailed Breakdown:**

#### **1. Basic Pointer Syntax**
```cpp
int age = 25;
int* ptr = &age;  // ptr stores the ADDRESS of age

// & = "address-of" operator
// * = "dereference" operator (access value at address)

cout << age;    // 25 (value)
cout << &age;   // 0x7ffe5367e044 (address)
cout << ptr;    // 0x7ffe5367e044 (same address)
cout << *ptr;   // 25 (value at that address)
```

**Visual representation:**
```
Memory:
Address    Variable    Value
0x1000     age         25
0x2000     ptr         0x1000 (points to age)
```

#### **2. Pointer Operations**
```cpp
int x = 10;
int* p = &x;

*p = 20;  // Changes x to 20 (through pointer)

int y = *p;  // y = 20 (read value through pointer)
```

#### **3. Dynamic Memory Allocation** (Heap)
```cpp
// Stack allocation (automatic, limited size)
int arr[1000];  // Fixed size

// Heap allocation (manual, larger available)
int* dynamicArr = new int[1000];  // Allocate 1000 integers

// Use it
dynamicArr[0] = 42;

// MUST free when done!
delete[] dynamicArr;
```

**Robotics example:**
```cpp
// Sensor data buffer - size unknown at compile time
int sensorCount = getSensorCount();  // Determined at runtime
float* sensorData = new float[sensorCount];

// Read all sensors
for (int i = 0; i < sensorCount; i++) {
    sensorData[i] = readSensor(i);
}

// Process data...

// Clean up
delete[] sensorData;
```

#### **4. Pointers and Arrays**
```cpp
int arr[5] = {10, 20, 30, 40, 50};
int* ptr = arr;  // Array name IS a pointer to first element

cout << *ptr;      // 10 (first element)
cout << *(ptr+1);  // 20 (second element)
cout << *(ptr+4);  // 50 (fifth element)

// These are equivalent:
arr[2] == *(arr + 2) == *(ptr + 2)
```

#### **5. Pointers and Functions** (Pass by reference)
```cpp
// Without pointer (pass by value) - COPY is made
void tryToChange(int x) {
    x = 100;  // Only changes the copy!
}

int value = 10;
tryToChange(value);
// value is still 10

// With pointer (pass by reference) - ACTUAL variable modified
void actuallyChange(int* x) {
    *x = 100;  // Changes original!
}

int value = 10;
actuallyChange(&value);
// value is now 100
```

**Robotics example:**
```cpp
// Update motor speeds through pointers
void updateMotors(int* leftSpeed, int* rightSpeed) {
    *leftSpeed += 10;
    *rightSpeed += 10;
}

int left = 100, right = 100;
updateMotors(&left, &right);
// left = 110, right = 110
```

#### **6. NULL/nullptr Pointers**
```cpp
int* ptr = nullptr;  // Points to nothing (safe)

if (ptr != nullptr) {
    *ptr = 10;  // Safe to use
} else {
    // Handle null case
}
```

#### **7. Pointer Arithmetic**
```cpp
int arr[5] = {10, 20, 30, 40, 50};
int* ptr = arr;

ptr++;     // Move to next element (now points to 20)
ptr += 2;  // Move forward 2 elements (now points to 40)
ptr--;     // Move back (now points to 30)

// Difference between pointers
int* start = arr;
int* end = arr + 5;
int size = end - start;  // 5 elements
```

---

## **SECTION 14: FUNCTIONS**

### **General Concept:**
Functions are reusable blocks of code. Instead of writing the same code multiple times, write it once in a function and call it whenever needed.

### **Detailed Breakdown:**

#### **1. Basic Function Structure**
```cpp
// returnType functionName(parameters) {
//     code
//     return value;
// }

int add(int a, int b) {
    int sum = a + b;
    return sum;
}

// Calling the function
int result = add(5, 3);  // result = 8
```

#### **2. Void Functions** (No return value)
```cpp
void moveForward() {
    digitalWrite(leftMotorPin, HIGH);
    digitalWrite(rightMotorPin, HIGH);
    // No return statement
}

moveForward();  // Just execute, no value returned
```

#### **3. Function Parameters**

**Pass by Value (Copy):**
```cpp
void process(int x) {
    x = x * 2;  // Only modifies the copy
}

int num = 10;
process(num);
// num is still 10
```

**Pass by Reference (Actual variable):**
```cpp
void process(int& x) {  // & means reference
    x = x * 2;  // Modifies original
}

int num = 10;
process(num);
// num is now 20!
```

**Pass by Pointer:**
```cpp
void process(int* x) {
    *x = *x * 2;
}

int num = 10;
process(&num);
// num is now 20
```

**Robotics example:**
```cpp
// Update PID values by reference (efficient, no copy)
void updatePID(float& p, float& i, float& d, float error) {
    p += error * 0.1;
    i += error * 0.01;
    d = error * 0.05;
}

float proportional = 0, integral = 0, derivative = 0;
updatePID(proportional, integral, derivative, sensorError);
```

#### **4. Function Overloading** (Same name, different parameters)
```cpp
int add(int a, int b) {
    return a + b;
}

float add(float a, float b) {
    return a + b;
}

int add(int a, int b, int c) {
    return a + b + c;
}

// Compiler chooses correct version:
add(5, 3);        // Calls int version
add(5.5f, 3.2f);  // Calls float version
add(1, 2, 3);     // Calls 3-parameter version
```

#### **5. Default Arguments**
```cpp
void setMotorSpeed(int speed, int acceleration = 10) {
    // acceleration defaults to 10 if not provided
}

setMotorSpeed(100);      // Uses default acceleration = 10
setMotorSpeed(100, 20);  // Overrides with acceleration = 20
```

#### **6. Function Templates** (Generic programming)
```cpp
template <typename T>
T getMax(T a, T b) {
    return (a > b) ? a : b;
}

int maxInt = getMax(10, 20);        // Works with int
float maxFloat = getMax(5.5f, 3.2f); // Works with float
```

**Robotics example:**
```cpp
// Generic sensor reading function
template <typename T>
T readSensor(int pin) {
    T value = analogRead(pin);
    return value;
}

int intValue = readSensor<int>(A0);
float floatValue = readSensor<float>(A1);
```

#### **7. Return by Reference** (Advanced)
```cpp
int& getElement(int arr[], int index) {
    return arr[index];  // Returns reference to element
}

int data[5] = {1, 2, 3, 4, 5};
getElement(data, 2) = 99;  // Directly modify array element!
// data is now {1, 2, 99, 4, 5}
```

---

## **SECTIONS 15-16: OBJECT-ORIENTED PROGRAMMING (OOP)**

### **General Concept:**
OOP organizes code around "objects" - things that have properties (data) and behaviors (functions). Think of a robot as an object: it has properties (battery level, position) and behaviors (move, turn, sense).

### **Why OOP in Robotics:**
- **Modularity**: Each component (motor, sensor) is a separate class
- **Reusability**: Write once, use in multiple robots
- **Maintainability**: Easy to update one component without breaking others
- **ROS2 is built on OOP**: Nodes, publishers, subscribers are all classes

### **Detailed Breakdown:**

#### **1. Classes and Objects**
```cpp
// Class = Blueprint
class Motor {
public:
    int speed;
    int pin;
    
    void start() {
        digitalWrite(pin, HIGH);
    }
    
    void stop() {
        digitalWrite(pin, LOW);
    }
};

// Objects = Instances of the class
Motor leftMotor;
Motor rightMotor;

leftMotor.speed = 100;
leftMotor.pin = 9;
leftMotor.start();

rightMotor.speed = 120;
rightMotor.pin = 10;
rightMotor.start();
```

#### **2. Encapsulation** (Data Hiding)
```cpp
class BatteryMonitor {
private:  // Can't access from outside
    float voltage;
    float minVoltage = 10.5;
    
public:   // Can access from outside
    void updateVoltage(float v) {
        if (v >= 0 && v <= 16.8) {  // Validation
            voltage = v;
        }
    }
    
    float getVoltage() {
        return voltage;
    }
    
    bool isLow() {
        return voltage < minVoltage;
    }
};

BatteryMonitor battery;
battery.updateVoltage(12.5);  // OK
// battery.voltage = -5;  // ERROR: private!
float v = battery.getVoltage();  // OK: public getter
```

**Why this matters:**
- Protected data (can't accidentally set invalid values)
- Clean interface (users don't need to know internal details)

#### **3. Constructors** (Initialize objects)
```cpp
class UltrasonicSensor {
private:
    int trigPin;
    int echoPin;
    
public:
    // Constructor - called when object is created
    UltrasonicSensor(int trig, int echo) {
        trigPin = trig;
        echoPin = echo;
        pinMode(trigPin, OUTPUT);
        pinMode(echoPin, INPUT);
    }
    
    float getDistance() {
        // Measure distance logic
        return distance;
    }
};

// Creating object automatically calls constructor
UltrasonicSensor frontSensor(7, 8);  // trig=7, echo=8
UltrasonicSensor backSensor(5, 6);   // trig=5, echo=6

float d = frontSensor.getDistance();
```

**Constructor Types:**
```cpp
class Robot {
public:
    int id;
    string name;
    
    // Default constructor
    Robot() {
        id = 0;
        name = "Unknown";
    }
    
    // Parameterized constructor
    Robot(int i, string n) {
        id = i;
        name = n;
    }
    
    // Copy constructor
    Robot(const Robot& other) {
        id = other.id;
        name = other.name;
    }
};

Robot r1;              // Calls default constructor
Robot r2(1, "Bot-A");  // Calls parameterized
Robot r3 = r2;         // Calls copy constructor
```

#### **4. Member Functions**
```cpp
class LineFollower {
private:
    int leftSensor;
    int rightSensor;
    int threshold;
    
public:
    LineFollower(int left, int right) {
        leftSensor = left;
        rightSensor = right;
        threshold = 500;
    }
    
    void calibrate() {
        // Calibration logic
    }
    
    bool onLine() {
        int leftValue = analogRead(leftSensor);
        int rightValue = analogRead(rightSensor);
        return (leftValue > threshold || rightValue > threshold);
    }
    
    char getDirection() {
        int left = analogRead(leftSensor);
        int right = analogRead(rightSensor);
        
        if (left > threshold && right > threshold) return 'F';
        if (left > threshold && right < threshold) return 'R';
        if (left < threshold && right > threshold) return 'L';
        return 'S';  // Stop
    }
};

LineFollower robot(A0, A1);
robot.calibrate();

while (true) {
    char direction = robot.getDirection();
    // Move based on direction
}
```

#### **5. This Pointer**
```cpp
class Motor {
private:
    int speed;
    
public:
    void setSpeed(int speed) {
        this->speed = speed;  // this->speed = member variable
                               // speed = parameter
    }
    
    Motor& increaseSpeed(int amount) {
        this->speed += amount;
        return *this;  // Return reference to current object
    }
};

Motor m;
m.increaseSpeed(10).increaseSpeed(5);  // Method chaining!
```

#### **6. Static Members** (Shared by all objects)
```cpp
class Robot {
private:
    int id;
    static int count;  // Shared by ALL robots
    
public:
    Robot() {
        id = count;
        count++;
    }
    
    static int getCount() {
        return count;
    }
};

// Must define static member outside class
int Robot::count = 0;

Robot r1;  // count = 1
Robot r2;  // count = 2
Robot r3;  // count = 3

cout << Robot::getCount();  // 3 (no object needed!)
```

---

## **SECTION 18: INHERITANCE**

### **General Concept:**
Inheritance allows you to create new classes based on existing classes. The new class "inherits" properties and methods from the parent class and can add its own.

### **Think of it like:**
- Vehicle (parent) → Car, Truck, Motorcycle (children)
- Sensor (parent) → UltrasonicSensor, IRSensor, LidarSensor (children)

### **Detailed Breakdown:**

#### **1. Basic Inheritance**
```cpp
// Base class (parent)
class Sensor {
protected:  // Accessible to child classes
    int pin;
    string type;
    
public:
    Sensor(int p, string t) {
        pin = p;
        type = t;
    }
    
    void initialize() {
        pinMode(pin, INPUT);
    }
    
    string getType() {
        return type;
    }
};

// Derived class (child)
class UltrasonicSensor : public Sensor {
private:
    int trigPin;
    
public:
    UltrasonicSensor(int trig, int echo) 
        : Sensor(echo, "Ultrasonic") {  // Call parent constructor
        trigPin = trig;
        pinMode(trigPin, OUTPUT);
    }
    
    float getDistance() {
        // Ultrasonic-specific code
        return distance;
    }
};

// Another derived class
class IRSensor : public Sensor {
public:
    IRSensor(int p) : Sensor(p, "IR") {}
    
    bool detectObject() {
        return digitalRead(pin) == HIGH;
    }
};

// Usage
UltrasonicSensor us(7, 8);
IRSensor ir(9);

us.initialize();  // Inherited from Sensor
float d = us.getDistance();  // UltrasonicSensor's own method

ir.initialize();  // Inherited from Sensor
bool detected = ir.detectObject();  // IRSensor's own method
```

**The power:**
- Write common code once in parent class
- Each child class adds its specific functionality
- All sensors can `initialize()` the same way

#### **2. Access Specifiers in Inheritance**
```cpp
class Base {
public:
    int publicVar;
protected:
    int protectedVar;
private:
    int privateVar;
};

class Derived : public Base {
    // publicVar → public (accessible)
    // protectedVar → protected (accessible)
    // privateVar → NOT accessible
};
```

#### **3. Constructor Chaining**
```cpp
class Vehicle {
protected:
    int wheels;
    
public:
    Vehicle(int w) {
        wheels = w;
        cout << "Vehicle created\n";
    }
};

class Car : public Vehicle {
private:
    string brand;
    
public:
    Car(int w, string b) : Vehicle(w) {  // Call parent first
        brand = b;
        cout << "Car created\n";
    }
};

Car myCar(4, "Tesla");
// Output:
// Vehicle created
// Car created
```

#### **4. Real Robotics Example**
```cpp
// Base class for all motors
class Motor {
protected:
    int pin;
    int speed;
    
public:
    Motor(int p) : pin(p), speed(0) {
        pinMode(pin, OUTPUT);
    }
    
    virtual void setSpeed(int s) {  // Virtual for overriding
        speed = s;
        analogWrite(pin, speed);
    }
    
    virtual void stop() {
        speed = 0;
        analogWrite(pin, 0);
    }
};

// Servo motor (different control method)
class ServoMotor : public Motor {
private:
    int angle;
    
public:
    ServoMotor(int p) : Motor(p), angle(90) {}
    
    void setSpeed(int s) override {
        // Servos use angle, not speed
        angle = map(s, 0, 255, 0, 180);
        // Servo library code here
    }
    
    void setAngle(int a) {
        angle = a;
        // Servo-specific control
    }
};

// DC motor with encoder
class EncoderMotor : public Motor {
private:
    int encoderPin;
    long encoderCount;
    
public:
    EncoderMotor(int motorPin, int encPin) 
        : Motor(motorPin), encoderPin(encPin), encoderCount(0) {
        pinMode(encoderPin, INPUT);
    }
    
    void setSpeed(int s) override {
        Motor::setSpeed(s);  // Call parent's setSpeed
        // Additional encoder logic
    }
    
    long getPosition() {
        return encoderCount;
    }
};

// Polymorphism in action
Motor* motors[4];
motors[0] = new Motor(3);
motors[1] = new ServoMotor(5);
motors[2] = new EncoderMotor(6, 7);

// Can control all motors the same way!
for (int i = 0; i < 3; i++) {
    motors[i]->setSpeed(150);
}
```

---

## **SECTION 19: POLYMORPHISM**

### **General Concept:**
Polymorphism means "many forms." Same function name, but different behavior depending on the object type. It's the culmination of OOP power.

### **Types of Polymorphism:**
1. **Compile-time** (Function overloading - you already learned)
2. **Runtime** (Virtual functions - this section)

### **Detailed Breakdown:**

#### **1. Function Overriding**
```cpp
class Shape {
public:
    void draw() {
        cout << "Drawing a shape\n";
    }
};

class Circle : public Shape {
public:
    void draw() {  // Overrides parent's draw()
        cout << "Drawing a circle\n";
    }
};

Shape s;
s.draw();  // "Drawing a shape"

Circle c;
c.draw();  // "Drawing a circle"
```

#### **2. Virtual Functions** (The game-changer)
```cpp
class Sensor {
public:
    virtual float read() {  // VIRTUAL keyword!
        return 0.0;
    }
};

class UltrasonicSensor : public Sensor {
public:
    float read() override {  // Override keyword (optional but good practice)
        // Ultrasonic-specific code
        return distance;
    }
};

class IRSensor : public Sensor {
public:
    float read() override {
        // IR-specific code
        return irValue;
    }
};

// THE MAGIC:
Sensor* sensors[3];
sensors[0] = new Sensor();
sensors[1] = new UltrasonicSensor();
sensors[2] = new IRSensor();

for (int i = 0; i < 3; i++) {
    float value = sensors[i]->read();  
    // Calls correct read() based on ACTUAL object type!
    // Runtime polymorphism
}
```

**Without `virtual`:**
```cpp
class Sensor {
public:
    float read() {  // NO virtual
        return 0.0;
    }
};

Sensor* s = new UltrasonicSensor();
s->read();  // Calls Sensor::read(), NOT UltrasonicSensor::read()!
```

#### **3. Pure Virtual Functions & Abstract Classes**
```cpp
class Actuator {  // Abstract class
public:
    virtual void activate() = 0;  // Pure virtual (= 0)
    // No implementation - MUST be overridden by children
};

// Can't do this:
// Actuator a;  // ERROR: Can't instantiate abstract class

class Motor : public Actuator {
public:
    void activate() override {
        // Start motor
    }
};

class Gripper : public Actuator {
public:
    void activate() override {
        // Close gripper
    }
};

// Now you can create Motor and Gripper objects
Motor m;
Gripper g;
m.activate();
g.activate();
```

**Why abstract classes?**
- Force children to implement specific functions
- Create interfaces/contracts
- Can't accidentally create incomplete objects

#### **4. Real Robotics Example: Plugin Architecture**
```cpp
// Abstract base class
class NavigationAlgorithm {
public:
    virtual void computePath() = 0;
    virtual void followPath() = 0;
    virtual ~NavigationAlgorithm() {}  // Virtual destructor
};

class AStarNavigation : public NavigationAlgorithm {
public:
    void computePath() override {
        // A* algorithm
    }
    
    void followPath() override {
        // A* path following
    }
};

class DijkstraNavigation : public NavigationAlgorithm {
public:
    void computePath() override {
        // Dijkstra algorithm
    }
    
    void followPath() override {
        // Dijkstra path following
    }
};

class RRTNavigation : public NavigationAlgorithm {
public:
    void computePath() override {
        // RRT algorithm
    }
    
    void followPath() override {
        // RRT path following
    }
};

// Robot class doesn't care which algorithm!
class Robot {
private:
    NavigationAlgorithm* nav;
    
public:
    void setNavigation(NavigationAlgorithm* n) {
        nav = n;
    }
    
    void navigateToGoal() {
        nav->computePath();
        nav->followPath();
    }
};

// Easy to switch algorithms at runtime!
Robot robot;
robot.setNavigation(new AStarNavigation());
robot.navigateToGoal();

// Change strategy
robot.setNavigation(new RRTNavigation());
robot.navigateToGoal();
```

#### **5. Virtual Destructors** (CRITICAL for memory management)
```cpp
class Base {
public:
    virtual ~Base() {  // Virtual destructor
        cout << "Base destructor\n";
    }
};

class Derived : public Base {
private:
    int* data;
    
public:
    Derived() {
        data = new int[1000];
    }
    
    ~Derived() {
        delete[] data;  // Clean up
        cout << "Derived destructor\n";
    }
};

// With virtual destructor:
Base* obj = new Derived();
delete obj;
// Output:
// Derived destructor  ← Correct! Memory freed
// Base destructor

// Without virtual destructor:
// Only Base destructor called → MEMORY LEAK!
```

**Rule:** If a class has virtual functions, it MUST have a virtual destructor!

---

## **KEY TAKEAWAYS FOR ROBOTICS:**

### **Memory Management (Section 12):**
```cpp
// Stack (automatic, fast, limited)
int localVar = 10;

// Heap (manual, larger, slower)
int* dynamicArray = new int[1000];
delete[] dynamicArray;  // MUST DELETE!

// Modern C++ (Smart pointers - you'll learn in C++11 features)
unique_ptr<int[]> smartArray(new int[1000]);
// Automatically deleted when out of scope
```

### **OOP in ROS2 Pattern:**
```cpp
class MyRobotNode : public rclcpp::Node {  // Inheritance
public:
    MyRobotNode() : Node("my_robot") {  // Constructor
        // Create publishers/subscribers
        pub_ = this->create_publisher<...>(...);
    }
    
private:
    void timerCallback() {  // Member function
        // Sensor processing
    }
    
    rclcpp::Publisher::SharedPtr pub_;
};

int main() {
    rclcpp::init(argc, argv);
    rclcpp::spin(make_shared<MyRobotNode>());  // Polymorphism
    rclcpp::shutdown();
}
```
