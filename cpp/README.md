# C++ Fundamentals for Robotics

## Why C++ in Robotics?

C++ is heavily used in robotics because:
- **Performance**: Direct hardware control and fast execution
- **Real-time capabilities**: Predictable timing for control loops
- **Memory management**: Fine control over resources
- **ROS2 compatibility**: Core ROS2 libraries are written in C++

---

## 1. Basic Structure & Syntax

### Hello World Anatomy
```cpp
#include <iostream>  // Include library
using namespace std; // Use standard namespace

int main() {         // Entry point
    cout << "Hello Robot!" << endl;
    return 0;        // Exit status
}
```

**Key Concepts:**
- `#include`: Brings in external code/libraries
- `namespace`: Organizes code to avoid naming conflicts
- `main()`: Where program execution begins
- `return 0`: Indicates successful execution

---

## 2. Variables & Data Types

### Fundamental Types
```cpp
int speed = 100;           // Integer: whole numbers
double distance = 10.5;    // Double: decimal numbers
float angle = 3.14f;       // Float: smaller decimal
bool is_moving = true;     // Boolean: true/false
char direction = 'N';      // Character: single letter
string robot_name = "R2D2"; // String: text
```

**Why This Matters:**
- **int**: Use for counts, IDs, array indices
- **double/float**: Use for sensor readings, coordinates, angles
- **bool**: Use for states (motor on/off, sensor triggered)
- **string**: Use for names, messages, file paths

### Variable Scope
```cpp
int global_var = 10;  // Available everywhere

void myFunction() {
    int local_var = 5;  // Only exists inside this function
    // Can access global_var here
}
// Cannot access local_var here
```

---

## 3. Pointers & References (IMPORTANT!)

### What Are Pointers?
Pointers store **memory addresses** rather than values.

```cpp
int speed = 50;
int* ptr = &speed;  // ptr stores the ADDRESS of speed

cout << speed;      // Prints: 50
cout << &speed;     // Prints: memory address (e.g., 0x7fff5fbff5ac)
cout << ptr;        // Prints: same memory address
cout << *ptr;       // Prints: 50 (dereference - get value at address)
```

**Why Use Pointers?**
- Pass large data without copying (efficient)
- Modify variables in other functions
- Dynamic memory allocation
- **ROS2 uses pointers extensively for nodes and messages**

### References
References are like "aliases" - another name for a variable.

```cpp
int speed = 50;
int& ref_speed = speed;  // ref_speed is another name for speed

ref_speed = 100;  // Changes speed to 100
cout << speed;    // Prints: 100
```

**Pointer vs Reference:**
- Pointers can be null, references cannot
- Pointers can change what they point to, references are fixed
- References are safer and easier to use

---

## 4. Functions

### Basic Function Structure
```cpp
// returnType functionName(parameters)
double calculateDistance(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return sqrt(dx*dx + dy*dy);
}
```

### Pass by Value vs Pass by Reference
```cpp
// Pass by value (copies the variable)
void incrementCopy(int x) {
    x++;  // Only changes the copy
}

// Pass by reference (works on original)
void incrementOriginal(int& x) {
    x++;  // Changes the original variable
}

int count = 5;
incrementCopy(count);      // count is still 5
incrementOriginal(count);  // count is now 6
```

**In Robotics:**
Use pass-by-reference for large data (sensor arrays, images) to avoid copying overhead.

---

## 5. Classes & Objects (CRUCIAL for ROS2)

### What is a Class?
A class is a blueprint for creating objects. It bundles data and functions together.

```cpp
class Robot {
private:
    double x_position;
    double y_position;
    
public:
    // Constructor
    Robot(double x, double y) {
        x_position = x;
        y_position = y;
    }
    
    // Method
    void moveTo(double new_x, double new_y) {
        x_position = new_x;
        y_position = new_y;
    }
    
    // Getter
    double getX() { return x_position; }
};

// Using the class
Robot my_robot(0.0, 0.0);
my_robot.moveTo(5.0, 3.0);
```

**Key Concepts:**
- **private**: Data only accessible inside the class (encapsulation)
- **public**: Methods/data accessible from outside
- **Constructor**: Special function that runs when object is created
- **Methods**: Functions that belong to the class

### Why Classes Matter in ROS2
ROS2 nodes are classes! You'll create classes that inherit from `rclcpp::Node`.

---

## 6. Inheritance

Inheritance lets you create new classes based on existing ones.

```cpp
class Vehicle {
protected:
    double speed;
    
public:
    void setSpeed(double s) { speed = s; }
};

class Robot : public Vehicle {  // Robot inherits from Vehicle
public:
    void executeTask() {
        // Can use speed from Vehicle
        cout << "Moving at speed: " << speed << endl;
    }
};
```

**In ROS2:**
Your node classes inherit from `rclcpp::Node` to get all ROS2 functionality.

---

## 7. Smart Pointers (ESSENTIAL for ROS2)

Smart pointers automatically manage memory - they delete objects when no longer needed.

### Types of Smart Pointers

```cpp
#include <memory>

// unique_ptr: Only one owner
std::unique_ptr<Robot> robot1 = std::make_unique<Robot>(0, 0);

// shared_ptr: Multiple owners, deleted when last owner is done
std::shared_ptr<Robot> robot2 = std::make_shared<Robot>(1, 1);
std::shared_ptr<Robot> robot3 = robot2;  // Both point to same robot
```

**Why Smart Pointers?**
- No memory leaks (automatic cleanup)
- Safer than raw pointers
- **ROS2 uses shared_ptr for nodes, publishers, subscribers**

---

## 8. Lambda Functions

Lambdas are anonymous functions, often used for callbacks.

```cpp
// Basic lambda syntax
auto greet = []() {
    cout << "Hello!" << endl;
};

// Lambda with parameters
auto add = [](int a, int b) {
    return a + b;
};

// Lambda capturing variables
int multiplier = 5;
auto multiply = [multiplier](int x) {
    return x * multiplier;
};
```

**In ROS2:**
Lambdas are used for timer callbacks and subscription callbacks.

```cpp
// Example: Timer callback in ROS2
timer_ = create_wall_timer(
    500ms, 
    [this]() { this->publishMessage(); }  // Lambda callback
);
```

---

## 9. Standard Template Library (STL)

### Vectors (Dynamic Arrays)
```cpp
#include <vector>

std::vector<double> sensor_readings;
sensor_readings.push_back(3.5);  // Add element
sensor_readings.push_back(4.2);

for (double reading : sensor_readings) {  // Range-based for loop
    cout << reading << endl;
}
```

### Maps (Key-Value Pairs)
```cpp
#include <map>

std::map<std::string, double> robot_speeds;
robot_speeds["robot1"] = 2.5;
robot_speeds["robot2"] = 3.0;

// Access
double speed = robot_speeds["robot1"];
```

**Why STL Matters:**
- Pre-built, tested data structures
- Efficient implementations
- Used throughout ROS2 code

---

## 10. Namespaces

Namespaces prevent naming conflicts in large projects.

```cpp
namespace robotics {
    class Sensor {
        // ...
    };
}

namespace vision {
    class Sensor {  // Different Sensor class
        // ...
    };
}

// Usage
robotics::Sensor lidar;
vision::Sensor camera;
```

**In ROS2:**
Everything is in the `rclcpp` namespace (ROS Client Library for C++).

---

## 11. Header Files & Compilation

### Header File (.hpp)
```cpp
// robot.hpp
#ifndef ROBOT_HPP
#define ROBOT_HPP

class Robot {
public:
    void move();
};

#endif
```

### Implementation File (.cpp)
```cpp
// robot.cpp
#include "robot.hpp"

void Robot::move() {
    // Implementation
}
```

**Why Separate Files?**
- **Declaration** (header): Tells compiler what exists
- **Implementation** (cpp): Tells compiler how it works
- Allows code reuse and faster compilation

---

## 12. CMakeLists.txt Basics

CMake is the build system used by ROS2.

```cmake
cmake_minimum_required(VERSION 3.5)
project(my_robot_package)

# Find dependencies
find_package(rclcpp REQUIRED)

# Create executable
add_executable(my_node src/my_node.cpp)

# Link libraries
ament_target_dependencies(my_node rclcpp)
```

**What This Does:**
- Finds required libraries (ROS2 packages)
- Compiles your source code
- Links everything together into an executable

---

## Essential C++ Concepts for ROS2 Summary

✅ **Must Know:**
1. Pointers and references (used everywhere)
2. Classes and objects (nodes are classes)
3. Smart pointers (shared_ptr for ROS2 objects)
4. Inheritance (your nodes inherit from rclcpp::Node)
5. Lambda functions (for callbacks)
6. STL containers (vectors, maps)
7. Namespaces (rclcpp::)

✅ **Important for Understanding:**
- Variable scope and lifetime
- Pass by reference for efficiency
- Header vs implementation files
- CMake basics

---

## Next Steps

1. Practice basic C++ syntax
2. Understand pointers and references thoroughly
3. Get comfortable with classes
4. Learn smart pointers
5. Move to ROS2 tutorials

**Remember:** You don't need to master everything before starting ROS2. Learn the basics, then learn more as you encounter it in ROS2 code!
