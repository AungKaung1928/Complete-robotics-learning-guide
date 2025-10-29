# Python Fundamentals for Robotics

## Why Python in Robotics?

Python is popular in robotics for:
- **Easy to learn**: Simple, readable syntax
- **Rapid prototyping**: Quick testing of ideas
- **Rich libraries**: NumPy, OpenCV, TensorFlow for robotics tasks
- **ROS2 compatible**: Full support through rclpy

---

## 1. Basic Syntax & Structure

### Python Simplicity
```python
# No semicolons, no curly braces
print("Hello Robot!")

# Variables - no type declaration needed
speed = 100
distance = 10.5
is_moving = True
robot_name = "R2D2"
```

**Key Differences from C++:**
- No type declarations (dynamic typing)
- Indentation defines code blocks (not `{}`)
- No compilation step (interpreted language)

---

## 2. Data Types & Structures

### Basic Types
```python
# Numbers
count = 42              # Integer
distance = 10.5         # Float
complex_num = 3 + 4j    # Complex number

# Strings
name = "Robot"
message = 'Hello'
multiline = """This is a
multi-line string"""

# Boolean
is_active = True
is_stopped = False
```

### Lists (Like C++ vectors)
```python
sensor_readings = [3.5, 4.2, 5.1, 3.8]
sensor_readings.append(6.0)     # Add item
first = sensor_readings[0]       # Access by index
last = sensor_readings[-1]       # Negative indexing from end

# Slicing
first_three = sensor_readings[0:3]  # Get first 3 items
```

### Dictionaries (Like C++ maps)
```python
robot_speeds = {
    "robot1": 2.5,
    "robot2": 3.0,
    "robot3": 1.8
}

# Access
speed = robot_speeds["robot1"]

# Add/modify
robot_speeds["robot4"] = 2.2
```

### Tuples (Immutable sequences)
```python
coordinates = (10.5, 20.3)  # Cannot be changed after creation
x, y = coordinates           # Unpacking
```

### Sets (Unique items)
```python
active_sensors = {"lidar", "camera", "imu"}
active_sensors.add("gps")
```

---

## 3. Functions

### Basic Function
```python
def calculate_distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return (dx**2 + dy**2) ** 0.5

# Call function
dist = calculate_distance(0, 0, 3, 4)
```

### Default Parameters
```python
def move_robot(distance, speed=1.0):
    time = distance / speed
    return time

move_robot(10)        # Uses default speed=1.0
move_robot(10, 2.0)   # Overrides speed
```

### Multiple Return Values
```python
def get_position():
    x = 10.5
    y = 20.3
    return x, y

x_pos, y_pos = get_position()
```

### Lambda Functions
```python
# Anonymous function
square = lambda x: x**2

# Used in callbacks (similar to C++ lambdas)
numbers = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, numbers))
```

---

## 4. Classes & Objects

### Basic Class Structure
```python
class Robot:
    def __init__(self, x, y):  # Constructor
        self.x_position = x     # Instance variable
        self.y_position = y
        
    def move_to(self, new_x, new_y):  # Method
        self.x_position = new_x
        self.y_position = new_y
        
    def get_position(self):
        return self.x_position, self.y_position

# Usage
my_robot = Robot(0.0, 0.0)
my_robot.move_to(5.0, 3.0)
x, y = my_robot.get_position()
```

**Key Concepts:**
- `self`: Refers to the instance (like `this` in C++)
- `__init__`: Constructor (special method)
- All methods need `self` as first parameter

### Inheritance
```python
class Vehicle:
    def __init__(self, speed):
        self.speed = speed
        
    def show_speed(self):
        print(f"Speed: {self.speed}")

class Robot(Vehicle):  # Inherits from Vehicle
    def __init__(self, speed, name):
        super().__init__(speed)  # Call parent constructor
        self.name = name
        
    def execute_task(self):
        print(f"{self.name} moving at {self.speed}")

robot = Robot(2.5, "R2D2")
robot.show_speed()      # Method from Vehicle
robot.execute_task()    # Method from Robot
```

---

## 5. Important Python Concepts

### List Comprehensions
```python
# Traditional way
squared = []
for x in range(10):
    squared.append(x**2)

# List comprehension (Pythonic way)
squared = [x**2 for x in range(10)]

# With condition
even_squared = [x**2 for x in range(10) if x % 2 == 0]
```

### Enumerate
```python
sensors = ["lidar", "camera", "imu"]

for index, sensor in enumerate(sensors):
    print(f"Sensor {index}: {sensor}")
```

### Zip
```python
x_coords = [1, 2, 3]
y_coords = [4, 5, 6]

for x, y in zip(x_coords, y_coords):
    print(f"Position: ({x}, {y})")
```

### F-strings (String Formatting)
```python
name = "Robot"
speed = 2.5

# Old way
message = "Robot speed: " + str(speed)

# F-string way (Python 3.6+)
message = f"{name} speed: {speed} m/s"
message = f"{name} speed: {speed:.2f} m/s"  # 2 decimal places
```

---

## 6. Error Handling

### Try-Except Blocks
```python
try:
    speed = 10 / 0  # Division by zero
except ZeroDivisionError:
    print("Cannot divide by zero!")
    speed = 0

# Multiple exceptions
try:
    value = int("not a number")
except ValueError:
    print("Invalid number format")
except Exception as e:  # Catch all other exceptions
    print(f"Error occurred: {e}")
```

**In Robotics:**
Use try-except to handle:
- Sensor failures
- Communication errors
- Invalid data

---

## 7. File Operations

### Reading Files
```python
# Read entire file
with open("config.txt", "r") as file:
    content = file.read()

# Read line by line
with open("data.txt", "r") as file:
    for line in file:
        print(line.strip())
```

### Writing Files
```python
with open("output.txt", "w") as file:
    file.write("Robot data\n")
    file.write(f"Position: {x}, {y}\n")
```

**Why `with`?**
Automatically closes the file, even if errors occur.

---

## 8. Important Libraries for Robotics

### NumPy (Numerical Computing)
```python
import numpy as np

# Arrays (like vectors in C++)
position = np.array([1.0, 2.0, 3.0])
velocity = np.array([0.5, 0.3, 0.1])

# Operations
acceleration = velocity * 2
magnitude = np.linalg.norm(position)

# Matrix operations
rotation_matrix = np.array([[0, -1], [1, 0]])
point = np.array([1, 0])
rotated = rotation_matrix @ point  # Matrix multiplication
```

### Math Module
```python
import math

angle = math.pi / 4
sin_value = math.sin(angle)
distance = math.sqrt(x**2 + y**2)
```

---

## 9. Object-Oriented Concepts for ROS2

### Properties and Methods
```python
class RobotController:
    def __init__(self):
        self._speed = 0.0  # Private-like variable (convention)
        
    @property
    def speed(self):  # Getter
        return self._speed
        
    @speed.setter
    def speed(self, value):  # Setter
        if value < 0:
            raise ValueError("Speed cannot be negative")
        self._speed = value

# Usage
controller = RobotController()
controller.speed = 2.5  # Calls setter
print(controller.speed)  # Calls getter
```

### Static and Class Methods
```python
class Robot:
    robot_count = 0  # Class variable
    
    def __init__(self, name):
        self.name = name
        Robot.robot_count += 1
        
    @classmethod
    def get_robot_count(cls):
        return cls.robot_count
        
    @staticmethod
    def calculate_distance(x1, y1, x2, y2):
        return ((x2-x1)**2 + (y2-y1)**2)**0.5

# Usage
robot1 = Robot("R1")
robot2 = Robot("R2")
print(Robot.get_robot_count())  # 2
dist = Robot.calculate_distance(0, 0, 3, 4)
```

---

## 10. Modules and Packages

### Importing
```python
# Import entire module
import math
result = math.sqrt(16)

# Import specific functions
from math import sqrt, pi
result = sqrt(16)

# Import with alias
import numpy as np
array = np.array([1, 2, 3])

# Import from your own module
from my_robot_code import RobotController
```

### Creating Your Own Module
```python
# File: robot_utils.py
def calculate_speed(distance, time):
    return distance / time

# File: main.py
from robot_utils import calculate_speed
speed = calculate_speed(10, 2)
```

---

## 11. Type Hints (Python 3.5+)

Type hints make code more readable and help catch errors.

```python
def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    return (dx**2 + dy**2) ** 0.5

def get_position() -> tuple[float, float]:
    return 10.5, 20.3

# For complex types
from typing import List, Dict, Optional

def process_readings(readings: List[float]) -> Dict[str, float]:
    return {"avg": sum(readings) / len(readings)}

def find_robot(name: str) -> Optional[Robot]:
    # Returns Robot or None
    pass
```

**Why Use Type Hints?**
- Better code documentation
- IDEs can provide better autocomplete
- Easier to catch bugs
- Common in ROS2 Python code

---

## 12. Asynchronous Programming Basics

ROS2 uses asyncio for some operations.

```python
import asyncio

async def fetch_sensor_data():
    await asyncio.sleep(1)  # Simulate delay
    return 42

async def main():
    data = await fetch_sensor_data()
    print(f"Sensor data: {data}")

# Run async function
asyncio.run(main())
```

**When You'll See This:**
- ROS2 lifecycle nodes
- Asynchronous service calls
- Parallel operations

---

## 13. Python vs C++ in ROS2

### When to Use Python:
✅ Rapid prototyping and testing
✅ Data analysis and visualization
✅ High-level logic and decision making
✅ Interfacing with ML libraries (TensorFlow, PyTorch)
✅ Quick scripts and tools

### When to Use C++:
✅ Performance-critical nodes (control loops)
✅ Low-level hardware interfaces
✅ Real-time operations
✅ Large-scale production systems

### Mixed Approach:
Most ROS2 projects use **both**:
- C++ for performance-critical nodes
- Python for high-level coordination and analysis

---

## Essential Python Concepts for ROS2 Summary

✅ **Must Know:**
1. Classes and objects (nodes are classes)
2. Functions and methods
3. Lists and dictionaries
4. Inheritance
5. Error handling (try-except)
6. Imports and modules

✅ **Important:**
- Type hints (for clarity)
- Lambda functions (callbacks)
- Properties (@property)
- File operations
- NumPy basics

✅ **Nice to Have:**
- List comprehensions
- Async/await basics
- Static/class methods

---

## Python Best Practices for ROS2

1. **Use type hints** - Makes code clearer
2. **Follow PEP 8** - Python style guide (use `snake_case`)
3. **Use meaningful names** - `sensor_reading` not `sr`
4. **Keep functions small** - One clear purpose each
5. **Handle exceptions** - Don't let errors crash your robot
6. **Use `with` for files** - Automatic cleanup

---

## Next Steps

1. Practice Python basics (variables, functions, classes)
2. Learn NumPy for numerical operations
3. Understand object-oriented programming in Python
4. Get comfortable with type hints
5. Move to ROS2 Python (rclpy) tutorials

**Remember:** Python is easier to learn than C++, so use it to understand ROS2 concepts first, then transition to C++ if needed for performance!
