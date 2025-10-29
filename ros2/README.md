# ROS2 Fundamentals

## What is ROS2?

**ROS2 (Robot Operating System 2)** is a framework for building robot software. It's NOT an operating system - it runs on top of Linux (or Windows/macOS).

### Why ROS2 Exists:
- **Modularity**: Break complex robot systems into small, manageable pieces
- **Reusability**: Use existing packages instead of reinventing the wheel
- **Communication**: Easy message passing between different parts of your system
- **Community**: Huge ecosystem of tools and libraries

### ROS1 vs ROS2:
ROS2 is the modern version with:
- Better real-time performance
- Built-in security
- Multi-robot support
- Works on more platforms

---

## Core Concepts

### 1. Nodes

**What is a Node?**
A node is a single-purpose program that does one specific job.

**Examples:**
- Camera node: Publishes camera images
- Motor controller node: Controls wheel motors
- Path planner node: Calculates routes
- Sensor filter node: Processes sensor data

**Why Multiple Nodes?**
- Easier to debug (isolate problems)
- Can restart individual nodes
- Reuse nodes in different projects
- Run on different computers

---

### 2. Topics (Publisher-Subscriber Pattern)

**What is a Topic?**
A named channel for sending messages. Think of it like a radio station.

**How it Works:**
```
Publisher Node → Topic → Subscriber Node(s)
```

**Example:**
```
Camera Node (publishes to "/camera/image")
    ↓
"/camera/image" topic
    ↓
Object Detector Node (subscribes to "/camera/image")
```

**Key Points:**
- One-way communication
- Can have multiple subscribers
- Can have multiple publishers
- Asynchronous (non-blocking)

**Common Topics:**
- `/cmd_vel` - Velocity commands
- `/odom` - Odometry data
- `/scan` - Laser scan data
- `/camera/image` - Camera images

---

### 3. Messages

**What is a Message?**
The data structure sent over topics.

**Standard Messages:**
```python
# geometry_msgs/msg/Twist
linear:
  x: 0.5
  y: 0.0
  z: 0.0
angular:
  z: 0.3

# sensor_msgs/msg/LaserScan
ranges: [1.2, 1.3, 1.5, ...]
angle_min: -3.14
angle_max: 3.14
```

---

### 4. Services (Request-Response)

**What is a Service?**
Synchronous communication - ask a question, wait for answer.

**When to Use:**
- One-time queries
- Trigger actions
- Get current state

---

### 5. Actions (Long-Running Tasks)

**What is an Action?**
Like a service, but for tasks that take time with progress feedback.

**Components:**
- **Goal**: What you want
- **Feedback**: Progress updates
- **Result**: Final outcome

---

### 6. Parameters

**What are Parameters?**
Configuration values changeable at runtime.

**Examples:**
- Maximum speed
- PID gains
- Update rates

---

## Important ROS2 Commands

### Building
```bash
colcon build
colcon build --packages-select my_package
colcon build --symlink-install
```

### Sourcing
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

### Running
```bash
ros2 run package_name node_name
ros2 launch package_name launch_file.py
```

### Topics
```bash
ros2 topic list
ros2 topic echo /topic_name
ros2 topic hz /topic_name
```

### Nodes
```bash
ros2 node list
ros2 node info /node_name
```

---

## Essential Concepts Summary

✅ **Must Understand:**
1. Nodes - Single-purpose programs
2. Topics - Continuous data streams
3. Services - Request-response
4. Actions - Long tasks with feedback
5. Parameters - Runtime configuration
6. Launch files - Start multiple nodes
7. TF2 - Coordinate transformations

✅ **Key Skills:**
- Creating publisher/subscriber nodes
- Using standard message types
- Understanding QoS profiles
- Working with TF2 transforms
