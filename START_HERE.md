# 🤖 START HERE - Robotics Fundamentals Guide

## Welcome! 👋

This repository contains everything you need to go from **complete beginner** to **intermediate robotics developer** with ROS2.

---

## 📦 What's Inside?

### 📘 **7 Complete Guides** 
Detailed explanations for beginners to intermediate learners:

| Topic | Lines | Focus |
|-------|-------|-------|
| **C++** | 414 | Programming for ROS2 (pointers, classes, smart pointers) |
| **Python** | 528 | Alternative to C++ (classes, type hints, async) |
| **ROS2** | 188 | Robot software framework (nodes, topics, TF) |
| **Linux** | 474 | Command-line essentials (files, permissions, bash) |
| **Docker** | 618 | Containerization (images, volumes, compose) |
| **SLAM** | 543 | Mapping & localization (algorithms, sensors) |
| **Nav2** | 627 | Autonomous navigation (planning, control) |

---

## 🎯 Choose Your Path

### Path 1: Complete Beginner
**"I'm new to programming and Linux"**

1. Start with [linux/README.md](linux/README.md)
2. Learn [python/README.md](python/README.md) *(easier)* OR [cpp/README.md](cpp/README.md)

---

### Path 2: Have Programming Experience
**"I know Python/C++ but new to ROS2"**

1. Quick review: [linux/README.md](linux/README.md) (Sections 8, 12)
2. Deep dive: [ros2/README.md](ros2/README.md)
3. Learn [slam/README.md](slam/README.md)
4. Master [nav2/README.md](nav2/README.md)
5. Optional: [docker/README.md](docker/README.md)

---

### Path 3: Quick Reference
**"I need to look something up"**

Common lookups:
- ROS2 commands
- Linux commands
- Docker commands
- SLAM setup
- Nav2 configuration

---

## 🚀 Quick Start (Right Now!)

### If you want to start immediately:

#### Option A: Learn Linux Basics (15 minutes)
```bash
# Open your terminal and practice:
pwd                 # Where am I?
ls                  # What files are here?
mkdir test_folder   # Create folder
cd test_folder      # Enter folder
touch test.txt      # Create file
cd ..               # Go back
rm -r test_folder   # Remove folder
```

**Next:** Read [linux/README.md](linux/README.md) Section 1-2

---

#### Option B: Start Python (30 minutes)
```python
# Create a file: robot.py
class Robot:
    def __init__(self, name):
        self.name = name
        self.position = (0, 0)
    
    def move(self, x, y):
        self.position = (x, y)
        print(f"{self.name} moved to {self.position}")

# Create and use robot
my_robot = Robot("R2D2")
my_robot.move(5, 3)
```

**Next:** Read [python/README.md](python/README.md) Sections 1-4

---

#### Option C: Understand ROS2 (Read first)
Read this quick overview:
- **Node**: A program that does one thing (like "camera reader" or "motor controller")
- **Topic**: A channel for sending messages (like "/camera/image" or "/cmd_vel")
- **Message**: The data being sent (like images or velocity commands)

**Next:** Read [ros2/README.md](ros2/README.md) Section on Core Concepts

---
