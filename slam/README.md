# SLAM Fundamentals
**Simultaneous Localization and Mapping**

## What is SLAM?

**SLAM** = Building a map while figuring out where you are in that map.

### The Problem:
- You can't localize without a map
- You can't map without knowing your location
- SLAM solves both simultaneously!

### Why SLAM Matters:
- Autonomous navigation
- Indoor mapping
- Unknown environment exploration
- GPS-denied areas (buildings, caves)

---

## Core Concepts

### 1. Localization

**What is Localization?**
Determining the robot's position and orientation (pose) in space.

**Pose Components:**
- **Position**: (x, y, z) coordinates
- **Orientation**: (roll, pitch, yaw) angles or quaternion

**Example:**
```
Pose: x=2.5m, y=1.3m, yaw=45°
"The robot is 2.5m forward, 1.3m left, facing 45 degrees"
```

---

### 2. Mapping

**What is Mapping?**
Creating a representation of the environment.

**Map Types:**

#### Occupancy Grid Map
- Grid of cells (like pixels)
- Each cell: Free, Occupied, or Unknown
- Used for: Navigation, obstacle avoidance

```
# = Obstacle (occupied)
. = Free space
? = Unknown

##########
#........#
#..###...#
#........#
##########
```

#### Feature Map
- Collection of landmarks
- Points, lines, or objects
- Used for: Accurate localization

```
Landmark 1: Corner at (5, 3)
Landmark 2: Doorway at (10, 7)
Landmark 3: Pillar at (8, 2)
```

---

### 3. Sensors for SLAM

#### LiDAR (Laser Scanner)
- Measures distances with lasers
- Creates 2D or 3D point clouds
- Very accurate (cm-level)
- Expensive
- **Best for:** Indoor mapping

#### Camera (Visual SLAM)
- Uses images
- Can detect features, colors, textures
- Cheaper than LiDAR
- Affected by lighting
- **Best for:** Rich environments

#### IMU (Inertial Measurement Unit)
- Measures acceleration and rotation
- Doesn't measure position directly
- Helps between sensor updates
- **Best for:** Smoothing motion

#### Wheel Odometry
- Counts wheel rotations
- Estimates distance traveled
- Accumulates error (drift)
- **Best for:** Short-term motion

---

## SLAM Process

### 1. Motion Prediction
```
Old Pose + Motion Command → Predicted New Pose
```

**Example:**
```
Old: x=0, y=0, yaw=0°
Command: Move forward 1m
Predicted: x=1, y=0, yaw=0°
```

**Problem:** Wheels slip, measurements have noise → prediction is uncertain!

---

### 2. Sensor Measurement
```
Sensors → Observe environment
```

**Example:**
```
LiDAR detects:
- Wall at 2m ahead
- Corner at 1.5m, 45° right
- Opening at 3m, 30° left
```

---

### 3. Data Association
```
Match current observations to map
```

**Question:** "Is this wall the same wall I saw before?"

**Challenges:**
- Similar-looking features
- Sensor noise
- Moving objects

---

### 4. Pose Update
```
Predicted Pose + Sensor Measurements → Updated Pose
```

**Combines:**
- Where we think we are (prediction)
- What we actually see (measurements)

---

### 5. Map Update
```
Add new information to map
```

**Actions:**
- Add new obstacles
- Update confidence in existing features
- Remove incorrect data

---

## SLAM Algorithms

### 1. EKF-SLAM (Extended Kalman Filter)
**How it works:**
- Maintains one best estimate of pose and map
- Updates estimate with each measurement
- Uses Gaussian distributions for uncertainty

**Pros:**
- Mathematically elegant
- Well understood
- Good for small environments

**Cons:**
- Computationally expensive for large maps
- Assumes Gaussian noise
- Can diverge if wrong associations

**When to use:** Small environments, few landmarks

---

### 2. Particle Filter SLAM (FastSLAM)
**How it works:**
- Maintains multiple pose hypotheses (particles)
- Each particle has its own map
- Best particles survive over time

**Pros:**
- Can handle non-Gaussian noise
- More robust to wrong associations
- Good for large environments

**Cons:**
- Needs many particles
- More memory usage
- Slower than some methods

**When to use:** Complex environments, ambiguous features

---

### 3. Graph-Based SLAM
**How it works:**
- Robot poses and landmarks are nodes in a graph
- Sensor measurements are edges (constraints)
- Optimize entire graph to find best solution

**Pros:**
- Can handle loop closures well
- Very accurate
- Scalable to large environments

**Cons:**
- Requires more computation
- Delayed correction
- Complex implementation

**When to use:** Large environments, need high accuracy

**Popular implementations:**
- SLAM Toolbox (ROS2)
- Cartographer
- RTAB-Map

---

### 4. Visual SLAM (vSLAM)
**How it works:**
- Uses camera images
- Extracts visual features (corners, edges)
- Tracks features between frames

**Pros:**
- Cheap sensor (camera)
- Rich information
- Can recognize places

**Cons:**
- Sensitive to lighting
- Computationally intensive
- Scale ambiguity (monocular)

**Popular implementations:**
- ORB-SLAM
- RTAB-Map (with camera)

---

## Key Challenges in SLAM

### 1. Loop Closure
**Problem:** Recognizing you've returned to a previously visited place

**Why it matters:**
- Corrects accumulated drift
- Improves map consistency

**Example:**
```
Start → Explore → Return to start
Without loop closure: Map doesn't connect properly
With loop closure: Map is corrected and consistent
```

### 2. Data Association
**Problem:** Matching current observations to map features

**Challenges:**
- Similar-looking features
- Sensor noise
- Moving objects

**Solution:**
- Feature descriptors
- Probabilistic matching
- Geometric consistency checks

### 3. Computational Complexity
**Problem:** Processing grows with map size

**Map with N landmarks:**
- EKF-SLAM: O(N²) - very slow for large N
- Graph-based: More efficient
- Modern SLAM: Various optimizations

### 4. Dynamic Environments
**Problem:** Moving objects confuse SLAM

**Solutions:**
- Ignore moving objects
- Model dynamics separately
- Use semantic information

---

## SLAM in ROS2

### Common Packages

#### 1. SLAM Toolbox
```bash
sudo apt install ros-humble-slam-toolbox
```

**Features:**
- Graph-based SLAM
- Works with LiDAR
- Supports loop closure
- Can load/save maps
- Good for indoor environments

**Launch:**
```bash
ros2 launch slam_toolbox online_async_launch.py
```

#### 2. Cartographer
```bash
sudo apt install ros-humble-cartographer-ros
```

**Features:**
- Google's SLAM algorithm
- 2D and 3D SLAM
- Real-time and offline modes
- Very accurate

#### 3. RTAB-Map
```bash
sudo apt install ros-humble-rtabmap-ros
```

**Features:**
- Visual and LiDAR SLAM
- 3D mapping
- Loop closure detection
- Memory management for large maps

---

## SLAM Topics in ROS2

### Required Topics

```bash
# Input to SLAM
/scan          # sensor_msgs/LaserScan - LiDAR data
/odom          # nav_msgs/Odometry - Wheel odometry
/tf            # tf2_msgs/TFMessage - Transforms

# Output from SLAM
/map           # nav_msgs/OccupancyGrid - The map
/map_metadata  # nav_msgs/MapMetaData - Map info
```

### Coordinate Frames (TF)

```
map → odom → base_link → laser_frame
 ↑      ↑       ↑           ↑
 │      │       │           └─ Sensor location
 │      │       └──────────── Robot center
 │      └──────────────────── Odometry frame (drifts)
 └─────────────────────────── Global map frame (corrected by SLAM)
```

**Key Points:**
- `map → odom`: SLAM provides this (corrects drift)
- `odom → base_link`: Odometry provides this
- `base_link → laser`: Static transform (sensor mounting)

---

## Running SLAM (Example)

### 1. Start SLAM Toolbox
```bash
ros2 launch slam_toolbox online_async_launch.py
```

### 2. Start Robot (or Simulator)
```bash
ros2 launch my_robot robot_bringup.py
```

### 3. Visualize in RViz
```bash
rviz2
```

**Add displays:**
- Map
- LaserScan
- RobotModel
- TF

### 4. Drive Robot to Explore
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### 5. Save Map
```bash
ros2 run nav2_map_server map_saver_cli -f my_map
```

**Generates:**
- `my_map.pgm` - Image file
- `my_map.yaml` - Metadata

---

## Map File Format

### YAML File (my_map.yaml)
```yaml
image: my_map.pgm
resolution: 0.05          # meters per pixel
origin: [-10.0, -10.0, 0.0]  # x, y, yaw
occupied_thresh: 0.65     # Threshold for occupied
free_thresh: 0.25         # Threshold for free
negate: 0                 # Whether to invert colors
```

### PGM File
- Grayscale image
- White = free space
- Black = occupied
- Gray = unknown

---

## SLAM Parameters to Tune

### Resolution
- **Lower** (e.g., 0.01): More detail, larger file, slower
- **Higher** (e.g., 0.1): Less detail, smaller file, faster

### Update Rate
- **Higher**: More responsive, more CPU
- **Lower**: Less CPU, may miss details

### Loop Closure Threshold
- **Lower**: More aggressive, may false match
- **Higher**: More conservative, may miss loops

### Particle Count (if using FastSLAM)
- **More**: Better accuracy, more memory/CPU
- **Fewer**: Faster, may lose track

---

## SLAM Best Practices

✅ **Do:**
- Drive slowly and smoothly
- Make multiple passes over areas
- Close loops when possible
- Provide good odometry
- Use consistent sensor data
- Save maps frequently

❌ **Don't:**
- Drive too fast (sensors can't keep up)
- Make sudden turns
- Map in crowded areas (dynamic objects)
- Forget to tune parameters
- Map very large areas in one session

---

## SLAM vs Localization

### SLAM (Mapping Mode)
- Building map while localizing
- Exploring unknown environment
- Map updates continuously
- More computational load

### Localization (Using Existing Map)
- Using pre-built map
- Only determining pose
- Map is fixed
- Faster, more reliable
- Used with Nav2 for navigation

**Typical Workflow:**
1. Use SLAM to build map
2. Save map
3. Use localization for navigation

---

## Essential SLAM Concepts Summary

✅ **Must Understand:**
1. SLAM = Simultaneous Localization + Mapping
2. Sensors: LiDAR, cameras, odometry, IMU
3. Map types: Occupancy grid, feature map
4. Loop closure corrects drift
5. TF frames: map → odom → base_link

✅ **Important for ROS2:**
- `/scan` and `/odom` topics are required
- SLAM provides `map → odom` transform
- Save maps with `map_saver_cli`
- Use SLAM Toolbox or Cartographer

✅ **Key Skills:**
- Run SLAM packages
- Tune parameters
- Save and load maps
- Understand TF relationships

---

## Next Steps

1. Understand sensor data (LaserScan, Odometry)
2. Learn TF2 coordinate frames
3. Run SLAM Toolbox in simulation
4. Practice map building
5. Learn map saving/loading
6. Move to Nav2 for navigation with saved maps
