# SLAM (Simultaneous Localization and Mapping)

## What is SLAM?

**SLAM** = Building a map while figuring out where you are in that map simultaneously.

**The Core Problem:**
- To localize, you need a map
- To map, you need to know your location
- SLAM solves both at the same time!

**Why SLAM Matters:**
- Autonomous vehicles (unknown roads)
- Indoor robots (warehouses, homes)
- GPS-denied areas (buildings, caves, Mars)
- Dynamic environment adaptation

---

## Types of SLAM

### 1. **LiDAR SLAM (2D/3D)**
**Sensor:** Laser scanner  
**Data:** Point clouds (distance measurements)  
**Pros:** Accurate, works in darkness  
**Cons:** Expensive, mechanical parts  
**Use:** Indoor mapping, autonomous vehicles

### 2. **Visual SLAM (vSLAM)**
**Sensor:** Camera (mono/stereo/RGB-D)  
**Data:** Images, visual features  
**Pros:** Cheap, rich information  
**Cons:** Lighting sensitive, computationally heavy  
**Use:** AR/VR, drones, budget robots

### 3. **RGB-D SLAM**
**Sensor:** Depth camera (Kinect, RealSense)  
**Data:** Color + Depth  
**Pros:** Direct depth, good indoors  
**Cons:** Limited range (~10m)  
**Use:** Indoor mapping, 3D reconstruction

### 4. **Sensor Fusion**
**Sensors:** LiDAR + Camera + IMU + Odometry  
**Pros:** Robust, accurate  
**Cons:** Complex, expensive  
**Use:** Self-driving cars, advanced robots

---

## Core Components

### 1. **Pose (Robot Position)**
```
Pose = Position + Orientation
Position: (x, y, z) in meters
Orientation: (roll, pitch, yaw) in radians/degrees

Example: x=2.5m, y=1.3m, yaw=45°
```

### 2. **Map Types**

**Occupancy Grid (2D):**
```
Grid of cells: Free (0), Occupied (100), Unknown (-1)

##########
#........#    # = Wall (100)
#..###...#    . = Free (0)
#........#    ? = Unknown (-1)
##########
```

**Feature Map:**
```
Landmarks: corners, edges, objects
Example: Corner at (5, 3), Door at (10, 7)
```

**Point Cloud (3D):**
```
Collection of 3D points (x, y, z)
Dense representation of environment
```

---

## SLAM Process Flow

### Basic SLAM Loop:

```
1. Prediction (Motion Model)
   Old Pose + Motion Command → Predicted Pose
   
2. Sensor Measurement
   LiDAR/Camera → Observe Environment
   
3. Data Association
   Match current observations to map features
   
4. Pose Correction
   Predicted Pose + Measurements → Corrected Pose
   
5. Map Update
   Add/Update map with new observations
   
6. Loop Closure (Optional)
   Detect revisited places → Correct drift
```

### Detailed Flow:

```python
# Pseudo-code
while robot_active:
    # 1. Predict pose from motion
    predicted_pose = motion_model(prev_pose, odometry)
    
    # 2. Get sensor data
    scan_data = lidar.get_scan()
    
    # 3. Extract features
    features = extract_features(scan_data)
    
    # 4. Match to existing map
    matches = match_features(features, map_features)
    
    # 5. Correct pose using matches
    corrected_pose = update_pose(predicted_pose, matches)
    
    # 6. Update map
    map = update_map(map, corrected_pose, features)
    
    # 7. Check for loop closure
    if loop_detected(corrected_pose, map):
        optimize_graph(poses, map)  # Global correction
    
    prev_pose = corrected_pose
```

---

## Key SLAM Algorithms

### 1. **Graph-Based SLAM** (Most Common)
**How it works:**
- Poses and landmarks = nodes
- Measurements = edges (constraints)
- Optimize entire graph for consistency

**Popular:**
- SLAM Toolbox (ROS2)
- Cartographer (Google)
- RTAB-Map

**Code Example (Conceptual):**
```python
class GraphSLAM:
    def __init__(self):
        self.poses = []        # Robot poses
        self.landmarks = []    # Map features
        self.constraints = []  # Measurements
    
    def add_pose(self, pose):
        self.poses.append(pose)
    
    def add_constraint(self, pose1_id, pose2_id, measurement):
        self.constraints.append((pose1_id, pose2_id, measurement))
    
    def optimize(self):
        # Minimize error across all constraints
        # Using: g2o, Ceres, GTSAM libraries
        optimized_poses = graph_optimize(self.poses, self.constraints)
        return optimized_poses
```

---

### 2. **Particle Filter SLAM (FastSLAM)**
**How it works:**
- Multiple pose hypotheses (particles)
- Each particle has its own map
- Best particles survive

**Code Example:**
```python
class ParticleSLAM:
    def __init__(self, num_particles=100):
        self.particles = [Particle() for _ in range(num_particles)]
    
    def update(self, odometry, scan):
        # 1. Predict: move all particles
        for p in self.particles:
            p.pose = motion_model(p.pose, odometry)
        
        # 2. Update: weight particles by scan match
        for p in self.particles:
            p.weight = scan_likelihood(scan, p.map, p.pose)
        
        # 3. Resample: keep best particles
        self.particles = resample(self.particles)
        
        # 4. Update maps
        for p in self.particles:
            p.map = update_map(p.map, p.pose, scan)
```

---

### 3. **Visual SLAM (ORB-SLAM)**
**How it works:**
- Extract visual features (ORB, SIFT)
- Track features across frames
- Triangulate 3D positions

**Code Example:**
```python
import cv2
import numpy as np

class VisualSLAM:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.prev_keypoints = None
        self.prev_descriptors = None
    
    def process_frame(self, image):
        # 1. Detect features
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        
        if self.prev_descriptors is not None:
            # 2. Match features
            matches = self.matcher.match(descriptors, self.prev_descriptors)
            
            # 3. Estimate motion
            src_pts = np.float32([keypoints[m.queryIdx].pt for m in matches])
            dst_pts = np.float32([self.prev_keypoints[m.trainIdx].pt for m in matches])
            
            # Essential matrix (camera motion)
            E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0,0))
            _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)
            
            # 4. Update pose
            # 5. Update map (triangulate new points)
        
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
```

---

## ROS2 SLAM Implementation

### Installation

```bash
# SLAM Toolbox (2D LiDAR - Best for beginners)
sudo apt install ros-humble-slam-toolbox

# Cartographer (2D/3D LiDAR)
sudo apt install ros-humble-cartographer ros-humble-cartographer-ros

# RTAB-Map (Visual + LiDAR)
sudo apt install ros-humble-rtabmap-ros
```

---

### Basic SLAM Launch

**1. Start SLAM Toolbox:**
```bash
ros2 launch slam_toolbox online_async_launch.py
```

**2. Required Topics:**
```bash
# Input (your robot must publish)
/scan   # sensor_msgs/LaserScan
/odom   # nav_msgs/Odometry
/tf     # TF transforms

# Output (SLAM publishes)
/map    # nav_msgs/OccupancyGrid
```

**3. Visualize:**
```bash
rviz2
# Add: Map, LaserScan, RobotModel, TF
```

**4. Save Map:**
```bash
ros2 run nav2_map_server map_saver_cli -f my_map
```

---

### TF (Transform) Tree

**Critical Concept:**
```
map → odom → base_link → laser_frame

map:        Global fixed frame (SLAM corrects this)
odom:       Odometry frame (drifts over time)
base_link:  Robot center
laser_frame: LiDAR sensor position
```

**Who publishes what:**
- `map → odom`: SLAM (corrects drift)
- `odom → base_link`: Wheel odometry / Robot
- `base_link → laser`: Static transform (sensor mount)

---

### SLAM Configuration (YAML)

**slam_toolbox_params.yaml:**
```yaml
slam_toolbox:
  ros__parameters:
    # Frames
    odom_frame: odom
    map_frame: map
    base_frame: base_footprint
    scan_topic: /scan
    
    # Mode
    mode: mapping  # or localization
    
    # Scan matching
    use_scan_matching: true
    minimum_travel_distance: 0.2  # meters
    minimum_travel_heading: 0.2   # radians
    scan_buffer_size: 10
    
    # Loop closure
    do_loop_closing: true
    loop_search_maximum_distance: 3.0
    loop_match_minimum_chain_size: 10
    
    # Map
    resolution: 0.05  # meters per pixel
    max_laser_range: 20.0
    
    # Performance
    minimum_time_interval: 0.5  # seconds between updates
```

---

### Simple SLAM Node (Python)

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np

class SimpleSLAM(Node):
    def __init__(self):
        super().__init__('simple_slam')
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        # Publishers
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/slam_pose', 10)
        
        # State
        self.pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.map = np.zeros((200, 200), dtype=np.int8)  # 200x200 grid
        self.resolution = 0.1  # meters
        
        self.get_logger().info('SLAM node started')
    
    def odom_callback(self, msg):
        # Update pose estimate from odometry
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # Extract yaw from quaternion (simplified)
        self.pose = np.array([x, y, 0.0])
    
    def scan_callback(self, msg):
        # Simple occupancy grid update
        robot_x = int(self.pose[0] / self.resolution) + 100
        robot_y = int(self.pose[1] / self.resolution) + 100
        
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                # Convert to grid coordinates
                x = robot_x + int((r * np.cos(angle)) / self.resolution)
                y = robot_y + int((r * np.sin(angle)) / self.resolution)
                
                if 0 <= x < 200 and 0 <= y < 200:
                    self.map[y, x] = 100  # Occupied
                
                # Mark cells between robot and obstacle as free
                for i in range(1, int(r / self.resolution)):
                    fx = robot_x + int((i * self.resolution * np.cos(angle)) / self.resolution)
                    fy = robot_y + int((i * self.resolution * np.sin(angle)) / self.resolution)
                    if 0 <= fx < 200 and 0 <= fy < 200:
                        if self.map[fy, fx] == 0:  # Only if unknown
                            self.map[fy, fx] = 0  # Free (actually already 0)
            
            angle += msg.angle_increment
        
        self.publish_map()
    
    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = self.resolution
        msg.info.width = 200
        msg.info.height = 200
        msg.info.origin.position.x = -10.0
        msg.info.origin.position.y = -10.0
        msg.data = self.map.flatten().tolist()
        self.map_pub.publish(msg)

def main():
    rclpy.init()
    slam = SimpleSLAM()
    rclpy.spin(slam)
    slam.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## Loop Closure

**What:** Recognizing you've returned to a previously visited place

**Why:** Corrects accumulated drift

**Visual:**
```
Without Loop Closure:      With Loop Closure:
Start ≠ End (drift)        Start = End (corrected)
  ↓                           ↓
  → → →                       → → →
  ↓   ↑                       ↓   ↑
  ← ← ← (error grows)         ← ← ← (global optimization)
```

**Implementation:**
```python
def detect_loop_closure(current_pose, pose_history, threshold=2.0):
    """Check if current pose is close to any previous pose"""
    for i, prev_pose in enumerate(pose_history[:-10]):  # Skip recent poses
        distance = np.linalg.norm(current_pose[:2] - prev_pose[:2])
        if distance < threshold:
            return True, i  # Loop detected
    return False, -1

# In SLAM loop
if loop_detected:
    optimize_pose_graph()  # Adjust all poses globally
```

---

## Sensor Data Processing

### LiDAR Scan Processing

```python
def process_lidar_scan(scan_msg):
    """Convert LaserScan to Cartesian points"""
    points = []
    angle = scan_msg.angle_min
    
    for r in scan_msg.ranges:
        if scan_msg.range_min < r < scan_msg.range_max:
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append([x, y])
        angle += scan_msg.angle_increment
    
    return np.array(points)

def scan_matching(scan1, scan2):
    """Simple ICP-like scan matching"""
    # Find transformation that best aligns scan2 to scan1
    # Use libraries like: open3d, pcl, or implement ICP
    from scipy.spatial import procrustes
    
    mtx1, mtx2, disparity = procrustes(scan1, scan2)
    return mtx2  # Transformed scan2
```

---

### Feature Extraction (Visual SLAM)

```python
def extract_visual_features(image):
    """Extract ORB features from image"""
    import cv2
    
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    # Convert to numpy array
    points = np.array([kp.pt for kp in keypoints])
    
    return points, descriptors

def match_features(desc1, desc2, ratio_threshold=0.75):
    """Match features between two frames"""
    import cv2
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    
    return good_matches
```

---

## Map Saving/Loading

### Save Map

```bash
# Command line
ros2 run nav2_map_server map_saver_cli -f my_map

# Programmatically
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '/path/to/map'}"
```

**Generates:**
- `my_map.pgm` - Grayscale image (map)
- `my_map.yaml` - Metadata

**YAML Format:**
```yaml
image: my_map.pgm
resolution: 0.05
origin: [-10.0, -10.0, 0.0]
occupied_thresh: 0.65
free_thresh: 0.25
negate: 0
```

---

### Load Map

```python
import yaml
from PIL import Image
import numpy as np

def load_map(yaml_file):
    """Load occupancy grid map"""
    with open(yaml_file, 'r') as f:
        map_metadata = yaml.safe_load(f)
    
    # Load image
    image_path = map_metadata['image']
    img = Image.open(image_path)
    map_data = np.array(img)
    
    # Convert to occupancy values
    # White (255) = Free (0)
    # Black (0) = Occupied (100)
    # Gray = Unknown (-1)
    occupancy_grid = 100 - (map_data / 255.0 * 100)
    
    return occupancy_grid, map_metadata
```

---
