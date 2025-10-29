# Nav2 Fundamentals
**ROS2 Navigation Stack**

## What is Nav2?

**Nav2 (Navigation2)** is the ROS2 navigation framework that enables robots to autonomously move from point A to point B while avoiding obstacles.

### What Nav2 Does:
- Plans paths around obstacles
- Follows paths accurately
- Avoids dynamic obstacles
- Recovers from failures
- Manages navigation behaviors

### What You Need Before Nav2:
1. ✅ Working robot (physical or simulated)
2. ✅ Map of the environment (from SLAM)
3. ✅ Localization (knowing where robot is)
4. ✅ Sensor data (LaserScan, camera, etc.)
5. ✅ Odometry (wheel encoders)

---

## Core Concepts

### 1. Costmaps

**What is a Costmap?**
A map that represents how "dangerous" or "costly" it is to be in each location.

**Cost Values:**
- **0**: Free space (safe)
- **1-252**: Inflated cost (near obstacles)
- **253**: Inscribed (robot barely fits)
- **254**: Lethal (obstacle)
- **255**: Unknown

**Visual Representation:**
```
Black = Lethal obstacle
Red = High cost (close to obstacle)
Yellow = Medium cost
Green = Low cost
White = Free space
```

**Why Inflation?**
- Robot has physical size
- Safety buffer around obstacles
- Smooth paths (not right against walls)

### Two Costmaps:

#### Global Costmap
- **Purpose**: Long-term path planning
- **Updates**: Slow (1-2 Hz)
- **Size**: Large (entire map or large area)
- **Used by**: Global planner

#### Local Costmap
- **Purpose**: Short-term obstacle avoidance
- **Updates**: Fast (5-10 Hz)
- **Size**: Small (just around robot)
- **Used by**: Local planner

---

### 2. Planning

#### Global Planner
**What it does:** Plans path from start to goal considering the full map.

**Algorithm Options:**
- **NavFn**: Classic Dijkstra-based
- **Smac Planner**: Hybrid A*, considers robot kinematics
- **Theta Star**: Any-angle paths (smoother)

**Output:** Complete path from current position to goal

**Example:**
```
Start: (0, 0)
Goal: (10, 10)
Output: [(0,0), (1,1), (2,2), ... (10,10)]
```

#### Local Planner (Controller)
**What it does:** Follows global path while avoiding new obstacles.

**Popular Controllers:**
- **DWB (Dynamic Window Approach)**: Fast, smooth
- **TEB (Timed Elastic Band)**: Considers dynamics
- **RPP (Regulated Pure Pursuit)**: Simple, reliable
- **MPPI**: Model Predictive Path Integral

**Output:** Velocity commands (Twist message)

**Example:**
```
linear.x: 0.5 m/s  (forward speed)
angular.z: 0.2 rad/s  (rotation speed)
```

---

### 3. Behavior Tree

**What is a Behavior Tree?**
A decision-making structure that controls navigation logic.

**Nav2 Default Tree:**
```
Navigation
├── Compute Path (global planning)
├── Follow Path (local control)
├── Recovery Actions
│   ├── Clear Costmap
│   ├── Spin
│   ├── Back Up
│   └── Wait
└── Goal Checker
```

**How it Works:**
1. Compute path to goal
2. Follow path
3. If stuck → Try recovery behaviors
4. If recovered → Continue
5. If goal reached → Success!

---

### 4. Localization (AMCL)

**AMCL (Adaptive Monte Carlo Localization)**
Determines robot's position on a known map.

**How it Works:**
1. Start with many pose guesses (particles)
2. Move robot → update all guesses
3. Get sensor reading → weight guesses by how well they match
4. Keep best guesses, discard bad ones
5. Converge to true pose

**Visual:**
```
Initial: Many particles spread across map
After movement: Particles follow motion
After sensor update: Particles cluster where likely
Converged: Tight cluster at true position
```

**Why AMCL Matters:**
- Nav2 needs to know where robot is
- Corrects odometry drift
- Updates continuously during navigation

---

## Nav2 Architecture

### Key Nodes:

1. **bt_navigator** - Behavior tree executor
2. **planner_server** - Global path planning
3. **controller_server** - Local path following
4. **recoveries_server** - Recovery behaviors
5. **waypoint_follower** - Multi-goal navigation
6. **lifecycle_manager** - Node management

### Data Flow:
```
User → Goal → BT Navigator
              ↓
        Planner Server → Global Path
              ↓
        Controller Server → Velocity Commands → Robot
              ↑
        Costmap (Sensors + Map + Odometry)
```

---

## Required Topics & TF Frames

### Input Topics:
```bash
/scan          # sensor_msgs/LaserScan
/odom          # nav_msgs/Odometry
/map           # nav_msgs/OccupancyGrid (from map_server)
```

### Output Topics:
```bash
/cmd_vel       # geometry_msgs/Twist (velocity commands)
/plan          # nav_msgs/Path (planned path)
/global_costmap/costmap  # Costmaps for visualization
/local_costmap/costmap
```

### Required TF Frames:
```
map → odom → base_link → base_footprint
              ↓
         (sensors: laser, camera, etc.)
```

**Who Provides What:**
- `map → odom`: AMCL (localization)
- `odom → base_link`: Robot odometry
- `base_link → sensors`: Static transforms (URDF)

---

## Setting Up Nav2

### 1. Install Nav2
```bash
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup
```

### 2. Create Nav2 Parameters File
```yaml
# nav2_params.yaml
bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link

controller_server:
  ros__parameters:
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.0
    min_theta_velocity_threshold: 0.001
    
    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"
      max_vel_x: 0.5
      min_vel_x: -0.2
      max_vel_theta: 1.0
      min_speed_xy: 0.0

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.3

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      robot_radius: 0.3
      resolution: 0.05
```

### 3. Launch Nav2
```bash
ros2 launch nav2_bringup navigation_launch.py \
  params_file:=/path/to/nav2_params.yaml \
  map:=/path/to/my_map.yaml
```

---

## Using Nav2

### Method 1: RViz
1. Open RViz
2. Add "Nav2 Goal" tool
3. Click "2D Goal Pose" button
4. Click on map to set goal
5. Robot navigates automatically!

### Method 2: Command Line
```bash
ros2 action send_goal /navigate_to_pose \
  nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, 
   pose: {position: {x: 2.0, y: 3.0}, 
   orientation: {w: 1.0}}}}"
```

### Method 3: Python Node
```python
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped

def create_pose(x, y, yaw=0.0):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.w = 1.0  # Simplified
    return pose

navigator = BasicNavigator()
goal_pose = create_pose(2.0, 3.0)

navigator.goToPose(goal_pose)

while not navigator.isTaskComplete():
    feedback = navigator.getFeedback()
    print(f"Distance remaining: {feedback.distance_remaining}")

result = navigator.getResult()
if result == TaskResult.SUCCEEDED:
    print('Goal reached!')
```

---

## Navigation Behaviors

### 1. Navigate to Pose
- **Goal**: Single position and orientation
- **Use**: Go to specific location

### 2. Navigate Through Poses (Waypoints)
- **Goal**: List of poses
- **Use**: Patrol route, complex paths

```python
waypoints = [
    create_pose(1.0, 1.0),
    create_pose(2.0, 3.0),
    create_pose(0.0, 0.0)
]
navigator.followWaypoints(waypoints)
```

### 3. Spin
- **Goal**: Rotate in place
- **Use**: Look around, recovery behavior

### 4. Back Up
- **Goal**: Move backward
- **Use**: Get out of tight spots

---

## Recovery Behaviors

When robot gets stuck, Nav2 tries:

1. **Clear Costmap** - Reset costmap data
2. **Spin** - Rotate to get new sensor data
3. **Back Up** - Move backward
4. **Wait** - Pause, let environment change

**Configurable:**
```yaml
recoveries_server:
  ros__parameters:
    recovery_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_recoveries/Spin"
      simulation_time_step: 0.02
      max_rotational_vel: 1.0
    backup:
      plugin: "nav2_recoveries/BackUp"
      backup_dist: 0.3
      backup_speed: 0.15
```

---

## Tuning Nav2

### Critical Parameters:

#### Robot Dimensions
```yaml
robot_radius: 0.3          # For circular robot
# OR
footprint: "[[0.2, 0.2], [0.2, -0.2], [-0.2, -0.2], [-0.2, 0.2]]"
```

#### Velocity Limits
```yaml
max_vel_x: 0.5            # Max forward speed (m/s)
min_vel_x: -0.2           # Max backward speed
max_vel_theta: 1.0        # Max rotation speed (rad/s)
```

#### Costmap Inflation
```yaml
inflation_radius: 0.55     # How far to inflate obstacles
cost_scaling_factor: 3.0   # How fast cost decreases
```

#### Controller Tuning
```yaml
controller_frequency: 20.0  # Hz - higher = smoother
lookahead_dist: 0.5        # How far ahead to look
```

---

## Common Issues & Solutions

### Issue: Robot Doesn't Move
**Check:**
- Is Nav2 getting /scan and /odom?
- Is map → odom → base_link TF chain complete?
- Are velocity limits too restrictive?
- Is robot_radius correct?

### Issue: Robot Stuck on Obstacle
**Solutions:**
- Increase inflation_radius
- Tune recovery behaviors
- Check sensor data quality
- Reduce speed

### Issue: Path Goes Through Obstacles
**Solutions:**
- Update costmap more frequently
- Check sensor mounting/TF
- Increase obstacle cost
- Verify map accuracy

### Issue: Jerky Motion
**Solutions:**
- Increase controller_frequency
- Tune velocity limits
- Smooth planner output
- Check odometry quality

### Issue: Can't Reach Goal
**Solutions:**
- Increase goal tolerance
- Check if goal is in obstacle
- Verify goal is reachable
- Check costmap configuration

---

## Nav2 vs Nav1 (ROS1)

### Key Improvements:
- **Behavior Trees** (vs fixed state machine)
- **Plugin-based** (easy to customize)
- **Better recovery** (multiple strategies)
- **Multi-robot ready**
- **Lifecycle management** (better startup/shutdown)

---

## Essential Nav2 Concepts Summary

✅ **Must Understand:**
1. Costmaps - Represent environment danger
2. Global planner - Long-term path
3. Local controller - Short-term control
4. Behavior tree - Decision logic
5. AMCL - Localization on known map
6. Recovery behaviors - Handle failures

✅ **Important for Setup:**
- Accurate robot dimensions (radius/footprint)
- Good sensor data (/scan, /odom)
- Complete TF tree (map → odom → base_link)
- Tuned velocity limits
- Proper costmap inflation

✅ **Key Skills:**
- Configure nav2_params.yaml
- Set navigation goals
- Tune costmaps
- Debug with RViz
- Handle navigation failures

---

## Nav2 Workflow

### One-Time Setup:
1. Build map with SLAM
2. Save map
3. Configure Nav2 parameters
4. Define robot URDF/TF

### Every Run:
1. Start robot
2. Load map (map_server)
3. Start AMCL (localization)
4. Start Nav2
5. Set initial pose (if needed)
6. Send navigation goals!

---

## Visualization in RViz

**Essential displays:**
- Map - The occupancy grid
- Costmap - Global and local
- Path - Planned path
- LaserScan - Sensor data
- TF - Coordinate frames
- RobotModel - Visual representation

**Interactive tools:**
- 2D Pose Estimate - Set robot initial position
- 2D Goal Pose - Set navigation goal
- Nav2 Goal - Advanced goal setting

---

## Advanced Topics

### Custom Planners & Controllers
- Write your own planning algorithm
- Implement as Nav2 plugin
- Configure in params file

### Behavior Tree Customization
- Create custom behaviors
- Modify decision logic
- Add new recovery strategies

### Multi-Robot Navigation
- Separate namespace for each robot
- Shared or independent maps
- Collision avoidance between robots

### Dynamic Obstacles
- Temporal costmap layers
- Predictive planning
- Social navigation

---

## Best Practices

✅ **Do:**
- Start with default parameters
- Tune one parameter at a time
- Test in simulation first
- Visualize everything in RViz
- Log navigation data
- Provide good odometry

❌ **Don't:**
- Set unrealistic velocity limits
- Make robot_radius too small
- Forget to tune inflation
- Skip localization verification
- Ignore TF warnings

---

## Debugging Tools

### Command Line:
```bash
# Check topics
ros2 topic list | grep nav

# Echo costmap
ros2 topic echo /local_costmap/costmap

# Check TF
ros2 run tf2_tools view_frames

# Send test goal
ros2 action send_goal /navigate_to_pose ...
```

### RViz Displays:
- View costmaps
- See planned paths
- Monitor robot pose
- Check sensor data

### Logs:
```bash
ros2 launch nav2_bringup navigation_launch.py log_level:=debug
```

---

## Next Steps

1. Complete SLAM to get a map
2. Install and configure Nav2
3. Test in simulation (Gazebo + Nav2)
4. Tune basic parameters (robot size, velocities)
5. Test single-goal navigation
6. Try waypoint navigation
7. Move to real robot

**Pro Tip:** Good localization = Good navigation. Bad odometry = Bad navigation!

---

## Resources

- Nav2 Documentation: navigation.ros.org
- Parameters Guide: docs.nav2.org/configuration
- Tutorials: docs.nav2.org/tutorials
- Community: discourse.ros.org

**Remember:** Nav2 is complex, but start simple and build up!
