# Robotics Transforms Fundamentals

Essential coordinate frame transformations for ROS2 robotics development.

## Core Concept

Every sensor, actuator, and reference point in a robot has its own **coordinate frame**. Transforms define spatial relationships between these frames—enabling questions like "where is the detected object relative to the robot base?"

## TF2 Overview

ROS2's transform library for tracking coordinate frames over time.

**Common frames:** `map` → `odom` → `base_link` → `sensor_link`

```
map (global fixed)
 └── odom (odometry origin)
      └── base_link (robot center)
           ├── camera_link
           ├── lidar_link
           └── arm_base_link
```

## Transform Representations

### 1. Homogeneous Transformation Matrix (4×4)

Combines rotation and translation in single matrix. Used internally by TF2.

```
| R11 R12 R13 Tx |
| R21 R22 R23 Ty |
| R31 R32 R33 Tz |
|  0   0   0   1 |
```

- **R (3×3):** Rotation matrix
- **T (3×1):** Translation vector
- Chain transforms via matrix multiplication: `T_map_to_gripper = T_map_to_base @ T_base_to_arm @ T_arm_to_gripper`

### 2. Quaternion (Rotation)

4D representation: `[x, y, z, w]`

```python
# Identity (no rotation)
rotation.x = 0.0
rotation.y = 0.0
rotation.z = 0.0
rotation.w = 1.0
```

**Why quaternions?**
- No gimbal lock
- Smooth interpolation (SLERP)
- Numerically stable
- Compact (4 values vs 9 for rotation matrix)

**Constraint:** Must be normalized → `x² + y² + z² + w² = 1`

### 3. Euler Angles

Human-readable: Roll (X), Pitch (Y), Yaw (Z)

```python
from tf_transformations import quaternion_from_euler, euler_from_quaternion

# Euler → Quaternion (for ROS2 messages)
q = quaternion_from_euler(roll, pitch, yaw)  # radians

# Quaternion → Euler (for debugging/display)
roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
```

**⚠️ Gimbal lock:** Avoid for computation; use only for human I/O.

### 4. Translation Vector

Position offset between frame origins: `[x, y, z]` in meters.

```python
transform.transform.translation.x = 0.5   # 50cm forward
transform.transform.translation.y = -0.1  # 10cm right
transform.transform.translation.z = 0.2   # 20cm up
```

## TF2 API Essentials

### Publishing Transforms

```python
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

# Dynamic (moving) transforms
broadcaster = TransformBroadcaster(node)
broadcaster.sendTransform(transform_stamped)

# Static (fixed) transforms - use for sensor mounts
static_broadcaster = StaticTransformBroadcaster(node)
static_broadcaster.sendTransform(transform_stamped)
```

### Listening to Transforms

```python
from tf2_ros import Buffer, TransformListener

tf_buffer = Buffer()
tf_listener = TransformListener(tf_buffer, node)

# Lookup transform (blocking)
try:
    trans = tf_buffer.lookup_transform(
        'base_link',      # target frame
        'camera_link',    # source frame
        rclpy.time.Time() # latest available
    )
except TransformException as e:
    node.get_logger().warn(f'Transform failed: {e}')
```

### Transforming Data

```python
from tf2_geometry_msgs import do_transform_point

# Transform point from camera frame to base frame
point_in_base = do_transform_point(point_in_camera, transform)
```

## tf_transformations Utilities

```python
import tf_transformations as tf

# Conversions
tf.quaternion_from_euler(r, p, y)
tf.euler_from_quaternion(q)
tf.quaternion_from_matrix(matrix_4x4)
tf.quaternion_matrix(q)  # → 4x4 matrix

# Operations
tf.quaternion_multiply(q1, q2)
tf.quaternion_inverse(q)
tf.quaternion_slerp(q0, q1, fraction)  # interpolation

# Inspect available functions
print(dir(tf_transformations))
help(tf_transformations.quaternion_from_euler)
```

## Common Patterns

### Static Sensor Transform (URDF/Launch)

```xml
<node pkg="tf2_ros" exec="static_transform_publisher" 
      args="0.1 0 0.2 0 0 0 base_link camera_link"/>
<!--    Tx Ty Tz Yaw Pitch Roll parent child -->
```

### Transform Stamped Message

```python
t = TransformStamped()
t.header.stamp = node.get_clock().now().to_msg()
t.header.frame_id = 'odom'        # parent
t.child_frame_id = 'base_link'    # child

t.transform.translation.x = 1.0
t.transform.translation.y = 0.0
t.transform.translation.z = 0.0

t.transform.rotation.x = 0.0
t.transform.rotation.y = 0.0
t.transform.rotation.z = 0.0
t.transform.rotation.w = 1.0
```

## Debugging

```bash
# View TF tree
ros2 run tf2_tools view_frames

# Echo specific transform
ros2 run tf2_ros tf2_echo base_link camera_link

# Monitor TF broadcast rate
ros2 run tf2_ros tf2_monitor
```

## Key Rules

1. **Frame naming:** Use `_link` suffix for rigid bodies, `_frame` for virtual references
2. **Timestamp:** Always use sensor timestamp, not processing time
3. **Tree structure:** TF2 requires single-parent tree (no cycles, no multiple parents)
4. **Units:** Meters for translation, radians for Euler angles
5. **Convention:** Right-hand rule; X-forward, Y-left, Z-up (REP-103)

## References

- [REP-103: Standard Units](https://www.ros.org/reps/rep-0103.html)
- [REP-105: Coordinate Frames](https://www.ros.org/reps/rep-0105.html)
- [TF2 Tutorials](https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Tf2-Main.html)
