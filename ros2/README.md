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

#### C++ Example: Basic Node Structure
```cpp
#include "rclcpp/rclcpp.hpp"

class MinimalNode : public rclcpp::Node
{
public:
  MinimalNode() : Node("minimal_node")
  {
    RCLCPP_INFO(this->get_logger(), "Node started!");
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalNode>());
  rclcpp::shutdown();
  return 0;
}
```

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

#### C++ Example: Simple Publisher
```cpp
#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"

using namespace std::chrono_literals;

class VelocityPublisher : public rclcpp::Node
{
public:
  VelocityPublisher() : Node("velocity_publisher")
  {
    publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    timer_ = this->create_wall_timer(
      100ms, std::bind(&VelocityPublisher::timer_callback, this));
    RCLCPP_INFO(this->get_logger(), "Velocity publisher started");
  }

private:
  void timer_callback()
  {
    auto msg = geometry_msgs::msg::Twist();
    msg.linear.x = 0.2;  // Move forward at 0.2 m/s
    msg.angular.z = 0.0; // No rotation
    publisher_->publish(msg);
  }

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VelocityPublisher>());
  rclcpp::shutdown();
  return 0;
}
```

#### C++ Example: Simple Subscriber
```cpp
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

class LaserSubscriber : public rclcpp::Node
{
public:
  LaserSubscriber() : Node("laser_subscriber")
  {
    subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10, std::bind(&LaserSubscriber::scan_callback, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Laser subscriber started");
  }

private:
  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    float min_distance = *std::min_element(msg->ranges.begin(), msg->ranges.end());
    
    if (min_distance < 0.5) {
      RCLCPP_WARN(this->get_logger(), "Obstacle detected at %.2f meters!", min_distance);
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LaserSubscriber>());
  rclcpp::shutdown();
  return 0;
}
```

#### C++ Example: Publisher + Subscriber (Obstacle Avoider)
```cpp
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "geometry_msgs/msg/twist.hpp"

class ObstacleAvoider : public rclcpp::Node
{
public:
  ObstacleAvoider() : Node("obstacle_avoider")
  {
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10, std::bind(&ObstacleAvoider::scan_callback, this, std::placeholders::_1));
    vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    RCLCPP_INFO(this->get_logger(), "Obstacle avoider started");
  }

private:
  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg)
  {
    auto vel_msg = geometry_msgs::msg::Twist();
    
    // Check front center (middle 30% of scan)
    size_t center_start = scan_msg->ranges.size() * 0.35;
    size_t center_end = scan_msg->ranges.size() * 0.65;
    
    float min_front_distance = std::numeric_limits<float>::max();
    for (size_t i = center_start; i < center_end; ++i) {
      if (scan_msg->ranges[i] < min_front_distance) {
        min_front_distance = scan_msg->ranges[i];
      }
    }
    
    // Simple avoidance logic
    if (min_front_distance > 0.8) {
      vel_msg.linear.x = 0.2;
      vel_msg.angular.z = 0.0;
    } else {
      vel_msg.linear.x = 0.0;
      vel_msg.angular.z = 0.5;  // Turn left
      RCLCPP_INFO(this->get_logger(), "Avoiding obstacle at %.2f m", min_front_distance);
    }
    
    vel_pub_->publish(vel_msg);
  }

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr vel_pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObstacleAvoider>());
  rclcpp::shutdown();
  return 0;
}
```

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

#### C++ Example: Service Server
```cpp
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/set_bool.hpp"

class SimpleServiceServer : public rclcpp::Node
{
public:
  SimpleServiceServer() : Node("service_server"), is_enabled_(false)
  {
    service_ = this->create_service<std_srvs::srv::SetBool>(
      "enable_system",
      std::bind(&SimpleServiceServer::handle_service, this, 
                std::placeholders::_1, std::placeholders::_2));
    RCLCPP_INFO(this->get_logger(), "Service server ready");
  }

private:
  void handle_service(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response)
  {
    is_enabled_ = request->data;
    response->success = true;
    response->message = is_enabled_ ? "System enabled" : "System disabled";
    RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
  }

  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr service_;
  bool is_enabled_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimpleServiceServer>());
  rclcpp::shutdown();
  return 0;
}
```

#### C++ Example: Service Client
```cpp
#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/set_bool.hpp"

using namespace std::chrono_literals;

class SimpleServiceClient : public rclcpp::Node
{
public:
  SimpleServiceClient() : Node("service_client")
  {
    client_ = this->create_client<std_srvs::srv::SetBool>("enable_system");
    
    while (!client_->wait_for_service(1s)) {
      RCLCPP_INFO(this->get_logger(), "Waiting for service...");
    }
    
    call_service(true);
  }

private:
  void call_service(bool enable)
  {
    auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
    request->data = enable;
    
    auto result = client_->async_send_request(request);
    
    if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), result) ==
        rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_INFO(this->get_logger(), "Response: %s", result.get()->message.c_str());
    }
  }

  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr client_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SimpleServiceClient>();
  rclcpp::shutdown();
  return 0;
}
```

**Call service from terminal:**
```bash
ros2 service call /enable_system std_srvs/srv/SetBool "{data: true}"
```

---

### 5. Actions (Long-Running Tasks)
**What is an Action?**
Like a service, but for tasks that take time with progress feedback.

**Components:**
- **Goal**: What you want
- **Feedback**: Progress updates
- **Result**: Final outcome

**When to Use Actions:**
- Navigation to a goal
- Picking up an object
- Any task that takes >1 second

#### C++ Example: Action Server (Fibonacci)
```cpp
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "example_interfaces/action/fibonacci.hpp"

class FibonacciActionServer : public rclcpp::Node
{
public:
  using Fibonacci = example_interfaces::action::Fibonacci;
  using GoalHandleFibonacci = rclcpp_action::ServerGoalHandle<Fibonacci>;

  FibonacciActionServer() : Node("fibonacci_action_server")
  {
    action_server_ = rclcpp_action::create_server<Fibonacci>(
      this, "fibonacci",
      std::bind(&FibonacciActionServer::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&FibonacciActionServer::handle_cancel, this, std::placeholders::_1),
      std::bind(&FibonacciActionServer::handle_accepted, this, std::placeholders::_1));
    
    RCLCPP_INFO(this->get_logger(), "Action server ready");
  }

private:
  rclcpp_action::Server<Fibonacci>::SharedPtr action_server_;

  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const Fibonacci::Goal> goal)
  {
    RCLCPP_INFO(this->get_logger(), "Received goal request with order %d", goal->order);
    (void)uuid;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleFibonacci> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
    (void)goal_handle;
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandleFibonacci> goal_handle)
  {
    std::thread{std::bind(&FibonacciActionServer::execute, this, std::placeholders::_1), goal_handle}.detach();
  }

  void execute(const std::shared_ptr<GoalHandleFibonacci> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing goal");
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<Fibonacci::Feedback>();
    auto result = std::make_shared<Fibonacci::Result>();
    
    auto & sequence = feedback->sequence;
    sequence.push_back(0);
    sequence.push_back(1);

    for (int i = 1; (i < goal->order) && rclcpp::ok(); ++i) {
      if (goal_handle->is_canceling()) {
        result->sequence = sequence;
        goal_handle->canceled(result);
        RCLCPP_INFO(this->get_logger(), "Goal canceled");
        return;
      }
      
      sequence.push_back(sequence[i] + sequence[i - 1]);
      goal_handle->publish_feedback(feedback);
      RCLCPP_INFO(this->get_logger(), "Publishing feedback");
      
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    if (rclcpp::ok()) {
      result->sequence = sequence;
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(), "Goal succeeded");
    }
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FibonacciActionServer>());
  rclcpp::shutdown();
  return 0;
}
```

#### C++ Example: Action Client
```cpp
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "example_interfaces/action/fibonacci.hpp"

class FibonacciActionClient : public rclcpp::Node
{
public:
  using Fibonacci = example_interfaces::action::Fibonacci;
  using GoalHandleFibonacci = rclcpp_action::ClientGoalHandle<Fibonacci>;

  FibonacciActionClient() : Node("fibonacci_action_client")
  {
    client_ = rclcpp_action::create_client<Fibonacci>(this, "fibonacci");
    
    while (!client_->wait_for_action_server(std::chrono::seconds(1))) {
      RCLCPP_INFO(this->get_logger(), "Waiting for action server...");
    }
    
    send_goal();
  }

private:
  rclcpp_action::Client<Fibonacci>::SharedPtr client_;

  void send_goal()
  {
    auto goal_msg = Fibonacci::Goal();
    goal_msg.order = 10;

    RCLCPP_INFO(this->get_logger(), "Sending goal");

    auto send_goal_options = rclcpp_action::Client<Fibonacci>::SendGoalOptions();
    send_goal_options.feedback_callback =
      std::bind(&FibonacciActionClient::feedback_callback, this, std::placeholders::_1, std::placeholders::_2);
    send_goal_options.result_callback =
      std::bind(&FibonacciActionClient::result_callback, this, std::placeholders::_1);

    client_->async_send_goal(goal_msg, send_goal_options);
  }

  void feedback_callback(
    GoalHandleFibonacci::SharedPtr,
    const std::shared_ptr<const Fibonacci::Feedback> feedback)
  {
    RCLCPP_INFO(this->get_logger(), "Feedback: Latest value = %d", feedback->sequence.back());
  }

  void result_callback(const GoalHandleFibonacci::WrappedResult & result)
  {
    switch (result.code) {
      case rclcpp_action::ResultCode::SUCCEEDED:
        RCLCPP_INFO(this->get_logger(), "Goal succeeded!");
        break;
      case rclcpp_action::ResultCode::ABORTED:
        RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
        return;
      case rclcpp_action::ResultCode::CANCELED:
        RCLCPP_ERROR(this->get_logger(), "Goal was canceled");
        return;
      default:
        RCLCPP_ERROR(this->get_logger(), "Unknown result code");
        return;
    }
    
    RCLCPP_INFO(this->get_logger(), "Result received with %zu values", result.result->sequence.size());
    rclcpp::shutdown();
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FibonacciActionClient>());
  return 0;
}
```

**Action commands:**
```bash
# List actions
ros2 action list

# Send action goal from terminal
ros2 action send_goal /fibonacci example_interfaces/action/Fibonacci "{order: 5}"
```

---

### 6. Parameters
**What are Parameters?**
Configuration values changeable at runtime.

**Examples:**
- Maximum speed
- PID gains
- Update rates

#### C++ Example: Using Parameters
```cpp
#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"

using namespace std::chrono_literals;

class ConfigurablePublisher : public rclcpp::Node
{
public:
  ConfigurablePublisher() : Node("configurable_publisher")
  {
    // Declare parameters with defaults
    this->declare_parameter("linear_speed", 0.2);
    this->declare_parameter("angular_speed", 0.5);
    this->declare_parameter("publish_rate", 10);
    
    // Get parameter values
    linear_speed_ = this->get_parameter("linear_speed").as_double();
    angular_speed_ = this->get_parameter("angular_speed").as_double();
    int rate = this->get_parameter("publish_rate").as_int();
    
    publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    
    auto period = std::chrono::milliseconds(1000 / rate);
    timer_ = this->create_wall_timer(
      period, std::bind(&ConfigurablePublisher::timer_callback, this));
    
    RCLCPP_INFO(this->get_logger(), "Started with linear: %.2f, angular: %.2f, rate: %d Hz",
                linear_speed_, angular_speed_, rate);
  }

private:
  void timer_callback()
  {
    auto msg = geometry_msgs::msg::Twist();
    msg.linear.x = linear_speed_;
    msg.angular.z = angular_speed_;
    publisher_->publish(msg);
  }

  double linear_speed_;
  double angular_speed_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ConfigurablePublisher>());
  rclcpp::shutdown();
  return 0;
}
```

**Run with custom parameters:**
```bash
ros2 run my_package configurable_publisher --ros-args -p linear_speed:=0.5 -p angular_speed:=1.0
```

**Get/Set parameters at runtime:**
```bash
ros2 param list /configurable_publisher
ros2 param get /configurable_publisher linear_speed
ros2 param set /configurable_publisher linear_speed 0.3
```

---

### 7. Launch Files
**What are Launch Files?**
Python scripts to start multiple nodes with configurations.

**Why Use Launch Files?**
- Start entire system with one command
- Set parameters for multiple nodes
- Organize complex robot systems

#### Python Example: Basic Launch File
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='velocity_publisher',
            name='vel_pub',
            output='screen',
            parameters=[{
                'linear_speed': 0.3,
                'angular_speed': 0.5
            }]
        ),
        Node(
            package='my_package',
            executable='laser_subscriber',
            name='laser_sub',
            output='screen'
        ),
    ])
```

**Save as:** `my_package/launch/my_robot.launch.py`

#### Python Example: Launch with Parameters from YAML
```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('my_package'),
        'config',
        'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='my_package',
            executable='configurable_publisher',
            name='config_pub',
            parameters=[config]
        ),
    ])
```

**YAML file** (`config/params.yaml`):
```yaml
configurable_publisher:
  ros__parameters:
    linear_speed: 0.5
    angular_speed: 1.0
    publish_rate: 20
```

**Run launch file:**
```bash
ros2 launch my_package my_robot.launch.py
```

**Update CMakeLists.txt:**
```cmake
# Install launch files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)
```

---

### 8. TF2 (Coordinate Frame Transformations)
**What is TF2?**
Manages coordinate frames and transformations between them.

**Why TF2?**
- Know where sensors are relative to robot
- Transform data between coordinate frames
- Handle moving parts (cameras, arms)

**Frame Tree Example:**
```
world → map → odom → base_link → camera_link
  ↑       ↑      ↑        ↑           ↑
Fixed   Fixed  Moving  Robot      Sensor
```

**Common Frames:**
- `map` - Global fixed frame
- `odom` - Odometry frame (drifts over time)
- `base_link` - Robot center
- `laser_link`, `camera_link` - Sensor frames

#### C++ Example: TF2 Listener
```cpp
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

class TFListener : public rclcpp::Node
{
public:
  TFListener() : Node("tf_listener")
  {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&TFListener::timer_callback, this));
  }

private:
  void timer_callback()
  {
    try {
      // Get transform from laser to base_link
      auto transform = tf_buffer_->lookupTransform(
        "base_link", "laser",
        tf2::TimePointZero);
      
      RCLCPP_INFO(this->get_logger(), 
        "Transform: x=%.2f, y=%.2f, z=%.2f",
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z);
        
    } catch (tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "Could not transform: %s", ex.what());
    }
  }

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TFListener>());
  rclcpp::shutdown();
  return 0;
}
```

#### C++ Example: TF2 Broadcaster
```cpp
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

class TFBroadcaster : public rclcpp::Node
{
public:
  TFBroadcaster() : Node("tf_broadcaster")
  {
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&TFBroadcaster::broadcast_transform, this));
  }

private:
  void broadcast_transform()
  {
    geometry_msgs::msg::TransformStamped transform;
    
    transform.header.stamp = this->now();
    transform.header.frame_id = "base_link";
    transform.child_frame_id = "sensor_link";
    
    // Position: 0.1m forward, 0.2m up
    transform.transform.translation.x = 0.1;
    transform.transform.translation.y = 0.0;
    transform.transform.translation.z = 0.2;
    
    // No rotation (identity quaternion)
    transform.transform.rotation.x = 0.0;
    transform.transform.rotation.y = 0.0;
    transform.transform.rotation.z = 0.0;
    transform.transform.rotation.w = 1.0;
    
    tf_broadcaster_->sendTransform(transform);
  }

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TFBroadcaster>());
  rclcpp::shutdown();
  return 0;
}
```

**TF Commands:**
```bash
# View TF tree
ros2 run tf2_tools view_frames

# Echo transform between frames
ros2 run tf2_ros tf2_echo base_link camera_link

# Visualize in RViz2
rviz2
# Add TF display
```

---

### 9. Custom Messages
**What are Custom Messages?**
Define your own message types for specific data.

**When to Use:**
- Standard messages don't fit your needs
- Complex data structures
- Domain-specific data

#### Example: Create Custom Message

**1. Create message file** (`msg/SensorData.msg`):
```
# Custom sensor data message
Header header
float32 temperature
float32 humidity
float32 pressure
bool is_valid
```

**2. Update `package.xml`:**
```xml
<build_depend>rosidl_default_generators</build_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

**3. Update `CMakeLists.txt`:**
```cmake
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SensorData.msg"
  DEPENDENCIES std_msgs
)
```

**4. Use in C++ code:**
```cpp
#include "rclcpp/rclcpp.hpp"
#include "my_package/msg/sensor_data.hpp"

class CustomPublisher : public rclcpp::Node
{
public:
  CustomPublisher() : Node("custom_publisher")
  {
    publisher_ = this->create_publisher<my_package::msg::SensorData>("sensor_data", 10);
    timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&CustomPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    auto msg = my_package::msg::SensorData();
    msg.header.stamp = this->now();
    msg.header.frame_id = "sensor_frame";
    msg.temperature = 25.5;
    msg.humidity = 60.0;
    msg.pressure = 1013.25;
    msg.is_valid = true;
    
    publisher_->publish(msg);
    RCLCPP_INFO(this->get_logger(), "Published sensor data");
  }

  rclcpp::Publisher<my_package::msg::SensorData>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CustomPublisher>());
  rclcpp::shutdown();
  return 0;
}
```

**Check your custom message:**
```bash
ros2 interface show my_package/msg/SensorData
ros2 topic echo /sensor_data
```

---

### 10. MoveIt2 (Robot Arm Manipulation)
**What is MoveIt2?**
Motion planning framework for robot arms.

**Key Features:**
- Path planning (avoid obstacles)
- Inverse kinematics
- Motion execution
- Collision detection

**Basic Workflow:**
1. Define goal pose
2. Plan path
3. Execute motion

#### C++ Example: MoveIt2 Basic Motion
```cpp
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "moveit/move_group_interface/move_group_interface.h"
#include "geometry_msgs/msg/pose.hpp"

class MoveItExample : public rclcpp::Node
{
public:
  MoveItExample() : Node("moveit_example")
  {
    // Create MoveGroup interface
    auto move_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), "arm");
    
    // Set target pose
    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = 0.3;
    target_pose.position.y = 0.0;
    target_pose.position.z = 0.5;
    target_pose.orientation.w = 1.0;
    
    move_group->setPoseTarget(target_pose);
    
    // Plan and execute
    auto const [success, plan] = [&move_group]{
      moveit::planning_interface::MoveGroupInterface::Plan msg;
      auto const ok = static_cast<bool>(move_group->plan(msg));
      return std::make_pair(ok, msg);
    }();
    
    if(success) {
      RCLCPP_INFO(this->get_logger(), "Planning successful! Executing...");
      move_group->execute(plan);
    } else {
      RCLCPP_ERROR(this->get_logger(), "Planning failed!");
    }
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MoveItExample>();
  rclcpp::shutdown();
  return 0;
}
```

**MoveIt2 Launch:**
```bash
# Launch MoveIt2 with robot
ros2 launch my_robot_moveit_config demo.launch.py

# Run motion planning node
ros2 run my_package moveit_example
```

**Key Concepts:**
- **Planning Group**: Set of joints (e.g., "arm", "gripper")
- **End Effector**: Robot's hand/gripper
- **Planning Scene**: Environment with obstacles
- **Joint Space**: Plan in joint angles
- **Cartesian Space**: Plan in XYZ position

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
ros2 topic info /topic_name
```

### Nodes
```bash
ros2 node list
ros2 node info /node_name
```

### Services
```bash
ros2 service list
ros2 service type /service_name
ros2 service call /service_name service_type "request_data"
```

### Actions
```bash
ros2 action list
ros2 action info /action_name
ros2 action send_goal /action_name action_type "goal_data"
```

### Parameters
```bash
ros2 param list
ros2 param get /node_name parameter_name
ros2 param set /node_name parameter_name value
```

### TF
```bash
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo frame1 frame2
```

---

## Package Setup Guide

### Create Package
```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake \
  --dependencies rclcpp geometry_msgs sensor_msgs std_srvs tf2_ros tf2_geometry_msgs example_interfaces \
  my_robot_package
```

### CMakeLists.txt Template
```cmake
cmake_minimum_required(VERSION 3.8)
project(my_robot_package)

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(example_interfaces REQUIRED)
find_package(rclcpp_action REQUIRED)

# For custom messages
find_package(rosidl_default_generators REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SensorData.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Executables
add_executable(velocity_publisher src/velocity_publisher.cpp)
ament_target_dependencies(velocity_publisher rclcpp geometry_msgs)

add_executable(tf_listener src/tf_listener.cpp)
ament_target_dependencies(tf_listener rclcpp tf2_ros tf2_geometry_msgs geometry_msgs)

add_executable(action_server src/action_server.cpp)
ament_target_dependencies(action_server rclcpp rclcpp_action example_interfaces)

# Install
install(TARGETS
  velocity_publisher
  tf_listener
  action_server
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
```

### package.xml Template
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.1</version>
  <description>My robot package</description>
  <maintainer email="your@email.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>std_srvs</depend>
  <depend>tf2_ros</depend>
  <depend>tf2_geometry_msgs</depend>
  <depend>example_interfaces</depend>
  <depend>rclcpp_action</depend>

  <!-- For custom messages -->
  <build_depend>rosidl_default_generators</build_depend>
  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

---

## Essential Concepts Summary

✅ **Core Communication:**
1. **Nodes** - Single-purpose programs
2. **Topics** - Continuous data streams (pub/sub)
3. **Services** - Request-response (one-time calls)
4. **Actions** - Long tasks with feedback
5. **Parameters** - Runtime configuration

✅ **Advanced Concepts:**
6. **Launch Files** - Start multiple nodes
7. **TF2** - Coordinate transformations
8. **Custom Messages** - Your own data types
9. **MoveIt2** - Robot arm manipulation

✅ **Key Patterns:**
- **Timer-based Publishing**: Fixed-rate commands
- **Callback Subscribing**: React to sensor data
- **Sense-Think-Act**: Navigation loop
- **Parameterized Nodes**: Reusable code

---

## Pro Tips for Beginners

1. **Naming Conventions:**
   - Nodes: `snake_case_node`
   - Topics: `/robot/sensor/data`
   - Frames: `base_link`, `camera_link`

2. **Always Log Important Events:**
   ```cpp
   RCLCPP_INFO(this->get_logger(), "Status: OK");
   RCLCPP_WARN(this->get_logger(), "Warning message");
   RCLCPP_ERROR(this->get_logger(), "Error occurred");
   ```

3. **Check for Valid Data:**
   ```cpp
   if (!msg || msg->ranges.empty()) {
     RCLCPP_WARN(this->get_logger(), "Invalid data");
     return;
   }
   ```

4. **Use const References:**
   ```cpp
   void callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
   ```

5. **Handle Exceptions:**
   ```cpp
   try {
     // Your code
   } catch (const std::exception& e) {
     RCLCPP_ERROR(this->get_logger(), "Error: %s", e.what());
   }
   ```

---

## Testing Your Code

### With TurtleBot3 in Gazebo:
```bash
# Terminal 1: Launch simulation
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2: Run your node
ros2 run my_robot_package obstacle_avoider

# Terminal 3: Monitor
ros2 topic echo /cmd_vel
ros2 node info /obstacle_avoider
```

### With RViz2:
```bash
# Launch RViz2
rviz2

# Add displays:
# - RobotModel
# - TF
# - LaserScan
# - Camera
```

---
