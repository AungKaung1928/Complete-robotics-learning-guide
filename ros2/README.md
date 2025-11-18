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

### Parameters
```bash
ros2 param list
ros2 param get /node_name parameter_name
ros2 param set /node_name parameter_name value
```

---

## How to Use These Examples

### 1. Create a ROS2 Package
```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake --dependencies rclcpp geometry_msgs sensor_msgs std_srvs my_robot_basics
```

### 2. Add Your Code
- Put `.cpp` files in `src/` folder
- Update `CMakeLists.txt`:

```cmake
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# Add executable for each node
add_executable(velocity_publisher src/velocity_publisher.cpp)
ament_target_dependencies(velocity_publisher rclcpp geometry_msgs)

add_executable(laser_subscriber src/laser_subscriber.cpp)
ament_target_dependencies(laser_subscriber rclcpp sensor_msgs)

add_executable(obstacle_avoider src/obstacle_avoider.cpp)
ament_target_dependencies(obstacle_avoider rclcpp geometry_msgs sensor_msgs)

add_executable(service_server src/service_server.cpp)
ament_target_dependencies(service_server rclcpp std_srvs)

add_executable(configurable_publisher src/configurable_publisher.cpp)
ament_target_dependencies(configurable_publisher rclcpp geometry_msgs)

# Install executables
install(TARGETS
  velocity_publisher
  laser_subscriber
  obstacle_avoider
  service_server
  configurable_publisher
  DESTINATION lib/${PROJECT_NAME}
)
```

### 3. Build and Run
```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_basics
source install/setup.bash
ros2 run my_robot_basics velocity_publisher
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

✅ **Common Patterns:**
- **Timer-based Publishing**: Send commands at fixed rates (motor control)
- **Callback-based Subscribing**: React to incoming data (sensor processing)
- **Sense-Think-Act Loop**: Read sensors → Process → Control (navigation)
- **Parameterized Nodes**: Reusable, configurable components

---

## Pro Tips for Beginners

1. **Always use meaningful variable names** - Not `a`, `b`, `x`
2. **Log important events** - Use `RCLCPP_INFO`, `RCLCPP_WARN`, `RCLCPP_ERROR`
3. **Check for valid data** - Especially with sensor inputs
4. **Use const for read-only data** - `const auto& msg` in callbacks
5. **Avoid magic numbers** - Use parameters or named constants
6. **Test in simulation first** - Use Gazebo with TurtleBot3
7. **Start simple, add complexity** - Get basic pub/sub working before complex logic

---

## Testing Your Code

### With TurtleBot3 in Gazebo:
```bash
# Terminal 1: Launch Gazebo simulation
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2: Run your node
ros2 run my_robot_basics obstacle_avoider

# Terminal 3: Monitor topics
ros2 topic echo /cmd_vel
ros2 topic echo /scan
```

---

## Next Steps

Once comfortable with these basics, explore:
- **Navigation Stack (Nav2)**: Autonomous navigation
- **MoveIt2**: Robot arm manipulation
- **TF2**: Coordinate frame transformations
- **Launch Files**: Start multiple nodes together
- **Custom Messages**: Create your own message types
- **Actions**: Implement goal-based behaviors

---
