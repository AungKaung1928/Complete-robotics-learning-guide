# Docker Fundamentals for Robotics

## What is Docker?

**Docker** is a platform for running applications in isolated containers. Think of it as lightweight virtual machines.

### Why Docker in Robotics?

- **Consistency**: "Works on my machine" → Works everywhere
- **Isolation**: No conflicts between different ROS versions
- **Easy setup**: Share entire development environment
- **Deployment**: Same container from dev to production
- **Testing**: Test on different ROS distros easily

---

## Core Concepts

### 1. Images

**What is an Image?**
A blueprint/template for creating containers. Like a class in programming.

**Think of it as:**
- A snapshot of a filesystem
- Includes OS, ROS2, your code, dependencies
- Read-only template
- Can be shared and reused

**Example Images:**
- `ubuntu:22.04` - Ubuntu operating system
- `ros:humble` - ROS2 Humble installation
- `osrf/ros:humble-desktop` - ROS2 with GUI tools

---

### 2. Containers

**What is a Container?**
A running instance of an image. Like an object from a class.

**Characteristics:**
- Isolated environment
- Has its own filesystem
- Can run processes
- Lightweight (not a full VM)
- Can be stopped, started, deleted

**Container Lifecycle:**
```
Image → Create Container → Run → Stop → Remove
```

---

### 3. Dockerfile

**What is a Dockerfile?**
A recipe for building an image. Lists all steps to create your environment.

**Simple Example:**
```dockerfile
# Start from ROS2 Humble image
FROM ros:humble

# Install additional packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy your code
COPY ros2_ws /root/ros2_ws

# Set working directory
WORKDIR /root/ros2_ws

# Build ROS2 workspace
RUN . /opt/ros/humble/setup.sh && colcon build

# Source workspace on container start
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

# Default command
CMD ["bash"]
```

---

### 4. Volumes

**What are Volumes?**
Shared folders between your computer and container.

**Why Use Volumes?**
- Edit code on your computer, run in container
- Persist data after container stops
- Share data between containers

**Example:**
```bash
# Mount current directory to container
docker run -v $(pwd):/root/ws ros:humble
```

---

### 5. Networks

**What are Networks?**
Allow containers to communicate with each other.

**Use Cases:**
- Multiple ROS2 nodes in different containers
- Separate containers for different robots
- Microservices architecture

---

## Basic Docker Commands

### Working with Images

```bash
# List images
docker images

# Pull image from Docker Hub
docker pull ros:humble

# Build image from Dockerfile
docker build -t my_robot_image .

# Remove image
docker rmi image_name

# Remove unused images
docker image prune
```

### Working with Containers

```bash
# Run container (create and start)
docker run ros:humble

# Run with name
docker run --name my_robot ros:humble

# Run in interactive mode with terminal
docker run -it ros:humble bash

# Run in background (detached)
docker run -d ros:humble

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop container
docker stop container_name

# Start stopped container
docker start container_name

# Remove container
docker rm container_name

# Remove all stopped containers
docker container prune
```

### Executing Commands in Running Container

```bash
# Execute command in running container
docker exec container_name ls

# Open terminal in running container
docker exec -it container_name bash

# Run ROS2 node
docker exec -it my_robot ros2 run package_name node_name
```

### Container Logs

```bash
# View logs
docker logs container_name

# Follow logs (like tail -f)
docker logs -f container_name

# Last 100 lines
docker logs --tail 100 container_name
```

---

## Common Docker Options

### Essential Flags

```bash
# -it: Interactive terminal
docker run -it ros:humble bash

# -d: Detached (background)
docker run -d ros:humble

# --name: Give container a name
docker run --name my_robot ros:humble

# -v: Mount volume
docker run -v /host/path:/container/path ros:humble

# -p: Port mapping
docker run -p 8080:80 ros:humble

# --rm: Auto-remove container when stopped
docker run --rm ros:humble

# -e: Set environment variable
docker run -e ROS_DOMAIN_ID=5 ros:humble

# --network: Specify network
docker run --network=host ros:humble
```

---

## Docker Compose

**What is Docker Compose?**
Tool for defining and running multi-container applications using YAML file.

### docker-compose.yml Example

```yaml
version: '3.8'

services:
  robot_brain:
    image: ros:humble
    container_name: robot_brain
    volumes:
      - ./ros2_ws:/root/ros2_ws
    environment:
      - ROS_DOMAIN_ID=0
    command: ros2 run my_package brain_node

  robot_sensors:
    image: ros:humble
    container_name: robot_sensors
    volumes:
      - ./ros2_ws:/root/ros2_ws
    environment:
      - ROS_DOMAIN_ID=0
    command: ros2 run my_package sensor_node

  visualization:
    image: osrf/ros:humble-desktop
    container_name: rviz
    environment:
      - DISPLAY=$DISPLAY
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix
    command: rviz2
```

### Docker Compose Commands

```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs

# Follow logs
docker-compose logs -f

# Restart service
docker-compose restart service_name

# Build images
docker-compose build
```

---

## Dockerfile Best Practices

### 1. Multi-Stage Build
```dockerfile
# Build stage
FROM ros:humble as builder
WORKDIR /root/ws
COPY src/ ./src
RUN . /opt/ros/humble/setup.sh && colcon build

# Runtime stage
FROM ros:humble
COPY --from=builder /root/ws/install /root/ws/install
WORKDIR /root/ws
```

### 2. Minimize Layers
```dockerfile
# ❌ Bad: Multiple RUN commands
RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y git

# ✅ Good: Combine commands
RUN apt-get update && apt-get install -y \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*
```

### 3. Use .dockerignore
```
# .dockerignore file
__pycache__/
*.pyc
.git/
build/
install/
log/
.vscode/
```

### 4. Order Layers by Change Frequency
```dockerfile
# Things that rarely change first
FROM ros:humble
RUN apt-get update && apt-get install -y build-essential

# Things that change often last
COPY src/ ./src
RUN colcon build
```

---

## ROS2 + Docker Specific

### Running ROS2 in Docker

#### Basic ROS2 Container
```bash
# Start container with ROS2
docker run -it --name ros2_dev \
    --network host \
    -e ROS_DOMAIN_ID=0 \
    ros:humble bash

# Inside container
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp talker
```

#### With GUI (RViz, Gazebo)
```bash
# Allow X server connection
xhost +local:docker

# Run with display
docker run -it \
    --env="DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --network host \
    osrf/ros:humble-desktop \
    bash

# Run RViz
rviz2
```

#### With Your Workspace
```bash
docker run -it \
    -v ~/ros2_ws:/root/ros2_ws \
    --network host \
    -e ROS_DOMAIN_ID=0 \
    ros:humble bash
```

### Complete Development Dockerfile

```dockerfile
FROM ros:humble

# Install development tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
RUN mkdir -p /root/ros2_ws/src

# Set working directory
WORKDIR /root/ros2_ws

# Copy source code
COPY src/ ./src/

# Install dependencies
RUN apt-get update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*

# Build workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# Setup environment
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

# Environment variables
ENV ROS_DOMAIN_ID=0

CMD ["bash"]
```

---

## Common Use Cases

### 1. Development Environment
```bash
# Create dev container
docker run -it --name ros2_dev \
    -v ~/ros2_ws:/root/ros2_ws \
    --network host \
    ros:humble bash

# Work normally inside container
cd /root/ros2_ws
colcon build
ros2 run my_package my_node
```

### 2. Testing Different ROS Versions
```bash
# Test on Humble
docker run -it -v $(pwd):/ws ros:humble bash

# Test on Iron
docker run -it -v $(pwd):/ws ros:iron bash
```

### 3. CI/CD Pipeline
```dockerfile
# Used in automated testing
FROM ros:humble
COPY . /ws
WORKDIR /ws
RUN . /opt/ros/humble/setup.sh && \
    colcon build && \
    colcon test
```

### 4. Multi-Robot Simulation
```yaml
# docker-compose.yml
version: '3.8'

services:
  robot1:
    image: my_robot_image
    environment:
      - ROS_DOMAIN_ID=1
      - ROBOT_NAME=robot1

  robot2:
    image: my_robot_image
    environment:
      - ROS_DOMAIN_ID=2
      - ROBOT_NAME=robot2
```

---

## Troubleshooting

### Issue: Permission Denied
```bash
# Run as current user
docker run -it --user $(id -u):$(id -g) ros:humble

# Or give permissions
chmod -R 777 shared_folder
```

### Issue: Can't See GUI Applications
```bash
# Allow X server
xhost +local:docker

# Run with display
docker run -it -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    ros:humble
```

### Issue: ROS2 Nodes Can't Communicate
```bash
# Use host network
docker run --network host ros:humble

# Or same ROS_DOMAIN_ID
docker run -e ROS_DOMAIN_ID=0 ros:humble
```

### Issue: Container Keeps Exiting
```bash
# Keep container alive
docker run -d --name my_robot ros:humble tail -f /dev/null

# Then execute commands
docker exec -it my_robot bash
```

---

## Essential Docker Concepts Summary

✅ **Must Understand:**
1. Images vs Containers (blueprint vs instance)
2. Dockerfile (recipe for image)
3. Volumes (share files)
4. Basic commands (run, exec, ps, logs)
5. --network host (for ROS2)

✅ **Important for ROS2:**
- Use `--network host` for node communication
- Mount workspace with `-v`
- Set ROS_DOMAIN_ID with `-e`
- Use `osrf/ros` images for GUI applications

✅ **Good to Know:**
- Docker Compose for multi-container setups
- Multi-stage builds for smaller images
- .dockerignore for efficiency

---

## Docker Workflow for ROS2

1. **Pull base image**
   ```bash
   docker pull ros:humble
   ```

2. **Create Dockerfile for your project**
   ```dockerfile
   FROM ros:humble
   # Add your setup
   ```

3. **Build image**
   ```bash
   docker build -t my_robot .
   ```

4. **Run container**
   ```bash
   docker run -it -v ~/ros2_ws:/ws my_robot
   ```

5. **Develop and test**
   - Edit code on host
   - Build and run in container

---

## When to Use Docker

✅ **Use Docker When:**
- Need consistent environment across team
- Testing on multiple ROS versions
- Deploying to production
- Want clean, isolated development
- Sharing your project with others

❌ **Maybe Skip Docker When:**
- Just learning ROS2 basics
- Simple, single-machine projects
- Need maximum performance
- Debugging low-level hardware

---

## Next Steps

1. Install Docker on your system
2. Pull ROS2 image: `docker pull ros:humble`
3. Practice basic commands
4. Create simple Dockerfile
5. Try Docker Compose with multiple nodes

**Pro Tip:** Start simple! Use Docker when you need it, not because you can!
