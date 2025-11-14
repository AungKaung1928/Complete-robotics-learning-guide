# Unity Simulation Fundamentals for Robotics & Autonomous Vehicles

## Table of Contents
1. [What is Unity?](#what-is-unity)
2. [Why Unity for Simulation?](#why-unity-for-simulation)
3. [Installation & Setup](#installation--setup)
4. [Unity Interface Basics](#unity-interface-basics)
5. [Core Concepts](#core-concepts)
6. [Essential C# for Unity](#essential-c-for-unity)
7. [Physics & Sensors](#physics--sensors)
8. [Vehicle Simulation](#vehicle-simulation)
9. [Camera & Perception](#camera--perception)
10. [Unity ML-Agents (AI)](#unity-ml-agents-ai)
11. [Common Tasks](#common-tasks)
12. [Best Practices](#best-practices)

---

## What is Unity?

**Unity** is a real-time 3D development platform primarily used for:
- ✅ Game development
- ✅ Robotics simulation
- ✅ Autonomous vehicle testing
- ✅ Digital twins
- ✅ Virtual environments for AI training

**For Robotics/Autonomous Vehicles:**
Unity provides realistic physics, sensors (cameras, LiDAR, radar), and environment simulation without physical hardware.

---

## Why Unity for Simulation?

### Advantages:
✅ **Realistic Physics** - Built-in physics engine (NVIDIA PhysX)  
✅ **Sensor Simulation** - Camera, LiDAR, IMU, GPS  
✅ **Fast Iteration** - Test algorithms without real hardware  
✅ **Safe Testing** - No risk of damaging vehicles/robots  
✅ **Scalable** - Run multiple simulations in parallel  
✅ **ML-Agents** - Train AI with reinforcement learning  
✅ **Cross-platform** - Windows, Linux, Mac  

### Use Cases:
- Autonomous vehicle path planning
- Sensor data collection
- Collision avoidance testing
- Traffic scenario simulation
- AI training (reinforcement learning)
- Hardware-in-the-loop (HIL) testing

---

## Installation & Setup

### 1. Install Unity Hub
```
1. Download: https://unity.com/download
2. Install Unity Hub (manages Unity versions)
3. Create Unity account (free)
```

### 2. Install Unity Editor
```
1. Open Unity Hub
2. Click "Installs" → "Install Editor"
3. Choose version: 2022.3 LTS (Long Term Support)
4. Add modules:
   - Visual Studio (for C# scripting)
   - Linux Build Support (optional)
   - Android/iOS (if needed)
```

### 3. Create First Project
```
1. Unity Hub → "Projects" → "New Project"
2. Template: 3D (URP - Universal Render Pipeline)
3. Name: "MyRobotSim"
4. Location: Choose folder
5. Click "Create Project"
```

**System Requirements:**
- Windows 10/11, macOS, or Ubuntu
- 8GB RAM minimum (16GB+ recommended)
- GPU with DirectX 11 support
- 20GB+ free disk space

---

## Unity Interface Basics

### Main Windows:

```
┌─────────────────────────────────────┐
│         Hierarchy (left)            │  List of objects in scene
├─────────────────────────────────────┤
│         Scene View (center)         │  3D view for editing
├─────────────────────────────────────┤
│         Game View (center)          │  Runtime view
├─────────────────────────────────────┤
│         Inspector (right)           │  Object properties
├─────────────────────────────────────┤
│         Project (bottom)            │  Assets/files
└─────────────────────────────────────┘
```

---

### Essential Shortcuts:
```
F                   Focus on selected object
W, E, R             Move, Rotate, Scale tools
Q                   Hand tool (pan view)
Ctrl + D            Duplicate object
Ctrl + S            Save scene
Ctrl + P            Play/Stop simulation
Space               Pause simulation
```

---

## Core Concepts

### 1. GameObject
**What:** Everything in Unity is a GameObject (vehicles, cameras, lights, terrain)

**Creating:**
```
Hierarchy → Right-click → Create Empty
Or
GameObject menu → 3D Object → Cube
```

---

### 2. Component
**What:** Functionality attached to GameObjects (scripts, physics, rendering)

**Example Components:**
- Transform (position, rotation, scale)
- Rigidbody (physics)
- Collider (collision detection)
- Camera (rendering)
- Scripts (custom behavior)

---

### 3. Transform
**What:** Position, Rotation, Scale of GameObject

```
Position: (X, Y, Z) in meters
Rotation: (X, Y, Z) in degrees (Euler angles)
Scale: (X, Y, Z) multiplier
```

**In Inspector:**
```
Transform
  Position: X: 0  Y: 0  Z: 0
  Rotation: X: 0  Y: 90  Z: 0
  Scale:    X: 1  Y: 1  Z: 1
```

---

### 4. Prefab
**What:** Reusable template for GameObjects

**Use Case:** Create one vehicle, reuse many times

**Creating:**
```
1. Create and configure GameObject
2. Drag from Hierarchy to Project window
3. Blue icon = Prefab
4. Drag into scene to instantiate
```

---

### 5. Scene
**What:** Container for GameObjects (like a level or environment)

**Usage:**
- Save: Ctrl+S
- New: File → New Scene
- Load: Double-click in Project window

---

## Essential C# for Unity

### Script Structure

```csharp
using UnityEngine;

public class VehicleController : MonoBehaviour
{
    // Variables (visible in Inspector if public)
    public float speed = 10f;
    private float currentSpeed;
    
    // Called once at start
    void Start()
    {
        Debug.Log("Vehicle initialized");
        currentSpeed = 0f;
    }
    
    // Called every frame
    void Update()
    {
        // Input and non-physics updates
        if (Input.GetKey(KeyCode.W))
        {
            currentSpeed = speed;
        }
    }
    
    // Called at fixed time steps (for physics)
    void FixedUpdate()
    {
        // Physics updates here
        transform.Translate(Vector3.forward * currentSpeed * Time.deltaTime);
    }
}
```

---

### Common Unity Methods

```csharp
void Start()           // Once at start
void Update()          // Every frame (~60 FPS)
void FixedUpdate()     // Fixed intervals (for physics)
void LateUpdate()      // After all Update() calls
void OnCollisionEnter(Collision col)  // When collision starts
void OnTriggerEnter(Collider other)   // When enters trigger
```

---

### Variables & Types

```csharp
// Basic types
int count = 5;
float speed = 10.5f;
bool isActive = true;
string name = "Vehicle";

// Unity types
Vector3 position = new Vector3(0, 0, 0);
Quaternion rotation = Quaternion.identity;
Transform carTransform;
Rigidbody rb;

// Arrays
float[] sensors = new float[8];
GameObject[] vehicles;

// Inspector-visible
public float maxSpeed = 20f;         // Shows in Inspector
[SerializeField] private float minSpeed = 0f;  // Private but shows
```

---

### Common Operations

```csharp
// Movement
transform.Translate(Vector3.forward * speed * Time.deltaTime);
transform.Rotate(Vector3.up * turnSpeed * Time.deltaTime);

// Position
transform.position = new Vector3(0, 1, 0);
Vector3 pos = transform.position;

// Rotation
transform.rotation = Quaternion.Euler(0, 90, 0);

// Find objects
GameObject car = GameObject.Find("Car");
Rigidbody rb = GetComponent<Rigidbody>();

// Instantiate (spawn)
GameObject newCar = Instantiate(carPrefab, position, rotation);

// Destroy
Destroy(gameObject);
Destroy(gameObject, 5f);  // After 5 seconds

// Debug
Debug.Log("Speed: " + speed);
Debug.LogWarning("Low fuel");
Debug.DrawLine(start, end, Color.red);
```

---

## Physics & Sensors

### Rigidbody Component

**What:** Gives GameObject physics (gravity, forces, collisions)

**Add:** Inspector → Add Component → Physics → Rigidbody

**Properties:**
```
Mass: 1500 (kg, for car)
Drag: 0.05 (air resistance)
Angular Drag: 0.5 (rotation resistance)
Use Gravity: ✓
Is Kinematic: ☐ (uncheck for physics)
```

**Code:**
```csharp
Rigidbody rb;

void Start()
{
    rb = GetComponent<Rigidbody>();
}

void FixedUpdate()
{
    // Apply force
    rb.AddForce(Vector3.forward * thrust);
    
    // Apply torque (rotation)
    rb.AddTorque(Vector3.up * steering);
    
    // Set velocity directly
    rb.velocity = new Vector3(0, 0, 10f);
}
```

---

### Colliders

**Types:**
- Box Collider (cubes, buildings)
- Sphere Collider (balls, wheels)
- Capsule Collider (characters)
- Mesh Collider (complex shapes)

**Trigger vs Collider:**
- Collider: Physical collision (bounces)
- Trigger: Detection only (no physics)

**Code:**
```csharp
void OnCollisionEnter(Collision collision)
{
    if (collision.gameObject.tag == "Obstacle")
    {
        Debug.Log("Hit obstacle!");
    }
}

void OnTriggerEnter(Collider other)
{
    if (other.tag == "Checkpoint")
    {
        Debug.Log("Passed checkpoint");
    }
}
```

---

### Raycasting (Sensor Simulation)

**What:** Cast invisible ray to detect objects (like LiDAR/ultrasonic)

```csharp
// Simple raycast
RaycastHit hit;
if (Physics.Raycast(transform.position, transform.forward, out hit, 10f))
{
    Debug.Log("Hit: " + hit.collider.name);
    Debug.Log("Distance: " + hit.distance);
}

// Multiple sensors (like LiDAR)
float[] GetLidarDistances()
{
    float[] distances = new float[8];
    float angleStep = 360f / 8;
    
    for (int i = 0; i < 8; i++)
    {
        float angle = angleStep * i;
        Vector3 direction = Quaternion.Euler(0, angle, 0) * Vector3.forward;
        
        RaycastHit hit;
        if (Physics.Raycast(transform.position, direction, out hit, maxRange))
        {
            distances[i] = hit.distance;
        }
        else
        {
            distances[i] = maxRange;
        }
        
        // Visualize ray
        Debug.DrawRay(transform.position, direction * maxRange, Color.red);
    }
    
    return distances;
}
```

---

## Vehicle Simulation

### Simple Car Controller

```csharp
using UnityEngine;

public class SimpleCarController : MonoBehaviour
{
    public float motorForce = 1500f;
    public float steerAngle = 30f;
    public float brakeForce = 3000f;
    
    public WheelCollider frontLeft, frontRight;
    public WheelCollider rearLeft, rearRight;
    
    private float throttle, steering, brake;
    
    void Update()
    {
        // Get input
        throttle = Input.GetAxis("Vertical");    // W/S or Up/Down
        steering = Input.GetAxis("Horizontal");  // A/D or Left/Right
        brake = Input.GetKey(KeyCode.Space) ? 1f : 0f;
    }
    
    void FixedUpdate()
    {
        // Apply steering
        frontLeft.steerAngle = steering * steerAngle;
        frontRight.steerAngle = steering * steerAngle;
        
        // Apply throttle
        rearLeft.motorTorque = throttle * motorForce;
        rearRight.motorTorque = throttle * motorForce;
        
        // Apply brakes
        float currentBrake = brake * brakeForce;
        frontLeft.brakeTorque = currentBrake;
        frontRight.brakeTorque = currentBrake;
        rearLeft.brakeTorque = currentBrake;
        rearRight.brakeTorque = currentBrake;
    }
}
```

---

### Wheel Collider Setup

**Steps:**
1. Create 4 empty GameObjects for wheels
2. Position at wheel locations
3. Add WheelCollider component to each
4. Configure:
```
Mass: 20
Radius: 0.35
Wheel Damping Rate: 0.25
Suspension Distance: 0.3
Spring: 35000
Damper: 4500
```

---

## Camera & Perception

### Follow Camera

```csharp
using UnityEngine;

public class FollowCamera : MonoBehaviour
{
    public Transform target;           // Vehicle to follow
    public Vector3 offset = new Vector3(0, 5, -10);
    public float smoothSpeed = 0.125f;
    
    void LateUpdate()
    {
        Vector3 desiredPosition = target.position + offset;
        Vector3 smoothedPosition = Vector3.Lerp(transform.position, desiredPosition, smoothSpeed);
        transform.position = smoothedPosition;
        
        transform.LookAt(target);
    }
}
```

---

### Camera Sensor (RGB)

```csharp
using UnityEngine;

public class CameraSensor : MonoBehaviour
{
    private Camera cam;
    private RenderTexture renderTexture;
    
    void Start()
    {
        cam = GetComponent<Camera>();
        renderTexture = new RenderTexture(640, 480, 24);
        cam.targetTexture = renderTexture;
    }
    
    public Texture2D CaptureImage()
    {
        RenderTexture.active = renderTexture;
        cam.Render();
        
        Texture2D image = new Texture2D(renderTexture.width, renderTexture.height);
        image.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        image.Apply();
        
        RenderTexture.active = null;
        return image;
    }
}
```

---

### Depth Camera

```csharp
using UnityEngine;

public class DepthCamera : MonoBehaviour
{
    private Camera cam;
    
    void Start()
    {
        cam = GetComponent<Camera>();
        cam.depthTextureMode = DepthTextureMode.Depth;
    }
    
    // Use shader to render depth
    // Attach shader with _CameraDepthTexture
}
```

---

## Unity ML-Agents (AI)

### What is ML-Agents?

**Purpose:** Train AI agents using reinforcement learning in Unity

**Use Cases:**
- Autonomous navigation
- Obstacle avoidance
- Path planning
- Decision making

---

### Installation

```bash
# Install Python package
pip install mlagents

# In Unity:
# Window → Package Manager → Add package from git URL
# https://github.com/Unity-Technologies/ml-agents.git?path=com.unity.ml-agents
```

---

### Simple Agent Example

```csharp
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class CarAgent : Agent
{
    public Transform target;
    private Rigidbody rb;
    
    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
    }
    
    public override void OnEpisodeBegin()
    {
        // Reset position
        transform.position = new Vector3(0, 0.5f, 0);
        rb.velocity = Vector3.zero;
        
        // Randomize target
        target.position = new Vector3(Random.Range(-8f, 8f), 0.5f, Random.Range(-8f, 8f));
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        // Agent position and velocity
        sensor.AddObservation(transform.position);
        sensor.AddObservation(rb.velocity);
        
        // Target position
        sensor.AddObservation(target.position);
    }
    
    public override void OnActionReceived(ActionBuffers actions)
    {
        // Get actions (continuous: steering, throttle)
        float steering = actions.ContinuousActions[0];
        float throttle = actions.ContinuousActions[1];
        
        // Apply to vehicle
        rb.AddForce(transform.forward * throttle * 10f);
        transform.Rotate(Vector3.up, steering * 100f * Time.deltaTime);
        
        // Reward for reaching target
        float distance = Vector3.Distance(transform.position, target.position);
        if (distance < 1.5f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        
        // Penalty for falling off
        if (transform.position.y < 0)
        {
            SetReward(-1.0f);
            EndEpisode();
        }
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Manual control for testing
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxis("Horizontal");
        continuousActions[1] = Input.GetAxis("Vertical");
    }
}
```

---

### Training

```bash
# Start training
mlagents-learn config.yaml --run-id=car_test

# In Unity: Click Play button
# Agent will train automatically
# Model saved in results/ folder
```

---

## Common Tasks

### 1. Create Terrain/Road

```
GameObject → 3D Object → Terrain
Tools: Raise/Lower, Paint Texture, Place Trees

Or use ProBuilder:
Window → Package Manager → ProBuilder
Create roads, buildings with ProBuilder tools
```

---

### 2. Add Physics Materials

```
Create → Physics Material
Name: "Road"
Properties:
  Dynamic Friction: 0.6
  Static Friction: 0.6
  Bounciness: 0

Apply to road collider
```

---

### 3. Randomize Environment

```csharp
public class EnvironmentRandomizer : MonoBehaviour
{
    public GameObject[] obstacles;
    public int obstacleCount = 10;
    public float spawnRadius = 50f;
    
    void Start()
    {
        for (int i = 0; i < obstacleCount; i++)
        {
            Vector3 randomPos = Random.insideUnitSphere * spawnRadius;
            randomPos.y = 0;
            
            int randomIndex = Random.Range(0, obstacles.Length);
            Instantiate(obstacles[randomIndex], randomPos, Quaternion.identity);
        }
    }
}
```

---

### 4. Data Collection

```csharp
using System.IO;
using UnityEngine;

public class DataLogger : MonoBehaviour
{
    private StreamWriter writer;
    
    void Start()
    {
        writer = new StreamWriter("simulation_data.csv");
        writer.WriteLine("Time,PosX,PosY,PosZ,VelX,VelY,VelZ");
    }
    
    void FixedUpdate()
    {
        Vector3 pos = transform.position;
        Vector3 vel = GetComponent<Rigidbody>().velocity;
        
        writer.WriteLine($"{Time.time},{pos.x},{pos.y},{pos.z},{vel.x},{vel.y},{vel.z}");
    }
    
    void OnDestroy()
    {
        writer.Close();
    }
}
```

---

### 5. Simple Traffic System

```csharp
using UnityEngine;

public class TrafficVehicle : MonoBehaviour
{
    public Transform[] waypoints;
    public float speed = 5f;
    private int currentWaypoint = 0;
    
    void Update()
    {
        if (waypoints.Length == 0) return;
        
        Transform target = waypoints[currentWaypoint];
        Vector3 direction = (target.position - transform.position).normalized;
        
        transform.position += direction * speed * Time.deltaTime;
        transform.LookAt(target);
        
        if (Vector3.Distance(transform.position, target.position) < 1f)
        {
            currentWaypoint = (currentWaypoint + 1) % waypoints.Length;
        }
    }
}
```

---

## Best Practices

### 1. Use Layers for Organization
```
Edit → Project Settings → Tags and Layers
Create layers: Vehicle, Road, Obstacle, Sensor
Assign to GameObjects
Use in raycasting: Physics.Raycast(..., layerMask)
```

---

### 2. Optimize Performance
```csharp
// Cache components
private Rigidbody rb;
void Start() { rb = GetComponent<Rigidbody>(); }

// Avoid GetComponent in Update()
void Update() {
    // Bad
    GetComponent<Rigidbody>().AddForce(...);
    
    // Good
    rb.AddForce(...);
}

// Use object pooling for spawning
// Use LOD (Level of Detail) for distant objects
```

---

### 3. Use Time.deltaTime
```csharp
// Frame-rate independent
transform.Translate(Vector3.forward * speed * Time.deltaTime);

// NOT frame-rate independent (wrong!)
transform.Translate(Vector3.forward * speed);
```

---

### 4. Debug Visualization
```csharp
// Draw rays in Scene view
Debug.DrawRay(position, direction * distance, Color.red);
Debug.DrawLine(start, end, Color.green);

// Gizmos (always visible)
void OnDrawGizmos()
{
    Gizmos.color = Color.blue;
    Gizmos.DrawSphere(target.position, 1f);
    Gizmos.DrawLine(transform.position, target.position);
}
```

---

### 5. Version Control (Git)
```
.gitignore contents:
[Ll]ibrary/
[Tt]emp/
[Oo]bj/
[Bb]uild/
[Bb]uilds/
[Ll]ogs/
*.csproj
*.sln
*.user
*.userprefs
```

---

## Quick Reference

### Essential Components
```
Rigidbody        - Physics
Collider         - Collision detection
Camera           - Rendering
Light            - Illumination
Audio Source     - Sound
Particle System  - Effects
```

### Common Scripts Pattern
```csharp
public class MyScript : MonoBehaviour
{
    [SerializeField] private float speed = 10f;
    private Rigidbody rb;
    
    void Start() { rb = GetComponent<Rigidbody>(); }
    void Update() { /* Input, non-physics */ }
    void FixedUpdate() { /* Physics */ }
}
```

### Input
```csharp
Input.GetKey(KeyCode.W)              // Held down
Input.GetKeyDown(KeyCode.Space)      // Pressed once
Input.GetAxis("Horizontal")          // -1 to 1 (smooth)
Input.GetMouseButton(0)              // Left click
```

### Vector Math
```csharp
Vector3.forward  // (0, 0, 1)
Vector3.up       // (0, 1, 0)
Vector3.right    // (1, 0, 0)
Vector3.Distance(a, b)
Vector3.Lerp(a, b, t)  // Interpolate
```

---

## Interview Questions

**Q: "What is Unity used for in robotics?"**
**A:** Simulation for testing algorithms, sensor data generation, AI training, and safe testing without physical hardware.

**Q: "Difference between Update and FixedUpdate?"**
**A:** Update() runs every frame (variable rate). FixedUpdate() runs at fixed intervals (good for physics, typically 50 times/second).

**Q: "What is a Rigidbody?"**
**A:** Component that enables physics simulation - gravity, forces, collisions. Required for realistic vehicle movement.

**Q: "How to detect collisions?"**
**A:** Use Collider components and OnCollisionEnter/OnTriggerEnter methods. Colliders for physics, Triggers for detection only.

**Q: "What is ML-Agents?"**
**A:** Unity package for training AI agents with reinforcement learning. Used for autonomous navigation, decision making.

**Q: "How to simulate LiDAR in Unity?"**
**A:** Use Physics.Raycast() in multiple directions (360°) to measure distances. Returns hit points like real LiDAR.

---

## Summary

**Essential Skills for Simulation:**
1. ✅ Create GameObjects and Components
2. ✅ Basic C# scripting (Start, Update, FixedUpdate)
3. ✅ Rigidbody physics and forces
4. ✅ Raycasting for sensors
5. ✅ Camera setup and following
6. ✅ Input handling
7. ✅ Data logging

**Recommended: Start with Unity Learn tutorials (learn.unity.com) for hands-on practice.**
