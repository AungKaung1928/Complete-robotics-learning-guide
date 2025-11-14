# Machine Learning Basics for Robotics

## Table of Contents
1. [What is Machine Learning?](#what-is-machine-learning)
2. [Types of Machine Learning](#types-of-machine-learning)
3. [ML in Robotics Use Cases](#ml-in-robotics-use-cases)
4. [Key Concepts](#key-concepts)
5. [Essential Python Libraries](#essential-python-libraries)
6. [Simple Examples](#simple-examples)
7. [Neural Networks Basics](#neural-networks-basics)
8. [Computer Vision for Robotics](#computer-vision-for-robotics)
9. [Reinforcement Learning](#reinforcement-learning)
10. [Quick Reference](#quick-reference)

---

## What is Machine Learning?

**Simple Definition:**
Teaching computers to learn from data instead of programming explicit rules.

**Traditional Programming:**
```
Rules + Data → Output
Example: IF distance < 10cm THEN stop
```

**Machine Learning:**
```
Data + Output → Rules (learned automatically)
Example: Show 1000 images of stop signs → learns to detect them
```

---

## Types of Machine Learning

### 1. Supervised Learning
**What:** Learn from labeled examples  
**Example:** Object detection (image + label)

```python
# Training data
images = [cat_image1, dog_image1, cat_image2, dog_image2]
labels = ['cat', 'dog', 'cat', 'dog']

# Model learns pattern
model.train(images, labels)

# Predict new image
prediction = model.predict(new_image)  # → 'cat'
```

**Robotics Use:**
- Object recognition
- Path prediction
- Sensor calibration

---

### 2. Unsupervised Learning
**What:** Find patterns in unlabeled data  
**Example:** Clustering similar objects

```python
# No labels provided
data = [point1, point2, point3, ...]

# Find groups
clusters = model.cluster(data)  # → Group similar points
```

**Robotics Use:**
- Anomaly detection
- Map clustering
- Pattern discovery

---

### 3. Reinforcement Learning (RL)
**What:** Learn by trial and error with rewards  
**Example:** Robot learns to walk

```python
# Agent tries action
action = agent.choose_action(state)

# Gets reward
reward = environment.step(action)

# Learns from result
agent.learn(state, action, reward)
```

**Robotics Use:**
- Navigation
- Manipulation
- Game playing
- Autonomous driving

---

## ML in Robotics Use Cases

| Task | ML Type | Example |
|------|---------|---------|
| **Object Detection** | Supervised | Detect pedestrians, traffic signs |
| **Path Planning** | Reinforcement | Navigate obstacle course |
| **Grasping** | Supervised/RL | Pick objects of different shapes |
| **SLAM** | Unsupervised | Build map from sensor data |
| **Imitation Learning** | Supervised | Learn from human demonstrations |
| **Anomaly Detection** | Unsupervised | Detect equipment failures |

---

## Key Concepts

### 1. Dataset
**What:** Collection of examples for training

```python
# Image dataset
images = 10000 images
labels = 10000 labels

# Split data
train_data = 70%    # Learn from this
val_data = 15%      # Tune parameters
test_data = 15%     # Final evaluation
```

---

### 2. Model
**What:** Mathematical function that makes predictions

```python
# Simple model
y = m * x + b

# Complex model (neural network)
y = neural_network(x)
```

---

### 3. Training
**What:** Adjust model to fit data

```python
for epoch in range(100):
    prediction = model(input)
    loss = calculate_error(prediction, true_label)
    model.update_parameters(loss)  # Learn!
```

---

### 4. Loss Function
**What:** Measures how wrong the model is

```python
# Example: Mean Squared Error
loss = (prediction - actual)^2

# Goal: Minimize loss
```

---

### 5. Overfitting vs Underfitting

```
Underfitting: Too simple, poor accuracy
Perfect Fit: Generalizes well
Overfitting: Memorizes training data, fails on new data
```

**Solution:** Validation set, regularization, more data

---

## Essential Python Libraries

### Installation
```bash
pip install numpy pandas scikit-learn
pip install tensorflow  # or pytorch
pip install opencv-python
pip install matplotlib
```

### Import Pattern
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
```

---

## Simple Examples

### Example 1: Linear Regression (Distance Prediction)

**Problem:** Predict stopping distance based on speed

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data: speed (km/h) vs stopping distance (m)
speed = np.array([10, 20, 30, 40, 50, 60]).reshape(-1, 1)
distance = np.array([5, 15, 30, 50, 75, 105])

# Create and train model
model = LinearRegression()
model.fit(speed, distance)

# Predict
new_speed = np.array([[35]])
predicted_distance = model.predict(new_speed)
print(f"At 35 km/h, stop distance: {predicted_distance[0]:.1f}m")

# Visualize
plt.scatter(speed, distance, color='blue', label='Data')
plt.plot(speed, model.predict(speed), color='red', label='Model')
plt.xlabel('Speed (km/h)')
plt.ylabel('Distance (m)')
plt.legend()
plt.show()
```

**Output:**
```
At 35 km/h, stop distance: 40.0m
```

---

### Example 2: Classification (Obstacle Detection)

**Problem:** Classify if sensor reading indicates obstacle

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Training data: [distance, speed]
X = np.array([
    [10, 5],   # Close, slow
    [50, 5],   # Far, slow
    [10, 20],  # Close, fast
    [50, 20],  # Far, fast
    [15, 10],
    [45, 15]
])

# Labels: 1 = stop, 0 = go
y = np.array([1, 0, 1, 0, 1, 0])

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict: distance=12m, speed=15km/h
new_reading = np.array([[12, 15]])
prediction = model.predict(new_reading)

if prediction[0] == 1:
    print("STOP - Obstacle detected!")
else:
    print("GO - Path clear")
```

**Output:**
```
STOP - Obstacle detected!
```

---

### Example 3: K-Means Clustering (Group Waypoints)

**Problem:** Group waypoints into clusters for efficient navigation

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Waypoints (x, y coordinates)
points = np.array([
    [1, 2], [1.5, 1.8], [2, 1.5],      # Cluster 1
    [8, 8], [8.5, 8.2], [9, 8.5],      # Cluster 2
    [5, 5], [5.5, 5.2], [5, 5.8]       # Cluster 3
])

# Find 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(points)

# Get cluster centers
centers = kmeans.cluster_centers_
print("Cluster centers:")
print(centers)

# Visualize
plt.scatter(points[:, 0], points[:, 1], c=kmeans.labels_)
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Waypoint Clusters')
plt.show()
```

---

## Neural Networks Basics

### What is a Neural Network?

**Structure:**
```
Input Layer → Hidden Layers → Output Layer
    [x1]                          [y]
    [x2]  →  [neurons]  →  [y]
    [x3]
```

**Simple Example: Sensor Fusion**

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Create simple neural network
model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(3,)),  # Hidden layer
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Output: 0 or 1
])

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training data: [distance, speed, light_level] → obstacle (0/1)
X_train = np.array([
    [10, 20, 100],
    [50, 20, 100],
    [10, 5, 50],
    [50, 5, 50]
])
y_train = np.array([1, 0, 1, 0])

# Train
model.fit(X_train, y_train, epochs=100, verbose=0)

# Predict
test_data = np.array([[15, 10, 75]])
prediction = model.predict(test_data)
print(f"Obstacle probability: {prediction[0][0]:.2%}")
```

---

### CNN (Convolutional Neural Network)

**What:** Specialized for images  
**Use:** Object detection, lane detection, sign recognition

```python
import tensorflow as tf
from tensorflow import keras

# Simple CNN for image classification
model = keras.Sequential([
    # Convolutional layers (extract features)
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    
    # Flatten and classify
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')  # 10 classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train with image data
# model.fit(images, labels, epochs=10)
```

---

## Computer Vision for Robotics

### Image Processing with OpenCV

```python
import cv2
import numpy as np

# Load image
image = cv2.imread('road.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Edge detection (for lane detection)
edges = cv2.Canny(gray, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(image, contours, -1, (0,255,0), 2)

# Display
cv2.imshow('Detected Edges', image)
cv2.waitKey(0)
```

---

### Object Detection with Pre-trained Model

```python
import cv2

# Load pre-trained model (YOLO, MobileNet, etc.)
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = open('coco.names').read().strip().split('\n')

# Load image
image = cv2.imread('street.jpg')
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB=True)

# Detect objects
net.setInput(blob)
outputs = net.forward(net.getUnconnectedOutLayersNames())

# Process detections
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.5:
            print(f"Detected: {classes[class_id]} ({confidence:.2%})")
```

---

## Reinforcement Learning

### Q-Learning (Simple Navigation)

**Problem:** Robot learns to reach goal

```python
import numpy as np

# Environment: 5x5 grid
# States: positions (0-24)
# Actions: up, down, left, right
# Reward: +100 at goal, -1 per step

class SimpleGridWorld:
    def __init__(self):
        self.position = 0
        self.goal = 24
        
    def step(self, action):
        # Move: 0=up, 1=down, 2=left, 3=right
        if action == 0 and self.position >= 5:
            self.position -= 5
        elif action == 1 and self.position < 20:
            self.position += 5
        elif action == 2 and self.position % 5 != 0:
            self.position -= 1
        elif action == 3 and self.position % 5 != 4:
            self.position += 1
        
        # Reward
        if self.position == self.goal:
            return self.position, 100, True  # state, reward, done
        else:
            return self.position, -1, False

# Q-Learning algorithm
Q = np.zeros((25, 4))  # 25 states, 4 actions
learning_rate = 0.1
discount = 0.9
epsilon = 0.1

env = SimpleGridWorld()

# Training
for episode in range(1000):
    state = 0
    env.position = 0
    
    for step in range(100):
        # Epsilon-greedy: explore vs exploit
        if np.random.random() < epsilon:
            action = np.random.randint(4)  # Explore
        else:
            action = np.argmax(Q[state])    # Exploit
        
        # Take action
        next_state, reward, done = env.step(action)
        
        # Update Q-value
        Q[state, action] = Q[state, action] + learning_rate * (
            reward + discount * np.max(Q[next_state]) - Q[state, action]
        )
        
        state = next_state
        
        if done:
            break

# Test learned policy
print("Learned path:")
state = 0
env.position = 0
path = [0]

for _ in range(20):
    action = np.argmax(Q[state])
    state, reward, done = env.step(action)
    path.append(state)
    if done:
        break

print(path)
```

---

### Policy Gradient (Robot Arm Control)

```python
import numpy as np

class RobotArm:
    def __init__(self):
        self.angle = 0  # Joint angle
        self.target = 45
        
    def step(self, action):
        # Action: change angle
        self.angle += action
        self.angle = np.clip(self.angle, 0, 90)
        
        # Reward: negative distance to target
        reward = -abs(self.target - self.angle)
        done = abs(self.target - self.angle) < 1
        
        return self.angle, reward, done

# Simple policy network (angle → action)
def policy(angle, theta):
    return theta[0] * angle + theta[1]

# Training
theta = np.random.randn(2)
learning_rate = 0.01
env = RobotArm()

for episode in range(100):
    state = 0
    env.angle = 0
    total_reward = 0
    
    for step in range(50):
        action = policy(state, theta)
        next_state, reward, done = env.step(action)
        
        # Update policy
        gradient = np.array([state, 1])
        theta += learning_rate * reward * gradient
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {total_reward:.1f}")
```

---

## Quick Reference

### Common ML Tasks in Robotics

**Perception:**
```python
# Object detection
objects = model.detect(camera_image)

# Semantic segmentation
road_mask = model.segment(image)

# Depth estimation
depth_map = model.estimate_depth(stereo_images)
```

**Control:**
```python
# Learn controller
action = policy_network.predict(state)

# Optimize trajectory
path = planner.optimize(start, goal, obstacles)
```

**Prediction:**
```python
# Predict trajectory
future_path = predictor.predict(object_history)

# Predict sensor value
next_reading = time_series_model.predict(sensor_history)
```

---

### Model Evaluation Metrics

**Classification:**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
```

**Regression:**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
```

---

### Training Tips

```python
# 1. Normalize input data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Save model
import joblib
joblib.dump(model, 'robot_model.pkl')

# 4. Load model
model = joblib.load('robot_model.pkl')
```

---

## Interview Questions

**Q: "What is machine learning?"**
**A:** Teaching computers to learn patterns from data instead of explicit programming. Used in robotics for perception, control, and decision making.

**Q: "Supervised vs Unsupervised learning?"**
**A:** Supervised uses labeled data (object detection). Unsupervised finds patterns without labels (clustering waypoints).

**Q: "What is a neural network?"**
**A:** Computational model inspired by brain neurons, organized in layers. Learns complex patterns through training.

**Q: "Why use ML in robotics?"**
**A:** Handle uncertainty, adapt to environments, learn from data, recognize patterns humans can't program explicitly.

**Q: "What is overfitting?"**
**A:** Model memorizes training data but fails on new data. Prevent with validation set, more data, regularization.

**Q: "Difference between CNN and regular neural network?"**
**A:** CNN specialized for images, uses convolution layers to extract spatial features. Better for vision tasks.

---

## Learning Path

### Week 1-2: Basics
- Python + NumPy basics
- scikit-learn simple examples
- Linear regression, classification

### Week 3-4: Neural Networks
- TensorFlow/PyTorch basics
- Simple neural network
- Train on small dataset

### Week 5-6: Computer Vision
- OpenCV basics
- Pre-trained models
- Object detection

### Week 7-8: RL Basics
- Q-learning
- Simple environment
- Train agent

---

## Resources

**Courses:**
- Coursera: "Machine Learning" by Andrew Ng
- fast.ai: "Practical Deep Learning for Coders"
- YouTube: sentdex, 3Blue1Brown

**Libraries:**
- scikit-learn (traditional ML)
- TensorFlow/Keras (deep learning)
- PyTorch (deep learning, research)
- OpenCV (computer vision)

**Datasets:**
- MNIST (handwritten digits)
- CIFAR-10 (objects)
- KITTI (autonomous driving)
- ImageNet (general images)

---

## Summary

**ML = Learning from Data**

**Three Types:**
1. Supervised (labeled data)
2. Unsupervised (no labels)
3. Reinforcement (trial & error)

**Robotics Applications:**
- Vision (detect, classify, segment)
- Control (navigate, grasp, balance)
- Prediction (trajectories, failures)

**Start Simple:**
1. Linear regression
2. Decision trees
3. Simple neural network
4. Pre-trained models
5. Custom models

---
