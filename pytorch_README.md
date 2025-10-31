# PyTorch Fundamentals for Robotics

## Why PyTorch in Robotics?

PyTorch is increasingly important in robotics because:

- **Computer Vision**: Object detection, segmentation, tracking
- **Deep Learning**: Neural networks for perception and control
- **Research-Friendly**: Easy prototyping and experimentation
- **Dynamic Computation**: Flexible model architectures
- **GPU Acceleration**: Fast training and inference
- **ROS Integration**: Can be integrated with ROS2 nodes

---

## 1. Installation & Setup

### Basic Installation
```bash
# CPU version
pip install torch torchvision torchaudio

# GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 2. Tensors (Core Data Structure)

### What Are Tensors?
Tensors are multi-dimensional arrays - the fundamental data structure in PyTorch. Think of them as NumPy arrays with GPU support.

```python
import torch

# Create tensors
scalar = torch.tensor(3.14)                    # 0D tensor (scalar)
vector = torch.tensor([1, 2, 3])              # 1D tensor (vector)
matrix = torch.tensor([[1, 2], [3, 4]])       # 2D tensor (matrix)
tensor_3d = torch.zeros(2, 3, 4)              # 3D tensor (2x3x4)

# Common shapes in robotics
image = torch.zeros(3, 480, 640)              # RGB image: [channels, height, width]
batch_images = torch.zeros(32, 3, 480, 640)   # Batch of 32 images
lidar_scan = torch.zeros(360)                 # 360-degree lidar scan
```

### Creating Tensors

```python
# Various creation methods
zeros = torch.zeros(3, 3)                     # All zeros
ones = torch.ones(2, 4)                       # All ones
random = torch.randn(3, 3)                    # Random normal distribution
range_tensor = torch.arange(0, 10, 2)         # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)            # 5 evenly spaced values

# From Python lists/NumPy
import numpy as np
from_list = torch.tensor([1, 2, 3])
from_numpy = torch.from_numpy(np.array([1, 2, 3]))
```

### Tensor Properties

```python
x = torch.randn(3, 4, 5)

print(x.shape)        # torch.Size([3, 4, 5])
print(x.dtype)        # torch.float32
print(x.device)       # cpu or cuda
print(x.requires_grad) # False (whether to track gradients)
```

---

## 3. Tensor Operations

### Basic Math Operations

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise operations
c = a + b              # [5., 7., 9.]
c = a * b              # [4., 10., 18.]
c = a / b              # Division
c = a ** 2             # [1., 4., 9.]

# In-place operations (modify original tensor)
a.add_(b)              # a is now [5., 7., 9.]
a.mul_(2)              # a is now [10., 14., 18.]
```

### Matrix Operations

```python
# Matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.matmul(A, B)  # or A @ B → shape: [3, 5]

# Transpose
A_T = A.T               # or A.transpose(0, 1)

# Dot product
v1 = torch.tensor([1., 2., 3.])
v2 = torch.tensor([4., 5., 6.])
dot = torch.dot(v1, v2)  # 32.0
```

### Reshaping & Indexing

```python
x = torch.arange(12)  # [0, 1, 2, ..., 11]

# Reshape
y = x.view(3, 4)      # 3x4 matrix
z = x.view(-1, 2)     # Auto-calculate first dimension → 6x2

# Indexing (like NumPy)
print(y[0, :])        # First row
print(y[:, 1])        # Second column
print(y[1:3, 2:4])    # Slice rows 1-2, columns 2-3

# Advanced indexing
mask = y > 5
filtered = y[mask]    # All elements > 5
```

---

## 4. GPU Acceleration

### Moving Tensors to GPU

```python
# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create tensor on GPU
x_gpu = torch.randn(1000, 1000, device=device)

# Move existing tensor to GPU
x_cpu = torch.randn(1000, 1000)
x_gpu = x_cpu.to(device)

# Move back to CPU
x_back = x_gpu.cpu()

# Common pattern for robotics
class RobotVision:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def process_image(self, image):
        # Move to GPU for processing
        img_tensor = torch.from_numpy(image).to(self.device)
        # ... process ...
        return result.cpu().numpy()  # Return as NumPy on CPU
```

---

## 5. Building Neural Networks

### Basic Network Structure

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Define forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on output (done in loss)
        return x

# Create model
model = SimpleClassifier(input_size=784, hidden_size=128, num_classes=10)
print(model)
```

### Convolutional Neural Network (for Images)

```python
class RobotVisionCNN(nn.Module):
    def __init__(self):
        super(RobotVisionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 3 input channels (RGB)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 60 * 80, 256)  # Depends on input image size
        self.fc2 = nn.Linear(256, 10)  # 10 classes (e.g., object types)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input: [batch, 3, 480, 640]
        
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 240, 320]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 120, 160]
        x = self.pool(F.relu(self.conv3(x)))  # [batch, 128, 60, 80]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [batch, 128*60*80]
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```

---

## 6. Training Loop

### Complete Training Example

```python
import torch.optim as optim

# Setup
model = RobotVisionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set to training mode
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}], Loss: {loss.item():.4f}')
    
    # Validation
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}] Validation Accuracy: {accuracy:.2f}%')
```

### Key Training Concepts

```python
# 1. Set model mode
model.train()  # Enables dropout, batch normalization training mode
model.eval()   # Disables dropout, batch normalization eval mode

# 2. Zero gradients (important!)
optimizer.zero_grad()  # Clear previous gradients

# 3. No gradient computation for inference
with torch.no_grad():
    output = model(input)  # Faster, uses less memory
```

---

## 7. Data Loading

### Dataset & DataLoader

```python
from torch.utils.data import Dataset, DataLoader
import cv2

class RobotImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Create DataLoader
dataset = RobotImageDataset(image_paths, labels, transform=None)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Iterate through batches
for images, labels in dataloader:
    # images: [32, 3, H, W]
    # labels: [32]
    pass
```

### Data Augmentation

```python
from torchvision import transforms

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## 8. Transfer Learning (Pre-trained Models)

### Using Pre-trained Models

```python
import torchvision.models as models

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Freeze all layers (don't train them)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for your task
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)  # Only train this layer

# Or fine-tune entire model
for param in model.parameters():
    param.requires_grad = True

model = model.to(device)
```

### Common Pre-trained Models for Robotics

```python
# Image Classification
resnet = models.resnet50(pretrained=True)
mobilenet = models.mobilenet_v2(pretrained=True)  # Lightweight for embedded systems

# Object Detection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
detector = fasterrcnn_resnet50_fpn(pretrained=True)

# Semantic Segmentation
from torchvision.models.segmentation import deeplabv3_resnet50
segmenter = deeplabv3_resnet50(pretrained=True)
```

---

## 9. Saving & Loading Models

### Save/Load Entire Model

```python
# Save
torch.save(model, 'robot_model.pth')

# Load
model = torch.load('robot_model.pth')
model.eval()
```

### Save/Load State Dict (Recommended)

```python
# Save (only weights, not architecture)
torch.save(model.state_dict(), 'robot_weights.pth')

# Load
model = RobotVisionCNN()  # Create model first
model.load_state_dict(torch.load('robot_weights.pth'))
model.eval()
```

### Save Training Checkpoint

```python
# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

---

## 10. Inference for Robotics

### Real-time Inference Example

```python
class ObjectDetector:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Pre-processing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, path):
        model = RobotVisionCNN()
        model.load_state_dict(torch.load(path))
        return model.to(self.device)
    
    def detect(self, image):
        """
        Args:
            image: NumPy array (H, W, 3) BGR format from OpenCV
        Returns:
            predictions: class probabilities
        """
        # Pre-process
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb)
        input_batch = input_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Inference
        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = F.softmax(output, dim=1)
        
        return probabilities.cpu().numpy()[0]

# Usage in ROS2 node
detector = ObjectDetector('model.pth')

def image_callback(msg):
    # Convert ROS image to NumPy
    image = bridge.imgmsg_to_cv2(msg, "bgr8")
    
    # Run detection
    predictions = detector.detect(image)
    
    # Get top prediction
    class_idx = predictions.argmax()
    confidence = predictions[class_idx]
    
    print(f"Detected: {class_names[class_idx]} ({confidence:.2f})")
```

---

## 11. Common Robotics Applications

### 1. Object Detection

```python
def detect_objects(image, model, confidence_threshold=0.5):
    """Detect objects in robot's camera view"""
    model.eval()
    with torch.no_grad():
        # Get predictions
        predictions = model(image)
        
        # Filter by confidence
        boxes = predictions['boxes'][predictions['scores'] > confidence_threshold]
        labels = predictions['labels'][predictions['scores'] > confidence_threshold]
        scores = predictions['scores'][predictions['scores'] > confidence_threshold]
    
    return boxes, labels, scores
```

### 2. Depth Estimation

```python
class DepthEstimator(nn.Module):
    def __init__(self):
        super(DepthEstimator, self).__init__()
        # Encoder
        self.encoder = models.resnet50(pretrained=True)
        # Decoder for depth map
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 1, 3, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth
```

### 3. Path Planning with CNN

```python
class PathPlanningNet(nn.Module):
    def __init__(self):
        super(PathPlanningNet, self).__init__()
        # Input: occupancy grid map
        # Output: waypoints or control commands
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * 50 * 50, 2)  # Output: (vx, omega)
    
    def forward(self, occupancy_map):
        x = self.conv_layers(occupancy_map)
        x = x.view(x.size(0), -1)
        control = self.fc(x)
        return control
```

---

## 12. PyTorch with ROS2 Integration

### Publisher Node with PyTorch

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
        # ROS setup
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(Image, '/detected_objects', 10)
        
        self.get_logger().info('Object Detection Node Started')
    
    def load_model(self):
        model = RobotVisionCNN()
        model.load_state_dict(torch.load('model.pth'))
        model.to(self.device)
        model.eval()
        return model
    
    def image_callback(self, msg):
        # Convert to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Run inference
        detections = self.detect(cv_image)
        
        # Publish results
        # ... (publish detection results)
    
    def detect(self, image):
        # Pre-process
        input_tensor = self.preprocess(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        return output

def main():
    rclpy.init()
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

---

## 13. Performance Optimization

### Mixed Precision Training (Faster Training)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in train_loader:
    optimizer.zero_grad()
    
    # Automatic mixed precision
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    # Scaled backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Model Quantization (Faster Inference)

```python
# Dynamic quantization (good for CPU)
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Static quantization (better accuracy)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)
```

---

## 14. Debugging Tips

```python
# Check tensor shapes
print(f"Shape: {tensor.shape}")

# Check for NaN or Inf
torch.isnan(tensor).any()  # True if any NaN
torch.isinf(tensor).any()  # True if any Inf

# Register hooks to monitor gradients
def print_grad(grad):
    print(f"Gradient: {grad}")

tensor.register_hook(print_grad)

# Visualize model architecture
from torchsummary import summary
summary(model, input_size=(3, 224, 224))
```

---

## Essential PyTorch Concepts for Robotics Summary

✅ **Must Know:**
- Tensors and tensor operations
- GPU acceleration (`.to(device)`)
- Building neural networks (`nn.Module`)
- Training loop structure
- Saving/loading models
- Inference optimization

✅ **Important for Robotics:**
- Transfer learning (pre-trained models)
- Real-time inference
- ROS2 integration
- Object detection/segmentation
- Model quantization for edge devices

✅ **Common Pitfall:**
- Forgetting `model.eval()` during inference
- Not moving data to same device as model
- Not using `torch.no_grad()` for inference
- Memory leaks from not detaching tensors

---

## Next Steps

1. Practice tensor operations and GPU usage
2. Build a simple image classifier
3. Integrate pre-trained model with ROS2
4. Implement real-time object detection
5. Optimize for embedded systems (Jetson Nano, etc.)
6. Explore advanced topics: reinforcement learning, SLAM with deep learning

---

## Useful Resources

- Official Docs: https://pytorch.org/docs/
- Tutorials: https://pytorch.org/tutorials/
- Pre-trained Models: https://pytorch.org/vision/stable/models.html
- ROS2 Integration: Custom implementation or existing packages
