# TensorFlow Fundamentals for Robotics

## Why TensorFlow in Robotics?

TensorFlow is widely used in robotics because:

- **Production-Ready**: Designed for deployment at scale
- **TensorFlow Lite**: Optimized for mobile and embedded devices
- **Keras API**: High-level, user-friendly interface
- **TensorBoard**: Powerful visualization tools
- **Model Optimization**: Built-in quantization and pruning
- **Cross-Platform**: Runs on CPU, GPU, TPU, and edge devices
- **ROS Integration**: Easy integration with ROS2 nodes

---

## 1. Installation & Setup

### Basic Installation
```bash
# CPU version
pip install tensorflow

# GPU version (includes CPU)
pip install tensorflow[and-cuda]

# TensorFlow Lite (for embedded systems)
pip install tensorflow-lite

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 2. Tensors (Core Data Structure)

### What Are Tensors?
Tensors are multi-dimensional arrays - similar to NumPy arrays but with GPU support and automatic differentiation.

```python
import tensorflow as tf
import numpy as np

# Create tensors
scalar = tf.constant(3.14)                          # 0D tensor
vector = tf.constant([1, 2, 3])                     # 1D tensor
matrix = tf.constant([[1, 2], [3, 4]])              # 2D tensor
tensor_3d = tf.zeros([2, 3, 4])                     # 3D tensor

# Common shapes in robotics
image = tf.zeros([480, 640, 3])                     # RGB image: [height, width, channels]
batch_images = tf.zeros([32, 480, 640, 3])          # Batch of 32 images
lidar_scan = tf.zeros([360])                        # 360-degree lidar scan
```

### Creating Tensors

```python
# Various creation methods
zeros = tf.zeros([3, 3])                            # All zeros
ones = tf.ones([2, 4])                              # All ones
random = tf.random.normal([3, 3])                   # Random normal distribution
range_tensor = tf.range(0, 10, 2)                   # [0, 2, 4, 6, 8]
linspace = tf.linspace(0.0, 1.0, 5)                 # 5 evenly spaced values

# From Python lists/NumPy
from_list = tf.constant([1, 2, 3])
numpy_array = np.array([1, 2, 3])
from_numpy = tf.constant(numpy_array)

# Convert back to NumPy
tensor = tf.constant([1, 2, 3])
numpy_array = tensor.numpy()
```

### Tensor Properties

```python
x = tf.random.normal([3, 4, 5])

print(x.shape)        # TensorShape([3, 4, 5])
print(x.dtype)        # <dtype: 'float32'>
print(x.device)       # /job:localhost/replica:0/task:0/device:GPU:0
```

---

## 3. Tensor Operations

### Basic Math Operations

```python
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])

# Element-wise operations
c = a + b              # or tf.add(a, b) → [5., 7., 9.]
c = a * b              # or tf.multiply(a, b) → [4., 10., 18.]
c = a / b              # or tf.divide(a, b)
c = tf.pow(a, 2)       # [1., 4., 9.]

# Reduction operations
sum_all = tf.reduce_sum(a)          # 6.0
mean = tf.reduce_mean(a)            # 2.0
max_val = tf.reduce_max(a)          # 3.0
```

### Matrix Operations

```python
# Matrix multiplication
A = tf.random.normal([3, 4])
B = tf.random.normal([4, 5])
C = tf.matmul(A, B)    # or A @ B → shape: [3, 5]

# Transpose
A_T = tf.transpose(A)  # shape: [4, 3]

# Dot product
v1 = tf.constant([1., 2., 3.])
v2 = tf.constant([4., 5., 6.])
dot = tf.tensordot(v1, v2, axes=1)  # 32.0
```

### Reshaping & Indexing

```python
x = tf.range(12)  # [0, 1, 2, ..., 11]

# Reshape
y = tf.reshape(x, [3, 4])      # 3x4 matrix
z = tf.reshape(x, [-1, 2])     # Auto-calculate first dimension → 6x2

# Indexing (similar to NumPy)
print(y[0, :])        # First row
print(y[:, 1])        # Second column
print(y[1:3, 2:4])    # Slice rows 1-2, columns 2-3

# Boolean masking
mask = y > 5
filtered = tf.boolean_mask(y, mask)  # All elements > 5
```

---

## 4. GPU Acceleration

### Automatic Device Placement

TensorFlow automatically places operations on GPU if available.

```python
# Check available devices
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

# Manual device placement
with tf.device('/GPU:0'):
    x = tf.random.normal([1000, 1000])
    y = tf.matmul(x, x)

with tf.device('/CPU:0'):
    x = tf.random.normal([1000, 1000])

# Memory growth (prevent TF from allocating all GPU memory)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

---

## 5. Building Neural Networks with Keras

### Sequential API (Simple Models)

```python
from tensorflow import keras
from tensorflow.keras import layers

# Simple feedforward network
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

print(model.summary())
```

### Functional API (More Flexible)

```python
# Input layer
inputs = keras.Input(shape=(784,))

# Hidden layers
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)

# Output layer
outputs = layers.Dense(10, activation='softmax')(x)

# Create model
model = keras.Model(inputs=inputs, outputs=outputs)
```

### Custom Model Class (Most Flexible)

```python
class RobotVisionModel(keras.Model):
    def __init__(self):
        super(RobotVisionModel, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.pool = layers.MaxPooling2D()
        self.conv2 = layers.Conv2D(64, 3, activation='relu')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropout(x)
        return self.dense2(x)

model = RobotVisionModel()
```

---

## 6. Convolutional Neural Networks (CNNs)

### CNN for Robot Vision

```python
def create_vision_model(input_shape=(480, 640, 3), num_classes=10):
    """
    CNN for robot vision tasks
    Input: RGB images from robot camera
    Output: Object classifications
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

model = create_vision_model()
model.summary()
```

### Important Layer Types

```python
# Convolutional layers
layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')

# Pooling layers
layers.MaxPooling2D(pool_size=(2, 2))
layers.AveragePooling2D(pool_size=(2, 2))

# Normalization
layers.BatchNormalization()

# Regularization
layers.Dropout(rate=0.5)

# Recurrent layers (for sequences)
layers.LSTM(units=128, return_sequences=True)
layers.GRU(units=64)
```

---

## 7. Training Models

### Compile & Train

```python
# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')
```

### Custom Training Loop

```python
# Define optimizer and loss
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

# Metrics
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()

@tf.function  # Compile for faster execution
def train_step(x, y):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    # Backward pass
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    
    # Update metrics
    train_acc_metric.update_state(y, predictions)
    return loss

# Training loop
epochs = 10
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Iterate over batches
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss = train_step(x_batch, y_batch)
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")
    
    # Display metrics
    train_acc = train_acc_metric.result()
    print(f"Training accuracy: {train_acc:.4f}")
    train_acc_metric.reset_states()
```

---

## 8. Data Pipeline with tf.data

### Creating Datasets

```python
# From NumPy arrays
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# From image files
def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0  # Normalize
    return image, label

image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
labels = [0, 1]
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_image)
```

### Data Pipeline Optimization

```python
# Create efficient data pipeline
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(buffer_size=1000)           # Shuffle data
    .batch(32)                           # Batch size
    .prefetch(buffer_size=AUTOTUNE)      # Prefetch for performance
    .cache()                             # Cache in memory
)

# For large datasets (can't fit in memory)
train_dataset = (
    tf.data.Dataset.from_tensor_slices(file_paths)
    .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    .shuffle(buffer_size=1000)
    .batch(32)
    .prefetch(buffer_size=AUTOTUNE)
)
```

### Data Augmentation

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Apply during training
def augment_image(image, label):
    image = data_augmentation(image, training=True)
    return image, label

train_dataset = train_dataset.map(augment_image)
```

---

## 9. Transfer Learning

### Using Pre-trained Models

```python
# Load pre-trained model (ImageNet weights)
base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,  # Remove final classification layer
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom layers
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Compile and train
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Fine-tuning

```python
# After initial training, unfreeze some layers
base_model.trainable = True

# Freeze early layers, fine-tune later layers
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Lower learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
history_fine = model.fit(train_dataset, epochs=10, validation_data=val_dataset)
```

### Common Pre-trained Models

```python
# MobileNetV2 (lightweight, good for embedded systems)
mobile_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False)

# EfficientNet (best accuracy-efficiency trade-off)
efficient_model = keras.applications.EfficientNetB0(weights='imagenet', include_top=False)

# ResNet (standard choice)
resnet_model = keras.applications.ResNet50(weights='imagenet', include_top=False)

# VGG16 (simpler architecture)
vgg_model = keras.applications.VGG16(weights='imagenet', include_top=False)
```

---

## 10. Saving & Loading Models

### Save/Load Entire Model

```python
# Save entire model (architecture + weights + optimizer state)
model.save('robot_model.h5')  # HDF5 format
model.save('robot_model')      # SavedModel format (recommended)

# Load
loaded_model = keras.models.load_model('robot_model')
loaded_model.summary()
```

### Save/Load Weights Only

```python
# Save weights only
model.save_weights('model_weights.h5')

# Load weights (need to create model first)
new_model = create_vision_model()
new_model.load_weights('model_weights.h5')
```

### Checkpointing During Training

```python
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model_epoch_{epoch:02d}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[checkpoint_callback]
)
```

---

## 11. TensorFlow Lite (For Embedded Systems)

### Convert Model to TFLite

```python
# Convert Keras model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimizations for embedded devices
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Quantization (reduce model size)
converter.target_spec.supported_types = [tf.float16]  # 16-bit float
# Or for 8-bit integer quantization:
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Convert
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Run TFLite Model (Inference)

```python
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get output
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
```

---

## 12. Object Detection

### Using Pre-trained Object Detection Models

```python
# Load pre-trained SSD MobileNet
model = tf.saved_model.load('ssd_mobilenet_v2/saved_model')

def detect_objects(image):
    """
    Detect objects in image
    Args:
        image: NumPy array (H, W, 3)
    Returns:
        boxes, classes, scores
    """
    # Preprocess
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    # Run detection
    detections = model(input_tensor)
    
    # Extract results
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    
    boxes = detections['detection_boxes']
    classes = detections['detection_classes'].astype(np.int64)
    scores = detections['detection_scores']
    
    return boxes, classes, scores

# Usage
image = load_image('robot_view.jpg')
boxes, classes, scores = detect_objects(image)

# Filter by confidence
confidence_threshold = 0.5
for i in range(len(scores)):
    if scores[i] > confidence_threshold:
        print(f"Detected class {classes[i]} with confidence {scores[i]:.2f}")
```

---

## 13. Integration with ROS2

### TensorFlow in ROS2 Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tensorflow as tf
import numpy as np

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        # Load TensorFlow model
        self.model = tf.keras.models.load_model('robot_model')
        self.bridge = CvBridge()
        
        # ROS2 setup
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(Image, '/detected_objects', 10)
        
        self.get_logger().info('Vision Node Started')
    
    def preprocess_image(self, cv_image):
        """Preprocess image for model"""
        image = cv2.resize(cv_image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image.astype(np.float32)
    
    def image_callback(self, msg):
        """Process incoming images"""
        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Preprocess
        input_tensor = self.preprocess_image(cv_image)
        
        # Run inference
        predictions = self.model.predict(input_tensor, verbose=0)
        class_id = np.argmax(predictions[0])
        confidence = predictions[0][class_id]
        
        self.get_logger().info(
            f'Detected class {class_id} with confidence {confidence:.2f}')
        
        # Publish results (example)
        # ...

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Optimized Inference Node

```python
class OptimizedVisionNode(Node):
    def __init__(self):
        super().__init__('optimized_vision_node')
        
        # Load TFLite model for faster inference
        self.interpreter = tf.lite.Interpreter(model_path='model.tflite')
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        # Process every Nth frame for efficiency
        self.frame_count = 0
        self.process_every_n_frames = 3
    
    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return
        
        # Convert and preprocess
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        input_tensor = self.preprocess_image(cv_image)
        
        # TFLite inference
        self.interpreter.set_tensor(
            self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(
            self.output_details[0]['index'])
        
        # Process results
        class_id = np.argmax(output[0])
        confidence = output[0][class_id]
        
        self.get_logger().info(f'Class: {class_id}, Conf: {confidence:.2f}')
```

---

## 14. Common Robotics Applications

### 1. Semantic Segmentation

```python
def create_segmentation_model(input_shape=(480, 640, 3), num_classes=20):
    """U-Net style architecture for semantic segmentation"""
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    
    # Bottleneck
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    
    # Decoder
    u1 = layers.UpSampling2D(2)(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(u1)
    
    u2 = layers.UpSampling2D(2)(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    
    # Output
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(c5)
    
    return keras.Model(inputs, outputs)
```

### 2. Depth Estimation

```python
def create_depth_estimation_model(input_shape=(480, 640, 3)):
    """Monocular depth estimation from RGB image"""
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Encoder (frozen)
    base_model.trainable = False
    
    # Decoder
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs)
    
    # Upsample to original resolution
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    
    # Output depth map
    outputs = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    
    return keras.Model(inputs, outputs)
```

### 3. Reinforcement Learning (Robot Control)

```python
class RobotControlNetwork(keras.Model):
    """Actor-Critic network for robot control"""
    def __init__(self, num_actions):
        super(RobotControlNetwork, self).__init__()
        
        # Shared layers
        self.conv1 = layers.Conv2D(32, 8, strides=4, activation='relu')
        self.conv2 = layers.Conv2D(64, 4, strides=2, activation='relu')
        self.conv3 = layers.Conv2D(64, 3, strides=1, activation='relu')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(512, activation='relu')
        
        # Actor head (policy)
        self.policy = layers.Dense(num_actions, activation='softmax')
        
        # Critic head (value function)
        self.value = layers.Dense(1)
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        
        policy = self.policy(x)
        value = self.value(x)
        
        return policy, value
```

---

## 15. Performance Optimization

### Mixed Precision Training

```python
from tensorflow.keras import mixed_precision

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Build model (automatically uses mixed precision)
model = create_vision_model()

# Use loss scaling
optimizer = keras.optimizers.Adam()
optimizer = mixed_precision.LossScaleOptimizer(optimizer)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
```

### Model Pruning (Reduce Size)

```python
import tensorflow_model_optimization as tfmot

# Prune model
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

model_for_pruning.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with pruning
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
model_for_pruning.fit(train_dataset, epochs=10, callbacks=callbacks)

# Export pruned model
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
```

---

## 16. TensorBoard Visualization

### Setup TensorBoard

```python
# Create callback
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch'
)

# Train with TensorBoard
model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[tensorboard_callback]
)

# Launch TensorBoard
# tensorboard --logdir=./logs
```

### Custom Metrics in TensorBoard

```python
# Create file writer
train_writer = tf.summary.create_file_writer('./logs/train')
val_writer = tf.summary.create_file_writer('./logs/val')

# Log custom metrics
with train_writer.as_default():
    tf.summary.scalar('custom_metric', value, step=epoch)
    tf.summary.image('input_images', images, step=epoch, max_outputs=3)
```

---

## 17. Debugging Tips

```python
# Enable eager execution (default in TF 2.x)
tf.config.run_functions_eagerly(True)

# Check tensor values
tf.debugging.assert_all_finite(tensor, message="Tensor contains NaN or Inf")

# Print tensor shape during model execution
class DebugLayer(keras.layers.Layer):
    def call(self, inputs):
        tf.print("Shape:", tf.shape(inputs))
        return inputs

# Check for NaN in gradients
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_weights)
    
    # Check for NaN
    for grad in gradients:
        tf.debugging.check_numerics(grad, "Gradient is NaN or Inf")
    
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss
```

---

## Essential TensorFlow Concepts for Robotics Summary

✅ **Must Know:**
- Tensors and operations
- Keras Sequential/Functional API
- Data pipelines (tf.data)
- Model training and evaluation
- Saving/loading models
- TensorFlow Lite for deployment

✅ **Important for Robotics:**
- Transfer learning with pre-trained models
- TFLite optimization for embedded systems
- ROS2 integration
- Real-time inference optimization
- Object detection and segmentation
- Model quantization and pruning

✅ **Common Pitfalls:**
- Not using `training=False` during inference
- Forgetting to normalize input data
- Memory issues with large models (use mixed precision)
- Not optimizing data pipeline (use prefetch, cache)

---

## PyTorch vs TensorFlow for Robotics

| Feature | PyTorch | TensorFlow |
|---------|---------|------------|
| **Ease of Use** | More Pythonic, intuitive | Keras API is user-friendly |
| **Deployment** | TorchScript, ONNX | TFLite (better for embedded) |
| **Production** | Growing ecosystem | More mature deployment tools |
| **Research** | Preferred in academia | Catching up with TF 2.x |
| **Mobile/Edge** | Limited support | Excellent (TFLite) |
| **Community** | Large, active | Very large, enterprise |

**Recommendation:** Use TensorFlow if deploying to embedded/edge devices (robots). Use PyTorch for research and flexibility.

---

## Next Steps

1. Build a simple image classifier with Keras
2. Convert model to TFLite and test on Raspberry Pi
3. Integrate with ROS2 for real-time vision
4. Implement object detection for navigation
5. Explore reinforcement learning for robot control
6. Optimize models for your target hardware

---
- Model Optimization: https://www.tensorflow.org/model_optimization
- TensorBoard: https://www.tensorflow.org/tensorboard
