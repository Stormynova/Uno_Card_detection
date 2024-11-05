# ==========================================
# UNO Card Detection using YOLOv8
# ==========================================
# This script implements a card detection system using YOLOv8 model
# It supports both training mode and inference mode
# The model is trained to detect and classify different UNO cards

# Import required libraries
from roboflow import Roboflow  # For dataset management and download
import matplotlib.pyplot as plt  # For visualization of results
import ultralytics  # For YOLO implementation
from ultralytics import YOLO  # Main YOLO model
import torch  # For deep learning operations
import os  # For file and directory operations
import glob  # For file pattern matching

# Perform system compatibility checks for YOLO
# This ensures all dependencies are properly installed
ultralytics.checks()

# Configuration settings
# training_mode: Controls whether to train new model or use pre-trained weights
training_mode = False  # Set to True to train model, False to use pre-trained weights

# Define paths for model and weights
model_path = "UNO_Card_Detection_YOLOv8/train2/"
model = YOLO(model_path + "weights/best.pt")  # Load pre-trained YOLO model
cwd = os.getcwd()  # Get current working directory for file operations

if training_mode:
    # Training Mode Operations
    # ======================
    
    # Initialize Roboflow and download dataset
    # API key is required for accessing Roboflow services
    rf = Roboflow(api_key="m5KUBEIZuQhSrvYgOJgU")
    project = rf.workspace("university-of-trento-bsapj").project("uno-dataset")
    version = project.version(3)
    dataset = version.download("yolov8")  # Download dataset in YOLOv8 format

    # Model Configuration
    # ==================
    model = YOLO("yolov8s.pt")  # Initialize with YOLOv8 small model
    img_size = 640  # Standard image size for YOLOv8
    
    # Device Selection
    # Automatically choose MPS (Metal Performance Shaders) if available, else CPU
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Training Configuration
    # =====================
    # Comprehensive training setup with hyperparameters optimized for card detection
    model.train(
        data=cwd + "/UNO-DATASET--3/data.yaml",  # Dataset configuration file
        epochs=100,  # Number of training epochs
        imgsz=img_size,  # Input image size
        batch=32,  # Batch size per iteration
        augment=True,  # Enable data augmentation
        plots=True,  # Generate training plots
        workers=4,  # Number of worker threads
        lr0=1e-3,  # Initial learning rate
        lrf=0.01,  # Final learning rate factor
        warmup_epochs=3,  # Epochs for warmup
        warmup_momentum=0.5,  # Initial warmup momentum
        box=0.05,  # Box loss gain
        cls=0.5,  # Class loss gain
        mosaic=1.0,  # Mosaic augmentation
        mixup=0.1,  # Mixup augmentation
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,  # HSV-Saturation augmentation
        hsv_v=0.4,  # HSV-Value augmentation
        perspective=0.001,  # Perspective augmentation
        flipud=0.5,  # Vertical flip probability
        fliplr=0.5,  # Horizontal flip probability
        copy_paste=0.2,  # Copy-paste augmentation
        device=device,  # Computing device
        project="UNO_Card_Detection_YOLOv8",  # Project name for saving results
    )

# Visualization and Analysis
# =========================

# Plot 1: Display Confusion Matrix
plt.figure(1)
plt.imshow(plt.imread(cwd + "/" + model_path + "confusion_matrix.png"))
plt.title('Confusion Matrix')

# Plot 2: Show Training Results
plt.figure(2) 
plt.imshow(plt.imread(cwd + "/" + model_path + "results.png"))
plt.title('Results')

# Plot 3: Display Validation Batch Labels
plt.figure(3)
plt.imshow(plt.imread(cwd + "/" + model_path + "val_batch0_labels.jpg"))
plt.title('Validation Batch Labels')
plt.show()

# Model Validation and Testing
# ===========================

# Validate model performance on test dataset
test_dataset = "UNO-DATASET--3"
model.val(data=cwd + "/" + test_dataset + "/data.yaml")

# Run predictions on test images
# Save results for visual inspection
model.predict(
    source=cwd + "/" + test_dataset + "/test/images",
    conf=0.25,  # Confidence threshold
    save=True  # Save predictions
)

# Visualization of Model Predictions
# ================================
# Display grid of first 10 prediction results
plt.figure(figsize=(15, 8))
for idx, image_path in enumerate(glob.glob(cwd + "/UNO_Card_Detection_YOLOv8/train24/*.jpg")[:10]):
    print(image_path)
    plt.subplot(2, 5, idx+1)
    plt.imshow(plt.imread(image_path))
    plt.axis('off')
    plt.title(f'Prediction {idx+1}')
plt.tight_layout()
plt.show()
