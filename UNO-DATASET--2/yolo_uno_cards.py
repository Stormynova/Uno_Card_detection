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

# Dictionary mapping the raw label names to more readable names
# Format: "internal_name": "display_name"
name_mapping = {
    "lblue-0": "blue 0",
    "lblue-1": "blue 1",
    "lblue-2": "blue 2",
    "lblue-20": "blue Draw two",
    "lblue-3": "blue 3",
    "lblue-4": "blue 4",
    "lblue-5": "blue 5",
    "lblue-6": "blue 6",
    "lblue-7": "blue 7",
    "lblue-8": "blue 8",
    "lblue-9": "blue 9",
    "lbluerevers-20": "blue reverse",
    "lblueskip-20": "blue skip",
    "lcolor-40": "Wild",
    "lcolor-400": "Wild draw 4",
    "lgreen-0": "green 0",
    "lgreen-1": "green 1",
    "lgreen-2": "green 2",
    "lgreen-20": "green Draw two",
    "lgreen-3": "green 3",
    "lgreen-4": "green 4",
    "lgreen-5": "green 5",
    "lgreen-6": "green 6",
    "lgreen-7": "green 7",
    "lgreen-8": "green 8",
    "lgreen-9": "green 9",
    "lgreenrevers-20": "green reverse",
    "lgreenskip-20": "green skip",
    "lred-0": "red 0",
    "lred-1": "red 1",
    "lred-2": "red 2",
    "lred-20": "red Draw two",
    "lred-3": "red 3",
    "lred-4": "red 4",
    "lred-5": "red 5",
    "lred-6": "red 6",
    "lred-7": "red 7",
    "lred-8": "red 8",
    "lred-9": "red 9",
    "lredrevers-20": "red reverse",
    "lredskip-20": "red skip",
    "lyellow-0": "yellow 0",
    "lyellow-1": "yellow 1",
    "lyellow-2": "yellow 2",
    "lyellow-20": "yellow Draw two",
    "lyellow-3": "yellow 3",
    "lyellow-4": "yellow 4",
    "lyellow-5": "yellow 5",
    "lyellow-6": "yellow 6",
    "lyellow-7": "yellow 7",
    "lyellow-8": "yellow 8",
    "lyellow-9": "yellow 9",
    "lyellowrevers-20": "yellow reverse",
    "lyellowskip-20": "yellow skip",
    "lbackground": "background",
}

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
test_dataset = "UNO-DATASET--2/tilted"
# model.val(data=cwd + "/" + test_dataset + "/data.yaml", save=True)

# Run predictions on test images
# Save results for visual inspection, get predictions as a list - compare with ground truth
predictions = model.predict(
    source=cwd + "/" + test_dataset + "/images",
    conf=0.25,  # Confidence threshold
    save=True  # Save predictions
)

# extract class ids, confidence scores and image names from predictions
class_ids = [pred.boxes.cls.numpy() for pred in predictions]
confidences = [pred.boxes.conf.numpy() for pred in predictions]
image_names = [pred.path for pred in predictions]

# Convert class IDs from numpy arrays to integers (54 is for background)
class_ids = [int(id[0]) if len(id) > 0 else 54 for id in class_ids]

# Convert confidence scores from numpy arrays to floats 
confidences = [float(conf[0]) if len(conf) > 0 else 0 for conf in confidences]

# Extract filenames from full paths
image_names = [os.path.basename(image_name) for image_name in image_names]

# labels are in the same directory as the images, with the same name but with .txt extension
# use image_names to find corresponding labels
labels = [os.path.join(test_dataset, "labels", os.path.splitext(image_name)[0] + '.txt') for image_name in image_names]

# extract class ids from label files - first number in the file
label_class_ids = []
for label in labels:
    with open(cwd + "/" + label, 'r') as file:
        label_class_ids.append(int(file.readline().split()[0]))

# Create confusion matrix and plot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Get unique class IDs and their corresponding names
unique_classes = sorted(list(set(label_class_ids + class_ids)))
class_names = [name_mapping[list(name_mapping.keys())[i]] if i < len(name_mapping) else "background" 
              for i in unique_classes]

cm = confusion_matrix(label_class_ids, class_ids)
# plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=90, values_format='')
plt.tight_layout()
plt.show()

# Plot F1 Score vs confidence
from sklearn.metrics import f1_score
import numpy as np

# Calculate F1 scores for different confidence thresholds
thresholds = np.arange(0.0, 1.0, 0.01)
f1_scores = []

for threshold in thresholds:
    # Filter predictions based on confidence threshold
    filtered_predictions = [class_id if conf >= threshold else -1 
                          for class_id, conf in zip(class_ids, confidences)]
    
    # Calculate F1 score
    f1 = f1_score(label_class_ids, filtered_predictions, average='macro')
    f1_scores.append(f1)

# Create subplots for all three plots
plt.figure(figsize=(15, 5))

# Plot F1 scores vs confidence thresholds
max_f1 = max(f1_scores)
max_threshold = thresholds[f1_scores.index(max_f1)]
plt.subplot(1, 3, 1)
plt.plot(thresholds, f1_scores)
plt.scatter(max_threshold, max_f1, color='red', label='Max F1 Score')
plt.title('F1 Score vs Confidence Threshold')
plt.xlabel('Confidence Threshold')
plt.ylabel('F1 Score')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(False)
plt.legend()


# Calculate precision and recall for different confidence thresholds
precisions = []
recalls = []

for threshold in thresholds:
    # Filter predictions based on confidence threshold
    filtered_predictions = [class_id if conf >= threshold else -1 
                          for class_id, conf in zip(class_ids, confidences)]
    
    # Calculate true positives, false positives, false negatives
    tp = sum((pred == true) and (pred != -1) for pred, true in zip(filtered_predictions, label_class_ids))
    fp = sum((pred != true) and (pred != -1) for pred, true in zip(filtered_predictions, label_class_ids))
    fn = sum((pred == -1) and (true != -1) for pred, true in zip(filtered_predictions, label_class_ids))
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    precisions.append(precision)
    recalls.append(recall)
    
# Plot precision vs recall curve
max_precision = max(precisions)
max_recall = recalls[precisions.index(max_precision)]
plt.subplot(1, 3, 2)
plt.plot(recalls, precisions)
plt.scatter(max_recall, max_precision, color='red', label='Max Precision')
plt.title('Precision vs Recall')
plt.xlabel('Recall')
plt.ylabel('Precision') 
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(False)
plt.legend()

# Plot Recall vs Confidence threshold
max_recall = max(recalls)
max_threshold = thresholds[recalls.index(max_recall)]
plt.subplot(1, 3, 3)
plt.plot(thresholds, recalls)
plt.scatter(max_threshold, max_recall, color='red', label='Max Recall')
plt.title('Recall vs Confidence Threshold')
plt.xlabel('Confidence Threshold')
plt.ylabel('Recall')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(False)
plt.legend()

plt.tight_layout()
plt.show()

# Visualization of Model Predictions
# ================================
# Display grid of first 10 prediction results
predict_folder = sorted(glob.glob(cwd + "/runs/detect/*"), key=os.path.getmtime)[-1]

plt.figure(figsize=(15, 8))
for idx, image_path in enumerate(glob.glob(predict_folder + "/*.jpg")[-10:]):
    plt.subplot(2, 5, idx+1)
    plt.imshow(plt.imread(image_path))
    plt.axis('off')
    plt.title(f'Prediction {idx+1}')
plt.tight_layout()
plt.show()
