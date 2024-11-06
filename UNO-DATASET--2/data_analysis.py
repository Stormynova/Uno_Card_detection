# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set matplotlib style for better looking plots
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

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
}

def analyze_dataset(images_dir, labels_dir):
    """
    Analyze and visualize the UNO card dataset statistics
    
    Args:
        images_dir (str): Path to directory containing images
        labels_dir (str): Path to directory containing labels
    """
    # Get list of all files in both directories
    images = os.listdir(images_dir)
    labels = os.listdir(labels_dir)

    # Print dataset statistics
    print(f"Number of images: {len(images)}")
    print(f"Number of labels: {len(labels)}")

    # Count occurrences of each class in the dataset
    class_counts = {}
    for label in labels:
        with open(os.path.join(labels_dir, label), 'r') as file:
            for line in file:
                # Extract class ID from each line and increment its count
                class_id = int(line.split()[0])
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

    # Create a mapping from numeric class IDs to human-readable names
    id_to_name = {i: name for i, name in enumerate(name_mapping.values())}

    # Prepare data for plotting
    names = [id_to_name[class_id] for class_id in class_counts.keys()]
    counts = [class_counts[class_id] for class_id in class_counts.keys()]

    # Create histogram visualization
    # plt.figure(figsize=(15, 6))
    # plt.bar(names, counts)
    # plt.xticks(rotation=45, ha='right')
    # plt.xlabel('Class Name')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of UNO Card Classes')
    # plt.tight_layout()
    
    # return plt
    return names, counts

# model
from ultralytics import YOLO
model = YOLO("UNO_Card_Detection_YOLOv8/train2/weights/best.pt")

# Example usage
images_dir = "UNO-DATASET--2/dark_bg/images"
labels_dir = "UNO-DATASET--2/dark_bg/labels"
analyze_dataset(images_dir, labels_dir)

cwd = os.getcwd()
img_dirs = [os.path.join(cwd, "UNO-DATASET--2/dark_bg/images"), os.path.join(cwd, "UNO-DATASET--2/non-plain_bg/images"), os.path.join(cwd, "UNO-DATASET--2/plain_bg/images"), os.path.join(cwd, "UNO-DATASET--2/tilted/images")]

# subplot each dataset  
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
colors = plt.cm.tab10.colors

for idx, img_dir in enumerate(img_dirs):
    names, counts = analyze_dataset(img_dir, img_dir.replace("images", "labels"))
    axes[idx // 2, idx % 2].bar(names, counts, color=colors[idx])
    axes[idx // 2, idx % 2].set_title(img_dir.split("/")[-1])
    axes[idx // 2, idx % 2].set_xlabel('Class Name')
    axes[idx // 2, idx % 2].set_ylabel('Frequency')
    axes[idx // 2, idx % 2].set_xticks(ticks=range(len(names)), labels=names, rotation=45, ha='right')
    axes[idx // 2, idx % 2].set_title(img_dir.split("/")[-2])
    
plt.tight_layout()
plt.show()

# Prediction analysis across datasets
def analyze_predictions(img_dirs):
    """Analyze model predictions across different datasets"""
    results = {}
    
    for img_dir in img_dirs:
        # Run predictions on dataset
        predictions = model.predict(
            source=img_dir,
            conf=0.25,
            save=False
        )
        
        # Extract metrics
        confidences = []
        num_detections = []
        class_distribution = {}
        
        for pred in predictions:
            # Get confidences for this image
            confs = pred.boxes.conf.cpu().numpy()
            confidences.extend(confs)
            num_detections.append(len(confs))
            
            # Get class distribution
            classes = pred.boxes.cls.cpu().numpy()
            for cls in classes:
                cls_name = name_mapping[model.names[int(cls)]]
                class_distribution[cls_name] = class_distribution.get(cls_name, 0) + 1
                
        results[img_dir] = {
            'mean_conf': np.mean(confidences),
            'std_conf': np.std(confidences),
            'avg_detections': np.mean(num_detections),
            'class_dist': class_distribution
        }
    
    # Visualize results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot mean confidences
    confs = [res['mean_conf'] for res in results.values()]
    conf_std = [res['std_conf'] for res in results.values()]
    dataset_names = [path.split('/')[-2] for path in results.keys()]
    ax1.bar(dataset_names, confs, yerr=conf_std)
    ax1.set_title('Mean Detection Confidence')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Confidence Score')
    ax1.set_xticklabels(dataset_names, rotation=45)
    
    # Plot average detections per image
    detections = [res['avg_detections'] for res in results.values()]
    ax2.bar(dataset_names, detections)
    ax2.set_title('Average Detections per Image')
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Number of Detections')
    ax2.set_xticklabels(dataset_names, rotation=45)
    
    # Plot class distribution for each dataset
    for i, (name, res) in enumerate(results.items()):
        dist = res['class_dist']
        ax3.plot(list(dist.values()), label=name.split('/')[-2], alpha=0.7)
    ax3.set_title('Class Distribution')
    ax3.set_xlabel('Class Index')
    ax3.set_ylabel('Number of Instances')
    ax3.legend()
    
    # Plot detection rate
    total_cards = [sum(res['class_dist'].values()) for res in results.values()]
    ax4.bar(dataset_names, total_cards)
    ax4.set_title('Total Cards Detected')
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Number of Cards')
    ax4.set_xticklabels(dataset_names, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run analysis
prediction_results = analyze_predictions(img_dirs)
