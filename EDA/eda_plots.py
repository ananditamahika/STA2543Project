import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from glob import glob
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib.patches import Patch

# Set Seaborn style for polished visuals
sns.set_theme(style="whitegrid", palette="pastel")

# Directories 
LABELS_DIR = "/Users/anandita/Downloads/MNR_sample/obj_train_data"  # Directory containing .txt annotation files
IMAGES_DIR = "CFC_dataset/nushagak"  # Directory containing image files

# Image dimensions 
IMAGE_WIDTH = 1086  
IMAGE_HEIGHT = 2125   

def get_fish_counts(annotations_dir):
    """
    Reads annotation files and counts the number of fish (bounding boxes) per image.
    
    Parameters:
        annotations_dir (str): Path to the directory containing .txt annotation files.

    Returns:
        list: A list of fish counts per image.
    """
    fish_counts = []

    # List all annotation files
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith(".txt")]

    # Iterate through each annotation file
    for file in annotation_files:
        file_path = os.path.join(annotations_dir, file)

        # Count the number of lines in the file (each line represents a fish annotation)
        with open(file_path, "r") as f:
            num_fish = sum(1 for _ in f)  

        fish_counts.append(num_fish)

    return fish_counts


def plot_fish_count_histogram(fish_counts, bins=20):
    """
    Plots a histogram of fish counts per image with enhanced aesthetics.

    Parameters:
        fish_counts (list): List of fish counts per image.
        bins (int): Number of bins for the histogram.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(fish_counts, bins=bins, kde=True, color="royalblue", alpha=0.7, edgecolor="black")

    plt.xlabel("Number of Fish per Image", fontsize=14, fontweight='bold')
    plt.ylabel("Frequency", fontsize=14, fontweight='bold')
    plt.title("Distribution of Fish Counts per Image", fontsize=16, fontweight='bold', pad=15)

    # Add grid and adjust layout
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Annotate key statistics
    mean_count = np.mean(fish_counts)
    median_count = np.median(fish_counts)
    
    plt.axvline(mean_count, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_count:.1f}')
    plt.axvline(median_count, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_count:.1f}')
    
    plt.legend()
    plt.show()


def plot_fish_count_boxplot(fish_counts):
    """
    Plots a box plot of fish counts per image with a refined style.

    Parameters:
        fish_counts (list): List of fish counts per image.
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=fish_counts, color="lightcoral", width=0.4, linewidth=2, fliersize=6, boxprops=dict(facecolor='lightpink'))

    plt.xlabel("Fish Counts", fontsize=14, fontweight='bold')
    plt.title("Box Plot of Fish Counts per Image", fontsize=16, fontweight='bold', pad=15)

    # Show statistics directly on the plot
    median = np.median(fish_counts)
    plt.axhline(median, color='blue', linestyle='dashed', linewidth=2, label=f'Median: {median:.1f}')
    plt.legend()
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

def generate_heatmap():
    heatmap = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    all_fish_counts = []
    fish_positions = []
    
    for label_file in sorted(glob(os.path.join(LABELS_DIR, "*.txt"))):
        with open(label_file, 'r') as f:
            bboxes = [list(map(float, line.strip().split())) for line in f.readlines()]
        fish_count = len(bboxes)
        all_fish_counts.append(fish_count)
        
        for bbox in bboxes:
            _, x_center, y_center, width, height = bbox  # Assuming YOLO format
            x_pixel = int(x_center * IMAGE_WIDTH)
            y_pixel = int(y_center * IMAGE_HEIGHT)
            heatmap[y_pixel, x_pixel] += 1
            fish_positions.append([x_pixel, y_pixel])
    
    # Apply Gaussian blur for smoother visualization
    heatmap = gaussian_filter(heatmap, sigma=15)
    
    # Plot Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap, cmap='magma', alpha=0.75)
    plt.title("Heatmap of Fish Positions Across Images")
    plt.xlabel("Image Width")
    plt.ylabel("Image Height")
    plt.show()
    
    return all_fish_counts, np.array(fish_positions)

def compare_extreme_cases(all_fish_counts):
    image_paths = sorted(glob(os.path.join(IMAGES_DIR, "*.jpg")))
    low_count_idx = np.argmin(all_fish_counts)
    high_count_idx = np.argmax(all_fish_counts)
    
    low_count_img = cv2.imread(image_paths[low_count_idx])
    high_count_img = cv2.imread(image_paths[high_count_idx])
    
    # Convert BGR to RGB for correct color display
    low_count_img = cv2.cvtColor(low_count_img, cv2.COLOR_BGR2RGB)
    high_count_img = cv2.cvtColor(high_count_img, cv2.COLOR_BGR2RGB)
    
    # Display Images Side-by-Side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(low_count_img)
    axes[0].set_title(f"Low Fish Count Image ({all_fish_counts[low_count_idx]} fish)")
    axes[0].axis("off")
    
    axes[1].imshow(high_count_img)
    axes[1].set_title(f"High Fish Count Image ({all_fish_counts[high_count_idx]} fish)")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()


def cluster_fish_positions(fish_positions, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(fish_positions)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(fish_positions[:, 0], fish_positions[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=100, label='Cluster Centers')
    plt.title("K-Means Clustering of Fish Positions")
    plt.xlabel("Image Width")
    plt.ylabel("Image Height")
    plt.legend()
    plt.gca().invert_yaxis()  # Match image coordinate system
    plt.show()

def get_fish_counts_with_label(annotations_dir, dataset_name):
    """
    Reads annotation files and counts the number of fish per image,
    tagging them with the dataset name.

    Returns:
        list of tuples: (fish_count, dataset_name)
    """
    fish_counts = []
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith(".txt")]
    for file in annotation_files:
        file_path = os.path.join(annotations_dir, file)
        with open(file_path, "r") as f:
            num_fish = sum(1 for _ in f)
        fish_counts.append((num_fish, dataset_name))
    return fish_counts


def plot_dual_fish_count_histogram(alaska_counts, ontario_counts, bins=20):
    # Convert lists of tuples to DataFrames
    alaska_df = pd.DataFrame(alaska_counts, columns=['Fish Count', 'Dataset'])
    ontario_df = pd.DataFrame(ontario_counts, columns=['Fish Count', 'Dataset'])

    # Combine
    combined_df = pd.concat([alaska_df, ontario_df], ignore_index=True)

    # Ensure numeric and drop bad data
    combined_df['Fish Count'] = pd.to_numeric(combined_df['Fish Count'], errors='coerce')
    combined_df = combined_df.dropna(subset=['Fish Count'])

    # Plot
    plt.figure(figsize=(12, 7))
    plot = sns.histplot(
        data=combined_df,
        x='Fish Count',
        hue='Dataset',
        stat='percent',
        bins=bins,
        multiple='dodge',
        palette={'Alaska': 'skyblue', 'Ontario': 'salmon'},
        edgecolor='black',
        alpha=0.8
    )

    # Enhancements
    plt.xlabel("Number of Fish per Image", fontsize=14, fontweight='bold')
    plt.ylabel("Percentage of Images (%)", fontsize=14, fontweight='bold')
    plt.title("Normalized Fish Count Distribution: Alaska vs Ontario", fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add legend explicitly
    custom_legend = [
        Patch(facecolor='skyblue', edgecolor='black', label='Alaska'),
        Patch(facecolor='salmon', edgecolor='black', label='Ontario')
    ]
    plt.legend(handles=custom_legend, title='Dataset', fontsize=12, title_fontsize=13)

    # Make small bars more visible
    plot.set_ylim(top=max(plot.get_ylim()[1], 5))  # set a minimum visible y-range

    plt.tight_layout()
    plt.show()







#generating the plots
#all_fish_counts, fish_positions = generate_heatmap()
#plot_fish_count_histogram(all_fish_counts)
#plot_fish_count_boxplot(all_fish_counts)
#compare_extreme_cases(all_fish_counts)
#cluster_fish_positions(fish_positions)

alaska_counts = get_fish_counts_with_label('/Users/anandita/Desktop/STA2453/CFC_dataset/labels/nushagak', "Alaska")
ontario_counts = get_fish_counts_with_label("/Users/anandita/Downloads/MNR_sample/obj_train_data", "Ontario")

plot_dual_fish_count_histogram(alaska_counts, ontario_counts)
