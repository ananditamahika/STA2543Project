import os
import numpy as np

# Path to the directory containing annotation text files
annotations_dir = "/Users/anandita/Downloads/MNR_sample/obj_train_data"

# List all annotation files
annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith(".txt")]

# Initialize a list to store fish counts per image
fish_counts = []

# Iterate through each annotation file
for file in annotation_files:
    file_path = os.path.join(annotations_dir, file)
    
    # Read the file and count the number of lines (each line is a fish annotation)
    with open(file_path, "r") as f:
        num_fish = sum(1 for _ in f)  # Count number of lines in file
    
    # Store the count for this image
    fish_counts.append(num_fish)

# Compute statistics
total_fish_annotations = sum(fish_counts)
average_fish_per_image = np.mean(fish_counts)
min_fish_count = np.min(fish_counts)
max_fish_count = np.max(fish_counts)

# Print results
print("Total number of fish annotations:", total_fish_annotations)
print("Average number of fish per image:", average_fish_per_image)
print("Minimum fish count per image:", min_fish_count)
print("Maximum fish count per image:", max_fish_count)
