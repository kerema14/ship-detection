import os
import random

# Set random seed for reproducibility
random.seed(42)

# Path to images directory
images_dir = os.path.join(os.path.dirname(__file__), "images")

# Get all .png filenames (without extension)
all_images = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(".png")]

# Shuffle the list
random.shuffle(all_images)

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

n_total = len(all_images)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)

train_files = all_images[:n_train]
val_files = all_images[n_train:n_train + n_val]
test_files = all_images[n_train + n_val:]

# Helper to write list to file
def write_list(filename, items):
    with open(filename, "w") as f:
        for item in items:
            f.write(f"{item}\n")

# Write to txt files
base_dir = os.path.dirname(__file__)
write_list(os.path.join(base_dir, "train.txt"), train_files)
write_list(os.path.join(base_dir, "val.txt"), val_files)
write_list(os.path.join(base_dir, "test.txt"), test_files)

print(f"Split {n_total} images into {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")