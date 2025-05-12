import os
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

# Path to the images folder
images_dir = "images"

# List to store resolutions
pixel_counts = []

# Iterate over all files in the images directory
for filename in os.listdir(images_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(images_dir, filename)
        with Image.open(img_path) as img:
            w, h = img.size
            pixel_counts.append(w * h)

# Define bins (in pixels)
bins = [
    (0, 90_000, "<0.09MP"),
    (90_000, 120_000, "0.09-0.12MP"),
    (120_000, 160_000, "0.12-0.16MP"),
    (160_000, 250_000, "0.16-0.25MP"),
    (250_000, 300_000, "0.25-0.30MP"),
    (300_000, 360_000, "0.30-0.36MP"),
]

# Group counts by bins
bin_counts = Counter()
for count in pixel_counts:
    for low, high, label in bins:
        if low <= count < high:
            bin_counts[label] += 1
            break

# Prepare data for plotting
labels = [label for _, _, label in bins]
counts = [bin_counts[label] for label in labels]

# Plot
plt.figure(figsize=(8, 6))
plt.bar(labels, counts)
plt.xlabel('Image Size (Megapixels)')
plt.ylabel('Number of Images')
plt.title('Number of Images Grouped by Size')
plt.tight_layout()
plt.show()