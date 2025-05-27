import json
import re

# Load your JSON file
with open(r'C:\Users\kerem\OneDrive\Masaüstü\Politechnika Warszawska\EARIN - Intro to Artificial Intelligence\ship-detection\test.json', 'r') as f:
    data = json.load(f)

# Build a mapping from old image id to new integer id
id_map = {}
for img in data['images']:
    match = re.search(r'(\d+)$', img['id'])
    if match:
        new_id = int(match.group(1))
        id_map[img['id']] = new_id
        img['id'] = new_id
    else:
        raise ValueError(f"Image id {img['id']} does not end with digits.")

# Update annotations
for ann in data['annotations']:
    old_image_id = ann['image_id']
    if old_image_id in id_map:
        ann['image_id'] = id_map[old_image_id]
    else:
        raise ValueError(f"Annotation image_id {old_image_id} not found in images.")

# Save the updated JSON
with open('val_output.json', 'w') as f:
    json.dump(data, f, indent=2)