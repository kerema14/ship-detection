import os
import xml.etree.ElementTree as ET
import seaborn as sns
import matplotlib.pyplot as plt

def get_bbox_coverages(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    image_area = width * height

    coverages = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bbox_area = (xmax - xmin) * (ymax - ymin)
        coverage = (bbox_area / image_area) * 100
        coverages.append(coverage)
    return coverages

def main():
    annotations_dir = 'annotations'  # Change if your path is different
    all_coverages = []
    for filename in os.listdir(annotations_dir):
        if filename.endswith('.xml'):
            xml_path = os.path.join(annotations_dir, filename)
            coverages = get_bbox_coverages(xml_path)
            for i, cov in enumerate(coverages):
                print(f"{filename} - bbox {i+1}: {cov:.2f}% covered")
            all_coverages.extend(coverages)

    # Plot box plot
    if all_coverages:
        sns.boxplot(y=all_coverages)
        plt.ylabel('Bounding Box Coverage (%)')
        plt.title('Distribution of Bounding Box Coverage')
        plt.show()

    # Plot box plot without extreme values (outliers)
    if all_coverages:
        sns.boxplot(y=all_coverages, showfliers=False)
        plt.ylabel('Bounding Box Coverage (%)')
        plt.title('Distribution of Bounding Box Coverage (No Outliers)')
        plt.show()

if __name__ == "__main__":
    main()