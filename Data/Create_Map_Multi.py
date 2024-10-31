import os
import json
import numpy as np
from PIL import Image, ImageDraw

# Define paths
project_path = '../'

dataset_path = os.path.join(project_path, 'supervisely')
dataset_path = os.path.join(dataset_path, 'wrist')
images_path = os.path.join(project_path, 'Images')
annotations_path = os.path.join(dataset_path, 'ann')

ClassTitle = ['fracture', 'boneanomaly', 'bonelesion', 'foreignbody', 'metal', 'periostealreaction', 'pronatorsign', 'softtissue']
ClassMap = {
    'fracture' : 1,
    'boneanomaly' : 2,
    'foreignbody' : 3,
    'metal' : 4,
    'pronatorsign' : 5,
    'softtissue' : 6,
    'periostealreaction' : 7,
    'bonelesion' : 8
}

# Load image and annotation file paths
image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.jpg') or f.endswith('.png')])
annotation_files = sorted([f for f in os.listdir(annotations_path) if f.endswith('.json')])

# Function to load an image
def load_image(image_path):
    return Image.open(image_path)

# Function to load annotation
def load_annotation(annotation_path):
    with open(annotation_path, 'r') as file:
        return json.load(file)

# Function to extract bounding boxes from annotation
def extract_bounding_boxes(annotation):
    bounding_boxes = []
    for obj in annotation['objects']:
        if obj['classTitle'] in ClassTitle:
            bbox = obj['points']['exterior']
            # Flatten bbox list
            coord = bbox
            bounding_boxes.append((coord, ClassMap[obj['classTitle']]))
    return bounding_boxes

# Function to create a segmentation map
def create_segmentation_map(image_size, bounding_boxes):
    segmentation_map = Image.new('1', image_size, 0)  # Create a new black image
    draw = ImageDraw.Draw(segmentation_map)
    # print(bounding_boxes)
    for bbox in bounding_boxes:
        # print("coord: ", bbox)
        coord, classTitle = bbox
        # print(classTitle)
        if classTitle < 7:
            # print(classTitle)
            draw.rectangle([coord[0][0], coord[0][1], coord[1][0], coord[1][1]], outline=1, fill=1)
        else:
            # print(classTitle)
            polygon = [(x, y) for [x, y] in coord]
            draw.polygon(polygon, outline=1, fill=1)
    # print("\n")
    segmentation_map = np.array(segmentation_map)
    # print("Unique values in the segmentation map:", np.unique(segmentation_map))
    # print("Labels in the mask are : ", np.unique(segmentation_map))
    # print("map shape: ", segmentation_map)
    return segmentation_map

def main():
    # Process all images and annotations
    # i = 0
    for image_file, annotation_file in zip(image_files, annotation_files):
        # Load image and annotation
        image = load_image(os.path.join(images_path, image_file))
        annotation = load_annotation(os.path.join(annotations_path, annotation_file))
        
        # Extract bounding boxes
        bounding_boxes = extract_bounding_boxes(annotation)
        
        # Create segmentation map
        # print("image size: ", image.size[::-1])
        segmentation_map = create_segmentation_map(image.size[::1], bounding_boxes)
        # print("image size: ", image.size[::-1])
        # Save segmentation map (optional)
        segmentation_map_image = Image.fromarray(segmentation_map)
        segmentation_map_image.save(f'../Segmentation_maps_multi/{image_file}')
        # i += 1
        # if i == 10:
        #     break

if __name__ == "__main__": 
    main()