import sys
sys.path.append('../')

import torch
from torch.utils.data import DataLoader, random_split, Subset
from Model.Confidence_new import Confidence
from Data.Dataset_multi import GrazData
import matplotlib.pyplot as plt
from Model.UNet import UNet, UNet_mod
import torch.optim as optim
import numpy as np
import json
import cv2
import os
from torchvision import transforms

RED = "\033[91m"
RESET = "\033[0m"

model_path = "../Model/Weights"
unet_name = "UNet_multi"
conf_name = "Confidence_multi"
data_path = '../Data/data.json' #path to json file

device = "mps" if torch.backends.mps.is_available() else "cpu"
print ("Device used: ", device)

def calculate_iou(mask1, mask2):

    # Ensure input masks are binary
    assert np.all(np.logical_or(mask1 == 0, mask1 == 1)), "Mask1 should be binary (0 or 1)"
    assert np.all(np.logical_or(mask2 == 0, mask2 == 1)), "Mask2 should be binary (0 or 1)"

    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Handle special case of union being 0
    if union == 0:
        return 0.0

    # Calculate IoU score
    iou = intersection / union

    return iou

def get_top_100_intensities(image, i):
    # Flatten the image to a 1D array
    flat_image = image.flatten()
    
    # Find the indices of the 100 highest intensities
    if len(flat_image) > i:
        top_100_indices = np.argpartition(flat_image, -i)[-i:]
    else:
        top_100_indices = np.argsort(flat_image)[-i:]  # If there are less than 100 elements, just sort them

    # Create an output matrix with the same shape as the input image
    output = np.zeros_like(image)
    
    # Set the positions of the highest intensities to their values in the original image
    for idx in top_100_indices:
        # Convert the 1D index back to 2D index
        x, y = np.unravel_index(idx, image.shape)
        output[x, y] = image[x, y]
    
    return output

def normalize_image(image):
    # Convert the image to float32 for precision
    image = image.astype(np.float32)
    
    # Calculate the mean and standard deviation
    mean = np.mean(image)
    std = np.std(image)
    
    # Normalize to zero mean and unit variance
    normalized_image = (image - mean) / std
    
    # Scale to have mean of 0.5
    desired_mean = 0.5
    normalized_image = normalized_image * std + mean  # Reverting back to original range
    normalized_image = (normalized_image - np.min(normalized_image)) / (np.max(normalized_image) - np.min(normalized_image))
    normalized_image = normalized_image * 0.5 + desired_mean
    
    # Clip values to ensure they are within [0, 1]
    normalized_image = np.clip(normalized_image, 0, 1)
    
    return normalized_image


def Inference():
    branch1 = UNet_mod()
    branch1.load_state_dict(torch.load(f'{model_path}/{unet_name}.pth', map_location=device))

    branch2 = Confidence()
    branch2.load_state_dict(torch.load(f'{model_path}/{conf_name}.pth', map_location=device))

    branch1.eval()
    branch2.eval()

    branch1 = branch1.to(device)
    branch2 = branch2.to(device)

    preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1024, 512)),  # Resize to match the input size expected by the model
        ])

    dataset = GrazData(data_path, transform=preprocess)

    # Load the indices
    #train_dataset = Subset(dataset, np.load('../Training/train_indices.npy').tolist())
    test_dataset = Subset(dataset, np.load('../Training/test_indices.npy').tolist())
    #val_dataset = Subset(dataset, np.load('../Training/val_indices.npy').tolist())

    val_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)

    # Directory to save segmentation maps
    output_dir = './Segmentation/'
    os.makedirs(output_dir, exist_ok=True)

    for _, batch in enumerate(val_loader):
        image, target, frc = batch['jpg'], batch['sgmt'], batch['frc']
        if frc[0][0] : print(f'{RED}FRACTURE VISIBLE{RESET}')
        else: print("NO FRACTURE VISIBLE")
        print(frc[0])
        # print("initial:", image.shape)
        input_tensor = image[0].unsqueeze(0)
        # print("after:", input_tensor.shape)
        input_tensor = input_tensor.to(device)
        image = image.cpu().numpy()
        image  = np.transpose(image[0], (1, 2, 0))
        target = target[0].squeeze(0)
        target = target.cpu().numpy()
        # target  = np.transpose(target[0], (1, 2, 0))
    
        # output = (output * 255).astype(np.uint8)
        # print("TENSOR:", input_tensor, "SHAPE: ",input_tensor.shape)

        # print("IMAGE :" ,image, "SHAPE: ", image.shape)
        # cv2.imwrite('./src.png', image*255)
        # print("TAARGET :" ,target*255, "SHAPE: ", target.shape)
        # cv2.imwrite('./seg.png', target*255)
        # Perform inference
        with torch.no_grad():
            conf_json = []
            output, bottleneck = branch1(input_tensor)
            print("OUTPUT shape before: ", output.shape)
            conf = branch2(bottleneck)
            print("CONFIDENCE: ", conf)
            output = output.squeeze(0).cpu().numpy()
            output = np.transpose(output, (1, 2, 0))
            output = np.reshape(output, output.shape[:2])
            # print(output.shape)
            output = get_top_100_intensities(output,10000)
            # print("OUTPUT shape: ", np.unique(output))

            output = output[:,:] > 0.2
            # print(output)
            mask = output > 0
            # cv2.imwrite('./out.png', mask*255)
            # print("MASK", mask)
            # save = ((output>0)*255).astype(np.uint8)
            # iou = calculate_iou(mask, target)
            # print("IOU: ", iou)
            # print(save.shape)
            # cv2.imwrite("./raw.png", save)
            plt.figure(figsize=(18, 6))
            plt.subplot(131)  # Change 121 to 131 for 3 subplots
            plt.imshow(image[:,:,0], cmap='gray')
            plt.title('Input Image')

            plt.subplot(132)  # Change 122 to 132 for 3 subplots
            plt.imshow(target[:,:], cmap='gray')
            plt.title('Target')

            plt.subplot(133)  # Change 123 to 133 for 3 subplots
            plt.imshow(output[:,:], cmap='gray')
            plt.title('Output')

            plt.show()
            # output = (output > 0.52)
            # print("OUTPUT :" ,output, "SHAPE: ", output.shape)
            # output = (output * 255.0).astype(np.uint8)
            # cv2.imwrite('./out.png', output)
            break
            # for batch_idx, batch in enumerate(val_loader):
            #     id, source, target, frc = batch['id'], batch['jpg'], batch['sgmt'], batch['frc'].float()
                
            #     source = np.transpose(source, (0, 3, 1, 2)).to(device)
            #     target = np.transpose(target, (0, 3, 1, 2)).to(device)
            #     frc = frc.reshape(-1,1).to(device)

            #     # Forward pass
            #     outputs, bottleneck = branch1(source)
            #     conf = branch2(bottleneck)

            #     # Post-process the outputs
            #     outputs = outputs.squeeze(1).cpu().numpy()  # Convert to numpy array and remove channel dimension
            #     print("output: \n",outputs)
            #     # outputs = (outputs > 0.5).astype(np.uint8)  # Apply threshold to get binary mask

            #     for i in range(outputs.shape[0]):
            #         image = source[i]
            #         segmentation_map = outputs[i]
            #         print("output: \n",segmentation_map)
            #         output_path = os.path.join(output_dir, f'seg_{conf[i].item()}_'+id[i][10:])

            #         # Ensure segmentation map has the same shape as the image
            #         # print(segmentation_map.shape)
            #         # segmentation_map = np.expand_dims(segmentation_map, axis=-1)  # Shape: (1024, 512, 1)
            #         # print(segmentation_map.shape)
            #         # segmentation_map = np.tile(segmentation_map, (1, 1, 3))  # Shape: (1024, 512, 3)
            #         # print(segmentation_map.shape)

            #         # Overlay the segmentation map on the original image with alpha blending
            #         # alpha = 0.5  # Transparency factor (adjust as needed)
            #         # segmentation_map = segmentation_map.cpu().numpy()
            #         # print(segmentation_map.shape)
            #         # segmentation_map = segmentation_map.float().clone()
            #         # overlayed_image = cv2.addWeighted(overlayed_image.astype(np.float32), 1 - alpha, segmentation_map.astype(np.float32), alpha, 0)
            #         segmentation_map = np.transpose(segmentation_map, (1, 2, 0))
            #         # print(overlayed_image.shape)
            #         # print(output_path)
            #         cv2.imwrite(output_path, segmentation_map)
            #         break

if __name__ == "__main__":
    Inference()