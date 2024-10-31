import sys
sys.path.append('../')
import torch
from Data.Dataset import GrazData
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Define your dataset and transformation
data_path = '../Data/data.json' #path to json file

dataset = GrazData(data_path)

# Use a DataLoader to iterate through the dataset and compute mean and std
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize variables to accumulate sum and sum of squares
channel_sum = torch.zeros(3)  # Assuming RGB images, change dimension if your images are grayscale or have more channels
channel_sum_sq = torch.zeros(3)
num_batches = 0

# Iterate through each batch in the DataLoader
for _, batch in enumerate(loader):
    images = batch['jpg']
    batch_size = images.size(0)  # Number of images in the batch
    height = images.size(1)      # Image height
    width = images.size(2)       # Image width
    channels = images.size(3)    # Number of channels (e.g., 3 for RGB)
    
    # Flatten images into [batch_size, channels, pixels]
    images = images.view(batch_size, channels, -1)
    
    # Calculate sum and sum of squares for each channel
    channel_sum += images.sum(dim=2).sum(dim=0)
    channel_sum_sq += (images ** 2).sum(dim=2).sum(dim=0)
    
    num_batches += batch_size

# Calculate mean and std across all batches
mean = channel_sum / (num_batches * height * width)
std = torch.sqrt((channel_sum_sq / (num_batches * height * width)) - (mean ** 2))

print(mean, " ", std)

