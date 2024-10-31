import sys
sys.path.append('../')

import torch
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn.functional as F
from Data.Dataset_multi import GrazData
from Model.UNet import UNet, UNet_mod
from Model.Confidence_new import Confidence
from torchvision import transforms
import torch.optim as optim
from PIL import Image
import numpy as np
import cv2
import os

batch_size = 12
train_split = 0.7
test_split = 0.25
learning_rate = 1e-4
num_epochs = 50
data_path = '../Data/data.json' #path to json file

mean=[0.3122, 0.2817, 0.3070]
std=[0.1946, 0.1755, 0.1958]

SIZE = (1024, 512)

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print ("Device used: ", device)

def Train():
    branch1 = UNet_mod().to(device)
    branch2 = Confidence(out_features=8).to(device) 

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.35),
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize(SIZE),  # Resize to match the input size expected by the model
        # transforms.Normalize(mean, std)
    ])

    dataset = GrazData(data_path, transform=preprocess)

    train_dataset = Subset(dataset, np.load('./train_indices.npy').tolist())
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # length = len(dataset)
    # train_split = int(0.7 * length)
    # test_split = int(0.25 * length)
    # val_split = length - train_split - test_split

    # generator1 = torch.Generator().manual_seed(42)
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_split, val_split, test_split], generator=generator1)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # train_indices = train_dataset.indices
    # test_indices = test_dataset.indices
    # val_indices = val_dataset.indices

    # np.save('train_indices_mod.npy', train_indices)
    # np.save('test_indices_mod.npy', test_indices)
    # np.save('val_indices_mod.npy', val_indices)

    optimizer1 = optim.Adam(branch1.parameters(), lr=learning_rate)
    optimizer2 = optim.Adam(branch2.parameters(), lr=learning_rate)

    CrossEntropy = torch.nn.CrossEntropyLoss()
    BinaryCrossEntropy = torch.nn.BCELoss()
    # CrossEntropy = F.cross_entropy()

    for epoch in range(num_epochs):
        branch1.train()
        branch2.train()
        mapping_loss = 0.0
        confidence_loss = 0.0
        batch_m_loss = 0.0
        batch_c_loss = 0.0
        count = 0
        for batch_idx, batch in enumerate(train_loader):
            count = count + 1
            # img = batch['jpg']
            # img = img[0].cpu().numpy()
            # img  = np.transpose(img, (1, 2, 0))
            # print(img.shape)
            # cv2.imwrite("check.png", img)
            # break
            source, target, frc = batch['jpg'].to(device), batch['sgmt'].to(device), batch['frc'].to(device)
            print(source.shape, " ", target.shape)
            # print(target)
            # source = np.transpose(source, (0, 3, 1, 2)).to(device)
            # target = np.transpose(target, (0, 3, 1, 2)).to(device)
            # frc = frc.reshape(-1,1).to(device)

            # print(frc)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            map_output, bottleneck = branch1(source)
            # print(map_output)
            # target = target.unsqueeze(1)
            # print(map_output.shape, " ", target.shape)
            # target = target.squeeze(1) # for softmax, wants dimension as [C, H, W]
            # print(target.shape)
            # map_output = (map_output > 0.5)

            # #DICE
            # probs = map_output.view(-1)
            # targets = target.view(-1)
            # # Calculate the Dice coefficient
            # intersection = (probs * targets).sum()
            # dice_coeff = (2. * intersection + 1) / (probs.sum() + targets.sum() + 1)
            
            # # Dice loss
            # map_loss = 1 - dice_coeff
            # print(type(map_output), type(target))

            map_loss = BinaryCrossEntropy(map_output, target)
            confidence = branch2(bottleneck)
            # print(confidence, " ", frc)
            conf_loss = BinaryCrossEntropy(confidence, frc)

            combined_loss = (map_loss + conf_loss)

            combined_loss.backward()

            optimizer1.step()
            optimizer2.step()
            mapping_loss += map_loss.item()
            confidence_loss += conf_loss.item()
            batch_m_loss += map_loss.item()
            batch_c_loss += conf_loss.item()

            # if batch_idx % batch_size == (batch_size - 1):
            #     print(f'Step [{batch_idx + 1}/{len(train_loader)}], Mapping_Loss: {batch_m_loss / 4:.4f}, Confidence_Loss: {batch_c_loss / 4:.4f}')
            #     batch_m_loss = 0.0
            #     batch_c_loss = 0.0
        print(f'\nEPOCH: [{epoch + 1}/{num_epochs}], Mapping_Loss: {mapping_loss / count:.4f}, Confidence_Loss: {confidence_loss / count:.4f}\n')
                # mapping_loss = 0.0
                # confidence_loss = 0.0
    torch.save(branch1.state_dict(), '../Model/Weights/UNet_multi.pth')
    torch.save(branch2.state_dict(), '../Model/Weights/Confidence_multi.pth')

if __name__ == "__main__":
    try:
        Train()
    except Exception as e:
        print(f"Error: {e}")
    else:
        print("TRAINING COMPLETED\n")