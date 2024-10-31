import sys
sys.path.append('../')
import json
import torch
import numpy as np
from Model.UNet import UNet, UNet_mod
import matplotlib.pyplot as plt
from Data.Dataset_multi import GrazData
from torchvision import transforms
from Model.Confidence_new import Confidence
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, random_split, Subset

model_path = "../Model/Weights"
unet_name = "UNet_multi"
conf_name = "Confidence_multi"
data_path = '../Data/data.json' #path to json file
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 8

# Define evaluation function
def evaluate_roc(branch1, branch2, dataloader):
    branch1.eval()
    branch2.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            inputs, labels = batch['jpg'].to(device), batch['frc'].float().to(device)
            # Forward pass
            _, bottleneck = branch1(inputs)
            outputs = branch2(bottleneck)
            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs).cpu().numpy()
            # Move labels to CPU and convert to numpy array
            labels = labels.cpu().numpy()
            # Append predictions and labels to lists
            all_preds.append(preds)
            all_labels.append(labels)

    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute ROC curve and ROC AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    optimal_thresholds = dict()
    for i in range(num_classes):
        # Check if there are both positive and negative samples
        if len(np.unique(all_labels[:, i])) == 2:
            fpr[i], tpr[i], thresholds[i] = roc_curve(all_labels[:, i], all_preds[:, i])
            roc_auc[i] = roc_auc_score(all_labels[:, i], all_preds[:, i])
            
            # Compute the optimal threshold
            optimal_idx = np.argmax(tpr[i] - fpr[i])
            optimal_thresholds[i] = thresholds[i][optimal_idx]
        else:
            fpr[i], tpr[i], thresholds[i], roc_auc[i], optimal_thresholds[i] = [None] * 5
            print(f"Class {i} does not have both positive and negative samples, skipping ROC AUC calculation.")

    return fpr, tpr, roc_auc, optimal_thresholds

def Eval():
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
    #train_dataset = Subset(dataset, np.load('../Training/train_indices_mod.npy').tolist())
    # test_dataset = Subset(dataset, np.load('../Training/test_indices_mod.npy').tolist())
    val_dataset = Subset(dataset, np.load('../Training/val_indices_mod.npy').tolist())

    loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # Evaluate ROC and AUC
    fpr, tpr, roc_auc, threshold = evaluate_roc(branch1, branch2, loader)

    print("Thresholds: ", threshold)

    # Plotting ROC curves for each class
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Multi-Label Classification')
    plt.legend(loc='lower right')
    plt.savefig('./ROC_multi.png')
    plt.show()

if __name__ == "__main__":
    Eval()
