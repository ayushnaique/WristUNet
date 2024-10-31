import sys
sys.path.append('../')
import json
import torch
import numpy as np
from Model.UNet import UNet, UNet_mod
import matplotlib.pyplot as plt
from Data.Dataset import GrazData
from torchvision import transforms
from Model.Confidence import Confidence
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, random_split, Subset

model_path = "../Model/Weights"
unet_name = "UNet_mod"
conf_name = "Confidence_mod"
data_path = '../Data/data.json' #path to json file
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Define evaluation function
def evaluate_roc(branch1, branch2, dataloader):
    branch1.eval()
    branch2.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            inputs, labels = batch['jpg'].to(device), batch['frc'].float()
            print(inputs.shape)
            _, bottleneck = branch1(inputs)
            outputs = branch2(bottleneck)
            preds = torch.sigmoid(outputs).cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    # Compute AUC
    auc = roc_auc_score(all_labels, all_preds)
    
    # Compute Youden's J statistic
    J = tpr - fpr
    # Find the index of the optimal threshold
    optimal_idx = np.argmax(J)
    # Get the optimal threshold
    optimal_threshold = thresholds[optimal_idx]
    
    return fpr, tpr, thresholds, auc, optimal_threshold

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
    #train_dataset = Subset(dataset, np.load('../Training/train_indices.npy').tolist())
    test_dataset = Subset(dataset, np.load('../Training/test_indices.npy').tolist())
    #val_dataset = Subset(dataset, np.load('../Training/val_indices.npy').tolist())

    loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Evaluate ROC and AUC
    fpr, tpr, thresholds, auc, optimal = evaluate_roc(branch1, branch2, loader)
    print(f"AUC: {auc:.4f}")
    print(f"Optimal Threshold: {optimal:.4f}")

    # Plot ROC curve (optional)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig("./ROC.png")
    plt.show()

if __name__ == "__main__":
    Eval()
