import torch
import torch.nn as nn

class Confidence(nn.Module):
  def __init__(self, out_features=8):
    super().__init__()

    self.globalMP = nn.AdaptiveMaxPool2d(output_size=(1, 1))
    self.dropout  = nn.Dropout(p=0.5)
    self.fullyConnected = nn.Linear(in_features=1024, out_features=out_features)
  
  def forward(self, x):
    x = self.globalMP(x)

    x = torch.flatten(x, start_dim=1)

    x = self.dropout(x)

    x = self.fullyConnected(x)
    
    return torch.sigmoid(x)
  
if __name__ == "__main__" : 
    conf = Confidence()
    print(conf)