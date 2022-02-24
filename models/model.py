from torchvision import models
from torch import nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(in_features=512, 
                             out_features=18, bias=True)
    
    def forward(self, x):
        x = self.model(x)
        
        return x        