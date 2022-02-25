# from torchvision import models
from torch import nn
from efficientnet_pytorch import EfficientNet

# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self.model = models.resnet18(pretrained=True)
#         self.model.fc = nn.Linear(in_features=512, 
#                              out_features=18, bias=True)
    
#     def forward(self, x):
#         x = self.model(x)
        
#         return x       

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=18)
    
    def forward(self, x):
        x = self.model(x)
        
        return x     