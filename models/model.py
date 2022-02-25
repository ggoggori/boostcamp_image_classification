from torchvision import models
from torch import nn
from efficientnet_pytorch import EfficientNet

class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.config = config
        self.model = self.get_model(self.config['model']['model_name'])
        
    def forward(self, x):
        x = self.model(x)
        
        return x     

    def get_model(self, model_name):
        pretrained = self.config['model']['pretrained']
        n_classes = self.config['model']['num_classes']

        if 'efficientnet' in model_name:
            if pretrained:
                model = EfficientNet.from_pretrained(model_name)
            else: 
                model = EfficientNet.from_name(model_name)
            
            num_ftrs = model._fc.in_features
            model._fc = nn.Linear(num_ftrs, n_classes)
            
        elif 'resnet' in model_name:
            model = models.resnet18(pretrained=pretrained)
                
            num_ftrs = model.fc.in_features
            classifier = nn.Sequential(nn.Linear(num_ftrs, n_classes))
            model.fc = classifier
        
        elif 'densenet' in model_name:
            model = models.densenet161(pretrained=pretrained)
            num_ftrs = model.classifier.in_features
            classifier = nn.Sequential(nn.Linear(num_ftrs, n_classes))
            model.classifier = classifier
        
        if self.config['model']['freezing'] == True:
                for param in model.parameters():
                    param.requires_grad = False

        return model