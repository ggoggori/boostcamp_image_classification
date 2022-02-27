from torchvision import models
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.config = config
        self.model = self.get_model(self.config['model']['model_name'])
        self.num_ftrs = 1000
        self.linear = nn.Linear(self.num_ftrs, 18)
        self.linear1 = nn.Linear(self.num_ftrs, 1)
        self.linear2 = nn.Linear(self.num_ftrs, 3)
        self.linear3 =  nn.Linear(self.num_ftrs, 6)
        
    def forward(self, x):
        x = self.model(x)
        
        if self.config['model']['output_structure'] == 'single':
            output = self.linear(x)
            return output
        
        elif self.config['model']['output_structure'] == 'multiple':
            output1 = F.sigmoid(self.linear1(x))
            output2 = self.linear2(x)
            output3 = self.linear3(x)

            return output1, output2, output3


    def get_model(self, model_name):
        pretrained = self.config['model']['pretrained']
        n_classes = self.config['model']['num_classes']

        if 'efficientnet' in model_name:
            if pretrained:
                model = EfficientNet.from_pretrained(model_name)
            else: 
                model = EfficientNet.from_name(model_name)
            
            # num_ftrs = model._fc.in_features
            # model._fc = nn.Linear(num_ftrs, n_classes)
            
        elif 'resnet' in model_name:
            model = models.resnet18(pretrained=pretrained)
                
            # num_ftrs = model.fc.in_features
            # classifier = nn.Sequential(nn.Linear(num_ftrs, n_classes))
            # model.fc = classifier
        
        elif 'densenet' in model_name:
            model = models.densenet161(pretrained=pretrained)

            # num_ftrs = model.classifier.in_features
            # classifier = nn.Sequential(nn.Linear(num_ftrs, n_classes))
            # model.classifier = classifier
        
        else:
            raise Exception('Model Name Error')
        
        if self.config['model']['freezing'] == True:
                for param in model.parameters():
                    param.requires_grad = False

        return model