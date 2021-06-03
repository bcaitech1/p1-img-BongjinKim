import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from torchsummary import summary
from efficientnet_pytorch import EfficientNet
    
class MyModel(nn.Module):
    def __init__(self, model_name):
        super(MyModel, self).__init__()
        if 'efficient' in model_name:
            self.model = EfficientNet.from_pretrained(model_name, num_classes=18)
        else:
            self.model = timm.create_model(model_name, pretrained=True, num_classes=18)
                
    def forward(self, x):
        return self.model(x)