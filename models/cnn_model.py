from torchvision.models import convnext_tiny
from transformers import FocalNetForImageClassification
# import timm
import sys
sys.path.append('../')
from torch import nn
import torch


class FocalNetB(nn.Module):
    def __init__(self, model_name='FocalNetB'):
        super(FocalNetB, self).__init__()
        self.model_name = model_name
        self.model = FocalNetForImageClassification.from_pretrained("microsoft/focalnet-base")
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 2)
        )
        for param in self.model.parameters():
            param.requires_grad=False
        
    def forward(self,x):
        return self.model(x).logits


class ConvNeXtTiny(nn.Module):
    def __init__(self, model_name='ConvNeXtTiny'):
        super(ConvNeXtTiny, self).__init__()
        self.model_name = model_name
        self.model = convnext_tiny(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.LayerNorm((768,1,1), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

        for param in self.model.parameters():
            param.requires_grad=False
        
    def forward(self,x):
        return self.model(x)


if __name__ == '__main__':
    model  = ConvNeXtTiny()
    data = torch.randn((5, 3, 224, 224))
    result = model(data)