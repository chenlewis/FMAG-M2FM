from torchvision.models import vit_b_16, ViT_B_16_Weights, swin_b, Swin_B_Weights, \
    convnext_tiny, ConvNeXt_Tiny_Weights
import sys
from transformers import BeitModel, BeitConfig, DinatForImageClassification, \
    FocalNetModel, FocalNetConfig, models, BeitForImageClassification, FocalNetForImageClassification
sys.path.append('../')
from models.BasicModule import BasicModule
from torch import nn
import torch
from torch.optim import Adam
import torchvision

class ViTB16(BasicModule):
    def __init__(self, model_name='ViTB16'):
        super(ViTB16, self).__init__()
        self.model_name = model_name
        self.model = vit_b_16(weights=ViT_B_16_Weights)
        self.model.heads = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay, freeze=True):
        if freeze:
            for param in self.model.parameters():  
                param.requires_grad = False
            for param in self.model.heads.parameters():  
                param.requires_grad = True
            print('ViTB16 model freeze cnn weights')
            return Adam(self.model.heads.parameters(), lr, weight_decay=weight_decay)
        else:
            print('ViTB16 model does not freeze cnn weights')
            return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class SwinB(BasicModule):
    def __init__(self, model_name='SwinB'):
        super(SwinB, self).__init__()
        self.model_name = model_name
        self.model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        self.model.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay, freeze=True):
        if freeze:
            for param in self.model.parameters(): 
                param.requires_grad = False
            for param in self.model.head.parameters():  
                param.requires_grad = True
            print('SwinB model freeze cnn weights')
            return Adam(self.model.head.parameters(), lr, weight_decay=weight_decay)
        else:
            print('SwinB model does not freeze cnn weights')
            return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class BeiTB(BasicModule):
    def __init__(self, model_name='BeiTB'):
        super(BeiTB, self).__init__()
        self.model_name = model_name
        self.model = BeitForImageClassification.from_pretrained("/home/lyj/.cache/huggingface/hub/models-microsoft-beit-base")
        self.model.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.model(x).logits

    def get_optimizer(self, lr, weight_decay, freeze=True):
        if freeze:
            for param in self.model.parameters():  
                param.requires_grad = False
            for param in self.model.classifier.parameters():  
                param.requires_grad = True
            print('BeiT-B model freeze cnn weights')
            return Adam(self.model.classifier.parameters(), lr, weight_decay=weight_decay)
        else:
            print('BeiT-B model does not freeze cnn weights')
            return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class DiNATB(nn.Module):
    def __init__(self, model_name='DiNATB'):
        super(DiNATB, self).__init__()
        self.model_name = model_name
        self.model = DinatForImageClassification.from_pretrained("/home/lipeiquan/.cache/huggingface/hub/models-shi-labs-dinat-base")
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.dinat.num_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x).logits

    def get_optimizer(self, lr, weight_decay, freeze=True):
        if freeze:
            for param in self.model.parameters():  
                param.requires_grad = False
            for param in self.model.classifier.parameters():  
                param.requires_grad = True
            print('DiNAT-B model freeze cnn weights')
            return Adam(self.model.classifier.parameters(), lr, weight_decay=weight_decay)
        else:
            print('DiNAT-B model does not freeze cnn weights')
            return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

class FocalNetB(BasicModule):
    def __init__(self, model_name='FocalNetB'):
        super(FocalNetB, self).__init__()
        self.model_name = model_name
        self.model = FocalNetForImageClassification.from_pretrained("/home/lyj/.cache/huggingface/hub/models-microsoft-focalnet")
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.focalnet.num_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2))

    def forward(self, x):
        return self.model(x).logits

    def get_optimizer(self, lr, weight_decay, freeze=True):
        if freeze:
            for param in self.model.parameters(): 
                param.requires_grad = False
            for param in self.model.classifier.parameters(): 
                param.requires_grad = True
            print('FocalNet-B model freeze cnn weights')
            return Adam(self.model.classifier.parameters(), lr, weight_decay=weight_decay)
        else:
            print('FocalNet-B model does not freeze cnn weights')
            return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class ConvNeXtTiny(BasicModule):
    def __init__(self, model_name='ConvNeXtTiny'):
        super(ConvNeXtTiny, self).__init__()
        self.model_name = model_name
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        original_layers = list(self.model.classifier.children())
        self.model.classifier = nn.Sequential(
            original_layers[0], 
            original_layers[1],  
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay, freeze=True):
        if freeze:
            for param in self.model.parameters():  
                param.requires_grad = False
            for param in self.model.classifier.parameters(): 
                param.requires_grad = True
            print('ConvNeXtTiny model freeze cnn weights')
            return Adam(self.model.classifier.parameters(), lr, weight_decay=weight_decay)
        else:
            print('ConvNeXtTiny model does not freeze cnn weights')
            return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


if __name__ == '__main__':
    model = ViTB16()  
    # model = FocalNetB()
    # model = ConvNeXtTiny()
    total_params = sum(p.numel() for p in model.parameters())  
    print("ViTB16模型的总参数数量:", total_params)
