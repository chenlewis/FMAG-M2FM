from torchvision import models
import torch.nn as nn
import torch as t
from models.BasicModule import BasicModule
import torch.nn.functional as F
import math
# from torchvision.models import VGG16_Weights,ResNet50_Weights

# import segmentation_models_pytorch as smp

class DenseNet121(BasicModule):
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.model = models.densenet121(pretrained=True)

        self.model.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.classifier.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):

        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out , (1, 1))
        output = t.flatten(out, 1)
        print(output.shape)
        # output = self.model(x)

        return output


class DenseNet169(BasicModule):
    def __init__(self):
        super(DenseNet169, self).__init__()
        self.model = models.densenet169(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(1664, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.classifier.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        # output = self.model.features(x)
        # print(output.shape)
        return output




class DenseNet201(BasicModule):
    def __init__(self):
        super(DenseNet201, self).__init__()
        self.model = models.densenet201(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(1920, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.classifier.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class MobileNet(BasicModule):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.classifier.parameters()):
            if index >= 1:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class VGG16(BasicModule):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.classifier.parameters()):
            if index >= 6:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class VGG19(BasicModule):
    def __init__(self):
        super(VGG19, self).__init__()
        self.model = models.vgg19(pretrained=True)
        self.model.classifier[6] = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.classifier.parameters()):
            if index >= 6:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output





class ResNet34(BasicModule):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.fc.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class ResNet50(BasicModule):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # self.model = models.resnet50(weights = ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.fc.parameters()):
            if index >= 0:
                param.requires_grad = True
    def forward(self, x):
        output = self.model(x)
        return output





class ResNet101(BasicModule):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.fc.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class ResNet152(BasicModule):
    def __init__(self):
        super(ResNet152, self).__init__()
        self.model = models.resnet152(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.fc.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class ResNeXt50(BasicModule):
    def __init__(self):
        super(ResNeXt50, self).__init__()
        self.model = models.resnext50_32x4d(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.fc.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class ResNeXt101(BasicModule):
    def __init__(self):
        super(ResNeXt101, self).__init__()
        self.model = models.resnext101_32x8d(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.fc.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class InceptionV3(BasicModule):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        self.model.AuxLogits.fc = nn.Linear(768, 2)
        self.model.aux_logits = False
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.fc.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

'''

--------------------PeleeNet------------------------

'''

class Conv_bn_relu(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, pad=1, use_relu=True):
        super(Conv_bn_relu, self).__init__()
        self.use_relu = use_relu
        if self.use_relu:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.convs(x)
        return out


class StemBlock(nn.Module):
    def __init__(self, inp=3, num_init_features=32):
        super(StemBlock, self).__init__()

        self.stem_1 = Conv_bn_relu(inp, num_init_features, 3, 2, 1)

        self.stem_2a = Conv_bn_relu(num_init_features, int(num_init_features / 2), 1, 1, 0)

        self.stem_2b = Conv_bn_relu(int(num_init_features / 2), num_init_features, 3, 2, 1)

        self.stem_2p = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stem_3 = Conv_bn_relu(num_init_features * 2, num_init_features, 1, 1, 0)

    def forward(self, x):
        stem_1_out = self.stem_1(x)

        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)

        stem_2p_out = self.stem_2p(stem_1_out)

        out = self.stem_3(t.cat((stem_2b_out, stem_2p_out), 1))

        return out


class DenseBlock(nn.Module):
    def __init__(self, inp, inter_channel, growth_rate):
        super(DenseBlock, self).__init__()

        self.cb1_a = Conv_bn_relu(inp, inter_channel, 1, 1, 0)
        self.cb1_b = Conv_bn_relu(inter_channel, growth_rate, 3, 1, 1)

        self.cb2_a = Conv_bn_relu(inp, inter_channel, 1, 1, 0)
        self.cb2_b = Conv_bn_relu(inter_channel, growth_rate, 3, 1, 1)
        self.cb2_c = Conv_bn_relu(growth_rate, growth_rate, 3, 1, 1)

    def forward(self, x):
        cb1_a_out = self.cb1_a(x)
        cb1_b_out = self.cb1_b(cb1_a_out)

        cb2_a_out = self.cb2_a(x)
        cb2_b_out = self.cb2_b(cb2_a_out)
        cb2_c_out = self.cb2_c(cb2_b_out)

        out = t.cat((x, cb1_b_out, cb2_c_out), 1)

        return out


class TransitionBlock(nn.Module):
    def __init__(self, inp, oup, with_pooling=True):
        super(TransitionBlock, self).__init__()
        if with_pooling:
            self.tb = nn.Sequential(Conv_bn_relu(inp, oup, 1, 1, 0),
                                    nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.tb = Conv_bn_relu(inp, oup, 1, 1, 0)

    def forward(self, x):
        out = self.tb(x)
        return out


class PeleeNet(BasicModule):
    def __init__(self, num_classes=2, num_init_features=32, growthRate=32, nDenseBlocks=[3, 4, 8, 6],
                 bottleneck_width=[1, 2, 4, 4]):
        super(PeleeNet, self).__init__()

        self.stage = nn.Sequential()
        self.num_classes = num_classes
        self.num_init_features = num_init_features

        inter_channel = list()
        total_filter = list()
        dense_inp = list()

        self.half_growth_rate = int(growthRate / 2)

        # building stemblock
        self.stage.add_module('stage_0', StemBlock(3, num_init_features))

        #
        for i, b_w in enumerate(bottleneck_width):

            inter_channel.append(int(self.half_growth_rate * b_w / 4) * 4)

            if i == 0:
                total_filter.append(num_init_features + growthRate * nDenseBlocks[i])
                dense_inp.append(self.num_init_features)
            else:
                total_filter.append(total_filter[i - 1] + growthRate * nDenseBlocks[i])
                dense_inp.append(total_filter[i - 1])

            if i == len(nDenseBlocks) - 1:
                with_pooling = False
            else:
                with_pooling = True

            # building middle stageblock
            self.stage.add_module('stage_{}'.format(i + 1), self._make_dense_transition(dense_inp[i], total_filter[i],
                                                                                        inter_channel[i],
                                                                                        nDenseBlocks[i],
                                                                                        with_pooling=with_pooling))

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(total_filter[len(nDenseBlocks) - 1], self.num_classes)
        )

        self._initialize_weights()

    def _make_dense_transition(self, dense_inp, total_filter, inter_channel, ndenseblocks, with_pooling=True):
        layers = []

        for i in range(ndenseblocks):
            layers.append(DenseBlock(dense_inp, inter_channel, self.half_growth_rate))
            dense_inp += self.half_growth_rate * 2

        # Transition Layer without Compression
        layers.append(TransitionBlock(dense_inp, total_filter, with_pooling))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.stage(x)

        # global average pooling layer
        x = F.avg_pool2d(x, kernel_size=7)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        out = F.log_softmax(x, dim=1)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class VGG_Feature(BasicModule):
    def __init__(self):
        super(VGG_Feature, self).__init__()
        self.model = models.vgg16(pretrained=False)
        # modules = list(self.model.classifier())[:-1]
        # self.model_1 = nn.Sequential(*modules)

    def forward(self, x):
        x = self.model.features(x)
        print(x.shape)
        x = self.model.avgpool(x)
        print(x.shape)
        x = t.flatten(x, 1)
        x = self.model.classifier(x)
        return x

#网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

net = ResNet50()

number = get_parameter_number(net)
# model = models.resnet18()
# print(model)
# net = VGG_Feature()
# x = t.rand(1, 3, 224, 224)
# y = net(x)
# print(net)
# print(y.shape)
# net = PeleeNet()
# x = t.randn(5, 3, 224, 224)
# out = net(x)
# net = DenseNet121()
# print(net)
# x = t.randn(1, 3, 224, 224)
# # y = net.model.features(x)
# y = net(x)
