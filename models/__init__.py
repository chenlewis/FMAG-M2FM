from models.model import DenseNet121
from models.model import DenseNet169
from models.model import DenseNet201
from models.model import MobileNet
from models.model import VGG16
from models.model import VGG19
from models.model import ResNet34
from models.model import ResNet50
from models.model import ResNet101
from models.model import ResNet152
from models.model import ResNeXt50
from models.model import ResNeXt101
from models.model import InceptionV3
from models.model import PeleeNet

from models.model_nn import NN_10
from models.model_nn import NN_20
from models.model_nn import NN_50
from models.model_nn import NN_100
from models.model_nn import NN_200
from models.model_nn import NN_500
from models.model_nn import NN_1000

from models.LeNet_Siamese import MISLNet
from models.LeNet_Siamese import LeNet_Siamese

#SSDG论文里面
from models.DGFAS import DG_model
from models.DGFAS import Discriminator

#SSDG我们自己的pretrained 模型
from models.DGFAS import DG_ResNet50
from models.DGFAS import DG_DenseNet121
from models.DGFAS import DG_ResNeXt101
from models.DGFAS import SSDG_Discriminator

#SSDG测试我们自己的pretrianed
from models.FS import ResNet50_FS
from models.FS import DenseNet121_FS
from models.FS import ResNeXt101_FS

from models.transformer_model import ViTB16, SwinB, BeiTB, DiNATB, FocalNetB, ConvNeXtTiny