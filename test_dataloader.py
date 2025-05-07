import models
from models.mfm_mask_eval import build_mfm_mask_eval
from config import _C, _update_config_from_file
import torch
from torchvision import transforms as T
import numpy as np
import time
import csv
import os
from PIL import Image
import pandas as pd
import tqdm
from tqdm import tqdm
config = _C.clone()
_update_config_from_file(config, './mfm_pretrain__vit_base__img224__300ep.yaml')
config.defrost()
config.DATA.BATCH_SIZE = 24
config.DATA.NUM_WORKERS = 4
config.MODEL.RESUME = './mfm_fag_ViT1.pth'

# config.MODEL.RESUME = '/home/lyj/MFM/ckpt_epoch_22.pth'
config.GPU = 0
config.MODEL.VIT.DECODER.DEPTH = 0
config.MODEL.VIT.DECODER.EMBED_DIM = 512
config.MODEL.VIT.DECODER.NUM_HEADS = 16
# config.MODEL_FAG = 'SwinB'
# config.MODEL_NAME = 'SwinB_noMoire_1'
# config.FAG_CHECKPOINT = './SwinB_FAG.pth'
config.MODEL_FAG = 'ViTB16'
config.MODEL_NAME = 'ViTB16_noMoire_1'
config.FAG_CHECKPOINT = './models/ViTB16_FAG.pth'
# about model

# from config import opt
import os
import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import models
from data.dataset import Copy_Detection, Copy_Detection_1
from torchnet import meter
# from utils.visualizer import Visualizer
import torch.nn as nn
from torch.nn import functional as T
from tqdm import tqdm
import random
import datetime
from sklearn import metrics
from collections import OrderedDict
from difflib import get_close_matches
from yacs.config import CfgNode

class DefaultConfig(object):
    env = 'default'
    vis_port = 8097

    model = 'ViTB16'
    # model = 'SwinB'

    test_data_root = '/home/data1/lyj/CMA/SRDID162_patch/images/'
    # load_model_path = '/home/lyj/MFM/models/ViTB16_FAG.pth'

    batch_size_0 = 32
    batch_size_1 = 32

    use_gpu = True
    use_multi_gpu = False

    num_workers = 0
    print_freq = 20
    debug_file = ''
    ##test

    result_file = '/home/data1/lyj/MFM/MFM-ViT-162.csv'
    max_epoch = 30
    lr = 0.0001
    lr_decay = 0.5
    # lr = 0.0001
    # lr_decay = 0.1
    weight_decay = 1e-3

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)
        # 修改设备选择逻辑以适应多卡
        if self.use_gpu:
            if self.use_multi_gpu:
                if t.cuda.device_count() > 1:
                    self.device = t.device('cuda')  # 使用 DataParallel 时，不指定具体的GPU
                else:
                    self.device = t.device('cuda:1')  # 如果只有一个GPU，默认使用 cuda:0
            else:
                self.device = t.device('cuda:0')  # 如果只有一个GPU，默认使用 cuda:0
        else:
            self.device = t.device('cpu')
        print('device', self.device)
        print('User config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()


def test(**kwargs):
    # model_name = 'SwinB'
    # # model_name = 'ViTB16'
    # model_checkpoint = './SwinB_FAG.pth'
    # # model_checkpoint = '/home/lyj/MFM/models/ViTB16_FAG.pth'
    # model = getattr(models, model_name)()
    # model.load_state_dict(torch.load(model_checkpoint, map_location='cpu'), strict=True)
    # model.cuda(0)
    # model.eval()
    ##原始
    opt._parse(kwargs)
    model = build_mfm_mask_eval(config)
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)

    del checkpoint

    model.FAG = getattr(models, config.MODEL_FAG)(config.MODEL_NAME)
    model.FAG.load_state_dict(torch.load(config.FAG_CHECKPOINT, map_location='cpu'), strict=True)
    # model.FAG.load_state_dict(torch.load(config.FAG_CHECKPOINT, map_location='cpu'), strict=False)
    model.to(opt.device)
    model.eval()

    log_file = '/home/data1/lyj/MFM/SRDID162-FM.txt'
    k = 1
    for j in range(k):  # Assuming k is the number of iterations
        opt._parse(kwargs)
        with open(log_file, 'a') as f:
            f.write(f"Device: {opt.device}\n")
            f.write("User config:\n")
            for k, v in opt.__class__.__dict__.items():
                if not k.startswith('_'):
                    f.write(f"{k}: {v}\n")
    print("opt device",opt.device)
    model = model.to(opt.device)

    if opt.use_multi_gpu:
        print("111")
        num_gpus = t.cuda.device_count()
        if num_gpus >= 2:
            device_ids = list(range(num_gpus))
        else:
            device_ids = [3]  # Default to the first GPU if only one available or using CPU
        net = t.nn.DataParallel(model, device_ids=device_ids)
        print(f"Using GPUs: {device_ids}")
    else:
        print("222")
        net = model

    test_data = Copy_Detection(opt.test_data_root, test=True)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size_1, shuffle=False, num_workers=opt.num_workers)
    all_probabilities = []
    all_labels = []
    results = []
    for ii, (data, label, path) in enumerate(tqdm(test_loader, desc="Processing")):  # 忽略标签
        input = data.to(opt.device)
        # score = net(input)
        score = net(input, None)
        # probability = F.softmax(score, dim=1)[:, 1].detach().cpu().tolist()
        probability = F.softmax(score, dim=1)[:, 1].detach().tolist()
        batch_results = [(path_, probability_, label_.item() if label_.numel() > 0 else None) for path_, label_, probability_ in zip(path, label, probability)]
        all_probabilities.extend(probability)
        all_labels.extend(label.cpu().tolist())  # 确保标签在CPU上，并转为列表
        # results += batch_results
        results.extend(batch_results)
    write_csv(results, opt.result_file)
    # 计算AUC
    auc_score = metrics.roc_auc_score(all_labels, all_probabilities)
    # 计算EER
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_probabilities)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

    print(f"AUC: {auc_score:.4f}")
    print(f"EER: {eer:.4f} at threshold {eer_threshold:.4f}")
    # return results

import pandas as pd
def write_csv(results, file_name):
    # 创建一个 DataFrame
    df = pd.DataFrame(results, columns=['Path', 'Score', 'label'])
    # 使用 DataFrame 的 to_csv 方法写入 CSV
    # df.to_csv(file_name, index=False)
    df.to_csv(file_name, mode='w', header=True, index=False)  # 使用覆盖模式写入文件

def compute_eer(fpr, tpr, thresholds):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, thresholds[min_index]

def ComputeMetric(y_true, y_score, pos_label=1):
    auc = metrics.roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=pos_label)
    eer, best_thresh = compute_eer(fpr, tpr, thresholds)

    return auc, eer

if __name__ == '__main__':
    import fire
    fire.Fire()

