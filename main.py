from config.config1 import opt
import os
import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader
import models
from data.dataset import Copy_Detection, Copy_Detection_1
# from torchnet import meter
from utils.visualizer import Visualizer
import torch.nn as nn
from torch.nn import functional as T
from tqdm import tqdm
import numpy as np
import random
import datetime
import torch
from sklearn import metrics

def train(**kwargs):
    log_file = '/home/lyj/CNN/log/ViT/log.txt'

    k = 1
    for j in range(k):  # k 迭代次数
        opt._parse(kwargs)  # 解析参数
        with open(log_file, 'a') as f:
            f.write(f"Device: {opt.device}\n")
            f.write("User config:\n")
            for k, v in opt.__class__.__dict__.items():
                if not k.startswith('_'):
                    f.write(f"{k}: {v}\n")
                    
        model = getattr(models, opt.model)()  # 根据模型名称获取模型实例
        if opt.use_multi_gpu:
            num_gpus = t.cuda.device_count()
            if num_gpus >= 2:  
                device_ids = list(range(num_gpus))
            else:
                device_ids = [2]  
            net = t.nn.DataParallel(model, device_ids=device_ids)  # 使用多 GPU 
            print(f"Using GPUs: {device_ids}")
        else:
            net = model.to(opt.device)  # 使用单个 GPU 或 CPU 运行模型
            device_str = f"{opt.device.type}:{opt.device.index}" if opt.device.type == 'cuda' else str(opt.device)
            print(f"Using device: {device_str}")

        net.to(opt.device)
        # 打印设备信息以便调试
        print(f"Model is on device: {opt.device}")

        train_data = Copy_Detection_1(root=opt.train_data_root_cnn_1, train=True)
        train_loader = DataLoader(train_data, batch_size=opt.batch_size_0, shuffle=True, num_workers=opt.num_workers)
        val_data = Copy_Detection_1(root=opt.train_data_root_cnn_1, train=False)
        val_loader = DataLoader(val_data, batch_size=opt.batch_size_1, shuffle=True, num_workers=opt.num_workers)
        test_data = Copy_Detection(opt.test_data_root, test=True)
        test_loader = DataLoader(test_data, batch_size=opt.batch_size_1, shuffle=False, num_workers=opt.num_workers)

        criterion = nn.CrossEntropyLoss()
        lr = opt.lr
        optimizer = model.get_optimizer(lr, opt.weight_decay)

        loss_meter = meter.AverageValueMeter()
        previous_loss = 1e10
        confusion_matrix = meter.ConfusionMeter(2)
        best_acc = 0.5
        val_acc = 0.0
        test_acc = 0.0
        for epoch in range(opt.max_epoch):
            loss_meter.reset()
            confusion_matrix.reset()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for ii, (data, label) in pbar:
                input = data.to(opt.device)
                target = label.to(opt.device)
                score = net(input)
                optimizer.zero_grad()
                loss = criterion(score, target)
                loss.backward()
                optimizer.step()

                loss_meter.add(loss.item())
                confusion_matrix.add(score.detach(), target.detach())

                # 更新进度条描述，显示当前epoch和经过的时间
                elapsed_time = str(datetime.timedelta(seconds=int(pbar.format_dict['elapsed'])))
                pbar.set_description(f"Epoch [{epoch + 1}/{opt.max_epoch}]")
                pbar.set_postfix({'Elapsed Time': elapsed_time})

            cm_accuracy = confusion_matrix.value()
            train_accuracy = 100 * (cm_accuracy[0][0] + cm_accuracy[1][1]) / (cm_accuracy.sum())

            val_cm, val_accuracy = val(net, val_loader)
            val_acc += val_accuracy

            print("epoch:{epoch}, lr:{lr}, loss:{loss}, train_accuracy:{train_accuracy}, val_accuracy:{val_accuracy}".format(
                epoch=epoch, lr=lr, loss=loss_meter.value()[0],
                train_accuracy=train_accuracy, val_accuracy=val_accuracy))

            if best_acc < val_accuracy:
                best_acc = val_accuracy
                t.save(model.state_dict(), '/home/lyj/CNN/weight/Original/ViT/ViT-{}.pth'.format(epoch))

            if loss_meter.value()[0] > previous_loss:
                lr = lr * opt.lr_decay
                # 第二种降低学习率的方法:不会有moment等信息的丢失
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            previous_loss = loss_meter.value()[0]

            # Evaluate on test set and print AUC, EER
            confusion_matrix, accuracy,test_auc, test_eer = val_test(model, test_loader)
            print(f'Test AUC: {test_auc}, Test EER: {test_eer}')
            with open(log_file, 'a') as f:
                f.write(
                    f"epoch:{epoch}, lr:{lr}, loss:{loss_meter.value()[0]}, train_accuracy:{train_accuracy}, val_accuracy:{val_accuracy}\n")
                f.write(f'Test AUC: {test_auc}, Test EER: {test_eer}\n')

def val(model, dataloader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label) in enumerate(dataloader):
        val_input = val_input.to(opt.device)
        label = label.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(t.LongTensor))
    model.train()

    cm_value = confusion_matrix.value()
    accuracy = 100 * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy

def val_test(model, test_loader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    y_true = []
    y_score = []

    for ii, (data, label, path) in enumerate(tqdm(test_loader, desc="Processing")):  # 忽略标签
        test_input = data.to(opt.device)
        label = label.to(opt.device)
        score = model(test_input)
        probability = F.softmax(score, dim=1)[:, 1].detach().cpu().tolist()
        confusion_matrix.add(score.detach().squeeze(), label.type(t.LongTensor))
        probability_tensor = t.tensor(probability, dtype=t.float32)
        y_score.extend(probability_tensor.cpu().numpy())
        # Collect true labels and predicted scores for metrics calculation
        y_true.extend(label.cpu().numpy())
        # y_score.extend(probability.detach().cpu().numpy())

    cm_value = confusion_matrix.value()
    accuracy = 100 * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())

    # Calculate additional evaluation metrics
    auc, eer = ComputeMetric(y_true, y_score)

    return confusion_matrix, accuracy, auc, eer

def test(**kwargs):

    log_file = '/home/lyj/CNN/log/ViT/log.txt'
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
    model = getattr(models, opt.model)().eval().to(opt.device)
    if opt.load_model_path:
        checkpoint = t.load(opt.load_model_path, map_location=opt.device)
        model.load_state_dict(checkpoint, strict=True)
        # model.load(opt.load_model_path)
    model.to(opt.device)
    model.eval()

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
        with torch.no_grad():
            input = data.to(opt.device)
            score = net(input)
            # score = net(input,None)
            # probability = F.softmax(score, dim=1)[:, 1].detach().cpu().tolist()
            probability = F.softmax(score, dim=1)[:, 1].detach().tolist()
            batch_results = [(path_, probability_, label_.item() if label_.numel() > 0 else None) for
                             path_, label_, probability_ in zip(path, label, probability)]
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

def cross_test(**kwargs):
    opt._parse(kwargs)
    model = getattr(models, opt.model)().eval()
    img = [os.path.join(opt.test_data_root, img) for img in os.listdir(opt.test_data_root)]
    test_model = [os.path.join(opt.load_model_path, model_1) for model_1 in os.listdir(opt.load_model_path)]
    results = []
    for i in range(len(test_model)):
        model.load(test_model[i])
        model.to(opt.device)
        test_data = Copy_Detection(opt.test_data_root, img, train=False, test=True)
        test_loader = DataLoader(test_data, batch_size=opt.batch_size_1, shuffle=False, num_workers=opt.num_workers)
        for ii, (data, path) in enumerate(test_loader):
            input = data.to(opt.device)
            score = model(input)
            probability = T.softmax(score, dim=1)[:,1].detach().tolist()
            batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]
            results += batch_results

    write_csv(results, opt.result_file)
    return results

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

