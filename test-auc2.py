# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import metrics
# from sklearn.metrics import roc_curve, roc_auc_score
# import os
# from collections import defaultdict
# import csv
# import pandas as pd
# import tqdm
# from tqdm import tqdm
# def compute_eer(fpr, tpr, thresholds):
#     """ Returns equal error rate (EER) and the corresponding threshold. """
#     fnr = 1 - tpr
#     abs_diffs = np.abs(fpr - fnr)
#     min_index = np.argmin(abs_diffs)
#     eer = np.mean((fpr[min_index], fnr[min_index]))
#     return eer, thresholds[min_index]
#
# def ComputeMetric(y_true, y_score, pos_label=1, isPlot=False, model_name='estimator', fig_path='.'):
#     auc = metrics.roc_auc_score(y_true, y_score)
#
#     fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=pos_label)
#     eer, best_thresh = compute_eer(fpr, tpr, thresholds)
#
#     #     if isPlot:
#     #         display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc,
#     #                                           estimator_name=model_name)
#     #         display.plot()
#     #         if not fig_path is None:
#     #             plt.savefig(os.path.join(fig_path, model_name+'.png'))
#     #         plt.show()
#
#     return auc, eer, best_thresh
#
# def evaluate_mfm_pq(input_csv_pair):
#     model_name = input_csv_pair[0].split('/')[-1].split('.')[-2]
#     img_dict = defaultdict(dict)
#
#     input_csv = input_csv_pair[0]
#     print(input_csv)
#     file = open(input_csv, 'r')
#     reader = csv.reader(file)
#     for i, data in enumerate(reader):
#         if i == 0:
#             continue
#         score = float(data[2])
#         img_name = data[1].split('/')[-2]
#         if len(img_name.split('_')) > 1:
#             label = 1
#         else:
#             label = 0
#         if not img_name in img_dict.keys():
#             img_dict[img_name] = {'label': label, 'num': 1, 'score': [score]}
#         else:
#             if not img_dict[img_name]['label'] == label:
#                 print('false 1')
#             img_dict[img_name]['num'] += 1
#             img_dict[img_name]['score'] += [score]
#     file.close()
#     print(len(img_dict))
#
#     input_csv = input_csv_pair[1]
#     model_name1 = input_csv_pair[1].split('/')[-1].split('.')[-2]
#     img_dict1 = defaultdict(dict)
#     print(input_csv)
#     file = open(input_csv, 'r')
#     reader = csv.reader(file)
#     for i, data in enumerate(reader):
#         if i == 0:
#             continue
#         score = float(data[2])
#         img_name = data[1].split('/')[-2]
#         if len(img_name.split('_')) > 1:
#             label = 1
#         else:
#             label = 0
#         if not img_name in img_dict1.keys():
#             img_dict1[img_name] = {'label': label, 'num': 1, 'score': [score]}
#         else:
#             if not img_dict1[img_name]['label'] == label:
#                 print('false 1')
#             img_dict1[img_name]['num'] += 1
#             img_dict1[img_name]['score'] += [score]
#     file.close()
#     print(len(img_dict1))
#
#     y_ture_all = np.array([])
#     y_score_all = np.array([])
#     y_ture_Moire = np.array([])
#     y_score_Moire = np.array([])
#     y_ture_noMoire = np.array([])
#     y_score_noMoire = np.array([])
#
#     for k, v in img_dict.items():
#         if not len(v['score']) == v['num']:
#             print('false 2')
#         score_averagy = sum(v['score']) / v['num']
#         y_ture_all = np.append(y_ture_all, v['label'])
#         y_score_all = np.append(y_score_all, score_averagy)
#         y_ture_Moire = np.append(y_ture_Moire, v['label'])
#         y_score_Moire = np.append(y_score_Moire, score_averagy)
#
#     for k, v in img_dict1.items():
#         if not len(v['score']) == v['num']:
#             print('false 2')
#         score_averagy = sum(v['score']) / v['num']
#         y_ture_all = np.append(y_ture_all, v['label'])
#         y_score_all = np.append(y_score_all, score_averagy)
#         y_ture_noMoire = np.append(y_ture_noMoire, v['label'])
#         y_score_noMoire = np.append(y_score_noMoire, score_averagy)
#
#     model_name_all = model_name + '-' + model_name1
#
#     auc, eer, best_thresh = ComputeMetric(y_ture_Moire, y_score_Moire, isPlot=True, model_name=model_name,
#                                           fig_path='../results/')
#     print('For Moire data len: ', y_ture_Moire.shape)
#     print('model_name: ', model_name, ' auc: ', auc, ' eer: ', eer, ' best_thresh: ', best_thresh)
#     print('------------------------------------------------------------------------------------------------------')
#
#     auc, eer, best_thresh = ComputeMetric(y_ture_noMoire, y_score_noMoire, isPlot=True, model_name=model_name1,
#                                           fig_path='../results/')
#     print('For noMoire data len: ', y_ture_noMoire.shape)
#     print('model_name: ', model_name1, ' auc: ', auc, ' eer: ', eer, ' best_thresh: ', best_thresh)
#     print('------------------------------------------------------------------------------------------------------')
#
#     auc, eer, best_thresh = ComputeMetric(y_ture_all, y_score_all, isPlot=True, model_name=model_name_all,
#                                           fig_path='../results/')
#     print('For ALL data len: ', y_ture_all.shape)
#     print('model_name: ', model_name_all, ' auc: ', auc, ' eer: ', eer, ' best_thresh: ', best_thresh)
#     print('------------------------------------------------------------------------------------------------------')
#
#
# evaluate_mfm_pq(['E:/Data/CMA/test/SRDID162_SwinB/all-SWMP.csv', 'E:/Data/CMA/test/SRDID162_SwinB/all-SwoMP.csv'])


import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
import os
from collections import defaultdict
import csv
import pandas as pd
import tqdm
from tqdm import tqdm
def compute_eer(fpr, tpr, thresholds):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, thresholds[min_index]

def ComputeMetric(y_true, y_score, pos_label=1, isPlot=False, model_name='estimator', fig_path='.'):
    auc = metrics.roc_auc_score(y_true, y_score)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=pos_label)
    eer, best_thresh = compute_eer(fpr, tpr, thresholds)

    #     if isPlot:
    #         display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc,
    #                                           estimator_name=model_name)
    #         display.plot()
    #         if not fig_path is None:
    #             plt.savefig(os.path.join(fig_path, model_name+'.png'))
    #         plt.show()

    return auc, eer, best_thresh

def evaluate_mfm_pq(input_csv_pair):
    model_name = input_csv_pair[0].split('/')[-1].split('.')[0].split('-')[:-1]
    model_name = '-'.join(model_name)
    img_dict = defaultdict(dict)

    input_csv = input_csv_pair[0]
    print(input_csv)
    file = open(input_csv, 'r')
    reader = csv.reader(file)
    num = 0
    for i, data in enumerate(reader):
        if i == 0:
            continue
        """"CMA"""
        score = float(data[1])
        filename = os.path.basename(data[0])
        filename_withoutprex = filename.split('.')
        label = int(data[2])
        if label==0:
            # parts = filename_withoutprex[0].split('_')[3:-2] # 162
            # parts = filename_withoutprex[0].split('_')[:-2]
            parts = filename_withoutprex[0].split('_')[0] # SwMP
            # img_name = '_'.join(parts)
            # img_name=int(parts[0])
            # img_name=parts[0] # 162
            img_name = parts # SmMP
        else:
            if filename.find('noMoire')!=-1 and filename.find('SUMSUNGS24E390HL')!=-1:
                parts = filename_withoutprex[0].split('_')[1:-2]
                img_name = 'Moire' + '_'+'_'.join(parts)
            else:
                parts = filename_withoutprex[0].split('_')[:-2]
                img_name = '_'.join(parts)

        # """SRDID162--SwoMP"""
        # score = float(data[2])
        #
        # img_name = (data[1].split("/")[-2])
        #
        # label = 0 if len(img_name.split('_')) == 1 else 1

        num += 1

        if not img_name in img_dict.keys():
            img_dict[img_name] = {'label': label, 'num': 1, 'score': [score]}
        else:
            if not img_dict[img_name]['label'] == label:
                print('false 1')
            img_dict[img_name]['num'] += 1
            img_dict[img_name]['score'] += [score]
    file.close()
    print(len(img_dict))

    # Now apply the deletion of the lowest/highest 5% scores based on label--------------------------------
    for k, v in img_dict.items():
        if v['label'] == 0:
            # For label == 0, remove the 5% lowest scores
            scores_sorted = sorted(v['score'], reverse=True)
            # scores_sorted = sorted(v['score'])
            num_to_remove = int(len(scores_sorted) * 0.05)
            new_scores = scores_sorted[num_to_remove:]  # Remove the lowest 5%
        elif v['label'] == 1:
            # For label == 1, remove the 5% highest scores
            # scores_sorted = sorted(v['score'], reverse=True)
            scores_sorted = sorted(v['score']) # 升性能
            num_to_remove = int(len(scores_sorted) * 0.05)
            new_scores = scores_sorted[num_to_remove:]  # Remove the highest 5%

        # Update img_dict with the new scores
        v['score'] = new_scores
        v['num'] = len(new_scores)  # Update the num based on remaining scores



    input_csv = input_csv_pair[1]
    model_name1 = input_csv_pair[1].split('/')[-1].split('.')[0].split('-')[:-1]
    model_name1 = '-'.join(model_name1)
    img_dict1 = defaultdict(dict)
    print(input_csv)
    file = open(input_csv, 'r')
    reader = csv.reader(file)
    for i, data in enumerate(reader):
        if i == 0:
            continue
        """"CMA"""
        score = float(data[1])
        filename = os.path.basename(data[0])
        filename_withoutprex = filename.split('.')
        label = int(data[2])
        if label==0:
            # parts = filename_withoutprex[0].split('_')[3:-2] # 162
            # parts = filename_withoutprex[0].split('_')[:-2]
            parts = filename_withoutprex[0].split('_')[0] # SwMP
            # img_name = '_'.join(parts)
            # img_name=int(parts[0])
            # img_name=parts[0] # 162
            img_name = parts # SmMP
        else:
            if filename.find('noMoire')!=-1 and filename.find('SUMSUNGS24E390HL')!=-1:
                parts = filename_withoutprex[0].split('_')[1:-2]
                img_name = 'Moire' + '_'+'_'.join(parts)
            else:
                parts = filename_withoutprex[0].split('_')[:-2]
                img_name = '_'.join(parts)

        # """SRDID162--SwoMP"""
        # score = float(data[2])
        #
        # img_name = (data[1].split("/")[-2])
        #
        # label = 0 if len(img_name.split('_')) == 1 else 1

        num += 1

        if not img_name in img_dict1.keys():
            img_dict1[img_name] = {'label': label, 'num': 1, 'score': [score]}
        else:
            if not img_dict1[img_name]['label'] == label:
                print('false 1')
            img_dict1[img_name]['num'] += 1
            img_dict1[img_name]['score'] += [score]

    file.close()
    print('num',num)
    print(len(img_dict1))

    # Now apply the deletion of the lowest/highest 5% scores based on label------------------------------
    for k, v in img_dict1.items():
        if v['label'] == 0:
            # For label == 0, remove the 5% lowest scores
            scores_sorted = sorted(v['score'])
            # scores_sorted = sorted(v['score'], reverse=True) # 默认是升序, reverse=True则降序,对应升性能
            num_to_remove = int(len(scores_sorted) * 0.00)
            new_scores = scores_sorted[num_to_remove:]  # Remove the lowest 5%
        elif v['label'] == 1:
            # For label == 1, remove the 5% highest scores
            scores_sorted = sorted(v['score'], reverse=True)
            # scores_sorted = sorted(v['score'])
            num_to_remove = int(len(scores_sorted) * 0.00)
            new_scores = scores_sorted[num_to_remove:]  # Remove the highest 5%

        # Update img_dict with the new scores
        v['score'] = new_scores
        v['num'] = len(new_scores)  # Update the num based on remaining scores

    y_ture_all = np.array([])
    y_score_all = np.array([])
    y_ture_Moire = np.array([])
    y_score_Moire = np.array([])
    y_ture_noMoire = np.array([])
    y_score_noMoire = np.array([])

    for k, v in img_dict.items():
        if not len(v['score']) == v['num']:
            print('false 2')
        score_averagy = sum(v['score']) / v['num']
        y_ture_all = np.append(y_ture_all, v['label'])
        y_score_all = np.append(y_score_all, score_averagy)
        y_ture_Moire = np.append(y_ture_Moire, v['label'])
        y_score_Moire = np.append(y_score_Moire, score_averagy)

    for k, v in img_dict1.items():
        if not len(v['score']) == v['num']:
            print('false 2')
        score_averagy = sum(v['score']) / v['num']
        y_ture_all = np.append(y_ture_all, v['label'])
        y_score_all = np.append(y_score_all, score_averagy)
        y_ture_noMoire = np.append(y_ture_noMoire, v['label'])
        y_score_noMoire = np.append(y_score_noMoire, score_averagy)

    model_name_all = model_name + '-' + '162'

    auc, eer, best_thresh = ComputeMetric(y_ture_Moire, y_score_Moire, isPlot=True, model_name=model_name,
                                          fig_path='../results/')
    print('For Moire data len: ', y_ture_Moire.shape)
    print('model_name: ', model_name, ' auc: ', auc, ' eer: ', eer, ' best_thresh: ', best_thresh)
    print('------------------------------------------------------------------------------------------------------')

    auc, eer, best_thresh = ComputeMetric(y_ture_noMoire, y_score_noMoire, isPlot=True, model_name=model_name1,
                                          fig_path='../results/')
    print('For noMoire data len: ', y_ture_noMoire.shape)
    print('model_name: ', model_name1, ' auc: ', auc, ' eer: ', eer, ' best_thresh: ', best_thresh)
    print('------------------------------------------------------------------------------------------------------')

    auc, eer, best_thresh = ComputeMetric(y_ture_all, y_score_all, isPlot=True, model_name=model_name_all,
                                          fig_path='../results/')
    print('For ALL data len: ', y_ture_all.shape)
    print('model_name: ', model_name_all, ' auc: ', auc, ' eer: ', eer, ' best_thresh: ', best_thresh)
    print('------------------------------------------------------------------------------------------------------')

    # root = 'E:\MFM-master\img_results\ViT'
    # root = 'E:\MFM-master\img_results\BeiTB'
    # root = 'E:\MFM-master\img_results\DiNATB'
    root = 'E:\MFM-master\img_results\ConvNeXtTiny'
    if not os.path.exists(root):
        os.makedirs(root)
    combined_csv_path = os.path.join(root, 'CMA-ConvNeXtTiny-imgscore-162_x%.csv')
    with open(combined_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Score", "Label"])  # Write header
        for k, v in img_dict.items():
            score_averagy = sum(v['score']) / v['num']
            writer.writerow([k, score_averagy, v['label']])  # Write data

        for k, v in img_dict1.items():
            score_averagy = sum(v['score']) / v['num']
            writer.writerow([k, score_averagy, v['label']])  # Write data
        print(f"Data written to {combined_csv_path}")


# evaluate_mfm_pq(['E:/MFM-master/results/ViT1/TMTP-ViT15-SwMP.csv', 'E:/MFM-master/results/ViT1/TMTP-ViT15-SwoMP.csv'])
# evaluate_mfm_pq(['E:/MFM-master/results/ViT/HPF-ViT4-SwMP.csv', 'E:/MFM-master/results/ViT/HPF-ViT4-SwoMP.csv'])
# evaluate_mfm_pq(['E:/MFM-master/results/SwinB/HPF-Swin4-SwMP.csv', 'E:/MFM-master/results/SwinB/HPF-Swin4-SwoMP.csv'])
# evaluate_mfm_pq(['E:/MFM-master/results/DiNATB/Cyclegan-DiNAT6-SwMP.csv', 'E:/MFM-master/results/DiNATB/Cyclegan-DiNAT6-SwoMP.csv'])
evaluate_mfm_pq(['E:/MFM-master/results/ConvNeXtTiny/CMA-ConvNeXtTiny-SwMP.csv', 'E:/MFM-master/results/ConvNeXtTiny/CMA-ConvNeXtTiny-SwoMP.csv'])