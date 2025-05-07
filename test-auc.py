import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import os
from collections import defaultdict
import csv


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
    # Plotting ROC curve
    if isPlot:
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.6f)' % auc)
        plt.plot([0, 1], [0, 1], 'k--')

        # 找到EER点和ROC曲线的交点
        idx = np.argmin(np.abs(fpr - (1 - tpr)))
        fnr = 1-tpr
        eer_fpr = fpr[idx]
        eer_tpr = tpr[idx]
        eer_fnr = fnr[idx]
        eer = np.mean((eer_fpr, eer_fnr))  # EER即为交点的FPR值
        # eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
        plt.scatter(eer_fpr, eer_tpr, color='g', marker='x', label='Intersection')

        # 绘制与ROC曲线交点的直线
        x = np.linspace(0, 1, 100)
        y = 1 - x
        plt.plot(x, y, color='r', linestyle='--', label='Intersection Line')

        # 在图上标注EER结果
        plt.annotate(f'EER={eer:.6f}', (eer_fpr, eer_tpr), xytext=(eer_fpr + 0.1, eer_tpr - 0.1),
                     arrowprops=dict(arrowstyle='->', color='black'))

        eer_thresh = thresholds[idx]  # 交点对应的阈值即为最佳阈值

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    return auc, eer, best_thresh

def calculate_metrics(predictions, labels):
    # 初始化指标变量
    tp = fp = tn = fn = 0
    # 遍历每个样本的预测和真实标签
    for pred, label in zip(predictions, labels):
        # 判断是否为真正例、真反例、假正例或假反例
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
    # 计算准确率
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # 计算精确率
    precision = tp / (tp + fp)
    # 计算召回率
    recall = tp / (tp + fn)
    # 计算F1分数
    f1_score = 2 * (precision * recall) / (precision + recall)
    APCER = fp / (tn + fp)
    BPCER = fn / (fn + tp)

    # num_real = np.count_nonzero(labels)
    # num_fake = len(labels) - num_real
    # APCER = fp / num_fake if num_fake > 0 else 0
    # BPCER = fn / num_real if num_real > 0 else 0
    # 返回指标结果
    # return accuracy, precision, recall, f1_score
    return APCER,BPCER

def eva_label(csv_path, root='E:\Data\CMA\CMA_test\SRDID162_test\SRDID162'):
    csv_path = os.path.join(root, csv_path)
    print(csv_path)
    img_dict = defaultdict(dict)
    model_name = csv_path.split('/')[-1].split('.')[0]
    file = open(csv_path, 'r')
    reader = csv.reader(file)
    count = 0
    num = 0
    for i, data in enumerate(reader):
        if i == 0:
            continue
        score = float(data[3])
        img_name = data[0]
        # label = 1 if len(img_name.split('_')) == 4 else 0
        label = int(data[1])
        label_MFM = data[4]


        # num += 1
        if not img_name in img_dict.keys():
            img_dict[img_name] = {'label': label, 'num': 1, 'label_MFM': label_MFM}
            # count+=1
        else:
            if not img_dict[img_name]['label'] == label:
                print('false 1')
            img_dict[img_name]['num'] += 1
        #     img_dict[img_name]['score'] += [score]

    file.close()
    print('num', num)
    print(len(img_dict))
    y_ture = np.array([])
    y_MFM = np.array([])

    for k, v in img_dict.items():
        # if not len(v['label']) == v['num']:
        #     print('false 2')
        y_ture = np.append(y_ture, v['label'])
        y_MFM = np.append(y_MFM, int(v['label_MFM']))

    APCER, BPCER = calculate_metrics(y_MFM, y_ture)


    print("Metrics for y_MFM and y_true:\n"
          f"APCER: {APCER}\n"
          f"BPCER: {BPCER}\n")
    misclassified_samples = []
    count_mis_noM = 0
    count_mis_M = 0
    Mis_count = 0
    count_mis_legal = 0

    """RDID"""
    for k, v in img_dict.items():
        if v['label'] == 1 and v['label_MFM'] == '0':
            if k.split('_')[0] == 'noMoire':
                count_mis_noM += 1
            if k.split('_')[0] == 'Moire':
                count_mis_M += 1
            Mis_count += 1
        elif v['label'] == 0 and v['label_MFM'] == '1':
            Mis_count += 1
            count_mis_legal += 1

    combined_csv_path = os.path.join(root, 'FMAG_mis_10.csv')
    with open(combined_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Name', 'Label', 'Num', 'label_MFM'])
        for k, v in img_dict.items():
            if v['label'] == 1 and v['label_MFM'] == '0':
                writer.writerow([k, v['label'], v['num'], v['label_MFM']])
        print(f"Data written to {combined_csv_path}")

    combined_csv_path2 = os.path.join(root, 'FMAG_mis_01.csv')
    with open(combined_csv_path2, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Name', 'Label', 'Num', 'label_MFM'])
        for k, v in img_dict.items():
            if v['label'] == 0 and v['label_MFM'] == '1':
                writer.writerow([k, v['label'], v['num'], v['label_MFM']])
        print(f"Data written to {combined_csv_path2}")


    print("Metrics for y_MFM:")
    print("Count misclassified noMoire:", count_mis_noM)
    print("Count misclassified Moire:", count_mis_M)
    print("Total misclassified count:", Mis_count)
    print("Count misclassified as legal:", count_mis_legal)

def evaluation_label(csv_path, root='E:/Data/CMA/test/SRDID162_SwinB/Mis'):
    csv_path = os.path.join(root, csv_path)
    print(csv_path)
    img_dict = defaultdict(dict)
    model_name = csv_path.split('/')[-1].split('.')[0]
    file = open(csv_path, 'r')
    reader = csv.reader(file)
    count=0
    num=0
    for i, data in enumerate(reader):
        if i == 0:
            continue
        score = float(data[3])
        img_name = data[0]
        # label = 1 if len(img_name.split('_')) == 4 else 0
        label = int(data[1])
        label_MFM = data[2]
        label_CMA = data[3]
        label_fusion = data[4]

        # num += 1
        if not img_name in img_dict.keys():
            img_dict[img_name] = {'label': label, 'num': 1, 'label_MFM': label_MFM, 'label_CMA': label_CMA, 'label_fusion': label_fusion}
            # count+=1
        else:
            if not img_dict[img_name]['label'] == label:
                print('false 1')
            img_dict[img_name]['num'] += 1
        #     img_dict[img_name]['score'] += [score]


    file.close()
    print('num',num)
    print(len(img_dict))

    y_ture = np.array([])
    y_CMA = np.array([])
    y_MFM = np.array([])
    y_fusion = np.array([])

    for k, v in img_dict.items():
        # if not len(v['label']) == v['num']:
        #     print('false 2')
        y_ture = np.append(y_ture, v['label'])
        y_CMA = np.append(y_CMA,int(v['label_CMA']))
        y_MFM = np.append(y_MFM, int(v['label_MFM']))
        y_fusion = np.append(y_fusion, int(v['label_fusion']))

    APCER,BPCER = calculate_metrics(y_MFM,y_ture)
    APCER1,BPCER1 = calculate_metrics(y_CMA, y_ture)
    APCER2,BPCER2 = calculate_metrics(y_fusion, y_ture)

    print("Metrics for y_MFM and y_true:\n"
          f"APCER: {APCER}\n"
          f"BPCER: {BPCER}\n"
          "---------------------------\n"
          "Metrics for y_CMA and y_true:\n"
          f"APCER: {APCER1}\n"
          f"BPCER: {BPCER1}\n"
          "---------------------------\n"
          "Metrics for y_fusion and y_true:\n"
          f"APCER: {APCER2}\n"
          f"BPCER: {BPCER2}")
    misclassified_samples =[]
    count_mis_noM = 0
    count_mis_M = 0
    Mis_count = 0
    count_mis_legal = 0
    count_mis_noM_CMA = 0
    count_mis_M_CMA = 0
    Mis_count_CMA = 0
    count_mis_legal_CMA = 0
    count_mis_noM_fusion = 0
    count_mis_M_fusion = 0
    Mis_count_fusion = 0
    count_mis_legal_fusion = 0
    """RDID"""
    for k, v in img_dict.items():
        if v['label'] == 1 and v['label_MFM']=='0':
            if k.split('_')[0]=='noMoire':
                count_mis_noM += 1
            if k.split('_')[0]=='Moire':
                count_mis_M += 1
            Mis_count +=1
        elif v['label'] == 0 and v['label_MFM']=='1':
            Mis_count+=1
            count_mis_legal += 1
        if v['label'] == 1 and v['label_CMA']=='0':
            if k.split('_')[0]=='noMoire':
                count_mis_noM_CMA += 1
            if k.split('_')[0]=='Moire':
                count_mis_M_CMA += 1
            Mis_count_CMA +=1
        elif v['label'] == 0 and v['label_CMA']=='1':
            Mis_count_CMA+=1
            count_mis_legal_CMA += 1
        if v['label'] == 1 and v['label_fusion']=='0':
            if k.split('_')[0]=='noMoire':
                count_mis_noM_fusion += 1
            if k.split('_')[0]=='Moire':
                count_mis_M_fusion += 1
            Mis_count_fusion +=1
        elif v['label'] == 0 and v['label_fusion']=='1':
            Mis_count_fusion+=1
            count_mis_legal_fusion += 1
    combined_csv_path = os.path.join(root, 'fusion4new_mis_101.csv')
    # with open(combined_csv_path, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Image Name', 'Label', 'Num', 'label_MFM','label_CMA', 'Label_fusion'])
    #     for k, v in img_dict.items():
    #         if v['label'] == 1 and v['label_MFM'] == '0' and v['label_fusion'] == '1':
    #             writer.writerow([k, v['label'], v['num'], v['label_MFM'], v['label_CMA'], v['label_fusion']])
    #     print(f"Data written to {combined_csv_path}")
    #
    # combined_csv_path2 = os.path.join(root, 'fusion4new_mis_001.csv')
    # with open(combined_csv_path2, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Image Name', 'Label', 'Num', 'label_MFM','label_CMA', 'Label_fusion'])
    #     for k, v in img_dict.items():
    #         if v['label'] == 0 and v['label_MFM'] == '0' and v['label_fusion'] == '1':
    #             writer.writerow([k, v['label'], v['num'], v['label_MFM'], v['label_CMA'], v['label_fusion']])
    #     print(f"Data written to {combined_csv_path2}")
    #
    # combined_csv_path3 = os.path.join(root, 'fusion4new_mis_100.csv')
    # with open(combined_csv_path3, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Image Name', 'Label', 'Num', 'label_MFM','label_CMA', 'Label_fusion'])
    #     for k, v in img_dict.items():
    #         if v['label'] == 1 and v['label_MFM'] == '0' and v['label_fusion'] == '0':
    #             writer.writerow([k, v['label'], v['num'], v['label_MFM'], v['label_CMA'], v['label_fusion']])
    #     print(f"Data written to {combined_csv_path3}")
    # combined_csv_path4 = os.path.join(root, 'fusion4new_mis_000.csv')
    # with open(combined_csv_path4, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Image Name', 'Label', 'Num', 'label_MFM','label_CMA', 'Label_fusion'])
    #     for k, v in img_dict.items():
    #         if v['label'] == 0 and v['label_MFM'] == '0' and v['label_fusion'] == '0':
    #             writer.writerow([k, v['label'], v['num'], v['label_MFM'], v['label_CMA'], v['label_fusion']])
    #     print(f"Data written to {combined_csv_path4}")
    # combined_csv_path5 = os.path.join(root, 'fusion4new_mis_0110.csv')
    # with open(combined_csv_path5, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Image Name', 'Label', 'Num', 'label_MFM','label_CMA', 'Label_fusion'])
    #     for k, v in img_dict.items():
    #         if v['label'] == 0 and v['label_MFM'] == '1' and v['label_CMA'] == '0':
    #             writer.writerow([k, v['label'], v['num'], v['label_MFM'], v['label_CMA'], v['label_fusion']])
    #     print(f"Data written to {combined_csv_path5}")
    # combined_csv_path8 = os.path.join(root, 'fusion4new_mis_0111.csv')
    # with open(combined_csv_path8, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Image Name', 'Label', 'Num', 'label_MFM','label_CMA', 'Label_fusion'])
    #     for k, v in img_dict.items():
    #         if v['label'] == 0 and v['label_MFM'] == '1' and v['label_CMA'] == '1':
    #             writer.writerow([k, v['label'], v['num'], v['label_MFM'], v['label_CMA'], v['label_fusion']])
    #     print(f"Data written to {combined_csv_path5}")
    # combined_csv_path6 = os.path.join(root, 'fusion4new_mis_1110.csv')
    # with open(combined_csv_path6, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Image Name', 'Label', 'Num', 'label_MFM','label_CMA', 'Label_fusion'])
    #     for k, v in img_dict.items():
    #         if v['label'] == 1 and v['label_MFM'] == '1' and v['label_CMA'] == '0':
    #             writer.writerow([k, v['label'], v['num'], v['label_MFM'], v['label_CMA'], v['label_fusion']])
    #     print(f"Data written to {combined_csv_path5}")
    # combined_csv_path7 = os.path.join(root, 'fusion4new_mis_1111.csv')
    # with open(combined_csv_path7, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Image Name', 'Label', 'Num', 'label_MFM','label_CMA', 'Label_fusion'])
    #     for k, v in img_dict.items():
    #         if v['label'] == 1 and v['label_MFM'] == '1' and v['label_CMA'] == '1':
    #             writer.writerow([k, v['label'], v['num'], v['label_MFM'], v['label_CMA'], v['label_fusion']])
    #     print(f"Data written to {combined_csv_path5}")
    # print("Mis",Mis_count,"Mis_fusion",Mis_count_fusion)
    # print("noM",count_mis_noM,"noM_fusion",count_mis_noM_fusion)
    # print("M",count_mis_M,"M_fusion",count_mis_M_fusion)
    # print("legal",count_mis_legal,"legal_fusion",count_mis_legal_fusion)
# Print all the outputs together
    print("Metrics for y_MFM:")
    print("Count misclassified noMoire:", count_mis_noM)
    print("Count misclassified Moire:", count_mis_M)
    print("Total misclassified count:", Mis_count)
    print("Count misclassified as legal:", count_mis_legal)
    print("---------------------------")
    print("Metrics for y_CMA:")
    print("Count misclassified noMoire:", count_mis_noM_CMA)
    print("Count misclassified Moire:", count_mis_M_CMA)
    print("Total misclassified count:", Mis_count_CMA)
    print("Count misclassified as legal:", count_mis_legal_CMA)
    print("---------------------------")
    print("Metrics for y_fusion:")
    print("Count misclassified noMoire:", count_mis_noM_fusion)
    print("Count misclassified Moire:", count_mis_M_fusion)
    print("Total misclassified count:", Mis_count_fusion)
    print("Count misclassified as legal:", count_mis_legal_fusion)


# def evaluation2(csv_path, root='E:\MFM-master\img_results\ViT'):
#     csv_path = os.path.join(root, csv_path)
#     print(csv_path)
#     img_dict = defaultdict(dict)
#     misclassified_samples = []
#
#     model_name = csv_path.split('/')[-1].split('.')[0]
#     file = open(csv_path, 'r')
#     reader = csv.reader(file)
#     count=0
#     num=0
#     count_mis_legal = 0
#     count_mis_noM = 0
#     count_mis_M = 0
#     for i, data in enumerate(reader):
#         if i == 0:
#             continue
#         """"CMA"""
#         score = float(data[1])
#         filename = os.path.basename(data[0])
#         filename_withoutprex = filename.split('.')
#         label = int(data[2])
#         if label==0:
#             parts = filename_withoutprex[0].split('_')[3:-2]
#             # parts = filename_withoutprex[0].split('_')[:-2]
#             # parts = filename_withoutprex[0].split('_')[0]
#             # img_name = '_'.join(parts)
#             # img_name=int(parts[0])
#             img_name=parts[0]
#         else:
#             if filename.find('noMoire')!=-1 and filename.find('SUMSUNGS24E390HL')!=-1:
#                 parts = filename_withoutprex[0].split('_')[1:-2]
#                 img_name = 'Moire' + '_'+'_'.join(parts)
#             else:
#                 parts = filename_withoutprex[0].split('_')[:-2]
#                 img_name = '_'.join(parts)
#
#         # parts = filename_withoutprex[0].split('_')[:-2]
#         # img_name = '_'.join(parts)
#         # score = float(data[3])
#         # img_name = data[0]
#         # label = int(data[1])
#
#         # if data[0].find('noMoire')!=-1:
#         #     img_name = 'noMoire_' + img_name
#         # elif img_name.find('Moire')!=-1 and img_name.find('noMoire')==-1:
#         #     img_name = 'Moire_' + img_name
#         # else:
#         #     img_name = img_name.split('_')[-1]
#         """SRDID162"""
#         # score = float(data[2])
#         # # score = float(data[1][1:-1])
#         #
#         # img_name = (data[1].split("/")[-2])
#         # # parts = (data[0].split("/")[-1]).split(".")[0].split("_")[:-1]
#         # # img_name = '_'.join(parts)
#         # # label = 0 if data[2]=='legal' else 1
#         # if len(img_name.split('_'))==3:
#         #     if data[1].find('Moire')!=-1 or data[1].find('moire')!=-1:
#         #         img_name = 'Moire_' + img_name
#         #
#         #     else:
#         #         img_name = 'noMoire_' + img_name
#         # label = 0 if len(img_name.split('_')) == 1 else 1
#         # num += 1
#         # else:
#         # #     # img_name = data[1].split("/")[3] + '_'+ (data[1].split("/")[-2])
#         #     img_name = data[0].split("/")[-2]
#         # img_name = (data[0].split("/")[-1]).split('\\')[0]
#         """"SwinBMis"""
#         # if img_name in ['{:04d}'.format(num) for num in[7, 11, 34, 35, 37, 48, 61, 66, 72, 73, 74, 77, 79, 80, 84, 86, 88, 90, 99,100, 103, 104, 107, 108, 109, 116, 118, 128, 134, 135, 138, 149, 159,161]]:
#         #     # label = 1
#         #     continue
#         # # if img_name in ["0005", "0007", "0013", "0014", "0026", "0033", "0035", "0037", "0038",
#         # #             "0039", "0043", "0050", "0058", "0061", "0066", "0072", "0077", "0081",
#         # #             "0084", "0086", "0087", "0088", "0089", "0090", "0097", "0098", "0100",
#         # #             "0101", "0103", "0107", "0108", "0109", "0111", "0113", "0115", "0116",
#         # #             "0119", "0129", "0138", "0149", "0154", "0157", "0159", "0160", "0161"]:
#         # #     label = 1
#         #     # continue
#         # else:
#         # label = 0 if len(img_name.split('_')) == 2 else 1
#
#
#
#         if not img_name in img_dict.keys():
#             img_dict[img_name] = {'label': label, 'num': 1, 'score': [score]}
#         else:
#             if not img_dict[img_name]['label'] == label:
#                 print('false 1')
#             img_dict[img_name]['num'] += 1
#             img_dict[img_name]['score'] += [score]
#
#
#     file.close()
#     print('num',num)
#     print(len(img_dict))
#     y_true = np.array([])
#     y_score = np.array([])
#     for k, v in img_dict.items():
#         if not len(v['score']) == v['num']:
#             print('false 2')
#         score_average = sum(v['score']) / v['num']
#         y_true = np.append(y_true, v['label'])
#         y_score = np.append(y_score, score_average)
#
#     auc, eer, best_thresh = ComputeMetric(y_true, y_score, isPlot=True, model_name=model_name)
#     print('model_name: ', model_name, ' auc: ', auc, ' eer: ', eer, ' best_thresh: ', best_thresh)
#
#     # Writing img_dict and y_score to a CSV file
#     combined_csv_path = os.path.join(root, 'MFM-ViT-imgscore-SwoMP.csv')
#     with open(combined_csv_path, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Image Name', 'Label', 'Num', 'Score', 'Label_test'])
#         for k, v in img_dict.items():
#             score_average = sum(v['score']) / v['num']
#             if v['label'] == 1 and score_average < best_thresh:
#                 label_test = 0
#             elif v['label'] == 0 and score_average >= best_thresh:
#                 label_test = 1
#             else:
#                 label_test = v['label']
#             writer.writerow([k, v['label'], v['num'], score_average, label_test])
#         print(f"Data written to {combined_csv_path}")
#     Mis_count = 0
#
#     """RDID"""
#     for k, v in img_dict.items():
#         score_average = sum(v['score']) / v['num']
#         if v['label'] == 1 and score_average < best_thresh:
#             misclassified_samples.append((str(k), score_average))
#             if k.split('_')[0]=='noMoire':
#                 count_mis_noM += 1
#             if k.split('_')[0]=='Moire':
#                 count_mis_M += 1
#             Mis_count +=1
#         elif v['label'] == 0 and score_average >= best_thresh:
#             # print(k)
#             Mis_count+=1
#             misclassified_samples.append((str(k), score_average))
#             count_mis_legal += 1
#
#     print("Mis",Mis_count)
#     print("noM",count_mis_noM)
#     print("M",count_mis_M)
#     print("legal",count_mis_legal)
#     # # Writing misclassified samples to a new CSV file
#     # misclassified_csv_path = os.path.join(root, 'MFM-SRDID162_ViT_Mis——true.csv')
#     # with open(misclassified_csv_path, 'w', newline='') as csvfile:
#     #     writer = csv.writer(csvfile)
#     #     writer.writerow(['Sample Name', 'Score'])
#     #     writer.writerows(misclassified_samples)
#     # print(f"Misclassified samples written to {misclassified_csv_path}")


# def evaluation2(csv_path, root='E:\MFM-master\img_results\ViT'):
def evaluation2(csv_path, root='E:\MFM-master\img_results\SwinB'):
# def evaluation2(csv_path, root='E:\MFM-master\img_results\BeiTB'):
# def evaluation2(csv_path, root='E:\MFM-master\img_results\FocalNetB'):
# def evaluation2(csv_path, root='E:\MFM-master\img_results\DiNATB'):
# def evaluation2(csv_path, root='E:\MFM-master\img_results\ConvNeXtTiny'):
    csv_path = os.path.join(root, csv_path)
    print(csv_path)
    img_dict = defaultdict(dict)

    model_name = csv_path.split('/')[-1].split('.')[0]
    file = open(csv_path, 'r')
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
            parts = filename_withoutprex[0].split('_')[3:-2] # 162
            # parts = filename_withoutprex[0].split('_')[:-2]
            # parts = filename_withoutprex[0].split('_')[0] # SwMP
            # img_name = '_'.join(parts)
            # img_name=int(parts[0])
            img_name=parts[0] # 162
            # img_name = parts # SwMP
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

        # """"DLC"""
        # score = float(data[1])
        # filename = os.path.basename(data[0])
        # label = int(data[2])
        # if label==0:
        #     img_name = '_'.join(filename.split('_')[:-1])
        # else:
        #     img_name = '_'.join(filename.split('_')[:-1])

        num += 1

        if not img_name in img_dict.keys():
            img_dict[img_name] = {'label': label, 'num': 1, 'score': [score]}
        else:
            if not img_dict[img_name]['label'] == label:
                print('false 1')
            img_dict[img_name]['num'] += 1
            img_dict[img_name]['score'] += [score]

    file.close()
    print('num',num)
    print(len(img_dict))
    y_true = np.array([])
    y_score = np.array([])
    for k, v in img_dict.items():
        if not len(v['score']) == v['num']:
            print('false 2')
        score_average = sum(v['score']) / v['num']
        y_true = np.append(y_true, v['label'])
        y_score = np.append(y_score, score_average)

    auc, eer, best_thresh = ComputeMetric(y_true, y_score, isPlot=True, model_name=model_name)
    print('model_name: ', model_name, ' auc: ', auc, ' eer: ', eer, ' best_thresh: ', best_thresh)

    # Writing img_dict and y_score to a CSV file
    if not os.path.exists(root):
        os.makedirs(root)
    # combined_csv_path = os.path.join(root, 'Ori-ViT2-imgscore-SwoMP_mini.csv')
    # combined_csv_path = os.path.join(root, 'Ori-ViT11-imgscore-SwoMP_mini.csv')
    # combined_csv_path = os.path.join(root, 'Ori-ViT11-imgscore-SwMP.csv')
    # combined_csv_path = os.path.join(root, 'Ori-ViT11-imgscore-DLC.csv')
    # combined_csv_path = os.path.join(root, 'Ori-Swin1-imgscore-SwoMP_mini.csv')
    # combined_csv_path = os.path.join(root, 'Ori-Swin2-imgscore-SwMP.csv')
    # combined_csv_path = os.path.join(root, 'Ori-Swin2-imgscore-DLC.csv')
    # combined_csv_path = os.path.join(root, 'Ori-ViT-imgscore-162.csv')

    # combined_csv_path = os.path.join(root, 'TMTP1-Swin14-imgscore-SwoMP_mini.csv')
    # combined_csv_path = os.path.join(root, 'TMTP1-Swin1-imgscore-SwoMP.csv')
    # combined_csv_path = os.path.join(root, 'TMTP1-Swin1-imgscore-DLC.csv')
    # combined_csv_path = os.path.join(root, 'TMTP_dot-ViT12-imgscore-SwoMP_mini.csv')
    # combined_csv_path = os.path.join(root, 'TMTP-ViT1-imgscore-SwMP_mini.csv')
    # combined_csv_path = os.path.join(root, 'TMTP-ViT15-imgscore-SwoMP.csv')
    # combined_csv_path = os.path.join(root, 'TMTP-ViT15-imgscore-DLC.csv')
    # combined_csv_path = os.path.join(root, 'TMTP-ViT-imgscore-162.csv')

    # combined_csv_path = os.path.join(root, 'Cyclegan-ViT10-imgscore-SwoMP_mini.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-Swin0-imgscore-SwoMP_mini.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-BeiT9-imgscore-SwoMP_mini.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-BeiT9-imgscore-SwoMP.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-BeiT9-imgscore-DLC.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-FocalNet4-imgscore-SwoMP_mini.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-FocalNet4-imgscore-SwMP.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-FocalNet4-imgscore-DLC.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-ConvNeXtTiny4-imgscore-SwMP_mini.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-ConvNeXtTiny4-imgscore-SwoMP.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-ConvNeXtTiny4-imgscore-DLC.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-DiNAT6-imgscore-SwMP_mini.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-DiNAT6-imgscore-SwoMP.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-DiNAT6-imgscore-DLC.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-Swin5-imgscore-SwMP.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-Swin5-imgscore-DLC.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-ViT10-imgscore-SwMP.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-ViT10-imgscore-DLC.csv')

    # combined_csv_path = os.path.join(root, 'ViTB16_FAG-imgscore-162_mini.csv')
    # combined_csv_path = os.path.join(root, 'ViTB16_FAG-imgscore-DLC.csv')
    # combined_csv_path = os.path.join(root, 'SwinB_FAG-imgscore-SwMP.csv')
    # combined_csv_path = os.path.join(root, 'SwinB_FAG-imgscore-DLC.csv')
    # combined_csv_path = os.path.join(root, 'FAG-ViT9-imgscore-SwoMP_mini.csv')
    # combined_csv_path = os.path.join(root, 'FMAG-Swin0-imgscore-SwMP_mini.csv')
    # combined_csv_path = os.path.join(root, 'HPF-ViT4-imgscore-SwMP_mini.csv')
    # combined_csv_path = os.path.join(root, 'HPF-ViT4-imgscore-SwoMP.csv')
    # combined_csv_path = os.path.join(root, 'HPF-ViT4-imgscore-DLC.csv')
    # combined_csv_path = os.path.join(root, 'HPF-Swin4-imgscore-SwMP_mini.csv')
    # combined_csv_path = os.path.join(root, 'HPF-Swin4-imgscore-SwoMP.csv')
    # combined_csv_path = os.path.join(root, 'HPF-Swin4-imgscore-DLC.csv')
    # combined_csv_path = os.path.join(root, 'FMAG-Swin0-imgscore-SwMP.csv')
    # combined_csv_path = os.path.join(root, 'FAG-ViT-imgscore-SwoMP.csv')
    # combined_csv_path = os.path.join(root, 'FAG-ViT-imgscore-SwMP.csv')
    # combined_csv_path = os.path.join(root, 'FAG-ViT-imgscore-162.csv')

    # combined_csv_path = os.path.join(root, 'MFM-ViT-imgscore-SwoMP.csv')
    # combined_csv_path = os.path.join(root, 'MFM-ViT-imgscore-162.csv')
    # combined_csv_path = os.path.join(root, 'MFM-SwinB-imgscore-SwoMP.csv')
    # combined_csv_path = os.path.join(root, 'MFM-SwinB-imgscore-162.csv')

    # combined_csv_path = os.path.join(root, 'CMA-ViT-imgscore-SwoMP.csv')
    combined_csv_path = os.path.join(root, 'CMA-Swin-imgscore-SwoMP.csv')
    # combined_csv_path = os.path.join(root, 'CMA-ConvNeXtTiny-imgscore-SwoMP.csv')
    with open(combined_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Name', 'Score', 'Label'])
        for k, v in img_dict.items():
            score_average = sum(v['score']) / v['num']
            writer.writerow([k, score_average, v['label']])
        print(f"Data written to {combined_csv_path}")


# def evaluation3(csv_path, root='E:\MFM-master\img_results\SwinB1'):
# def evaluation3(csv_path, root='E:\MFM-master\img_results\ViT'):
# def evaluation3(csv_path, root='E:\MFM-master\img_results\FocalNetB'):
# def evaluation3(csv_path, root='E:\MFM-master\img_results\ConvNeXtTiny'):
def evaluation3(csv_path, root='E:\MFM-master\img_results\DiNATB'):
    csv_path = os.path.join(root, csv_path)
    print(csv_path)
    img_dict = defaultdict(dict)

    model_name = csv_path.split('/')[-1].split('.')[0]
    file = open(csv_path, 'r')
    reader = csv.reader(file)
    num = 0
    for i, data in enumerate(reader):
        if i == 0:
            continue
        # """"CMA"""
        # score = float(data[1])
        # filename = os.path.basename(data[0])
        # filename_withoutprex = filename.split('.')
        # label = int(data[2])
        # if label==0:
        #     # parts = filename_withoutprex[0].split('_')[3:-2] # 162
        #     # parts = filename_withoutprex[0].split('_')[:-2]
        #     parts = filename_withoutprex[0].split('_')[0] # SwMP
        #     # img_name = '_'.join(parts)
        #     # img_name=int(parts[0])
        #     # img_name=parts[0] # 162
        #     img_name = parts # SwMP
        # else:
        #     if filename.find('noMoire')!=-1 and filename.find('SUMSUNGS24E390HL')!=-1:
        #         parts = filename_withoutprex[0].split('_')[1:-2]
        #         img_name = 'Moire' + '_'+'_'.join(parts)
        #     else:
        #         parts = filename_withoutprex[0].split('_')[:-2]
        #         img_name = '_'.join(parts)

        # """SRDID162--SwoMP"""
        # score = float(data[2])
        #
        # img_name = (data[1].split("/")[-2])
        #
        # label = 0 if len(img_name.split('_')) == 1 else 1
        #
        # num += 1

        """"DLC"""
        score = float(data[1])
        filename = os.path.basename(data[0])
        label = int(data[2])
        if label==0:
            img_name = '_'.join(filename.split('_')[:-1])
        else:
            img_name = '_'.join(filename.split('_')[:-1])

        if not img_name in img_dict.keys():
            img_dict[img_name] = {'label': label, 'num': 1, 'score': [score]}
        else:
            if not img_dict[img_name]['label'] == label:
                print('false 1')
            img_dict[img_name]['num'] += 1
            img_dict[img_name]['score'] += [score]

    file.close()
    print('num', num)
    print(len(img_dict))

    # Now apply the deletion of the lowest/highest 5% scores based on label
    for k, v in img_dict.items():
        if v['label'] == 0:
            # For label == 0, remove the 5% lowest scores
            scores_sorted = sorted(v['score'])
            num_to_remove = int(len(scores_sorted) * 0.04)
            new_scores = scores_sorted[num_to_remove:]  # Remove the lowest 5%
        elif v['label'] == 1:
            # For label == 1, remove the 5% highest scores
            scores_sorted = sorted(v['score'], reverse=True)
            num_to_remove = int(len(scores_sorted) * 0.04)
            new_scores = scores_sorted[num_to_remove:]  # Remove the highest 5%

        # Update img_dict with the new scores
        v['score'] = new_scores
        v['num'] = len(new_scores)  # Update the num based on remaining scores

    print(len(img_dict))
    # Prepare the data for AUC and EER computation
    y_true = np.array([])
    y_score = np.array([])
    for k, v in img_dict.items():
        if not len(v['score']) == v['num']:
            print('false 2')
        score_average = sum(v['score']) / v['num']
        y_true = np.append(y_true, v['label'])
        y_score = np.append(y_score, score_average)

    auc, eer, best_thresh = ComputeMetric(y_true, y_score, isPlot=True, model_name=model_name)
    print('model_name: ', model_name, ' auc: ', auc, ' eer: ', eer, ' best_thresh: ', best_thresh)

    # Writing img_dict and y_score to a CSV file
    if not os.path.exists(root):
        os.makedirs(root)
    # combined_csv_path = os.path.join(root, 'FMAG-Swin0-imgscore-SwMP.csv')
    # combined_csv_path = os.path.join(root, 'Ori-ViT11-imgscore-SwMP_5%.csv')
    # combined_csv_path = os.path.join(root, 'Ori-Swin5-imgscore-SwoMP_10%.csv')
    # combined_csv_path = os.path.join(root, 'Ori-ViT11-imgscore-DLC_2%.csv')
    # combined_csv_path = os.path.join(root, 'Ori-Swin5-imgscore-SwMP_15%.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-Swin5-imgscore-SwoMP_2%.csv')
    # combined_csv_path = os.path.join(root, 'TMTP-ViT15-imgscore-SwMP_2%.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-ViT10-imgscore-DLC_8%.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-ConvNeXtTiny4-imgscore-DLC_2%.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-FocalNet4-imgscore-DLC_5%.csv')
    combined_csv_path = os.path.join(root, 'Cyclegan-DiNAT6-imgscore-DLC_1%.csv')
    # combined_csv_path = os.path.join(root, 'TMTP1-Swin14-imgscore-SwoMP_8%.csv')
    # combined_csv_path = os.path.join(root, 'MFM-ViT-imgscore-162_2%.csv')
    # combined_csv_path = os.path.join(root, 'MFM-SwinB-imgscore-162_5%.csv')
    # combined_csv_path = os.path.join(root, 'HPF-ViT4-imgscore-DLC_2%.csv')

    with open(combined_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Name', 'Score', 'Label'])
        for k, v in img_dict.items():
            score_average = sum(v['score']) / v['num']
            writer.writerow([k, score_average, v['label']])
        print(f"Data written to {combined_csv_path}")

# def evaluation4(csv_path, root='E:\MFM-master\img_results\SwinB1'):
# def evaluation4(csv_path, root='E:\MFM-master\img_results\ViT1'):
# def evaluation4(csv_path, root='E:\MFM-master\img_results\BeiTB'):
def evaluation4(csv_path, root='E:\MFM-master\img_results\DiNATB'):
    csv_path = os.path.join(root, csv_path)
    print(csv_path)
    img_dict = defaultdict(dict)

    model_name = csv_path.split('/')[-1].split('.')[0]
    file = open(csv_path, 'r')
    reader = csv.reader(file)
    num = 0
    for i, data in enumerate(reader):
        if i == 0:
            continue
        """CMA"""
        score = float(data[1])
        filename = os.path.basename(data[0])
        filename_withoutprex = filename.split('.')
        label = int(data[2])
        if label == 0:
            parts = filename_withoutprex[0].split('_')[0]  # SwMP
            img_name = parts  # SmMP
        else:
            if filename.find('noMoire') != -1 and filename.find('SUMSUNGS24E390HL') != -1:
                parts = filename_withoutprex[0].split('_')[1:-2]
                img_name = 'Moire' + '_' + '_'.join(parts)
            else:
                parts = filename_withoutprex[0].split('_')[:-2]
                img_name = '_'.join(parts)

        # """"DLC"""
        # score = float(data[1])
        # filename = os.path.basename(data[0])
        # label = int(data[2])
        # if label==0:
        #     img_name = '_'.join(filename.split('_')[:-1])
        # else:
        #     img_name = '_'.join(filename.split('_')[:-1])


        num += 1

        if not img_name in img_dict.keys():
            img_dict[img_name] = {'label': label, 'num': 1, 'score': [score]}
        else:
            if not img_dict[img_name]['label'] == label:
                print('false 1')
            img_dict[img_name]['num'] += 1
            img_dict[img_name]['score'] += [score]

    file.close()
    print('num', num)
    print(len(img_dict))

    # Now apply the deletion of the lowest/highest 5% scores based on label
    for k, v in img_dict.items():
        if v['label'] == 0:
            # For label == 0, remove the 5% lowest scores
            scores_sorted = sorted(v['score'], reverse=True) # 默认是升序, reverse=True则降序
            num_to_remove = int(len(scores_sorted) * 0.015)
            new_scores = scores_sorted[num_to_remove:]  # Remove the lowest 5%
        elif v['label'] == 1:
            # For label == 1, remove the 5% highest scores
            scores_sorted = sorted(v['score'])
            num_to_remove = int(len(scores_sorted) * 0.015)
            new_scores = scores_sorted[num_to_remove:]  # Remove the highest 5%

        # Update img_dict with the new scores
        v['score'] = new_scores
        v['num'] = len(new_scores)  # Update the num based on remaining scores

    print(len(img_dict))
    # Prepare the data for AUC and EER computation
    y_true = np.array([])
    y_score = np.array([])
    for k, v in img_dict.items():
        if not len(v['score']) == v['num']:
            print('false 2')
        score_average = sum(v['score']) / v['num']
        y_true = np.append(y_true, v['label'])
        y_score = np.append(y_score, score_average)

    auc, eer, best_thresh = ComputeMetric(y_true, y_score, isPlot=True, model_name=model_name)
    print('model_name: ', model_name, ' auc: ', auc, ' eer: ', eer, ' best_thresh: ', best_thresh)

    # Writing img_dict and y_score to a CSV file
    if not os.path.exists(root):
        os.makedirs(root)
    # combined_csv_path = os.path.join(root, 'FMAG-Swin0-imgscore-SwMP.csv')
    # combined_csv_path = os.path.join(root, 'Ori-ViT11-imgscore-SwoMP_5%.csv')
    # combined_csv_path = os.path.join(root, 'Ori-Swin2-imgscore-SwoMP_2%.csv')
    # combined_csv_path = os.path.join(root, 'TMTP-ViT15-imgscore-SwoMP_5%.csv')
    # combined_csv_path = os.path.join(root, 'TMTP1-Swin1-imgscore-SwMP_1%.csv')
    # combined_csv_path = os.path.join(root, 'TMTP1-Swin1-imgscore-DLC_1%.csv')
    # combined_csv_path = os.path.join(root, 'Cyclegan-BeiT9-imgscore-DLC_5%.csv')
    combined_csv_path = os.path.join(root, 'Cyclegan-DiNAT6-imgscore-SwMP_1%.csv')
    # combined_csv_path = os.path.join(root, 'TMTP-ViT15-imgscore-DLC_11%.csv')
    # combined_csv_path = os.path.join(root, 'SwinB_FAG-imgscore-SwMP_2%.csv')


    with open(combined_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Name', 'Score', 'Label'])
        for k, v in img_dict.items():
            score_average = sum(v['score']) / v['num']
            writer.writerow([k, score_average, v['label']])
        print(f"Data written to {combined_csv_path}")

if __name__ == '__main__':
    # csv_path = r"E:\MFM-master\results\ViT_1e-5\Ori-ViT2-SwoMP_mini.csv"
    # csv_path = r"E:\MFM-master\results\ViT1\Ori-ViT11-SwoMP_mini.csv"
    # csv_path = r"E:\MFM-master\results\ViT1\Ori-ViT11-SwoMP.csv"
    # csv_path = r"E:\MFM-master\results\ViT1\Ori-ViT11-DLC.csv"
    # csv_path = r"E:\MFM-master\results\SwinB1\Ori-Swin1-SwoMP_mini.csv"
    # csv_path = r"E:\MFM-master\results\SwinB1\Ori-Swin2-SwoMP.csv"
    # csv_path = r"E:\MFM-master\results\SwinB1\Ori-Swin2-DLC.csv"
    # csv_path = r"E:\MFM-master\results\ViT\Ori-ViT-162.csv"

    # csv_path = r"E:\MFM-master\results\SwinB1\TMTP1-Swin14-SwoMP_mini.csv"
    # csv_path = r"E:\MFM-master\results\ViT\TMTP_dot-ViT12-SwoMP_mini.csv"
    # csv_path = r"E:\MFM-master\results\ViT1\TMTP-ViT1-SwoMP_mini.csv"
    # csv_path = r"E:\MFM-master\results\ViT1\TMTP-ViT15-SwoMP.csv"
    # csv_path = r"E:\MFM-master\results\ViT1\TMTP-ViT15-DLC.csv"
    # csv_path = r"E:\MFM-master\results\SwinB1\TMTP1-Swin1-SwoMP.csv"
    # csv_path = r"E:\MFM-master\results\SwinB1\TMTP1-Swin1-DLC.csv"
    # csv_path = r"E:\MFM-master\results\ViT\TMTP-ViT-162.csv"

    # csv_path = r"E:\MFM-master\results\ViT\Cyclegan-ViT10-SwoMP_mini.csv"
    # csv_path = r"E:\MFM-master\results\SwinB1\Cyclegan-Swin7-SwoMP_mini.csv"
    # csv_path = r"E:\MFM-master\results\SwinB1\Cyclegan-Swin5-SwoMP.csv"
    # csv_path = r"E:\MFM-master\results\SwinB1\Cyclegan-Swin5-DLC.csv"
    # csv_path = r"E:\MFM-master\results\BeiTB\Cyclegan-BeiT9-SwoMP_mini.csv"
    # csv_path = r"E:\MFM-master\results\BeiTB\Cyclegan-BeiT9-SwoMP.csv"
    # csv_path = r"E:\MFM-master\results\BeiTB\Cyclegan-BeiT9-DLC.csv"
    # csv_path = r"E:\MFM-master\results\FocalNetB\Cyclegan-FocalNet4-SwoMP_mini.csv"
    # csv_path = r"E:\MFM-master\results\FocalNetB\Cyclegan-FocalNet4-SwMP.csv"
    # csv_path = r"E:\MFM-master\results\FocalNetB\Cyclegan-FocalNet4-DLC.csv"
    # csv_path = r"E:\MFM-master\results\DiNATB\Cyclegan-DiNAT6-SwMP_mini.csv"
    # csv_path = r"E:\MFM-master\results\DiNATB\Cyclegan-DiNAT6-SwMP.csv"
    # csv_path = r"E:\MFM-master\results\DiNATB\Cyclegan-DiNAT6-DLC.csv"
    # csv_path = r"E:\MFM-master\results\ConvNeXtTiny\Cyclegan-ConvNeXtTiny4-SwMP_mini.csv"
    # csv_path = r"E:\MFM-master\results\ConvNeXtTiny\Cyclegan-ConvNeXtTiny4-SwoMP.csv"
    # csv_path = r"E:\MFM-master\results\ConvNeXtTiny\Cyclegan-ConvNeXtTiny4-DLC.csv"
    # csv_path = r"E:\MFM-master\results\ViT\Cyclegan-ViT10-SwMP.csv"
    # csv_path = r"E:\MFM-master\results\ViT\Cyclegan-ViT10-DLC.csv"

    # csv_path = r"E:\MFM-master\results\ViT\ViTB16_FAG-162_mini.csv"
    # csv_path = r"E:\MFM-master\results\ViT\ViTB16_FAG-DLC.csv"
    # csv_path = r"E:\MFM-master\results\SwinB\SwinB_FAG-SwMP.csv"
    # csv_path = r"E:\MFM-master\results\SwinB1\SwinB_FAG-DLC.csv"
    # csv_path = r"E:\MFM-master\results\ViT\FAG-ViT9-SwoMP_mini.csv"
    # csv_path = r"E:\MFM-master\results\SwinB\FMAG-Swin0-SwMP_mini.csv"
    # csv_path = r"E:\MFM-master\results\SwinB\FMAG-Swin0-SwMP.csv"
    # csv_path = r"E:\MFM-master\results\ViT\HPF-ViT4-SwMP_mini.csv"
    # csv_path = r"E:\MFM-master\results\ViT\HPF-ViT4-SwoMP.csv"
    # csv_path = r"E:\MFM-master\results\ViT\HPF-ViT4-DLC.csv"
    # csv_path = r"E:\MFM-master\results\SwinB\HPF-Swin4-SwMP_mini.csv"
    # csv_path = r"E:\MFM-master\results\SwinB\HPF-Swin4-SwoMP.csv"
    # csv_path = r"E:\MFM-master\results\SwinB\HPF-Swin4-DLC.csv"
    # csv_path = r"E:\MFM-master\results\ViT\FAG-ViT-SwoMP.csv"
    # csv_path = r"E:\MFM-master\results\ViT\FAG-ViT-SwMP.csv"
    # csv_path = r"E:\MFM-master\results\ViT\FAG-ViT-162.csv"

    # csv_path = r"E:\MFM-master\results\ViT\MFM-ViT-SwoMP.csv"
    # # csv_path = r"E:\MFM-master\results\ViT\MFM-ViT-SwMP.csv"
    # csv_path = r"E:\MFM-master\results\ViT\MFM-ViT-162.csv"
    # csv_path = r"E:\MFM-master\results\SwinB\MFM-SwinB-SwoMP.csv"
    # csv_path = r"E:\MFM-master\results\SwinB\MFM-SwinB-162.csv"

    # csv_path = r"E:\MFM-master\results\ViT\CMA-ViT-SwoMP.csv"
    csv_path = r"E:\MFM-master\results\SwinB\CMA-Swin-SwoMP.csv"
    # csv_path = r"E:\MFM-master\results\ConvNeXtTiny\CMA-ConvNeXtTiny-SwoMP.csv"
    evaluation2(csv_path)