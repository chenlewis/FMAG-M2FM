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

def evaluation2(csv_path, root='E:\MFM-master\img_results\ViT'):
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
        score = float(data[1])
        filename = os.path.basename(data[0])
        filename_withoutprex = filename.split('.')
        label = int(data[2])
        if label==0:
            # parts = filename_withoutprex[0].split('_')[3:-2] # 162
            parts = filename_withoutprex[0].split('_')[0] # SwMP
            # img_name = '_'.join(parts)
            # img_name=int(parts[0])
            # img_name=parts[0] # 162
            img_name = parts # SwMP
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
        
    combined_csv_path = os.path.join(root, 'FAG-ViT-imgscore-162.csv')

    with open(combined_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Name', 'Score', 'Label'])
        for k, v in img_dict.items():
            score_average = sum(v['score']) / v['num']
            writer.writerow([k, score_average, v['label']])
        print(f"Data written to {combined_csv_path}")


if __name__ == '__main__':
    csv_path = r"E:\MFM-master\results\ViT\FAG-ViT-SwoMP.csv"
    evaluation2(csv_path)
