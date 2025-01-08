from torch import nn
from torch.nn import init
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
from mmcv.ops import DeformConv2dPack as DCN


# weights_init
def weights_init(m):
    if isinstance(m, (nn.Conv2d)):
        init.kaiming_normal_(m.weight.data)
    elif isinstance(m, DCN):
        init.kaiming_normal_(m.weight.data)
    #    init.constant_(m.bias.data, 0)

def loadData(name, num_components):
    # load data amd map
    if name == 'HL':
        dataset = sio.loadmat('./data/Salinas.mat')# [150 150 204] 64
        data = dataset['data']
        map = dataset['map']
    elif name == 'HB':
        dataset = sio.loadmat('./data/Beach.mat')# [150 150 188] 128  128
        data = dataset['data']
        map = dataset['map']
    elif name == 'HP':
        dataset = sio.loadmat('./data/Pavia center.mat')# [150 150 102] 128
        data = dataset['data']
        map = dataset['map']
    elif name == 'HUI':
        dataset = sio.loadmat('./data/abu-urban-4.mat')# [100 100 205] 128
        data = dataset['data']
        map = dataset['map']
    elif name == 'HY':
        dataset = sio.loadmat('./data/Hyperion.mat')# [100 100 198] 96
        data = dataset['data']
        map = dataset['groundtruth']
    elif name == 'HC':
        dataset = sio.loadmat('./data/coast.mat')# [100 100 198] 128
        data = dataset['data']
        map = dataset['map']
    else:
        print("NO DATASET")
        exit()

    return data, map

#输入模型输出结果 模型输入数据data map
def accuracy(output, input, map, draw = False):
    """Computes the precision@k for the specified values of k"""
    row, col = map.shape[0], map.shape[1]
    res = input - output
    anomaly_degree = torch.mean(torch.square(res), axis=1)
    anomaly_map = torch.reshape(anomaly_degree, (row, col))
    anomaly_map = anomaly_map.data.cpu().numpy()

    if draw != False:
        plt.figure()
        plt.axis('off')
        plt.imshow(anomaly_map, cmap='coolwarm')
        plt.savefig("Net3HY.pdf", format='pdf', pad_inches=0.0, bbox_inches='tight', dpi=1200)

    PD, PF, AUC = ROC_AUC(anomaly_map, map, draw)  # coding=utf-8

    return anomaly_map, PD, PF, AUC

#输入异常图，map
def ROC_AUC(anomaly_map, map, draw=False):
    row, col = anomaly_map.shape
    predict = np.reshape(anomaly_map, (row * col))
    predict = (predict - np.amin(predict)) / (np.amax(predict) - np.amin(predict))

    map = np.reshape(map, (row * col))

    fpr, tpr, threshold = roc_curve(map, predict)
    roc_auc = auc(fpr, tpr)
    # print('roc=',roc_auc)

    if draw != False:
        # draw ROC
        lw = 2
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()

    return tpr, fpr, roc_auc

class ZScoreNorm:
    def __init__(self):
        self.means = None
        self.stds = None
    def fit(self, x):
        self.means = np.mean(x, axis=(0, 1))
        self.stds = np.std(x, axis=(0, 1))
        return self
    def transform(self, x):
        x_norm = np.zeros_like(x)
        for i in range(x.shape[2]):
            x_norm[:, :, i] = (x[:, :, i] - self.means[i]) / self.stds[i]
        return x_norm
    def inverse_transform(self, x_norm):
        x = np.zeros_like(x_norm)
        for i in range(x_norm.shape[2]):
            x[:, :, i] = x_norm[:, :, i] * self.stds[i] + self.means[i]
        return x
