import argparse

import numpy as np

import auxil
from hyper_pytorch import *
import torch
import torch.nn.parallel
import torch.nn.functional as F
from torchvision.transforms import *
import random
import os
import time
import scipy.io as sio
from bands_select import OPBS
from model import MWANet
torch.autograd.set_detect_anomaly(True)

seed=100
random.seed(seed)  
np.random.seed(seed)  
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed)  
torch.backends.cudnn.benchmark = False#
torch.backends.cudnn.deterministic = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.use_deterministic_algorithms(True)
torch.use_deterministic_algorithms(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#data_norm=True
#num_bs=128
#导入数据
def load_hyper(args):#处理数据
    data, map = auxil.loadData(args.dataset, num_components=args.components)#
    row, col, band = data.shape#100,100,191  band为通道数
    band_idx = OPBS(data, args.BS)
    data_bs = data[:, :, band_idx]
    data_bs = (data_bs - np.min(data_bs)) / (np.max(data_bs) - np.min(data_bs))
    if  (args.dataset=='HP' or args.dataset=='HB') and args.data_norm:#(args.dataset=='HP' or args.dataset=='HB') and
        data_bs = auxil.ZScoreNorm().fit(data_bs).transform(data_bs)
    x_train = data_bs[np.newaxis, :]
    # map = torch.tensor((map[np.newaxis,np.newaxis,:,:]))
    # map = F.interpolate(map, size=(128, 128), mode='bilinear', align_corners=False).squeeze().numpy()
    train_loader = torch.tensor(np.transpose(x_train, (0, 3, 1, 2)).astype("float32"))
    # train_loader = F.interpolate(train_loader, size=(128, 128), mode='bilinear', align_corners=False)
    train_loader = HyperData(train_loader)
    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=args.tr_bsize, shuffle=False)
    return train_loader, band, map
def train(trainloader, model, criterion1, optimizer, scheduler, use_cuda,draw=False):
    model.train()
    losses = np.zeros((len(trainloader)))  # 存储每个batch的loss

    for batch_idx, (input) in enumerate(trainloader):
        if use_cuda:
            input = input.cuda()
            input = input.float()
        # 前向传播，获取模型输出
        outputs = model(input,draw)
        # 计算简单的损失，不加自适应权重
        loss = criterion1(input, outputs)
        # 记录当前batch的loss
        losses[batch_idx] = loss.item()
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 更新学习率
        scheduler.step()
    # 返回平均损失
    return np.average(losses)
def test(test_loader, map, model, criterion, use_cuda, draw = False):
    model.eval()
    accs   = np.zeros((len(test_loader)))
    losses = np.zeros((len(test_loader)))
    for batch_idx, (input) in enumerate(test_loader):
        if use_cuda:
            input = input.cuda()
            input = input.float()

        outputs= model(input,draw)
        loss= criterion(outputs, input)
        losses[batch_idx] = loss.item()
        anomaly_map, PD, PF, accs[batch_idx] = auxil.accuracy(outputs.data, input.data, map, draw)
        test_result = {
                        'anomaly_map': anomaly_map,
                        'PD': PD,
                        'PF': PF,
                        'AUC' : np.average(accs)
                    }
        return (np.average(losses), test_result)
def main():
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument('--epochs', default=110, type=int, help='number of total epochs to run')  
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='initial learning rate')  
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,help='weight decay (default: 1e-4)')  
    parser.add_argument('--tr_bsize', default=1, type=int, help='mini-batch train size (default: 100)')  
    parser.add_argument('--components', default=False, type=int, help='dimensionality reduction') 
    parser.add_argument('--thre', default=1e-6, type=float, help='the threshold of stopping training')  
    parser.add_argument('--dataset', default='HL', type=str, help='dataset (options:HY HU HUI)')  
    parser.add_argument('--BS', default=64, type=int, help='num_bs')
    parser.add_argument('--data_norm', default=True, type=int, help='data_norm')
    args = parser.parse_args()
    train_loader, band, map = load_hyper(args)
    use_cuda = torch.cuda.is_available()
    if use_cuda: torch.backends.cudnn.benchmark = False
    model = MWANet.UMAD()
    model = model.float()

    model_dict = model.state_dict()
    model.load_state_dict(model_dict)

    if use_cuda: model = model.cuda()
    optimizer = torch.optim.AdamW \
         (model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)

    criterion1 = torch.nn.L1Loss()
    loss_np = np.zeros((1, 30), dtype=np.float32)
    loss_last = 100
    mean_loss = 1
    best_acc = 0
    loss_value = np.zeros(args.epochs)
    acc_value = np.zeros(args.epochs)
    time_train = 0
    time_test = 0
    for epoch in range(args.epochs):
        start0 = time.perf_counter()
        train_loss = train(train_loader, model, criterion1, optimizer, scheduler, use_cuda,draw=False)
        end0 = time.perf_counter()
        time_UMAD0 = end0 - start0
        time_train += time_UMAD0
        # print(time_train)
        test_loss, test_result = test(train_loader, map, model, criterion1, use_cuda)
        start1 = time.perf_counter()
        test_acc = test_result['AUC']
        if epoch % 1 == 0:
            print("EPOCH", epoch, "TRAIN LOSS", train_loss, end=',')
            print("TEST LOSS", test_loss, "ACCURACY", test_acc)

        loss_value[epoch] = train_loss
        acc_value[epoch] = test_acc

        # save model whose result is the best in the epochs
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch + 1,
                'loss': train_loss,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, "best_model.pth.tar")

            if epoch >= 1:
                index = epoch - int(epoch / 30) * 30
                loss_np[0][index - 1] = abs(train_loss - loss_last)
                if epoch % 10 == 0:
                    mean_loss = np.mean(loss_np)

        loss_last = train_loss

        if epoch == args.epochs - 1 or mean_loss < args.thre:
            checkpoint = torch.load("best_model.pth.tar")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            _, test_result = test(train_loader, map, model, criterion1, use_cuda, draw=args.data_norm)
            end1 = time.perf_counter()
            time_UMAD1 = end1 - start1
            time_test += time_UMAD1
            print("FINAL: LOSS", checkpoint['loss'], "ACCURACY", checkpoint['best_acc'])
            return test_result, time_train + time_test
if __name__ == '__main__':
    result,time_UMAD = main()
    sio.savemat('MWANet.mat', {'anomaly_map': result['anomaly_map']})
    print("AUC: ", result['AUC'], "Time: ",time_UMAD)
