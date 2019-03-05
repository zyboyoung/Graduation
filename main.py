import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
from torch.autograd import Variable
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
import argparse
import time
from functools import wraps
import matplotlib.pyplot as plt
import plot_evaluation

np.set_printoptions(threshold=10000)
if torch.cuda.is_available():
    gpu = True
    print('===> Using GPU')
else:
    gpu = False
    print('===> Using CPU')


# 计算程序运行时间的装饰器
def cal_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('-' * 8)
        print('start time: ', time.asctime(time.localtime(start)))
        print('end time:   ', time.asctime(time.localtime(end)))
        print('-' * 8)
        cost_time = end - start
        if cost_time < 1:
            print(func.__name__, '{:.5f}'.format(cost_time * 1000), 'ms')
        else:
            print(func.__name__, str(cost_time // 3600), 'h', str(cost_time % 3600 // 60), 'min',
                  '{:.2f}'.format(cost_time % 60), 's')
        print('-' * 8)
        return result
    return wrapper


# 对结构进行分类
def structure_encode(s):
    """
    f: free end, s: stem, h: hairpin, m: multi loops / internal loop, j: joint
    :param s: dot-bracket格式
    :return: 列表形式的结构，五种字母分别表示五种结构
    """
    start = 0
    result = []
    while start < len(s) and s[start] == '.':
        start += 1
        result.append('F')
    if start == len(s):
        return result
    else:
        s = s[start:]
        end_count = 0
        while s[-1] == '.':
            end_count += 1
            s = s[:-1]

        length = len(s)
        i = 0
        stack_count = 0
        while i < length:
            if s[i] == '(':
                stack_count += 1
                result.append('S')
                i += 1
            elif s[i] == ')':
                stack_count -= 1
                result.append('S')
                i += 1
            else:
                if s[i-1] == '(':
                    count = 1

                    while s[i+1] == '.':
                        count += 1
                        i += 1

                    if s[i+1] == ')':
                        for j in range(count):
                            result.append('H')
                        i += 1
                    else:
                        for j in range(count):
                            result.append('M')
                        i += 1
                else:
                    if stack_count == 0:
                        i += 1
                        result.append('J')
                    else:
                        i += 1
                        result.append('M')

        for j in range(end_count):
            result.append('F')

        return ''.join([i for i in result])


# 读取Graphprot序列，每个文件中包含大量数据，每条数据共三行，同时根据文件名可以判断标签
def read_seq(seq_file, label=1):
    """
    第一行表示注释，有蛋白质名称，训练集/测试集等等
    第二行表示一级结构，即序列信息
    第三行表示二级结构，即结构信息
    返回的是序列，结构，标签（0，1表示）
    :param seq_file: 文件名
    :param label:
    :return:
    """
    seq_list = []
    structure_list = []
    labels = []
    with open(seq_file, 'r') as f:
        for line in f:
            if line[0] == '>':
                continue
            elif line[0] == '.' or line[0] == '(':
                structure = line.split()[0]
                structure = structure_encode(s=structure)
                structure_list.append(structure)
            else:
                seq = line[:].split()[0].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)

    return seq_list, structure_list, labels


# 对于长度超过501bp的序列或结构，只选择其前501bp，对于不足501bp的序列或结构，使用'N'填充
def padding_s(s, max_len=501, key='N'):
    """
    :return: 长度为501p的序列或结构
    """
    s = str(s)
    s_len = len(s)
    if s_len < max_len:
        gap_len = max_len - s_len
        new_s = s + key * gap_len
    else:
        new_s = s[:max_len]
    return new_s


# 根据参数指定蛋白质种类、是否是训练集，读取graphprot文件所有数据
def load_data(protein, train=True, path='../Data/GraphProt_CLIP_ss/'):
    """
    :param protein: 指定蛋白质种类
    :param train: 指定是训练集/测试集
    :param path: 数据集所处目录
    :return: 字典，Key包括seq和Y，其中seq是指定蛋白质对应的正例和负例，Y是1和0
    """
    data = dict()
    file_list = os.listdir(path)

    key = '.train.'
    if not train:
        key = '.ls.'
    mix_label = []
    mix_seq = []
    mix_structure = []

    for file in file_list:
        if protein not in file:
            continue
        elif key in file:
            if 'positive' in file:
                label = 1
            else:
                label = 0
            seq_file = os.path.join(path, file)
            seqs, structure, labels = read_seq(seq_file=seq_file, label=label)
            mix_label = mix_label + labels
            mix_seq = mix_seq + seqs
            mix_structure = mix_structure + structure

    data['seq'] = mix_seq
    data['structure'] = mix_structure
    data['Y'] = np.array(mix_label)

    return data


# 按照DeepBind中的方法将原序列转化为(n+2m-2)*4的卷积array，加上了padding
def get_rna_seq_convolution_array(seq, motif_len=4):
    """
    :param seq: RNA序列
    :param motif_len: 设置的卷积核的长度，不是最后得到的motif长度
    :return: 卷积用的array格式 (507, 4)
    """
    seq = seq.upper().replace('T', 'U')
    if len(seq) != 501:
        seq = padding_s(s=seq, max_len=501)
    alpha = 'ACGU'
    row = len(seq) + 2 * motif_len - 2
    new_array = np.zeros((row, 4))

    for i in range(motif_len - 1):
        new_array[i] = np.array([0.25] * 4)
    for i in range(row - (motif_len - 1), row):
        new_array[i] = np.array([0.25] * 4)

    for i, val in enumerate(seq):
        i += motif_len - 1
        if val not in alpha:
            new_array[i] = np.array([0.25] * 4)
            continue

        index = alpha.index(val)
        new_array[i][index] = 1

    return new_array


# 按照处理序列同样的方法将原结构转化为(n+2m-2)*5的卷积array，加上了padding
def get_rna_structure_convolution_array(structure, motif_len=5):
    """
    五种结构: F: free end, S: stem, H: hairpin, M: multi loops / internal loop, J: joint
    :param structure: RNA二级结构
    :param motif_len: 设置的卷积核的长度
    :return: 卷积用的array格式 (509, 5)
    """
    if len(structure) != 501:
        structure = padding_s(s=structure, max_len=501)
    structure_info = structure_encode(structure)
    alpha = 'FSHMJ'
    row = len(structure_info) + 2 * motif_len - 2
    new_array = np.zeros((row, 5))

    for i in range(motif_len - 1):
        new_array[i] = np.array([0.2] * 5)
    for i in range(row - (motif_len - 1), row):
        new_array[i] = np.array([0.2] * 5)

    for i, value in enumerate(structure):
        i += motif_len - 1
        if value not in alpha:
            new_array[i] = np.array([0.2] * 5)
            continue

        index = alpha.index(value)
        new_array[i][index] = 1

    return new_array


# 将输入的data(即指定蛋白质对应的seq，structure和label)，转化为卷积array后，倒置，返回[seq_bags, structure_bags], labels
def get_bag_data(data) -> (list, np.ndarray):
    """
    :param data: 字典类型，由函数load_data返回
    :return: 经过处理编码的[seq_bags, structure_bags], labels
    """
    seq_bags = []
    structure_bags = []
    seqs = data['seq']
    structures = data['structure']
    labels = data['Y']
    for seq in seqs:
        fea = get_rna_seq_convolution_array(seq)
        seq_bags.append(np.array([fea.T]))

    for structure in structures:
        fea = get_rna_structure_convolution_array(structure)
        structure_bags.append(np.array([fea.T]))

    data_bags = [seq_bags, structure_bags]

    return data_bags, labels


# 对指定蛋白质数据集进行处理后，得到训练集数据和测试集数据
def get_all_data(protein, path) -> (list, np.ndarray, list, np.ndarray):
    """
    :param protein: 指定蛋白质种类
    :param path: 根据数据集，选择不同的生成策略
    :return:
        train_bags      :   [train_seq_bags, train_structure_bags]
        train_labels    :   np.ndarray
        test_bags       :   [test_seq_bags, test_structure_bags]
        test_labels     :   np.ndarray
    """
    train_data = load_data(protein=protein, train=True, path=path)
    test_data = load_data(protein=protein, train=False, path=path)

    train_bags, train_labels = get_bag_data(data=train_data)
    test_bags, test_labels = get_bag_data(data=test_data)

    return train_bags, train_labels, test_bags, test_labels


# 构建CNN模型
class CNN(nn.Module):
    def __init__(self, nb_filter, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), window_size=507,
                 hidden_size=200, stride=(1, 1), padding=0, drop=True):
        super(CNN, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=nb_filter, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=nb_filter),
            nn.ReLU()
        )

        self.pool_1 = nn.MaxPool2d(kernel_size=pool_size, stride=stride)
        out_1_size = (window_size + 2 * padding - (kernel_size[1] - 1) - 1) // stride[1] + 1
        max_pool_size = (out_1_size + 2 * padding - (pool_size[1] - 1) - 1) // stride[1] + 1

        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=nb_filter, out_channels=nb_filter, kernel_size=(1, 10), stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size, stride=stride)
        )
        out_2_size = (max_pool_size + 2 * padding - (kernel_size[1] - 1) - 1) // stride[1] + 1
        max_pool_2_size = (out_2_size + 2 * padding - (pool_size[1] - 1) - 1) // stride[1] + 1

        self.drop = drop
        self.drop_1 = nn.Dropout(p=0.25)
        self.fc_1 = nn.Linear(in_features=max_pool_2_size * nb_filter, out_features=hidden_size)
        self.drop_2 = nn.Dropout(p=0.25)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.pool_1(out)
        out = self.layer_2(out)
        out = out.view(out.size(0), -1)
        if self.drop:
            out = self.drop_1(out)
        out = self.fc_1(out)
        if self.drop:
            out = self.drop_2(out)
        out = self.relu_1(out)
        out = self.fc_2(out)
        out = torch.sigmoid(out)
        return out

    def layer_1_out(self, x):
        x = np.array(x)
        x = Variable(torch.from_numpy(x.astype(np.float32)))
        if gpu:
            x = x.cuda()
        out = self.layer_1(x)
        temp = out.data.cpu().numpy()
        return temp

    def predict_prob(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x)
        if gpu:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]


# 构建包含序列和结构信息的CNN模型
class CNN_ss(nn.Module):
    def __init__(self, nb_filter, num_classes=2, seq_kernel_size=(4, 10), st_kernel_size=(5, 10), pool_size=(1, 3),
                 seq_window_size=507, st_window_size=509, hidden_size=200, stride=(1, 1), padding=0,
                 drop=True):
        super(CNN_ss, self).__init__()

        self.layer_seq_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=nb_filter, kernel_size=seq_kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=nb_filter),
            nn.ReLU()
        )

        self.layer_structure_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=nb_filter, kernel_size=st_kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=nb_filter),
            nn.ReLU()
        )

        self.pool_1 = nn.MaxPool2d(kernel_size=pool_size, stride=stride)
        self.hidden_size = hidden_size
        self.drop = drop

        out_seq_1_size = (seq_window_size + 2 * padding - (seq_kernel_size[1] - 1) - 1) // stride[1] + 1
        max_pool_seq_size = (out_seq_1_size + 2 * padding - (pool_size[1] - 1) - 1) // stride[1] + 1

        out_st_1_size = (st_window_size + 2 * padding - (st_kernel_size[1] - 1) - 1) // stride[1] + 1
        max_pool_st_size = (out_st_1_size + 2 * padding - (pool_size[1] - 1) - 1) // stride[1] + 1

        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=nb_filter, out_channels=nb_filter, kernel_size=(1, 10), stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size, stride=stride)
        )

        out_seq_2_size = (max_pool_seq_size + 2 * padding - (10 - 1) - 1) // stride[1] + 1
        max_pool_seq_2_size = (out_seq_2_size + 2 * padding - (pool_size[1] - 1) - 1) // stride[1] + 1

        out_st_2_size = (max_pool_st_size + 2 * padding - (10 - 1) - 1) // stride[1] + 1
        max_pool_st_2_size = (out_st_2_size + 2 * padding - (pool_size[1] - 1) - 1) // stride[1] + 1

        self.drop_1 = nn.Dropout(p=0.25)
        self.fc_1 = nn.Linear(in_features=(max_pool_seq_2_size + max_pool_st_2_size) * nb_filter,
                              out_features=hidden_size)
        self.drop_2 = nn.Dropout(p=0.25)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
        seq_x = x[0]
        structure_x = x[1]

        seq_out = self.layer_seq_1(seq_x)
        seq_out = self.pool_1(seq_out)
        seq_out = self.layer_2(seq_out)
        seq_out = seq_out.view(seq_out.size(0), -1)

        structure_out = self.layer_structure_1(structure_x)
        structure_out = self.pool_1(structure_out)
        structure_out = self.layer_2(structure_out)
        structure_out = structure_out.view(structure_out.size(0), -1)

        if self.drop:
            seq_out = self.drop_1(seq_out)
            structure_out = self.drop_1(structure_out)

        out = torch.cat([seq_out, structure_out], -1)
        out = self.fc_1(out)

        if self.drop:
            out = self.drop_2(out)

        out = self.relu_1(out)
        out = self.fc_2(out)
        out = torch.sigmoid(out)
        return out

    def layer_seq_out(self, x):
        x = np.array(x[0])
        x = Variable(torch.from_numpy(x.astype(np.float32)))
        if gpu:
            x = x.cuda()
        out = self.layer_seq(x)
        temp = out.data.cpu().numpy()
        return temp

    def layer_structure_out(self, x):
        x = np.array(x[1])
        x = Variable(torch.from_numpy(x.astype(np.float32)))
        if gpu:
            x = x.cuda()
        out = self.layer_structure(x)
        temp = out.data.cpu().numpy()
        return temp

    def predict_prob(self, x):
        x1 = np.array(x[0])
        x2 = np.array(x[1])
        x1_v = Variable(torch.from_numpy(x1.astype(np.float32)))
        x2_v = Variable(torch.from_numpy(x2.astype(np.float32)))
        if gpu:
            x1_v = x1_v.cuda()
            x2_v = x2_v.cuda()
        x_v = [x1_v, x2_v]
        y = self.forward(x_v)
        temp = y.data.cpu().numpy()
        return temp[:, 1]


# 构建CNN_BLSTM模型
class CNN_BLSTM(nn.Module):
    def __init__(self, nb_filter, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), window_size=507,
                 hidden_size=200, stride=(1, 1), padding=0, num_layers=2, drop=True):
        super(CNN_BLSTM, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=nb_filter, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=nb_filter),
            nn.ReLU()
        )

        self.pool_1 = nn.MaxPool2d(kernel_size=pool_size, stride=stride)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        out_1_size = (window_size + 2 * padding - (kernel_size[1] - 1) - 1) // stride[1] + 1
        max_pool_size = (out_1_size + 2 * padding - (pool_size[1] - 1) - 1) // stride[1] + 1

        self.down_sample = nn.Conv2d(in_channels=nb_filter, out_channels=1, kernel_size=(1, 10),
                                     stride=stride, padding=padding)
        input_size = (max_pool_size + 2 * padding - (kernel_size[1] - 1) - 1) // stride[1] + 1

        self.layer_2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               batch_first=True, bidirectional=True)
        self.drop_1 = nn.Dropout(p=0.25)
        self.fc_1 = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size)
        self.drop_2 = nn.Dropout(p=0.25)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.drop = drop

    def forward(self, x):
        out = self.layer_1(x)
        out = self.pool_1(out)
        out = self.down_sample(out)
        out = torch.squeeze(out, 1)
        if gpu:
            h0 = Variable(torch.zeros(self.num_layers * 2, out.size(0), self.hidden_size)).cuda()
            c0 = Variable(torch.zeros(self.num_layers * 2, out.size(0), self.hidden_size)).cuda()
        else:
            h0 = Variable(torch.zeros(self.num_layers * 2, out.size(0), self.hidden_size))
            c0 = Variable(torch.zeros(self.num_layers * 2, out.size(0), self.hidden_size))
        out, _ = self.layer_2(out, (h0, c0))
        out = out[:, -1, :]
        if self.drop:
            out = self.drop_1(out)
        out = self.fc_1(out)
        if self.drop:
            out = self.drop_2(out)
        out = self.relu_1(out)
        out = self.fc_2(out)
        out = torch.sigmoid(out)
        return out

    def layer_1_out(self, x):
        x = np.array(x)
        x = Variable(torch.from_numpy(x.astype(np.float32)))
        if gpu:
            x = x.cuda()
        out = self.layer_1(x)
        temp = out.data.cpu().numpy()
        return temp

    def predict_prob(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x)
        if gpu:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        del x, y
        return temp[:, 1]


# 构建包含序列和结构信息的CNN_BLSTM模型
class CNN_BLSTM_ss(nn.Module):
    def __init__(self, nb_filter, num_classes=2, seq_kernel_size=(4, 10), st_kernel_size=(5, 10), pool_size=(1, 3),
                 seq_window_size=507, st_window_size=509, hidden_size=200, stride=(1, 1), padding=0, num_layers=2,
                 drop=True):
        super(CNN_BLSTM_ss, self).__init__()

        self.layer_seq = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=nb_filter, kernel_size=seq_kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=nb_filter),
            nn.ReLU()
        )

        self.layer_structure = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=nb_filter, kernel_size=st_kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=nb_filter),
            nn.ReLU()
        )

        self.pool_1 = nn.MaxPool2d(kernel_size=pool_size, stride=stride)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.drop = drop

        out_seq_size = (seq_window_size + 2 * padding - (seq_kernel_size[1] - 1) - 1) // stride[1] + 1
        max_pool_seq_size = (out_seq_size + 2 * padding - (pool_size[1] - 1) - 1) // stride[1] + 1
        out_st_size = (st_window_size + 2 * padding - (st_kernel_size[1] - 1) - 1) // stride[1] + 1
        max_pool_st_size = (out_st_size + 2 * padding - (pool_size[1] - 1) - 1) // stride[1] + 1

        self.down_sample = nn.Conv2d(in_channels=nb_filter, out_channels=1, kernel_size=(1, 10), stride=stride,
                                     padding=padding)

        input_seq_size = (max_pool_seq_size + 2 * padding - (seq_kernel_size[1] - 1) - 1) // stride[1] + 1
        input_st_size = (max_pool_st_size + 2 * padding - (st_kernel_size[1] - 1) - 1) // stride[1] + 1

        self.layer_LSTM = nn.LSTM(input_size=input_seq_size + input_st_size, hidden_size=hidden_size,
                                  num_layers=num_layers, batch_first=True, bidirectional=True)

        self.drop_1 = nn.Dropout(p=0.25)
        self.fc_1 = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size)
        self.drop_2 = nn.Dropout(p=0.25)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
        seq_x = x[0]
        structure_x = x[1]

        seq_out = self.layer_seq(seq_x)
        seq_out = self.pool_1(seq_out)
        seq_out = self.down_sample(seq_out)
        seq_out = torch.squeeze(seq_out, 1)

        structure_out = self.layer_structure(structure_x)
        structure_out = self.pool_1(structure_out)
        structure_out = self.down_sample(structure_out)
        structure_out = torch.squeeze(structure_out, 1)

        h0 = Variable(torch.zeros(self.num_layers * 2, seq_out.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers * 2, seq_out.size(0), self.hidden_size))
        if gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()

        out = torch.cat([seq_out, structure_out], -1)
        out, _ = self.layer_LSTM(out, (h0, c0))
        out = out[:, -1, :]
        if self.drop:
            out = self.drop_1(out)
        out = self.fc_1(out)
        if self.drop:
            out = self.drop_2(out)
        out = self.relu_1(out)
        out = self.fc_2(out)
        out = torch.sigmoid(out)
        return out

    def layer_seq_out(self, x):
        x = np.array(x[0])
        x = Variable(torch.from_numpy(x.astype(np.float32)))
        if gpu:
            x = x.cuda()
        out = self.layer_seq(x)
        temp = out.data.cpu().numpy()
        return temp

    def layer_structure_out(self, x):
        x = np.array(x[1])
        x = Variable(torch.from_numpy(x.astype(np.float32)))
        if gpu:
            x = x.cuda()
        out = self.layer_structure(x)
        temp = out.data.cpu().numpy()
        return temp

    def predict_prob(self, x):
        x1 = np.array(x[0])
        x2 = np.array(x[1])
        x1_v = Variable(torch.from_numpy(x1.astype(np.float32)))
        x2_v = Variable(torch.from_numpy(x2.astype(np.float32)))
        if gpu:
            x1_v = x1_v.cuda()
            x2_v = x2_v.cuda()
        x_v = [x1_v, x2_v]
        y = self.forward(x_v)
        temp = y.data.cpu().numpy()
        return temp[:, 1]


# 重写Dataset用于多输入
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, data_1, data_2, labels):
        self.data_1 = data_1
        self.data_2 = data_2
        self.labels = labels

    def __getitem__(self, index):
        input_1, input_2, target = self.data_1[index], self.data_2[index], self.labels[index]
        return [input_1, input_2], target

    def __len__(self):
        return len(self.data_1)


# 配置参数：指定优化器optimizer、损失函数loss_f...
class ModelCompile(object):
    def __init__(self, model, optimizer, loss):
        self.model = model
        self.optimizer = optimizer
        self.loss_f = loss

    def fit_one(self, train_loader):
        total_loss = []
        for idx, (x, y) in enumerate(train_loader):
            if len(x) != 2:
                x_v = Variable(x)
                y_v = Variable(y)
                if gpu:
                    x_v = x_v.cuda()
                    y_v = y_v.cuda()

                self.optimizer.zero_grad()
                y_pre = self.model(x_v)
                loss = self.loss_f(y_pre, y_v)
                loss.backward()
                self.optimizer.step()
                total_loss.append(loss.item())
                del y_pre
                del loss
            else:
                x1 = x[0]
                x2 = x[1]
                x1_v = Variable(x1)
                x2_v = Variable(x2)
                y_v = Variable(y)
                if gpu:
                    x1_v = x1_v.cuda()
                    x2_v = x2_v.cuda()
                    y_v = y_v.cuda()
                x_v = [x1_v, x2_v]
                self.optimizer.zero_grad()
                y_pre = self.model(x_v)
                loss = self.loss_f(y_pre, y_v)
                loss.backward()
                self.optimizer.step()
                total_loss.append(loss.item())
                del y_pre
                del loss
        return sum(total_loss) / len(total_loss)

    def fit(self, x, y, batch_size=32, nb_epoch=10, plot=False, protein=None, total_loss=None, count=0):
        if len(x) == 2:
            x[0] = np.array(x[0])
            x[1] = np.array(x[1])
            print('seq.shape: ', x[0].shape)
            print('structure.shape: ', x[1].shape)
            train_set = MyDataSet(torch.from_numpy(x[0].astype(np.float32)), torch.from_numpy(x[1].astype(np.float32)),
                                  torch.from_numpy(y.astype(np.float32)).long().view(-1))
        else:
            print('X.shape: ', x.shape)
            train_set = TensorDataset(torch.from_numpy(x.astype(np.float32)),
                                      torch.from_numpy(y.astype(np.float32)).long().view(-1))

        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

        self.model.train()
        plot_loss = []
        for t in range(nb_epoch):
            loss = self.fit_one(train_loader)
            print('第%d次迭代的loss为:' % (t + 1), loss)
            if plot:
                total_loss[count].append(loss)
                plot_loss.append(loss)

        if plot:
            x = list(range(nb_epoch))

            plt.figure()
            plt.title('Model Loss for %s' % protein)
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.plot(x, plot_loss, color='blue')
            plt.savefig('epochs/CNN_s/' + protein + '_train.png')


# 多种评价函数
def evaluation(y_true, y_pred, protein):
    """
    :param y_true: array类型，真实标签
    :param y_pred: array类型，预测值
    :param protein: 选择的数据集
    :return:
    """
    y_pred_int = [1 if i > 0.5 else 0 for i in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred_int).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 / (1 / precision + 1 / recall)

    print()
    print('%s的混淆矩阵如下：' % protein)
    print(confusion_matrix(y_true=y_true, y_pred=y_pred_int))
    print('Precision:', precision)
    print('Recall:', recall)
    print('F-score:', f1)
    print()

    cnf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred_int)
    np.set_printoptions(precision=2)

    plt.figure()
    plot_evaluation.plot_confusion_matrix(cnf_matrix, classes=[0, 1], protein=protein, normalize=True)
    plt.show()

    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred)
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    plot_evaluation.plot_roc(fpr=fpr, tpr=tpr, label='AUC=%0.4f' % auc, title='ROC of ', protein=protein)


# 指定网络模型进行训练，并且将训练好的模型参数进行保存，同时可以根据需要提取出对应数据集的motif
def train_network(model_type, x_train, y_train, model_file='model.pkl', batch_size=128, n_epochs=50, num_filters=16,
                  info=1, motif=False, motif_seqs=None, motif_out_dir='Motifs', plot=False):
    """
    :param model_type: 网络模型
    :param x_train: 训练集的输入
    :param y_train: 训练集的标签
    :param model_file: 用于存储模型参数的文件
    :param batch_size: 每次训练的样本个数
    :param n_epochs: 迭代次数
    :param num_filters: 卷积核个数参数
    :param info: 表明使用的信息种类数量，1表示只使用序列信息，2表示同时使用序列和结构信息
    :param motif: 指定是否需要提取motif
    :param motif_seqs:
    :param motif_out_dir:
    :param plot:
    :return: 保存训练好的模型参数
    """
    print('===>')
    if info == 1:
        comment = 'via Sequence'
    else:
        comment = 'via Sequence & Structure'
    print('model training for', model_type, comment)

    if model_type == 'CNN' and info == 1:
        model = CNN(nb_filter=num_filters, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), window_size=507,
                    hidden_size=200, stride=(1, 1), padding=0, drop=True)
    elif model_type == 'CNN' and info == 2:
        model = CNN_ss(nb_filter=num_filters, num_classes=2, seq_kernel_size=(4, 10), st_kernel_size=(5, 10),
                       pool_size=(1, 3), seq_window_size=507, st_window_size=509, hidden_size=200, stride=(1, 1),
                       padding=0, drop=True)
    elif model_type == 'CNN_BLSTM' and info == 1:
        model = CNN_BLSTM(nb_filter=num_filters, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), window_size=507,
                          hidden_size=200, stride=(1, 1), padding=0, num_layers=2, drop=True)
    else:
        model = CNN_BLSTM_ss(nb_filter=num_filters, num_classes=2, seq_kernel_size=(4, 10), st_kernel_size=(5, 10),
                             pool_size=(1, 3), seq_window_size=507, st_window_size=509, hidden_size=200, stride=(1, 1),
                             padding=0, drop=True)

    if gpu:
        model = model.cuda()

    # print(model)

    clf = ModelCompile(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001),
                       loss=nn.CrossEntropyLoss())
    clf.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epochs, plot=plot)

    if motif:
        print(motif_seqs, motif_out_dir)
        print('暂未完成')

    torch.save(model.state_dict(), model_file)


# 指定网络模型，直接提取对应的训练好的模型参数，对给定的序列（数据集）完成预测，返回预测值
def predict_network(model_type, x_test, model_file='model.pkl', num_filters=16, info=1, drop=True):
    """
    :param model_type: 网络模型
    :param x_test: 待预测的序列（数据集）
    :param model_file: 已经训练好的模型参数
    :param num_filters: 卷积核个数参数
    :param num_filters: 卷积核个数参数
    :param info: 表明使用的信息种类数量，1表示只使用序列信息，2表示同时使用序列和结构信息
    :param drop:
    :return: 预测值
    """
    print('===>')
    if info == 1:
        comment = 'via Sequence'
    else:
        comment = 'via Sequence & Structure'
    print('Predicting for', model_type, comment)

    if model_type == 'CNN' and info == 1:
        model = CNN(nb_filter=num_filters, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), window_size=507,
                    hidden_size=200, stride=(1, 1), padding=0, drop=drop)
    elif model_type == 'CNN' and info == 2:
        model = CNN_ss(nb_filter=num_filters, num_classes=2, seq_kernel_size=(4, 10), st_kernel_size=(5, 10),
                       pool_size=(1, 3), seq_window_size=507, st_window_size=509, hidden_size=200, stride=(1, 1),
                       padding=0, drop=True)
    elif model_type == 'CNN_BLSTM' and info == 1:
        model = CNN_BLSTM(nb_filter=num_filters, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), window_size=507,
                          hidden_size=200, stride=(1, 1), padding=0, num_layers=2, drop=drop)
    else:
        model = CNN_BLSTM_ss(nb_filter=num_filters, num_classes=2, seq_kernel_size=(4, 10), st_kernel_size=(5, 10),
                             pool_size=(1, 3), seq_window_size=507, st_window_size=509, hidden_size=200, stride=(1, 1),
                             padding=0, drop=True)
    if gpu:
        model = model.cuda()

    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(model_file))
    pre = model.predict_prob(x_test)

    return pre


# 使用graphprot数据集作为训练集和验证集运行模型
@cal_time
def run_deeps(model_type='CNN', data_dir='../Data/GraphProt_CLIP_sequences', info=1, batch_size=128, n_epochs=50,
              num_filters=16, motif=False, motif_seqs=None, motif_out_dir='Motifs'):
    """
    :param model_type: 网络模型
    :param data_dir: 指定数据集
    :param info: 表明使用的信息种类数量，1表示只使用序列信息，2表示同时使用序列和结构信息
    :param batch_size: 每次训练的样本个数
    :param n_epochs: 迭代次数
    :param num_filters: 卷积核个数参数
    :param motif: 指定是否需要提取motif
    :param motif_seqs:
    :param motif_out_dir:
    :return: 将验证集上的预测结果写入文件中
    """
    protein_set = set()
    start_time = time.time()

    if info == 1:
        output_file = '../Results/' + 'adam_' + model_type + '_s'
    else:
        output_file = '../Results/' + 'adam_' + model_type + '_ss'

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, 'w') as f:
        total_proteins = 24
        count = 1

        for protein in os.listdir(data_dir):
            protein = protein.split('.')[0]
            if protein in protein_set:
                continue

            protein_set.add(protein)
            print(protein, '\t', str(count) + '/' + str(total_proteins))
            count += 1
            f.write(protein + '\t')

            train_bags, train_labels, test_bags, test_labels = get_all_data(protein, path=data_dir)

            if info == 1:
                train_bags = train_bags[0]
                test_bags = test_bags[0]

                if model_type == 'CNN':
                    model_dir = '../Results/CNN_s_pkl/'
                else:
                    model_dir = '../Results/CNN_BLSTM_s_pkl/'

            else:
                if model_type == 'CNN':
                    model_dir = '../Results/CNN_ss_pkl/'
                else:
                    model_dir = '../Results/CNN_BLSTM_ss_pkl/'

            model_file = model_dir + protein + '_model.pkl'

            if not os.path.exists(model_file):
                if info == 1:
                    train_network(model_type=model_type, x_train=np.array(train_bags), y_train=np.array(train_labels),
                                  model_file=model_file, batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, info=info, motif=motif, motif_seqs=motif_seqs,
                                  motif_out_dir=motif_out_dir)
                else:
                    train_network(model_type=model_type, x_train=train_bags, y_train=train_labels,
                                  model_file=model_file, batch_size=batch_size, n_epochs=n_epochs,
                                  num_filters=num_filters, info=info, motif=motif, motif_seqs=motif_seqs,
                                  motif_out_dir=motif_out_dir)

            if info == 1:
                predict = predict_network(model_type=model_type, x_test=np.array(test_bags), model_file=model_file,
                                          num_filters=num_filters, info=info, drop=True)
            else:
                predict = predict_network(model_type=model_type, x_test=test_bags, model_file=model_file,
                                          num_filters=num_filters, info=info, drop=True)

            auc = roc_auc_score(test_labels, predict)
            del predict
            print('AUC: ', auc)
            f.write(str(auc) + '\n')

    end_time = time.time()
    print('Training final took: %.2f s' % (end_time - start_time))


# 用于对已经训练好的蛋白质的数据集进行验证，方便测试各种评价指标
@cal_time
def validation(protein, model_type='CNN', info=1, drop=False):
    if info == 1:
        if model_type == 'CNN':
            model_dir = '../Results/CNN_s_pkl/'
        else:
            model_dir = '../Results/CNN_BLSTM_s_pkl/'

    else:
        if model_type == 'CNN':
            model_dir = '../Results/CNN_ss_pkl/'
        else:
            model_dir = '../Results/CNN_BLSTM_ss_pkl/'

    model_file = model_dir + protein + '_model.pkl'
    if not os.path.exists(model_file):
        print('此类蛋白质的数据集还未训练！')
        return

    data = load_data(protein=protein, train=False)
    x, labels = get_bag_data(data=data)

    if info == 1:
        x = x[0]
        predict = predict_network(model_type=model_type, x_test=np.array(x), model_file=model_file, num_filters=16,
                                  info=info, drop=drop)
    else:
        predict = predict_network(model_type=model_type, x_test=x, model_file=model_file, num_filters=16,
                                  info=info, drop=drop)
    evaluation(y_true=labels, y_pred=predict, protein=protein)

    pre_int = [1 if i > 0.5 else 0 for i in predict]
    print(classification_report(labels, pre_int))


# 计算每一次完整数据迭代后的loss，并绘制loss-epochs图
def epochs(protein, train=True, model_type='CNN', batch_size=128, n_epochs=50, num_filters=16, info=1, total_loss=None,
           count=0):
    """
    :param protein: 指定的蛋白质
    :param train: 指定作图使用的是训练集还是测试集
    :param model_type: 指定使用的网络模型
    :param batch_size: 指定batch_size大小
    :param n_epochs: 指定一共迭代次数
    :param num_filters: 指定模型中所使用的卷积核数量
    :param info: 指定所使用的特征信息种类
    :param total_loss: 当需要绘制所有数据集的loss-epochs图时，用于存储loss
    :param count：
    :return: 绘制指定蛋白质的loss-epochs图，并保存
    """
    data = load_data(protein=protein, train=train)
    x_data, y_data = get_bag_data(data=data)

    print('===>')
    if info == 1:
        x_data = x_data[0]
        comment = 'via Sequence'
    else:
        comment = 'via Sequence & Structure'
    print('model training via', model_type, comment)

    if model_type == 'CNN' and info == 1:
        model = CNN(nb_filter=num_filters, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), window_size=507,
                    hidden_size=200, stride=(1, 1), padding=0, drop=True)
    elif model_type == 'CNN' and info == 2:
        model = CNN_ss(nb_filter=num_filters, num_classes=2, seq_kernel_size=(4, 10), st_kernel_size=(5, 10),
                       pool_size=(1, 3), seq_window_size=507, st_window_size=509, hidden_size=200, stride=(1, 1),
                       padding=0, drop=True)
    elif model_type == 'CNN_BLSTM' and info == 1:
        model = CNN_BLSTM(nb_filter=num_filters, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3),
                          window_size=507,
                          hidden_size=200, stride=(1, 1), padding=0, num_layers=2, drop=True)
    else:
        model = CNN_BLSTM_ss(nb_filter=num_filters, num_classes=2, seq_kernel_size=(4, 10), st_kernel_size=(5, 10),
                             pool_size=(1, 3), seq_window_size=507, st_window_size=509, hidden_size=200,
                             stride=(1, 1),
                             padding=0, drop=True)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    if gpu:
        model = model.cuda()

    clf = ModelCompile(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001),
                       loss=nn.CrossEntropyLoss())
    clf.fit(x_data, y_data, batch_size=batch_size, nb_epoch=n_epochs, plot=True, protein=protein,
            total_loss=total_loss, count=count)


# 绘制所有数据集的loss-epochs图
@cal_time
def plot_epochs(protein='all', train=True, model_type='CNN', n_epochs=100, info=1):
    """
    :param protein: 指定为all时，绘制所有数据的loss-epochs图
    :param train: 指定作图使用的是训练集还是测试集
    :param model_type: 指定使用的网络模型
    :param n_epochs: 指定一共迭代次数
    :param info: 指定所使用的特征信息种类
    :return: 绘制所有数据的loss-epochs图，并保存
    """
    if protein == 'all':
        train_total_loss = []
        for count, protein in enumerate(proteins):
            train_total_loss.append([])
            epochs(protein=protein, train=True, model_type=model_type, n_epochs=n_epochs, info=info,
                   total_loss=train_total_loss, count=count)
            print('====第%d个已经完成====' % count)
            torch.cuda.empty_cache()

        x = list(range(1, n_epochs + 1))
        plot_train_loss = [0] * n_epochs
        for i in range(n_epochs):
            for j in range(len(proteins)):
                plot_train_loss[i] += train_total_loss[j][i]

        plot_train_loss = [i / len(proteins) for i in plot_train_loss]

        plt.figure()
        plt.title('Model Total Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.plot(x, plot_train_loss, color='blue', label='train_data')
        plt.legend(loc='upper right')
        plt.savefig('epochs/CNN_s_pkl/' + 'total_loss_2.png')

        return plot_train_loss

    else:
        epochs(protein=protein, train=train, model_type=model_type, n_epochs=n_epochs, info=info)


# 用于存储所有数据集100次迭代中每次的模型参数结果，方便测试100次迭代中每次测试集的loss
def epochs_pkl():
    for protein in proteins:
        data = load_data(protein=protein, train=True)
        x_data, y_data = get_bag_data(data=data)
        x_data = x_data[0]
        model = CNN(nb_filter=16, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), window_size=507,
                    hidden_size=200, stride=(1, 1), padding=0, drop=True)
        if gpu:
            model = model.cuda()

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        loss_f = nn.CrossEntropyLoss()
        train_set = TensorDataset(torch.from_numpy(x_data.astype(np.float32)),
                                  torch.from_numpy(y_data.astype(np.float32)).long().view(-1))
        train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True)
        model.train()

        for epoch in range(100):
            total_loss = []
            for idx, (x, y) in enumerate(train_loader):
                x_v = Variable(x)
                y_v = Variable(y)
                if gpu:
                    x_v = x_v.cuda()
                    y_v = y_v.cuda()

                optimizer.zero_grad()
                y_pre = model(x_v)
                loss = loss_f(y_pre, y_v)
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())
                del y_pre
                del loss

            loss = sum(total_loss) / len(total_loss)

            print(protein + '第%d次迭代的loss为:' % (epoch + 1), loss)
            torch.save(model.state_dict(), '../Results/CNN_s_pkl/' + str(epoch+1) + '/' + protein + '_model.pkl')


# 绘制测试集在不同迭代次数下的ACC曲线
@cal_time
def plot_test_acc():
    total_tn = []
    total_fp = []
    total_fn = []
    total_tp = []
    for count, protein in enumerate(proteins):
        total_tn.append([])
        total_fp.append([])
        total_fn.append([])
        total_tp.append([])
        data = load_data(protein=protein, train=False)
        x_data, y_data = get_bag_data(data=data)
        x_data = np.array(x_data[0])
        for epoch in range(1, 101):
            model = CNN(nb_filter=16, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), window_size=507,
                        hidden_size=200, stride=(1, 1), padding=0, drop=False)
            model = model.cuda()
            model_file = '../Results/CNN_s_pkl/' + str(epoch) + '/' + protein + '_model.pkl'
            torch.cuda.empty_cache()
            model.load_state_dict(torch.load(model_file))
            y_pred = model.predict_prob(x_data)
            y_data = list(y_data)
            y_pred_int = [1 if i > 0.5 else 0 for i in y_pred]
            tn, fp, fn, tp = confusion_matrix(y_true=y_data, y_pred=y_pred_int).ravel()
            total_tn[count].append(tn)
            total_fp[count].append(fp)
            total_fn[count].append(fn)
            total_tp[count].append(tp)

    tn = [0] * 100
    fp = [0] * 100
    fn = [0] * 100
    tp = [0] * 100
    acc = []
    for i in range(100):
        for j in range(len(proteins)):
            tn[i] += total_tn[j][i]
            fp[i] += total_fp[j][i]
            fn[i] += total_fn[j][i]
            tp[i] += total_tp[j][i]
        acc.append((tp[i] + tn[i]) / (tp[i] + tn[i] + fn[i] + fp[i]))

    x = list(range(1, 101))
    plt.figure()
    plt.title('Model Total ACC')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(x, acc, color='blue', label='test_data')
    plt.legend(loc='lower right')
    plt.savefig('epochs/' + 'total_acc_test.png')


# 将train_loss和test_loss绘制在一个图中
@cal_time
def train_test_loss():
    test_total_loss = []
    for count, protein in enumerate(proteins):
        test_total_loss.append([])
        data = load_data(protein=protein, train=False)
        x_data, y_data = get_bag_data(data=data)
        x_data = x_data[0]
        test_set = TensorDataset(torch.from_numpy(np.array(x_data).astype(np.float32)),
                                 torch.from_numpy(y_data.astype(np.float32)).long().view(-1))
        test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False)
        model = CNN(nb_filter=16, num_classes=2, kernel_size=(4, 10), pool_size=(1, 3), window_size=507,
                    hidden_size=200, stride=(1, 1), padding=0, drop=False)
        model = model.cuda()
        loss = nn.CrossEntropyLoss()
        for epoch in range(1, 101):
            model_file = '../Results/CNN_s_pkl/' + str(epoch) + '/' + protein + '_model.pkl'
            torch.cuda.empty_cache()
            model.load_state_dict(torch.load(model_file))
            temp_loss_for_one_epoch = []
            for x, y in test_loader:
                x_v = Variable(x).cuda()
                y_v = Variable(y).cuda()
                y_pred = model(x_v)
                temp_loss_for_one_epoch.append(loss(y_pred, y_v).item())
            test_total_loss[count].append(sum(temp_loss_for_one_epoch) / len(temp_loss_for_one_epoch))
            print('%s测试集第%d次迭代已经完成' % (protein, epoch))
        print('========')
        print('%s测试集已经完成loss的计算' % protein)
        print('========')

    print('所有测试集的loss已经计算完成')
    test_loss = [0] * 100
    for i in range(100):
        for j in range(len(proteins)):
            test_loss[i] += test_total_loss[j][i]
    test_loss = [i / len(proteins) for i in test_loss]

    train_loss = plot_epochs(protein='all', train=True)
    x = list(range(1, 101))
    plt.figure()
    plt.title('Model Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(x, train_loss, color='red', label='train_loss')
    plt.plot(x, test_loss, color='blue', label='test_loss')
    plt.legend(loc='lower right')
    plt.savefig('epochs/' + 'train_test_loss.png')


if __name__ == '__main__':
    protein_list = []
    data_dir = '../Data/GraphProt_CLIP_sequences'
    for pro_id in os.listdir(data_dir):
        protein_list.append(pro_id.split('.')[0])
    proteins = list(set(protein_list))

    # pro = 'ICLIP_TDP43'
    # s1, train_y, s2, test_y = get_all_data(protein=pro, path='../Data/GraphProt_CLIP_ss/')
    # run_deeps(model_type='CNN_BLSTM', data_dir='../Data/GraphProt_CLIP_ss/', info=2)

    # for count, protein in enumerate(proteins):
    #     validation(protein=protein)
    #     print('====第%d个已经完成====: %s' % ((count + 1), protein))

    train_test_loss()
