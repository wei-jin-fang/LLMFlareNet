import os
import random
import sys

import pandas as pd
import  torch.backends.cudnn as cudnn
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, brier_score_loss
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return np.trunc(stepper * number) / stepper

def save_csv(y_pred, y_true,path):
    '''
    y_pred  ALL_PROBA.append([n_proba, c_proba, m_proba, x_proba])  save_csv(np.array(ALL_PROBA)
    y_true  test_y
    '''
    save_csv_dir = "Csv_pre_ture/"
    if not os.path.exists(save_csv_dir):
        os.makedirs(save_csv_dir)
    # 真实标签数据 - (160,)
    true_data = y_true.reshape(-1, 1)  # 转成 (160, 1) 的形状

    # 预测的概率数据 - (160, 2)
    pred_data = y_pred.reshape(-1, 2)  # 转成 (160, 2) 的形状

    # 合并真实标签和预测数据，形成 (160, 3) 的数据
    combined_data = np.hstack((true_data, pred_data))

    # 创建一个DataFrame，并添加表头
    df = pd.DataFrame(combined_data, columns=['y_true', 'zero', 'one'])

    # 保存为CSV文件，包含表头
    df.to_csv(os.path.join(save_csv_dir, "" + str(path) + ".csv",), header=True, index=False,encoding="utf-8-sig")

def compare_goes_class(class1, class2):
    """
    比较两个 GOES_Class 字段，返回较大的一个。
    比较规则：N < C < M < X，对于同一个字母，比较后面的数字部分，数字越大类别越大。
    """
    # 定义类别顺序
    order = {'A':-2,'B':-1,'N': 0, 'C': 1, 'M': 2, 'X': 3}

    # 获取字母和数字部分
    letter1, number1 = class1[0], float(class1[1:])
    letter2, number2 = class2[0], float(class2[1:])

    # 先按字母顺序比较
    if order[letter1] != order[letter2]:
        return class1 if order[letter1] > order[letter2] else class2
    else:
        # 字母相同时，比较数字部分
        return class1 if number1 > number2 else class2
def find_max_label_for_noaaARs_in_day(noaa_ars_dict):
    """
    在所有 NOAA_ARS 中找到整体最大类别的 GOES_Class 和对应的 NOAAID。
    :param noaa_ars_dict: 每个 NOAA_ARS 的最大 GOES_Class 和 NOAAID 字典
    :return: 整体最大 GOES_Class 和对应的 NOAAID
    """
    overall_max_goes_class = None
    overall_max_noaa_id = None

    for noaa_ars, (goes_class, noaa_id) in noaa_ars_dict.items():
        if overall_max_goes_class is None or compare_goes_class(goes_class, overall_max_goes_class) == goes_class:
            overall_max_goes_class = goes_class
            overall_max_noaa_id = noaa_id

    return overall_max_goes_class, overall_max_noaa_id


class Metric(object):
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

        self.__matrix = confusion_matrix(y_true, y_pred)

    def to_matrix(self, matrix=None):
        self.__matrix = np.asarray(matrix)
        return self

    def Matrix(self):
        return self.__matrix

    def TP(self):
        tp = np.diag(self.__matrix)
        return tp.astype(float)

    def TN(self):
        tn = self.__matrix.sum() - (self.FP() + self.FN() + self.TP())
        return tn.astype(float)

    def FP(self):
        fp = self.__matrix.sum(axis=0) - np.diag(self.__matrix)
        return fp.astype(float)

    def FN(self):
        fn = self.__matrix.sum(axis=1) - np.diag(self.__matrix)
        return fn.astype(float)

    def TPRate(self):
        return self.TP() / (self.TP() + self.FN() + sys.float_info.epsilon)

    def TNRate(self):
        return self.TN() / (self.TN() + self.FP() + sys.float_info.epsilon)

    def FPRate(self):
        return 1 - self.TNRate()

    def FNRate(self):
        return 1 - self.TPRate()

    def Accuracy(self):
        ALL = self.TP() + self.FP() + self.TN() + self.FN()
        RIGHT = self.TP() + self.TN()
        return RIGHT / (ALL + sys.float_info.epsilon)

    def Recall(self):
        return self.TP() / (self.TP() + self.FN() + sys.float_info.epsilon)

    def Precision(self):
        return self.TP() / (self.TP() + self.FP() + sys.float_info.epsilon)

    def TSS(self):
        return self.TPRate() - self.FPRate()

    def BSS(self):
        # print("########################")
        # print(self.y_true)
        # print("#########################")
        # print(self.y_pred)
        # print("########################")
        # print(type(self.y_true))

        BS = np.mean(np.square(self.y_true - self.y_pred), axis=0)
        # print(BS)
        # print("######################")
        y_ave = np.mean(self.y_true, axis=0)
        BSS = 1 - BS / np.mean(np.square(self.y_true - y_ave), axis=0)
        BSS = [BSS, BSS]
        return BSS

    def HSS(self):
        P = self.TP() + self.FN()
        N = self.TN() + self.FP()
        up = 2 * (self.TP() * self.TN() - self.FN() * self.FP())
        below = P * (self.FN() + self.TN()) + N * (self.TP() + self.FP())
        return up / (below + sys.float_info.epsilon)

    def FAR(self):  ###12/11新加
        return self.FP() / (self.FP() + self.TP() + sys.float_info.epsilon)

    def FPR(self):
        return self.FP() / (self.FP() + self.TN() + sys.float_info.epsilon)

def Rectify_binary(all_y, Class_NUM, TIME_STEPS):
    '''

      "开始修改y形状，目前只针对二分类
    '''

    all_y = all_y.reshape(-1, 1)  # 为了保证单独下标改成1
    # print(all_y.shape)(17800, 1)

    temp_y = []
    for i in range(0, all_y.shape[0], TIME_STEPS):
        temp_y.append(all_y[i][0])  # 两个一样的拿出一个就行
    time_y = np.array(temp_y)
    # print(time_y.shape)#(445,)

    return time_y

def Rectify_multiple (all_y,Class_NUM,TIME_STEPS):
    '''
        四分类使用这个
    '''
    print("all_y",all_y)
    all_y = all_y.reshape(-1, Class_NUM)#
    print("_y",all_y)
    temp_y = []
    for i in range(0, all_y.shape[0], TIME_STEPS):
        temp_y.append(all_y[i])
    time_y = np.array(temp_y)
    print(time_y.shape)
    # exit()
    return time_y


def get_batches(data, labels, batch_size):
    for i in range(0, len(data), batch_size):
        data_batch = data[i:i + batch_size]
        labels_batch = labels[i:i + batch_size]
        # 确保 data 和 labels 是 PyTorch 张量
        if not isinstance(data_batch, torch.Tensor):
            data_batch = torch.tensor(data_batch, dtype=torch.float32)
        if not isinstance(labels_batch, torch.Tensor):
            labels_batch = torch.tensor(labels_batch, dtype=torch.float32)
        yield data_batch, labels_batch
def get_batches_all(data, labels, batch_size):
    for i in range(0, len(data), batch_size):
        data_batch = data[i:i + batch_size]
        labels_batch = labels[i:i + batch_size]
        # 确保 data 和 labels 是 PyTorch 张量
        if not isinstance(data_batch, torch.Tensor):
            data_batch = torch.tensor(data_batch, dtype=torch.float32)
        if not isinstance(labels_batch, torch.Tensor):
            labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        yield data_batch, labels_batch

def get_batches_integer2(data, labels, batch_size):
    for i in range(0, len(data), batch_size):
        data_batch = data[i:i + batch_size]
        labels_batch = labels[i:i + batch_size]
        # 确保 data 和 labels 是 PyTorch 张量
        if not isinstance(data_batch, torch.Tensor):
            data_batch = torch.tensor(data_batch, dtype=torch.float32)
        if not isinstance(labels_batch, torch.Tensor):
            labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        yield data_batch, labels_batch
def get_batches_integer(data, labels, batch_size):
    data = np.array(data)  # 确保数据是 NumPy 数组格式
    labels = np.array(labels)

    for i in range(0, len(data), batch_size):
        data_batch = data[i:i + batch_size]
        labels_batch = labels[i:i + batch_size]

        # 如果最后一个 batch 的大小小于 batch_size，直接舍弃
        if len(data_batch) < batch_size:
            continue

        # 确保 data 和 labels 是 PyTorch 张量
        data_batch = torch.tensor(data_batch, dtype=torch.float32)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)

        yield data_batch, labels_batch
def getClass(everyclassnum_list):
    # Class1: ['N', 'C']
    # ____weight1: 73800
    # Class2: ['M', 'X']
    # ____weight2: 9360

    # [73800,9360]

    '''''''''
    everyclassnum_list: 每个元素在类中的个数

    返回的对应元素的权重
    '''''''''
    # all_samples: 总共的所有的样本数量 = everyclassnum_list中元素的求和
    all_samples = 0
    num_classes = len(everyclassnum_list)
    for i in everyclassnum_list:
        all_samples += i
    # 计算权重公式
    weight_list = []
    for i in everyclassnum_list:
        weight_list.append(all_samples / (i * num_classes))
    return weight_list

def shuffle_data(x, y):
    # 获取数据的长度
    indices = np.arange(x.shape[0])

    # 打乱索引
    np.random.shuffle(indices)
    # 返回打乱后的数据
    return x[indices], y[indices]
def plot_losses(epoch_losses, dataset_index, loss_type,model_type,path=""):
    # 如果 epoch_losses 是 PyTorch 张量，确保移到 CPU 并转换为 NumPy
    if isinstance(epoch_losses, torch.Tensor):
        epoch_losses = epoch_losses.cpu().numpy()
    elif isinstance(epoch_losses, list) and any(isinstance(x, torch.Tensor) for x in epoch_losses):
        epoch_losses = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in epoch_losses]

    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label=f'{loss_type.capitalize()} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type}_{loss_type.capitalize()} Loss  (Dataset {dataset_index})')
    plt.legend()
    plt.savefig(f'{path}/{model_type}_{loss_type}_loss_plot_dataset_{dataset_index}.png')
    plt.close()


def BSS_eval_np(y_true, y_pred):
    '''
        y_true, y_pred = val_y, models.predict(val_x)
        BS, BSS = BSS_eval_np(y_true, y_pred)
        y_true是热编码的二分类
        y_pred是模型输出的概率值（batch,2）
    '''
    '''
        用法
                    y_true = np.array(all_predictions_y_true)#拿出目标标签下表
                    y_true_onehot = np.zeros((y_true.shape[0], 2), dtype=int)
                    y_true_onehot[np.arange(y_true.shape[0]), y_true] = 1
                    y_true=y_true_onehot
                    # y_prob = np.array([row[1] for row in all_predictions_y_prob])#拿到正类概率
                    y_prob = np.array( all_predictions_y_prob)  # 拿到概率
                    BS, BSS = BSS_eval_np(y_true, y_prob)
    '''
    BS = np.mean(np.square(y_true - y_pred), axis=0)[1]
    y_ave = np.mean(y_true, axis=0)
    BSS = 1 - BS / np.mean(np.square(y_true - y_ave), axis=0)[1]
    return BS, BSS

def BS_BSS_score(y_true, y_prob):
    """
    :param y_true: one_hot格式
    :param y_prob: softmax输出的(m, 2)形状的

    y_true = y_test.argmax(axis=1)
    y_prob = models.predict(x_test_time_step)[:, 1]拿到正类

    :return: BS和BSS的值
    """
    # BSS开始计算

    BS = brier_score_loss(y_true, y_prob)
    # y_mean = y_prob.mean()
    y_mean = np.array(y_true).mean()

    # print(y_true)
    # print(y_mean)
    temp = y_true - y_mean
    # print(temp)
    temp = np.square(temp)
    # print(temp)
    temp = np.sum(temp) / float(len(y_true))
    BSS = 1 - BS / temp
    return BS, BSS

def save_torchModel(model, save_filename):
    with open(save_filename, "wb") as f:
        torch.save(model, f)
    print('Saved as %s' % save_filename)

def setup_seed_torch(seed):
    #  下面两个常规设置了，用来np和random的话要设置
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU训练需要设置这个
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(False)  # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法   修改了和原来相比
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.backends.cudnn.benchmark = False  # 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现。
def Write_txt(results_logfilepath):
    std_file_name = results_logfilepath
    # 标准输出（通常是终端或控制台）重定向到一个文件。这意味着在执行这行代码之后，
    # 所有的打印语句（print`）和其他写入标准输出的操作都将写入到指定的文件中，而不是显示在终端上。
    sys.stdout = open(std_file_name, "w+")

class DualOutput:
    def __init__(self, filepath):
        self.console = sys.stdout
        self.file = open(filepath, "w+")

    def write(self, message):
        self.console.write(message)  # 输出到控制台
        self.file.write(message)  # 输出到文件

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()














