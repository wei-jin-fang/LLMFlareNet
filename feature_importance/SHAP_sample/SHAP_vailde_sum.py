import argparse
import sys

import matplotlib
import torch

import random
import numpy as np
import os

import time

import openpyxl
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import os
import pandas as pd
from sklearn.utils import compute_class_weight
from tools import BS_BSS_score, BSS_eval_np

from tools import Metric, plot_losses
from tools import getClass
from tools import shuffle_data
from tools import get_batches_integer_test
from tools import Rectify_binary
from tools import save_torchModel
from tools import setup_seed_torch
from tools import DualOutput
from tools import save_csv

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


def load_data(train_csv_path, validate_csv_path, test_csv_path, Class_NUM):
    pd.set_option('display.max_columns', None)  # 意思是不限制显示的列数。这样设置后，无论 Pandas 数据帧有多少列
    List = []
    count_index = 0
    for path in [train_csv_path, validate_csv_path, test_csv_path]:
        count_index += 1
        csv = pd.read_csv(path)
        start = 0
        end = 0
        for Column in csv.columns.values:
            start += 1
            if Column.__eq__("TOTUSJH"):
                break

        for Column in csv.columns.values:
            end += 1
            if Column.__eq__("SHRGT45"):
                break
        List.append(csv.iloc[:, start - 1:end].values)  # train_x  test_x
        Classes = csv["CLASS"].copy()
        # 定义分类数组
        categories = ["NC", "MX"]
        # 初始化类别计数列表
        weights = [0] * len(categories)
        class_list = []

        # 遍历每个Class，计算每个类别的数量
        for Class_ in Classes:
            for i, category in enumerate(categories):
                if Class_ in category:
                    weights[i] += 1
                    class_list.append(i)
                    break  # 找到匹配的类别后退出内层循环
        class_tensor = torch.tensor(class_list, dtype=torch.long)
        List.append(np.array(class_tensor))

        print(f"{path}每一类的数量是:", weights)
        tempweight = []

        for weight in weights:
            tempweight.append(weight)

        weight_list = getClass(tempweight)
        print(f"{path}get_Class函数得到的的权重：", weight_list)
        # exit()
    return List[0], List[1], List[2], List[3], List[4], List[5]


def Preprocess(train_csv_path, validate_csv_path, test_csv_path):
    global FIRST

    train_x, train_y, validate_x, validate_y, test_x, test_y = load_data(train_csv_path, validate_csv_path,
                                                                         test_csv_path, Class_NUM)
    # print(train_x.shape)(17800, 10)
    train_x = train_x.reshape(-1, TIME_STEPS, INPUT_SIZE)
    # print(train_x.shape)(445, 40, 10)

    train_y = Rectify_binary(train_y, Class_NUM, TIME_STEPS)
    validate_x = validate_x.reshape(-1, TIME_STEPS, INPUT_SIZE)
    test_x = test_x.reshape(-1, TIME_STEPS, INPUT_SIZE)
    validate_y = Rectify_binary(validate_y, Class_NUM, TIME_STEPS)
    test_y = Rectify_binary(test_y, Class_NUM, TIME_STEPS)

    if FIRST == 1:
        print("train_x.shape : {} ".format(train_x.shape))
        print("train_y.shape : {} ".format(train_y.shape))
        print("validate_x.shape : {} ".format(validate_x.shape))
        print("validate_y.shape : {} ".format(validate_y.shape))
        print("test_x.shape : {} ".format(test_x.shape))
        print("test_y.shape : {} ".format(test_y.shape))
        FIRST = 0

    def class_weight(alpha, beta, zero=True):
        # 权重计算-修改位置开始
        classes = np.array([0, 1])
        # 权重计算-修改位置结束
        # weight = compute_class_weight(class_weight='balanced', classes=classes, y=train_y.argmax(axis=1))
        # exit()
        weight = compute_class_weight(class_weight='balanced', classes=classes, y=train_y)
        if zero:
            weight[0] = weight[0] * alpha
            print("zero:", weight[0])
            return weight[0]
        else:
            weight[1] = weight[1] * beta
            print("zero:", weight[1])
            return weight[1]

    class_weight = {0.: class_weight(alpha=1, beta=1, zero=True), 1.: class_weight(alpha=1, beta=1, zero=False)}
    print(class_weight)

    return train_x, train_y, validate_x, validate_y, test_x, test_y, class_weight


def train(ep, train_x, train_y, optimizer, model):
    global steps
    train_loss = 0
    model.train()
    batch_count = 0
    for batch_idx, (data, target) in enumerate(get_batches_integer_test(train_x, train_y, batch_size)):
        # exit()
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.long)
        # data = data.unsqueeze(1)  # 增加维度适应
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data = data.contiguous().view(-1, 1, INPUT_SIZE * TIME_STEPS)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            class_weights_cuda = torch.tensor([class_weight[0.], class_weight[1.]], dtype=torch.float32).cuda()
        else:
            class_weights_cuda = torch.tensor([class_weight[0.], class_weight[1.]], dtype=torch.float32)

        optimizer.zero_grad()
        # print("data.shape",data.shape) #data.shape torch.Size([4, 1, 400])
        output = model(data)
        '''
                loss = F.binary_cross_entropy_with_logits(output, target, weight=class_weights_cuda)
        '''
        # exit()
        loss = F.nll_loss(output, target, weight=class_weights_cuda)
        loss.backward()
        # if args.clip > 0:
        #     torch.nn.utils.clip_grad_norm_(models.parameters(), args.clip)
        optimizer.step()
        train_loss += loss.item()
        steps += len(data)
        message = ('Train Epoch: {} \tLoss: {:.6f}\tSteps: {}'.format(
            ep, train_loss / args.log_interval, steps))
        batch_count += 1
    return train_loss / batch_count


def evaluate(data_x, data_y, model):
    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []
    batch_count = 0

    all_predictions_y_true = []
    all_predictions_y_prob = []
    with torch.no_grad():
        for data, target in get_batches_integer_test(data_x, data_y, batch_size):

            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target, dtype=torch.long)

            # data = data.unsqueeze(1)  # 增加维度适应
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            # data = data.contiguous().view(-1, 1, INPUT_SIZE * TIME_STEPS)  # 调整为适合模型输入的形状
            if args.cuda:
                data, target = data.cuda(), target.cuda()
                class_weights_cuda = torch.tensor([class_weight[0.], class_weight[1.]], dtype=torch.float32).cuda()
            else:
                class_weights_cuda = torch.tensor([class_weight[0.], class_weight[1.]], dtype=torch.float32)
            output = model(data)
            # test_loss += F.nll_loss(output, target, weight=class_weights_cuda, size_average=False).item()
            test_loss += F.nll_loss(output, target, size_average=False).item()

            pred = torch.argmax(output, dim=1)  # 拿到概率最大的下标
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            all_targets.extend(target.cpu().numpy())  # 这一批次实际label加进去数组
            all_predictions.extend(torch.argmax(output, dim=1).cpu().numpy())  # 这一批次预测label加进去数组

            # 添加内容方便计算BSS BS
            probabilities = torch.exp(output)  # 拿到这一批次概率数值
            # tensor([[0.8139, 0.1861],
            #         [0.8321, 0.1679],
            # all_predictions_y_true.extend(torch.argmax(output, dim=1).cpu().numpy())  # 拿到预测的实际label
            all_predictions_y_true.extend(target.cpu().numpy())  # 拿到预测的实际label
            all_predictions_y_prob.extend(probabilities.cpu().numpy().tolist())  # 拿到概率数值加入数组

            batch_count += 1
        metric = Metric(all_targets, all_predictions)
        print(metric.Matrix())
        Tss = metric.TSS()[0]
        print(f"TSS: {Tss}")
        test_loss /= batch_count
        accuracy = correct / len(data_x)
        message = ('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(data_x),
            100. * accuracy))
        print("本轮评估完成", message)
        return {'loss': test_loss, 'accuracy': accuracy, 'tss': Tss,
                "metric": metric}, all_predictions_y_true, all_predictions_y_prob


import shap
import matplotlib.pyplot as plt

matplotlib.use('Agg')
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt


def SHAP_all_time_combined_force_plot(train_x, data_x, data_y, model, count, model_typ):
    # 确保输入数据为 NumPy 数组
    if isinstance(data_x, torch.Tensor):
        data_x = data_x.cpu().numpy()
    if isinstance(train_x, torch.Tensor):
        train_x = train_x.cpu().numpy()

    # 转换为 float32 类型
    data_x = data_x.astype(np.float32)

    # 选择背景数据
    background_data = train_x
    background_tensor = torch.tensor(background_data, dtype=torch.float32)

    # 如果有 CUDA，将背景数据和模型移到 GPU
    args = type('Args', (), {'cuda': torch.cuda.is_available()})()
    if args.cuda:
        background_tensor = background_tensor.cuda()
        model = model.cuda()

    # 手动计算 expected_value（正类的平均模型输出）
    model.eval()
    with torch.no_grad():
        background_outputs = model(background_tensor).cpu().numpy()  # 形状: (n_background, n_classes)
        expected_value = np.mean(background_outputs, axis=0)  # 形状: (n_classes,)
        classnumber = 1
        expected_value_positive = expected_value[classnumber]  # 正类的标量值
    # print(f"正类的预期值: {expected_value_positive}")

    # 创建特征名称
    feature_names = ["TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
                     "USFLUX", "AREA_ACR", "MEANPOT", "R_VALUE", "SHRGT45"]

    # 初始化 SHAP 解释器
    explainer = shap.GradientExplainer(model, background_tensor)

    # 循环处理每个样本
    n_samples = data_x.shape[0]
    time_steps = data_x.shape[1]  # 假设时间步为 40

    # 循环处理每个样本（此处仅处理 noaaid_number=62）
    n_samples = data_x.shape[0]
    time_steps = data_x.shape[1]  # 假设时间步为 40
    for noaaid_number in range(175):

        # 提取单个样本数据
        single_sample = data_x[noaaid_number:noaaid_number + 1]  # 形状: (1, 40, 10)
        single_tensor = torch.tensor(single_sample, dtype=torch.float32)
        if args.cuda:
            single_tensor = single_tensor.cuda()

        # 计算模型输出
        with torch.no_grad():
            nowprob = model(single_tensor).cpu().numpy()  # 形状: (1, n_classes)
            model_output_positive = nowprob # 正类的 logits
        # print(f"模型输出（正类 logits）: {model_output_positive}")

        # 计算 SHAP 值
        print(f"正在计算样本 {noaaid_number} 的 SHAP 值...")
        shap_values = explainer.shap_values(single_tensor)  # 形状: (n_classes, 1, 40, 10)
        # print(f"SHAP 值形状: {np.array(shap_values).shape}")

        # 提取正类的 SHAP 值
        shap_values_positive = shap_values[classnumber]  # 形状: (1, 40, 10)

        # 收集所有时间步的 SHAP 值和特征值
        shap_values_all_timesteps = shap_values_positive[0]  # 形状: (40, 10)
        data_x_all_timesteps = single_sample[0]  # 形状: (40, 10)

        # 转换为 DataFrame
        shap_values_df = pd.DataFrame(shap_values_all_timesteps, columns=feature_names)
        data_x_df = pd.DataFrame(data_x_all_timesteps, columns=feature_names)

        # 验证 SHAP 值：基线值 + SHAP 值之和 ≈ 模型输出
        shap_sum = np.sum(shap_values_all_timesteps)  # 所有 SHAP 值之和
        predicted_output = expected_value_positive + shap_sum  # 基线值 + SHAP 和
        # print(f"SHAP 值之和: {shap_sum}")
        print(f"==============验证=======================基线值 + SHAP 值之和: {predicted_output}")
        print(f"==============验证=======================模型实际输出（正类）: {model_output_positive}")


        # 如果模型输出是 logits，转换为概率进行额外验证
        def sigmoidold(x):
            return 1 / (1 + np.exp(-x))
        def exp(x):
            return np.exp(x)

        predicted_prob = exp(predicted_output)
        model_prob = exp(model_output_positive)
        print(f"==============验证=======================预测概率（基线值 + SHAP 值）: {predicted_prob}")
        print(f"==============验证=======================模型输出概率: {model_prob}")

parser = argparse.ArgumentParser(description='Time-LLM')

# basic config

parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')

# my参数
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--epochs', type=int, default=160)
parser.add_argument('--num_dataset', type=int, default=9)  # 循环处理0-9个数据集
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--num_hidden_layers', type=int, default=4)
parser.add_argument('--num_heads', type=int, default=5)  # 大模型层
parser.add_argument('--log-interval', type=int, default=100, metavar='N')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--seed', type=int, default=2021, help='random seed')
parser.add_argument('--model_type', type=str, default='VIT', help='VIT,LLM_VIT')
parser.add_argument('--embed_dim', type=str, default=100, help='VIT,LLM_VIT')  # 大模型层
parser.add_argument('--hidden_units', type=int, default=256, help='64')
parser.add_argument('--num_layers', type=int, default=2, help='64')
parser.add_argument('--print', type=int, default=1, help='是否第一次输出')
parser.add_argument('--datasetname', type=str, default="data", help='new_data_scaler,data')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = args.batch_size

FIRST = 1
# 参数设置-修改位置开始
TIME_STEPS = 40
INPUT_SIZE = 10
Class_NUM = 2

# 参数设置-修改位置结束

import LLM_VIT

if __name__ == "__main__":
    # for item in range(8,120,8):
    #     args.embed_dim=item
    #     for _modelType in ["VIT","LSTM_new","LLM","LSTM"]:
    for _modelType in ["LLM_VIT"]:
        # for _modelType in ["LLM_VIT"]:
        # for _modelType in ["LLM_VIT"]:
        args.model_type = _modelType
        other = "全局归一化数据集_监测TSS"
        if args.datasetname == "new_data_scaler":
            other = "第三种归一化数据集_监测TSS"
        commment = 'lr{}_epochs{}_datasetNum{}_modelType{}_batchsize{}_numhiddenlayers{}_lr{}_numheads{}_embeddim{}_hiddenunits{}_numlayers{}_datasetname{}'.format(
            args.lr,
            args.epochs,
            args.num_dataset,
            args.model_type,
            args.batch_size,
            args.num_hidden_layers,
            args.lr,
            args.num_heads,
            args.embed_dim,
            args.hidden_units,
            args.num_layers,
            args.datasetname,
            other
        )
        print(commment)
        start_time = time.time()

        timelabel = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        model_base = f"./model_output/{timelabel}"
        os.makedirs(f"{model_base}")
        os.makedirs(f"{model_base}/plot")
        results_filepath = f"{model_base}/important.txt"
        results_logfilepath = f"{model_base}/log.txt"

        sys.stdout = DualOutput(results_logfilepath)

        # 定义十个数据集合存储内容
        all_matrix = np.array([[0, 0], [0, 0]])
        data_Recall, data_Precision, data_Accuracy, data_TSS, data_BSS, data_HSS, data_FAR = [], [], [], [], [], [], []
        data_FPR = []
        # 打开文件准备写入
        with open(results_filepath, 'w') as results_file:
            # for count in range(10):  # 循环处理0-9个数据集
            # for count in range(args.num_dataset + 1):  # 循环处理0-9个数据集
            for count1 in range(args.num_dataset + 1):  # 循环处理0-9个数据集
                setup_seed_torch(args.seed)

                steps = 0
                print(args)
                count = 3
                # dataname="new_data_scaler"
                dataname = args.datasetname
                train_csv_path = rf"../../../{dataname}/{count}Train.csv"
                validate_csv_path = rf"../../../{dataname}/{count}Val.csv"
                test_csv_path = rf"../../../{dataname}/{count}Test.csv"
                train_x, train_y, validate_x, validate_y, test_x, test_y, class_weight = Preprocess(train_csv_path,
                                                                                                    validate_csv_path,
                                                                                                    test_csv_path)

                model_filename = rf"G:\本科\项目_比赛_论文资料\论文_LLM_VIT\Titan\TSS" \
                                 + f"\TSS_{args.model_type}" + f"\model_{count}.pt"

                # 加载最佳模型
                model_test = torch.load(model_filename)
                print(model_test.args)
                # exit()
                # 测试集评估
                print(f"====================数据集{count}测试集轮评估数据=============================================")
                # test_metrics, all_predictions_y_true, all_predictions_y_prob = evaluate(test_x, test_y,
                #                                                                         model_test)  # 使用测试集进行评估


                SHAP_all_time_combined_force_plot(train_x,test_x, test_y, model_test, count, args.model_type)  # 使用测试集进行评估
                exit()