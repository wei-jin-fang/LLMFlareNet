import argparse
import shap
import matplotlib.pyplot as plt
import matplotlib

from tools import *

matplotlib.use('Agg')
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import sys

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
import matplotlib
matplotlib.use('Agg')

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
            # 针对每一个数据
            for i, category in enumerate(categories):
                # 判断数据那个类别
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
        print(f"{path}get_Class函数得到的的权重：", weight_list)  # 用于测试

    return List[0], List[1], List[2], List[3], List[4], List[5]


def Preprocess(train_csv_path, validate_csv_path, test_csv_path):
    global FIRST

    train_x, train_y, validate_x, validate_y, test_x, test_y = \
        load_data(train_csv_path, validate_csv_path, test_csv_path, Class_NUM)

    train_x = train_x.reshape(-1, TIME_STEPS, INPUT_SIZE)
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


def SHAP(train_x,data_x, data_y, model, count, model_typ, plot_type,shap_mode):

    # https://grok.com/share/bGVnYWN5_1cf6c2f1-7ff5-44f7-87fb-5ccb8cac6e41
    # 确保输入数据为 NumPy 数组
    if isinstance(data_x, torch.Tensor):
        data_x = data_x.cpu().numpy()
    if isinstance(data_y, torch.Tensor):
        train_x = train_x.cpu().numpy()

    # 转换为 float32 类型
    data_x = data_x.astype(np.float32)

    # 选择整个数据集作为 SHAP_bar_and_dot 背景数据,换成trainx作为背景
    background_data = train_x
    background_tensor = torch.tensor(background_data, dtype=torch.float32)

    # 如果使用 CUDA，将背景数据和模型移动到 GPU
    if args.cuda and torch.cuda.is_available():
        background_tensor = background_tensor.cuda()
        model = model.cuda()

    # 使用 GradientExplainer
    explainer = shap.GradientExplainer(model, background_tensor)

    # 确保输入数据为张量
    data_tensor = torch.tensor(data_x, dtype=torch.float32)
    if args.cuda and torch.cuda.is_available():
        data_tensor = data_tensor.cuda()

    print("Computing SHAP_bar_and_dot values for the entire dataset...")
    shap_values = explainer.shap_values(data_tensor)
    print(f"SHAP_bar_and_dot values shape: {np.array(shap_values).shape}")
    # SHAP_bar_and_dot values shape: (175, 40, 10, 1)

    shap_values_positive = shap_values[:, :, :, 0]# 形状 (175, 40, 10)

    # 创建特征名称，仅包含 10 个特征
    feature_names = ["TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
                     "USFLUX", "AREA_ACR", "MEANPOT", "R_VALUE","SHRGT45"]

    # 按时间步（40）求均值，得到每个特征的 SHAP_bar_and_dot 值
    # shap_values_positive=np.abs(shap_values_positive)
    if shap_mode=="mean":
        shap_values_positive = np.mean(shap_values_positive, axis=1)
        data_x= np.mean(data_x, axis=1)
    elif shap_mode=="sum":
        shap_values_positive = np.sum(shap_values_positive, axis=1)
        data_x = np.sum(data_x, axis=1)

    if plot_type == "bar":
        shap.summary_plot(
            shap_values_positive,
            data_x,
            feature_names=feature_names,
            plot_type=plot_type,
            title="mean(|SHAP_bar_and_dot value|) (average impact on model output)"
        )
    else:
        shap.summary_plot(
            shap_values_positive,
            data_x,
            feature_names=feature_names,
            plot_type=plot_type
        )

    # 保存图形
    filename = fr"./feature_important/{plot_type}-{shap_mode}-{model_typ}-{count}.png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    print(f"Saved plot to {filename}")

    # 释放内
    plt.close()

    return None






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


if __name__ == "__main__":


    # for _modelType in ["Onefitall_16", "Onefitall_17", "Onefitall_18","LLMFlareNet_5", "LLMFlareNet_6","NN", "Transformer","LSTM"]:
    # for _modelType in ["NN",]:
    # for _modelType in ["Onefitall_16", "Onefitall_18", "Onefitall_26", "Transformer","LSTM"]:
    for _modelType in [ "Transformer","LSTM"]:
    # for _modelType in ["LLMFlareNet_5", "LLMFlareNet_6","NN", "Transformer","LSTM"]:

        start_time = time.time()
        timelabel = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        model_base = f"./model_output/{timelabel}"
        os.makedirs(f"{model_base}")
        os.makedirs(f"{model_base}/plot")
        results_filepath = f"{model_base}/important.txt"
        results_logfilepath = f"{model_base}/log.txt"

        sys.stdout = DualOutput(results_logfilepath)

        with open(results_filepath, 'w') as results_file:

            for count in range(args.num_dataset + 1):  # 循环处理0-9个数据集
                setup_seed_torch(args.seed)
                # dataname="new_data_scaler"
                dataname = args.datasetname
                train_csv_path = rf"../../{dataname}/{count}Train.csv"
                validate_csv_path = rf"../../{dataname}/{count}Val.csv"
                test_csv_path = rf"../../{dataname}/{count}Test.csv"
                train_x, train_y, validate_x, validate_y, test_x, test_y, class_weight = Preprocess(train_csv_path,
                                                                                                    validate_csv_path,
                                                                                                    test_csv_path)

                model_filename = f"../../weight/{_modelType}/model_{count}.pt"
                # 加载最佳模型
                model_test = torch.load(model_filename)

                print(f"===================={_modelType}的数据集{count}测试集进行SHAP=============================================")
                SHAP(train_x,test_x, test_y,model_test,count,_modelType,
                     plot_type="bar",shap_mode="sum")  # 使用测试集进行评估
                del model_test
