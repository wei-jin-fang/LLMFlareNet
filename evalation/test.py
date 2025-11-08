import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tools import DualOutput, setup_seed_torch, truncate


def global_data_by_mean_and_std(list_data):
    print("++++++++++++++++++++++++++++++++第四步归一化+++++++++++++++++++++++++++++++++++++++++++++")

    # 将数据加载到 DataFrame 中，并将所有数据转换为浮点数类型
    df = pd.DataFrame(list_data,
                      columns=["TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH",
                               "SAVNCPP", "USFLUX", "AREA_ACR", "MEANPOT", "R_VALUE", "SHRGT45"])
    # print(df)
    df = df.apply(pd.to_numeric)  # 转换为数值类型

    # 手动设置标准化参数（使用提供的均值和标准差）
    mean_values = np.array([
        1.19527086e+03, 2.68865850e+23, 2.45100336e+13, 1.50900269e+02,
        6.72536197e+12, 1.54981119e+22, 8.31207961e+02, 6.51523418e+03,
        3.13289555e+00, 2.62529642e+01
    ])
    scale_values = np.array([
        1.36034771e+03, 4.14845493e+23, 2.63605181e+13, 2.70139847e+02,
        1.00889221e+13, 1.73280368e+22, 8.09170152e+02, 4.20950529e+03,
        1.45805498e+00, 1.61363519e+01
    ])

    # 初始化 StandardScaler 并手动设置参数
    scaler = StandardScaler()
    scaler.mean_ = mean_values  # 设置均值
    scaler.scale_ = scale_values  # 设置标准差
    scaler.var_ = scale_values ** 2  # 方差是标准差的平方

    # 使用手动设置的 scaler 对新数据进行标准化
    df_normalized = df.copy()
    new_data_normalized_data = scaler.transform(df_normalized)

    return new_data_normalized_data

def inference_by_data(_modelTypeList, data, JianceType=None):
    parser = argparse.ArgumentParser(description='Time-LLM')
    parser.add_argument('--model_type', type=str, default='VIT', help='VIT,LLM_VIT')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    args = parser.parse_args()

    for _modelType in _modelTypeList:
        args.model_type = _modelType
        start_time = time.time()

        timelabel = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        # model_base = rf".\model_output\{timelabel}"
        # os.makedirs(rf"{model_base}")
        # os.makedirs(rf"{model_base}\plot")
        # results_filepath = rf"{model_base}\important.txt"
        # results_logfilepath = rf"{model_base}\log.txt"

        # sys.stdout = DualOutput(results_logfilepath)
        # with open(results_filepath, 'w') as results_file:
        probabilitiesList = []
        for count in range(10):  # 循环处理0-9个数据集
            setup_seed_torch(args.seed)
            # model_filename = rf"." \
            #                  + f"\{JianceType}_{args.model_type}" \
            #                  + f"\model_{count}.pt"
            model_filename = rf"/home/wjf/project/LLM_FlareNet/weight/{_modelType}/model_{count}.pt"


            # 加载最佳模型
            model_test = torch.load(model_filename, map_location=torch.device('cpu'), weights_only=False)
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            data_3d = data.unsqueeze(0).to("cpu")
            output = model_test(data_3d)
            # 添加内容方便计算BSS BS
            # probabilities = torch.exp(output)  # 拿到这一批次概率数值,现在这个版本以及不需啊哟概率阈值了，现在是输出头是1
            probabilitiesList.append(output)

        # print(probabilitiesList)
        #probabilitiesList:  [tensor([[0.2298]], grad_fn=<SigmoidBackward0>), tensor([[0.2386]], grad_fn=.3866]],
        # tensor[0, 0] 直接拿出这一个值
        new_list = [truncate(tensor[0, 0].item(), 3) for tensor in probabilitiesList]
        return new_list
if __name__ == '__main__':
    # 转换成双重列表
    _model_type="Onefitall_16"
    random_2d_list = [[random.random() for _ in range(10)] for _ in range(40)]
    # 接下来进行归一化
    globdata = global_data_by_mean_and_std(random_2d_list)
    probabilitiesList = inference_by_data([_model_type], globdata)
    print(probabilitiesList)