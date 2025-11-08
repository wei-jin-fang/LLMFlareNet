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
from tools import get_batches_all
from tools import Rectify_binary
from tools import save_torchModel
from tools import setup_seed_torch
from tools import DualOutput
from tools import save_csv

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

TIME_STEPS = 40
INPUT_SIZE = 10
Class_NUM = 2
FIRST = 1

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


def train_integer(ep, train_x, train_y, optimizer, model, batch_size):
    global steps
    train_loss = 0
    model.train()
    batch_count = 0
    for batch_idx, (data, target) in enumerate(get_batches_integer(train_x, train_y, batch_size)):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        # 修改 target 处理：无论是否为 Tensor，都强制转换为 float32
        target = torch.tensor(target, dtype=torch.float32) if not isinstance(target, torch.Tensor) else target.to(
            dtype=torch.float32)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
            class_weights_cuda = torch.tensor([class_weight[0.], class_weight[1.]], dtype=torch.float32).cuda()
        else:
            class_weights_cuda = torch.tensor([class_weight[0.], class_weight[1.]], dtype=torch.float32)

        optimizer.zero_grad()
        output = model(data)  # [batch_size, 1]，概率值
        target = target.view(-1, 1)  # 确保 target 形状为 [batch_size, 1]
        # 根据 target 生成每个样本的权重
        weights = torch.zeros_like(target, dtype=torch.float32, device=target.device)
        weights[target == 0] = class_weights_cuda[0]  # 类别 0 的权重
        weights[target == 1] = class_weights_cuda[1]  # 类别 1 的权重
        loss = F.binary_cross_entropy(output, target, weight=weights)  # 使用 BCE Loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        batch_count += 1

    message = ('Train Epoch: {} \t average Loss: {:.6f}'.format(ep, train_loss / batch_count))
    print(message)
    return train_loss / batch_count


def train_all(ep, train_x, train_y, optimizer, model, batch_size):
    global steps
    train_loss = 0
    model.train()
    batch_count = 0
    for batch_idx, (data, target) in enumerate(get_batches_all(train_x, train_y, batch_size)):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        # 修改 target 处理：无论是否为 Tensor，都强制转换为 float32
        target = torch.tensor(target, dtype=torch.float32) if not isinstance(target, torch.Tensor) else target.to(
            dtype=torch.float32)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
            class_weights_cuda = torch.tensor([class_weight[0.], class_weight[1.]], dtype=torch.float32).cuda()
        else:
            class_weights_cuda = torch.tensor([class_weight[0.], class_weight[1.]], dtype=torch.float32)

        optimizer.zero_grad()
        output = model(data)  # [batch_size, 1]，概率值
        target = target.view(-1, 1)  # 确保 target 形状为 [batch_size, 1]
        # 根据 target 生成每个样本的权重
        weights = torch.zeros_like(target, dtype=torch.float32, device=target.device)
        weights[target == 0] = class_weights_cuda[0]  # 类别 0 的权重
        weights[target == 1] = class_weights_cuda[1]  # 类别 1 的权重
        loss = F.binary_cross_entropy(output, target, weight=weights)  # 使用 BCE Loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        batch_count += 1

    message = ('Train Epoch: {} \t average Loss: {:.6f}'.format(ep, train_loss / batch_count))
    print(message)
    return train_loss / batch_count


def evaluate_integer(data_x, data_y, model, batch_size):
    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []
    batch_count = 0
    all_predictions_y_true = []
    all_predictions_y_prob = []

    with torch.no_grad():
        for data, target in get_batches_integer(data_x, data_y, batch_size):
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            # 修改 target 处理：无论是否为 Tensor，都强制转换为 float32
            target = torch.tensor(target, dtype=torch.float32) if not isinstance(target, torch.Tensor) else target.to(
                dtype=torch.float32)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
                class_weights_cuda = torch.tensor([class_weight[0.], class_weight[1.]], dtype=torch.float32).cuda()
            else:
                class_weights_cuda = torch.tensor([class_weight[0.], class_weight[1.]], dtype=torch.float32)

            output = model(data)  # [batch_size, 1]，概率值
            target = target.view(-1, 1)
            # 根据 target 生成每个样本的权重
            weights = torch.zeros_like(target, dtype=torch.float32, device=target.device)
            weights[target == 0] = class_weights_cuda[0]  # 类别 0 的权重
            weights[target == 1] = class_weights_cuda[1]  # 类别 1 的权重
            test_loss += F.binary_cross_entropy(output, target, weight=weights)  # 使用 BCE Loss.item()

            pred = (output > 0.5).float()  # 阈值 0.5 这个地方取值就是0或者1

            all_predictions.extend(pred.cpu().numpy().flatten())  # 对应之前的：这一批次预测laebl加进去数组
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            all_targets.extend(target.cpu().numpy().flatten())  # 对应之前的：这一批次实际label加进去数组

            # 添加内容方便计算 BSS BS
            all_predictions_y_true.extend(target.cpu().numpy().flatten())  # 对应之前的：拿到实际的label
            pos_prob = output  # 正类概率 [batch_size, 1]
            neg_prob = 1.0 - pos_prob  # 负类概率 [batch_size, 1]
            probabilities = torch.cat((neg_prob, pos_prob), dim=1)  # [batch_size, 2]，[负类概率, 正类概率]
            all_predictions_y_prob.extend(probabilities.cpu().numpy().tolist())  # 存储 [负类概率, 正类概率] 列表 #对应之前的： 拿到概率数值加入数组

            batch_count += 1

        metric = Metric(all_targets, all_predictions)
        print(metric.Matrix())
        TSS = metric.TSS()[0]
        print(f"TSS: {TSS}")
        test_loss /= batch_count
        accuracy = correct / len(data_x)
        message = (
            '\nTesting: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(data_x), 100. * accuracy))
        print("本轮评估完成", message)
        return {'loss': test_loss, 'accuracy': accuracy, 'tss': TSS, "metric": metric}, \
               all_predictions_y_true, all_predictions_y_prob


def evalual_all(data_x, data_y, model, batch_size):
    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []
    batch_count = 0
    all_predictions_y_true = []
    all_predictions_y_prob = []

    with torch.no_grad():
        for data, target in get_batches_all(data_x, data_y, batch_size):
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            # 修改 target 处理：无论是否为 Tensor，都强制转换为 float32
            target = torch.tensor(target, dtype=torch.float32) if not isinstance(target, torch.Tensor) else target.to(
                dtype=torch.float32)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
                class_weights_cuda = torch.tensor([class_weight[0.], class_weight[1.]], dtype=torch.float32).cuda()
            else:
                class_weights_cuda = torch.tensor([class_weight[0.], class_weight[1.]], dtype=torch.float32)

            output = model(data)  # [batch_size, 1]，概率值
            target = target.view(-1, 1)
            # 根据 target 生成每个样本的权重
            weights = torch.zeros_like(target, dtype=torch.float32, device=target.device)
            weights[target == 0] = class_weights_cuda[0]  # 类别 0 的权重
            weights[target == 1] = class_weights_cuda[1]  # 类别 1 的权重
            test_loss += F.binary_cross_entropy(output, target, weight=weights)

            pred = (output > 0.5).float()  # 阈值 0.5 这个地方取值就是0或者1

            all_predictions.extend(pred.cpu().numpy().flatten())  # 对应之前的：这一批次预测laebl加进去数组
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            all_targets.extend(target.cpu().numpy().flatten())  # 对应之前的：这一批次实际label加进去数组

            # 添加内容方便计算 BSS BS
            all_predictions_y_true.extend(target.cpu().numpy().flatten())  # 对应之前的：拿到实际的label
            pos_prob = output  # 正类概率 [batch_size, 1]
            neg_prob = 1.0 - pos_prob  # 负类概率 [batch_size, 1]
            probabilities = torch.cat((neg_prob, pos_prob), dim=1)  # [batch_size, 2]，[负类概率, 正类概率]
            all_predictions_y_prob.extend(probabilities.cpu().numpy().tolist())  # 存储 [负类概率, 正类概率] 列表 #对应之前的： 拿到概率数值加入数组

            batch_count += 1

        metric = Metric(all_targets, all_predictions)
        print(metric.Matrix())
        TSS = metric.TSS()[0]
        print(f"TSS: {TSS}")
        test_loss /= batch_count
        accuracy = correct / len(data_x)
        message = (
            '\nTesting: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(data_x), 100. * accuracy))
        print("本轮评估完成", message)
        return {'loss': test_loss, 'accuracy': accuracy, 'tss': TSS, "metric": metric}, \
               all_predictions_y_true, all_predictions_y_prob


import shap
import matplotlib.pyplot as plt

matplotlib.use('Agg')
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt

def SHAP_all_sample_of_all_time_combined_force_plot(train_x, data_x, data_y, model, count, model_typ):

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
        expected_value_positive = expected_value[0]  # 因为格式是一个list里面有个数字，提取出来
    print(f"正类的预期值: {expected_value_positive}")

    # 创建特征名称
    feature_names = ["TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
                     "USFLUX", "AREA_ACR", "MEANPOT", "R_VALUE", "SHRGT45"]

    # 初始化 SHAP 解释器
    explainer = shap.GradientExplainer(model, background_tensor)

    # 循环处理每个样本
    n_samples = data_x.shape[0]
    for noaaid_number in range(n_samples):
        # noaaid_number=62
        # 提取单个样本数据
        single_sample = data_x[noaaid_number:noaaid_number + 1]  # 形状: (1, 40, 10)
        single_tensor = torch.tensor(single_sample, dtype=torch.float32)
        if args.cuda:
            single_tensor = single_tensor.cuda()
        # 计算模型输出
        with torch.no_grad():
            nowprob = model(single_tensor).cpu().numpy()  # 形状: (1, )
        current_sample_laebl=data_y[noaaid_number]
        # 计算 SHAP 值
        print(f"Computing SHAP values for sample {noaaid_number}...")
        shap_values = explainer.shap_values(single_tensor)  # 形状: (n_classes, 1, 40, 10)
        print(f"SHAP values shape: {np.array(shap_values).shape}")

        # 提取的 SHAP 值
        shap_values_positive = shap_values.squeeze(axis=0)        # 形状: (1, 40, 10，1) 用0只是为了拿出数据变成1 40 10
        # print(shap_values_positive.shape)(40, 10, 1)
        # 收集所有时间步的 SHAP 值和特征值
        shap_values_all_timesteps = shap_values_positive[:, :, 0]  # 形状: (40, 10)
        data_x_all_timesteps = single_sample[0]  # 形状: (40, 10)
        # 转换为 DataFrame
        shap_values_df = pd.DataFrame(shap_values_all_timesteps, columns=feature_names)
        data_x_df = pd.DataFrame(data_x_all_timesteps, columns=feature_names)

        # # 绘制交互式力图并保存为 HTML
        plot_type = "force"
        shap.initjs()  # 初始化 JavaScript 可视化
        force_plot = shap.force_plot(
            expected_value_positive,  # 使用手动计算的基线
            shap_values_df.values,  # 所有时间步的 SHAP 值 (40, 10)
            data_x_df,  # 所有时间步的特征值 (40, 10)
            feature_names=feature_names
        )
        if not os.path.exists("force_combine_html"):
            os.mkdir("force_combine_html")
        # 保存为 HTML 文件
        filename = f"./force_combine_html/{plot_type}-{model_typ}--sample-{noaaid_number}-({nowprob})-label({current_sample_laebl}).html"
        shap.save_html(filename, force_plot)
        print(f"HTML 图像已保存至 {filename}")





def SHAP_all_sample_all_time_plot(train_x, data_x, data_y, model, count, model_typ):
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
        expected_value_positive = expected_value[0]  # 因为格式是一个list里面有个数字，提取出来
    print(f"正类的预期值: {expected_value_positive}")

    # 创建特征名称
    feature_names = ["TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
                     "USFLUX", "AREA_ACR", "MEANPOT", "R_VALUE", "SHRGT45"]

    # 初始化 SHAP 解释器
    explainer = shap.GradientExplainer(model, background_tensor)

    # 循环处理每个样本
    n_samples = data_x.shape[0]
    time_steps = data_x.shape[1]  # 假设时间步为 40

    for noaaid_number in range(n_samples):
        # 提取单个样本数据
        # noaaid_number=28
        single_sample = data_x[noaaid_number:noaaid_number + 1]  # 形状: (1, 40, 10)
        single_tensor = torch.tensor(single_sample, dtype=torch.float32)
        if args.cuda:
            single_tensor = single_tensor.cuda()

        # 计算模型输出
        with torch.no_grad():
            nowprob = model(single_tensor).cpu().numpy()[0][0]  # 形状: (1, )然后[0]为了拿出概率
        print("当前样本概率：",nowprob)
        current_sample_laebl = data_y[noaaid_number]
        # 计算 SHAP 值
        print(f"Computing SHAP values for sample {noaaid_number}...")
        shap_values = explainer.shap_values(single_tensor)  # 形状: (n_classes, 1, 40, 10)
        print(f"SHAP values shape: {np.array(shap_values).shape}")

        # 提取正类的 SHAP 值
        shap_values_positive = shap_values.squeeze(axis=0)   # 得到形状: ( 40, 10 1)
        # 收集所有时间步的 SHAP 值和特征值
        shap_values_all_timesteps = shap_values_positive[:, :, 0]  # 形状: (40, 10)
        # 按时间步循环绘制力图
        for t in range(time_steps):

            # 提取当前时间步的 SHAP 值和特征值
            shap_values_t = shap_values_all_timesteps[t, :]  # 形状: (10,)
            data_x_t = single_sample[0, t, :]  # 形状: (10,)

            # 转换为 DataFrame 以匹配特征名称
            shap_values_t_df = pd.DataFrame([shap_values_t], columns=feature_names)
            data_x_t_df = pd.DataFrame([data_x_t], columns=feature_names)

            # 绘制 SHAP 力图
            plot_type = "force"
            shap.initjs()  # 初始化 JavaScript 可视化
            shap.force_plot(
                expected_value_positive,  # 使用手动计算的基线
                shap_values_t_df.iloc[0].values,  # 当前时间步的 SHAP 值 (10,)
                data_x_t_df.iloc[0],  # 当前时间步的特征值 (10,)
                feature_names=feature_names,
                matplotlib=True  # 使用 Matplotlib 渲染
            )

            # 设置标题
            plt.title(f"SHAP Force Plot (AR{noaaid_number}, Time Step {t}, Model: {model_typ})")

            # 自定义特征值的小数位数（例如保留 2 位小数）并移除 base value
            decimal_places = 2
            for text in plt.gca().get_children():
                if isinstance(text, plt.Text):
                    text_content = text.get_text()
                    # 跳过包含 "base value" 的文本
                    if "base value" in text_content.lower():
                        text.set_text("")  # 清空 base value 文本
                        continue
                    # 处理特征值文本（包含 "="）
                    if "=" in text_content:
                        try:
                            feature_part, value_part = text_content.split("=")
                            value = float(value_part.strip())
                            # 格式化值为指定小数位数
                            new_text = f"{feature_part}= {value:.{decimal_places}f}"
                            text.set_text(new_text)
                        except ValueError:
                            continue  # 跳过无法解析的文本

            if not os.path.exists("force_all_sample_all_timestep"):
                os.mkdir("force_all_sample_all_timestep")
            # 保存图像
            filename = f"./force_all_sample_all_timestep/{plot_type}-{model_typ}-sample-{noaaid_number}-timestep-{t}-({nowprob})-label({current_sample_laebl}).png"
            plt.savefig(filename, bbox_inches="tight", dpi=300)
            print(f"图像已保存至 {filename}")

            # 释放内存
            plt.close()
            # exit()
        # exit()

def SHAP_single(data_x, data_y, model, count, model_typ,):
    # 确保输入数据为 NumPy 数组
    if isinstance(data_x, torch.Tensor):
        data_x = data_x.cpu().numpy()
    if isinstance(data_y, torch.Tensor):
        data_y = data_y.cpu().numpy()

    # 转换为 float32 类型
    data_x = data_x.astype(np.float32)

    # 选择背景数据（若数据量大，可用子集以减少计算量）
    background_data = data_x
    background_tensor = torch.tensor(background_data, dtype=torch.float32)

    # 如果有 CUDA，将背景数据和模型移到 GPU
    args = type('Args', (), {'cuda': torch.cuda.is_available()})()  # 模拟 args.cuda
    if args.cuda:
        background_tensor = background_tensor.cuda()
        model = model.cuda()

    # 提取单个样本数据
    noaaid_number=0
    single_sample = data_x[noaaid_number:noaaid_number+1]  # 形状: (1, 40, 10)
    single_tensor = torch.tensor(single_sample, dtype=torch.float32)
    if args.cuda:
        single_tensor = single_tensor.cuda()

    # 使用 GradientExplainer
    explainer = shap.GradientExplainer(model, background_tensor)

    # 计算单个样本的 SHAP 值
    print(f"Computing SHAP values for sample {noaaid_number}...")
    shap_values = explainer.shap_values(single_tensor)
    print(f"SHAP values shape: {np.array(shap_values).shape}")  # 形状: (2, 1, 40, 10)

    # 提取正类的 SHAP 值（正类为标签 1，索引 1）
    classnumber = 1
    shap_values_positive = shap_values[classnumber]  # 形状: (1, 40, 10)

    # 创建特征名称
    feature_names = ["TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
                     "USFLUX", "AREA_ACR", "MEANPOT", "R_VALUE", "SHRGT45"]

    # 按时间步（40）取均值，得到每个特征的 SHAP 值
    shap_values_positive_mean = np.mean(shap_values_positive, axis=1)  # 形状: (1, 10)
    shap_values_positive_mean = pd.DataFrame(shap_values_positive_mean, columns=feature_names)

    # 输入数据按时间步取均值
    data_x_mean = np.mean(single_sample, axis=1)  # 形状: (1, 10)
    data_x_mean = pd.DataFrame(data_x_mean, columns=feature_names)

    # 手动计算 expected_value（正类的平均模型输出）
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        background_outputs = model(background_tensor).cpu().numpy()  # 形状: (n_background, n_classes)
        expected_value = np.mean(background_outputs, axis=0)  # 形状: (n_classes,)
        expected_value_positive = expected_value[classnumber]  # 正类的标量值
    print(f"正类的预期值: {expected_value_positive}")

    # 绘制 SHAP 力图
    plot_type = "force"
    shap.initjs()  # 初始化 JavaScript 可视化
    shap.force_plot(
        expected_value_positive,  # 使用手动计算的基线
        shap_values_positive_mean.iloc[0].values,  # 单个样本的 SHAP 值 (10,)
        data_x_mean.iloc[0],  # 单个样本的特征值 (10,)
        feature_names=feature_names,
        matplotlib=True  # 使用 Matplotlib 渲染，适合保存
    )

    # 设置标题
    plt.title(f"SHAP Force Plot for Positive Class (Sample {noaaid_number}, Model: {model_typ})")

    # 保存图像
    filename = f"{plot_type}-{model_typ}-positive_class-{classnumber}-summary_plot-{count}({noaaid_number}).png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    print(f"图像已保存至 {filename}")

    # 释放内存
    plt.close()

    return shap_values_positive_mean

def read_parameters():
    # 初始化参数收集器
    parser = argparse.ArgumentParser(description='Time-LLM')
    # 数据集部分参数
    parser.add_argument('--num_dataset', type=int, default=9)  # 循环处理0-9个数据集
    parser.add_argument('--input_size', type=int, default=10)  # 循环处理0-9个数据集
    parser.add_argument('--time_step', type=int, default=40)  # 循环处理0-9个数据集
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--device', type=str, default="cuda")  # 循环处理0-9个数据集
    parser.add_argument('--datasetname', type=str, default="data", help='new_data_scaler,data')
    parser.add_argument('--epochs', type=int, default=50)
    # 公共训练参数
    parser.add_argument('--cuda', action='store_false')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--model_type', type=str, default='NN',
                        help='Onefitall,LLMFlareNet')

    # 备注参数
    parser.add_argument('--conmment', type=str, default="None")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = read_parameters()
    batch_size=16
    start_time = time.time()
    timelabel = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    model_base = f"./model_output/{timelabel}"
    os.makedirs(f"{model_base}")
    os.makedirs(f"{model_base}/plot")
    results_filepath = f"{model_base}/important.txt"
    results_logfilepath = f"{model_base}/log.txt"
    sys.stdout = DualOutput(results_logfilepath)
    # 打开文件准备写入
    with open(results_filepath, 'w') as results_file:
        for count in range(args.num_dataset + 1):  # 循环处理0-9个数据集
            setup_seed_torch(args.seed)
            count = 3
            dataname = args.datasetname
            train_csv_path = f"../../data/{count}Train.csv"
            validate_csv_path =f"../../data/{count}Val.csv"
            test_csv_path = f"../../data/{count}Test.csv"
            train_x, train_y, validate_x, validate_y, test_x, test_y, class_weight = Preprocess(train_csv_path,
                                                                                                validate_csv_path,
                                                                                                test_csv_path)

            model_filename = f'../../weight/NN/model_{count}.pt'

            # 加载最佳模型
            model_test = torch.load(model_filename)
            print(f"====================数据集{count}测试集轮评估数据=============================================")

            SHAP_all_sample_all_time_plot(train_x,test_x, test_y, model_test, count, args.model_type)  # 使用测试集进行评估
            exit()