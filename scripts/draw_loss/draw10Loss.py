import os
import pandas as pd
import matplotlib.pyplot as plt

# 设置全局字体和大小
plt.rcParams.update({
    'figure.figsize': (12, 8),  # 调整整体图像大小
    'axes.titlesize': 20,  # 标题字体大小
    'axes.labelsize': 18,  # x 轴和 y 轴标签字体大小
    'xtick.labelsize': 16,  # x 轴刻度字体大小
    'ytick.labelsize': 16,  # y 轴刻度字体大小
    'legend.fontsize': 10,  # 图例字体大小
    'lines.linewidth': 2,  # 线条宽度
})

# 定义要绘制的行范围（基于 0 索引，不包括表头）
start_epoch = 0  # 起始轮次（epoch 0）
start_row = start_epoch   # 起始行号（epoch 0 对应第2行，索引1）

end_epoch = 50   # 结束轮次（epoch 50）
end_row = end_epoch + 1      # 结束行号（epoch 50 对应第51行，不包含此行）
end_row = end_row + 1      # 实现闭区间

# 定义模型组
model_groups = {
    "Onefit": ["Onefitall_16", "Onefitall_18", "Onefitall_26"],
    # "TimeLLM": ["LLMFlareNet_5", "LLMFlareNet_6"],
    "Three": ["NN", "Transformer", "LSTM"]
}

for result_dir, model_types in model_groups.items():
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for _modelType in model_types:
        # 文件路径
        train_loss_path = fr"../../weight/{_modelType}/{_modelType}_train_loss.csv"
        validation_loss_path = fr"../../weight/{_modelType}/{_modelType}_validation_loss.csv"

        # 读取CSV文件，跳过第一行表头，并选择指定行范围
        train_loss_data = pd.read_csv(train_loss_path, header=None, skiprows=1)
        validation_loss_data = pd.read_csv(validation_loss_path, header=None, skiprows=1)

        # 选择指定行范围（基于 0 索引）
        train_loss_data = train_loss_data.iloc[start_row:end_row]
        validation_loss_data = validation_loss_data.iloc[start_row:end_row]

        # 确定 x 轴范围和刻度
        x_start = 0  # 数据从 epoch 0 开始
        x_range = range(x_start, len(train_loss_data) + x_start)
        # 绘制并保存训练损失曲线

        if _modelType=="Onefitall_16":
            _modelType="LLMFlareNet"
        if _modelType=="Onefitall_16":
            _modelType="LLMFlareNet"
        if _modelType=="Onefitall_26":
            _modelType="BERT (Random Parameters)"
        if _modelType=="Transformer":
            _modelType="BERT——>Transformer"
        plt.figure(figsize=(10, 6))
        for i in range(train_loss_data.shape[1]):
            plt.plot(x_range, train_loss_data[i], label=f'Dataset {i}')
        plt.title(f"Training Loss Curves ({_modelType})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.xlim(-2, end_epoch+2)  # 调整 x 轴起始点为 -0.5，使 0 刻度与 y 轴有距离
        plt.xticks(range(0, end_epoch + 1+5, 10))  # 明确设置 x 轴刻度，从 0 开始，步长为 5
        plt.legend()
        plt.savefig(f"./{result_dir}/{_modelType}_training_loss_curves_epochs_{start_epoch}_to_{end_epoch}.eps",format="eps")
        plt.close()

        # 绘制并保存验证损失曲线
        plt.figure(figsize=(10, 6))
        for i in range(validation_loss_data.shape[1]):
            plt.plot(x_range, validation_loss_data[i], label=f'Dataset {i}')
        plt.title(f"Validation Loss Curves ({_modelType})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.xlim(-2, end_epoch+5)  # 调整 x 轴起始点为 -0.5，使 0 刻度与 y 轴有距离
        plt.xticks(range(0, end_epoch + 1+2, 10))  # 明确设置 x 轴刻度，从 0 开始，步长为 5
        plt.legend()
        plt.savefig(f"./{result_dir}/{_modelType}_validation_loss_curves_epochs_{start_epoch}_to_{end_epoch}.eps",format="eps")
        plt.close()