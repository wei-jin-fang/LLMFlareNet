import os
import re
import pandas as pd


# 定义处理函数
def extract_value(cell):
    """从 tensor(...) 字符串中提取数值"""
    # 方法1：使用正则表达式提取数值
    match = re.search(r'tensor\(([\d.]+)', str(cell))
    if match:
        return float(match.group(1))

    # 方法2：使用字符串分割（备选）
    parts = str(cell).split('(')
    if len(parts) > 1:
        value_str = parts[1].split(',')[0]
        try:
            return float(value_str)
        except:
            return cell

    return cell  # 如果无法解析，返回原始值


# 定义模型组
model_groups = {
    "Onefit": ["Onefitall_16", "Onefitall_18", "Onefitall_26"],
    # "TimeLLM": ["LLMFlareNet_5", "LLMFlareNet_6"],
    "Three": ["NN", "Transformer", "LSTM"]
}

# 遍历模型组
for result_dir, model_types in model_groups.items():
    for _modelType in model_types:
        # 定义输入目录
        input_dir = os.path.join("../../weight", _modelType)

        # 检查输入目录是否存在
        if not os.path.exists(input_dir):
            print(f"Skipping {input_dir}: Directory not found")
            continue

        # 定义要处理的 CSV 文件类型（训练和验证）
        file_types = ['train_loss', 'validation_loss']

        for file_type in file_types:
            # 输入文件路径
            input_path = fr"../../weight/{_modelType}/{_modelType}_{file_type}.csv"
            # 输出文件路径（与输入路径相同，覆盖原文件）
            output_path = input_path

            # 检查输入文件是否存在
            if not os.path.exists(input_path):
                print(f"Skipping {input_path}: File not found")
                continue

            # 读取 CSV 文件
            df = pd.read_csv(input_path)

            # 应用处理函数到所有列
            for col in df.columns:
                df[col] = df[col].apply(extract_value)

            # 保存清理后的 CSV 到原路径
            df.to_csv(output_path, index=False)
            print(f"Cleaned and saved: {output_path}")