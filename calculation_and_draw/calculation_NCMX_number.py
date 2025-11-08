import os
import pandas as pd
from collections import defaultdict
import glob

# 设置工作目录
base_dir = "../evalation/daily_mode"

# 初始化统计字典，存储每个CSV文件的分布
csv_class_counts = defaultdict(lambda: {'N': 0, 'C': 0, 'M': 0, 'X': 0})

# 遍历以Onefit16开头的文件夹
for folder in glob.glob(os.path.join(base_dir, "Onefitall_16*")):
    print(folder)
    if os.path.isdir(folder):  # 确保是文件夹
        folder_name = os.path.basename(folder)
        print(f"处理文件夹: {folder_name}")

        # 遍历文件夹中以merger_开头的CSV文件
        for csv_file in glob.glob(os.path.join(folder, "merger_*.csv")):
            csv_name = os.path.basename(csv_file)  # 获取CSV文件名
            try:
                # 读取CSV文件
                df = pd.read_csv(csv_file)

                # 统计Class列的NCMX分布
                class_distribution = df['Class'].value_counts()

                # 更新统计字典，以文件夹名+CSV文件名作为键
                key = f"{folder_name}/{csv_name}"
                for class_label in ['N', 'C', 'M', 'X']:
                    count = class_distribution.get(class_label, 0)
                    csv_class_counts[key][class_label] = count  # 直接赋值，不累加

            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")

# 打印统计结果
print("\n统计结果:")
for csv_key, counts in csv_class_counts.items():
    print(f"\n文件: {csv_key}")
    print(f"N: {counts['N']}")
    print(f"C: {counts['C']}")
    print(f"M: {counts['M']}")
    print(f"X: {counts['X']}")

# 将结果保存到CSV文件
result_df = pd.DataFrame.from_dict(csv_class_counts, orient='index')
result_df.to_csv("class_distribution_summary.csv")
print("\n统计结果已保存到 class_distribution_summary.csv")