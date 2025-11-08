'''
我现在有一个目录，目录里面有若干CSV，命名格式如下：
0Test，...，9Test，
表头内容如下：
NOAA_AR	T_REC	NOAA_NUM	CLASS	TOTUSJH	TOTPOT	TOTUSJZ	ABSNJZH	SAVNCPP	USFLUX	AREA_ACR	MEANPOT	R_VALUE	SHRGT45

请你根据 NOAA_NUM字段，
d第一次筛选要求
便利每一个文件，把NOAA_NUM值为1的行，保存，不要改变原有数据的先后关系，因为我是时间序列数据，并且我的数据是40为一个窗口，不会出现NOAA_NUM不为1时候不是去掉40的倍数的行数的情况，
接下来对于上述保存的数据进行二次处理
要求：对于NOAA_AR	，如果数量不满足40为一个单位就舍弃，满足才保留，
最后然后把处理后的所有CSV重新保存到一个新目录下
'''
import os
import pandas as pd

# 输入和输出目录
input_dir = '../../data'  # 你的CSV文件所在的目录
output_dir_single = './data_fillter_single'  # 保存处理后的CSV文件的目录
output_dir_mutiple = './data_fillter_mutiple'  # 保存处理后的CSV文件的目录

# 创建输出目录（如果不存在）
if not os.path.exists(output_dir_single):
    os.makedirs(output_dir_single)
if not os.path.exists(output_dir_mutiple):
    os.makedirs(output_dir_mutiple)

# 列出输入目录中所有的CSV文件
csv_files = [f for f in os.listdir(input_dir) if f.endswith('Test.csv') and not f.startswith('-1')]


# 处理每个文件
for file_name in csv_files:
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir_single, file_name)

    # 读取CSV文件
    df = pd.read_csv(input_path, encoding='utf-8')
    # print(df.__len__())
    # 第一次筛选：只保留NOAA_NUM为1的行
    filtered_df = df[df['NOAA_NUM'] == 1]

    # 二次处理：检查每个NOAA_AR的行数是否为40的倍数
    final_df = pd.DataFrame()  # 用于存储最终要保存的数据
    for noaa_ar in filtered_df['NOAA_AR'].unique():#遍历所有唯一的NOAA_AR值去重
        subset = filtered_df[filtered_df['NOAA_AR'] == noaa_ar]#拿出每一个NOAA_AR 对应的行数所有数据
        if len(subset) % 40 == 0:
            final_df = pd.concat([final_df, subset], ignore_index=True)

    # 保存处理后的数据到新目录下
    final_df.to_csv(output_path, index=False)
    print(f"Processed and saved: {output_path}")


# # 处理每个文件
# for file_name in csv_files:
#     input_path = os.path.join(input_dir, file_name)
#     output_path = os.path.join(output_dir_mutiple, file_name)
#
#     # 读取CSV文件
#     df = pd.read_csv(input_path, encoding='utf-8')
#     # print(df.__len__())
#     # 第一次筛选：只保留NOAA_NUM不是1的行
#     filtered_df = df[df['NOAA_NUM'] != 1]
#
#     # 二次处理：检查每个NOAA_AR的行数是否为40的倍数
#     final_df = pd.DataFrame()  # 用于存储最终要保存的数据
#     for noaa_ar in filtered_df['NOAA_AR'].unique():#遍历所有唯一的NOAA_AR值去重
#         subset = filtered_df[filtered_df['NOAA_AR'] == noaa_ar]#拿出每一个NOAA_AR 对应的行数所有数据
#         if len(subset) % 40 == 0:
#             final_df = pd.concat([final_df, subset], ignore_index=True)
#
#     # 保存处理后的数据到新目录下
#     final_df.to_csv(output_path, index=False)
#     print(f"Processed and saved: {output_path}")


print("All files have been processed.")
