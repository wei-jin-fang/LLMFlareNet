'''
df = pd.read_csv('G:\本科\SHARP_23_24\ccmc_sorted.csv')
我现在有个csv，里面表头是这样的
id	noaaID	lat	lon	hasNOAAMeta	models	prediction_window_start	prediction_window_end	issue_time	CPlus	MPlus	C	M	X
其中一行数据格式如下
 ac8f50ee-e3bd-47a2-bcce-780c24382ac8	13180	19	-56	TRUE	AMOS_v1	2023/1/1 0:00	2023/1/2 0:00	2023/1/1 0:30	0.253199995	0.0317			0.0012
数据中每一列可能都有空单元格

请你第一步
遍历每一行，对于每一行的MPlus 进行判断，如果是空或者0，那么就把M列和X列相加
如果M列和X列两个都是空或者0，那么Mplus就是0
'''

import pandas as pd
import numpy as np
# 目前暂定不使用相加
'''
# # 定义函数计算 MPlus
# def calculate_mplus(row):
#     if pd.isna(row['MPlus']) or row['MPlus'] == 0:
#         m_value = row['M'] if not pd.isna(row['M']) else 0
#         x_value = row['X'] if not pd.isna(row['X']) else 0
#         if m_value == 0 and x_value == 0:
#             return 0
#         else:
#             return m_value + x_value
#     return row['MPlus']
#
# # 更新 MPlus 列
# df['MPlus'] = df.apply(calculate_mplus, axis=1)
'''
# 读取 CSV 文件
df = pd.read_csv(r'./compare_data/ccmc_ar_solar_flare_forecast.csv')

# 保留数值为0的,保留数值不为nan的
df = df[(df['MPlus'].notna()) & (df['MPlus'] >=0)]

# 转换 prediction_window_start 为日期格式并生成新列 prediction_date
df['prediction_date'] = pd.to_datetime(df['prediction_window_start']).dt.strftime('%Y%m%d')

# 对 prediction_date 和 noaaID 分组，计算 MPlus 的平均值
df['MPlus_avg'] = df.groupby(['prediction_date', 'noaaID'])['MPlus'].transform('mean')
df['MPlus_avg'] = np.trunc(df['MPlus_avg'] * 100000) / 100000

# 根据 prediction_date 和 noaaID 分组，保留分组内的第一行
df = df.drop_duplicates(subset=['prediction_date', 'noaaID'])

# 保存处理后的数据到新的 CSV 文件
output_path = r'./compare_data/ccmc_waitcompare.csv'
df.to_csv(output_path, index=False)
print(f"修改后的文件已保存到: {output_path}")


import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('./compare_data/scientific_reports_solar_flares_predictions.csv')

# Step 1: 转换 prediction_date 字段为指定格式
df['prediction_date'] = pd.to_datetime(df['prediction_date'], errors='coerce').dt.strftime('%Y%m%d')

# Step 2: 提取 noaaID 字段的数据部分（假设要提取的是数字部分）
df['noaaID'] = df['noaaID'].str.extract('(\d+)')
print(df[0:5])
# Step 3: 转换 MPlus 列中的百分数为小数，并丢弃 N/A 行
df = df[df['MPlus'] != 'N/A']  # 丢弃 N/A 行
df = df.dropna(subset=['MPlus'])  # 丢弃 MPlus 列为空的行
df['MPlus'] = df['MPlus'].str.rstrip('%').astype(float) / 100  # 转换百分数为小数

# 保存处理后的数据为新的 CSV 文件
df.to_csv('./compare_data/sc_waitcompare.csv', index=False)
