import csv
import os

import matplotlib
from matplotlib import pyplot as plt, ticker

from tools import truncate, Metric

matplotlib.use('TkAgg')
import numpy as np
import pandas as pd

def duiqiOneandMPLus(JianceType,mode_type,comparewith,kind):
    '''
    我现在有两个csv
    第一个是average_probabilities_path = rf"./result_forecast.csv_{JianceType}_cleaned.csv"
    第二个是SC_waitcompare.csv
    你需要拿出第一个csv的T_REC与noaaID列这两个字段和第二个表的prediction_date和noaaID列进行综合对比，
    只有某一行这两个字段均相等后，提取出第一个表这一行的Class字段数值和CV_average。提取出来第二个表的MPlus 字段，与他们共同的T_REC与noaaID拼成一行
    保存下来到一个新的csv
    '''
    '''
    Modified function to:
    1. Calculate NOAA_NUMS based on comma-separated values in NOAA_ARS
    2. Filter for NOAA_NUMS == 1 if kind is 'single'
    3. Merge result_forecast.csv with ccmc_waitcompare.csv based on predictday and noaaID
    4. Extract Class, CV_average from first CSV and MPlus from second CSV
    5. Add y_true column based on Class
    6. Save to a new CSV
    '''
    pre_path_name = f"{mode_type}_{comparewith}_{kind}"
    if not os.path.exists(pre_path_name):
        os.makedirs(pre_path_name)

    average_probabilities_path = f"./split_result_csvs/{mode_type}.csv"
    df1 = pd.read_csv(average_probabilities_path)

    df2=None
    if comparewith=="CCMC":
        sc_waitcompare_path = './compare_data/ccmc_waitcompare.csv'
        df2 = pd.read_csv(sc_waitcompare_path)
    elif comparewith=="SR":
        sc_waitcompare_path = './compare_data/sc_waitcompare.csv'
        df2 = pd.read_csv(sc_waitcompare_path)

    # Calculate NOAA_NUMS based on comma-separated values in NOAA_ARS
    df1['NOAA_NUMS'] = df1['NOAA_ARS'].apply(lambda x: len(str(x).split(',')))

    # If kind is 'single', filter for NOAA_NUMS == 1
    if kind == 'single':
        df1 = df1[df1['NOAA_NUMS'] == 1]
    elif kind == 'multiple':
        df1 = df1[df1['NOAA_NUMS'] != 1]
    elif kind == 'mixed':
        df1 = df1

    # Select and rename columns for df1
    df1 = df1[['predictday', 'new_noaaid', 'Class', 'CV_average', 'NOAA_NUMS']]  # Include NOAA_NUMS
    df1.rename(columns={'predictday': 'predictday', 'new_noaaid': 'noaaID'}, inplace=True)

    # Select and rename columns for df2
    if comparewith=="CCMC":
        df2 = df2[['prediction_date', 'noaaID', 'MPlus_avg']]  # Extract necessary columns
        df2.rename(columns={'prediction_date': 'predictday', 'MPlus_avg': 'MPlus'}, inplace=True)
    elif comparewith=="SR":
        df2 = df2[['prediction_date', 'noaaID', 'MPlus']]  # Extract necessary columns
        df2.rename(columns={'prediction_date': 'predictday', 'MPlus': 'MPlus'}, inplace=True)


    # Merge the two dataframes on predictday and noaaID
    merged_df = pd.merge(df1, df2, on=['predictday', 'noaaID'], how='inner')

    # Save merged result to CSV
    output_path = f"{pre_path_name}/merger_{mode_type}_{JianceType}_conpare_with_{comparewith}.csv"
    merged_df.to_csv(output_path, index=False)

    # Read the saved file
    combined_df = pd.read_csv(output_path)

    # Add y_true column based on Class
    combined_df['y_true'] = combined_df['Class'].apply(lambda x: 0 if x in ['N', 'C'] else 1)

    # Save the updated DataFrame with y_true to the same file
    combined_df.to_csv(output_path, index=False)
    print(combined_df)
    print(f"文件已更新，添加 y_true 列，并保存至 '{output_path}'")
    return pre_path_name,output_path

def drawTSSPlot(Path,threshold_name,JinaceType,model_type,pre_path_name):
    # 设置步长
    step_size = 0.05;start = 0.05
    # 使用for循环从0开始，增加至1（包含1）
    step_values = []
    for i in range(int(1 / step_size) + 1):  # +1 是为了确保1被包括进去
        step_value = start + i * step_size
        if step_value > 0.96:
            break  # 避免由于浮点精度问题超过1的情况
        step_values.append(step_value)

    # 遍历这些步长
    tssList=[]
    for threshold in step_values:
        def getTssFromFile(path, threshold, threshold_name):
            # 遍历目录中的所有CSV文件
            data_TSS = []
            true_labels = []
            predicted_labels = []
            # 读取CSV文件
            df = pd.read_csv(path, encoding='ISO-8859-1')
            # 获取真实标签
            df['y_true'] = df['y_true'].astype(int)
            # 获取真实标签
            true_labels.extend(df['y_true'].tolist())
            # 计算预测标签
            predicted = (df[threshold_name] > threshold).astype(int)  # 如果'one'列的值大于阈值，则预测为1，否则为0
            predicted_labels.extend(predicted.tolist())

            # 你现在有两个列表：true_labels 和 predicted_labels，分别包含所有CSV文件中的真实标签和预测标签

            metric = Metric(true_labels, predicted_labels)
            Tss = metric.TSS()[0]
            data_TSS.append(Tss)
            data_TSS = np.array(data_TSS)
            TSS_mean = data_TSS.mean(axis=0) #反正就一次取不取平均无所谓，这个地方写了是因为兼容之前的
            return TSS_mean
        # 指定包含CSV文件的目录
        thisThresholdTss = getTssFromFile(Path,threshold,threshold_name)
        print(thisThresholdTss)
        tssList.append(truncate(thisThresholdTss, 3))
    # 保存阈值和TSS数据到CSV文件
    csv_filename = fr'{pre_path_name}/all_threshold_{threshold_name}_of_{model_type}_{JinaceType}.csv'

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Threshold', 'TSS'])
        writer.writerows(zip(step_values, tssList))
    # 创建图表
    plt.figure(figsize=(10, 5))  # 可以调整图表大小
    plt.plot(step_values, tssList, marker='o')  # 使用圆点标记每个数据点

    # 设置图表的标题和轴标签
    plt.title(f'all_threshold_{threshold_name}_of_{model_type}_{JinaceType}')
    plt.xlabel('Threshold')
    plt.ylabel('TSS')

    # 设置横纵坐标的显示范围
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # 设置X轴和Y轴的刻度间隔为0.1
    ax = plt.gca()  # 获取当前轴对象
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))  # 设置X轴主刻度间隔
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))  # 设置Y轴主刻度间隔

    # 显示网格
    plt.grid(True)

    # 找到TSS值最大的点，并标记为红色，同时显示横纵坐标
    max_index = np.argmax(tssList)  # 获取最大TSS值的索引
    max_tss = tssList[max_index]  # 获取最大TSS值
    max_threshold = step_values[max_index]  # 获取对应的阈值
    plt.plot(max_threshold, max_tss, 'ro')  # 将最大的点标记为红色
    plt.text(max_threshold, max_tss, f'({truncate(max_threshold, 3):.3f}, {max_tss:.2f})', ha='left', va='bottom')
    # 显示图表
    # plt.show()
    # 保存图表到文件
    plt.savefig(fr'{pre_path_name}/all_threshold_{threshold_name}_of_{model_type}_{JinaceType}.png', dpi=300)  # 指定分辨率为300 DPI
    return max_threshold

def getBestTssMetrics(merge_path,threshold,threshold_name,JianceType,model_type,pre_path_name):

    results = []  # 存储所有结果的列表
    results_excel = []  # 存储所有结果的列表
    for threshold in [threshold]:
        # 获取当前阈值下的所有指标
        def getMetricsFromFile(path, threshold, threshold_name):
            # 读取CSV文件
            df = pd.read_csv(path, encoding='ISO-8859-1')
            df['y_true'] = df['y_true'].astype(int)  # 将真实标签转换为整数类型

            # 提取真实标签和预测标签
            true_labels = df['y_true'].tolist()
            predicted = (df[f'{threshold_name}'] > threshold).astype(int)  # 根据阈值计算预测标签
            predicted_labels = predicted.tolist()

            # 初始化 Metric 类，用于计算各项指标
            metric = Metric(true_labels, predicted_labels)
            print(metric.Matrix())

            # 计算每个指标，确保每个指标为包含类别 [0] 和 [1] 的列表
            metrics = {
                "Accuracy": [metric.Accuracy()[0], metric.Accuracy()[1]],  # 准确率
                "Recall": [metric.Recall()[0], metric.Recall()[1]],  # 召回率
                "Precision": [metric.Precision()[0], metric.Precision()[1]],  # 精确率
                "TSS": [metric.TSS()[0], metric.TSS()[1]],  # 真负率减去假正率
                # "BSS": [metric.BSS()[0], metric.BSS()[1]],                 # Brier技能得分
                "HSS": [metric.HSS()[0], metric.HSS()[1]],  # Heidke技能得分
                "FAR": [metric.FAR()[0], metric.FAR()[1]],  # 虚警率
                "FPR": [metric.FPR()[0], metric.FPR()[1]]  # 假正率
            }

            return metrics
        metrics = getMetricsFromFile(merge_path, threshold,threshold_name)
        for metric_name, metric_value in metrics.items():
            # 将每个指标的类别 [0] 和 [1] 值添加到结果列表中
            results.append([threshold, metric_name,     truncate(metric_value[0], 3),     truncate(metric_value[1], 3)])
            results_excel.append([metric_name,     truncate(metric_value[0], 3),     truncate(metric_value[1], 3)])
    # 创建包含指定结构的 DataFrame
    df_results = pd.DataFrame(results,
                              columns=["Threshold", "Metric Name/BS and BSS ", "Class 0 Value", "Class 1 Value"])
    print(df_results)
    # 保存为CSV文件
    output_path = f"{pre_path_name}/best_threshold_{threshold_name}_of_{model_type}_metrics_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"结果已保存到 {output_path}")

    # 创建包含指定结构的 DataFrame
    df_results = pd.DataFrame(results_excel, columns=["Metric", "负类", "正类"])
    # 保存为 Excel 文件
    output_excel_path = f"{pre_path_name}/best_threshold_{threshold_name}_of_{model_type}_metrics_results.xlsx"
    df_results.to_excel(output_excel_path, index=False, engine='openpyxl')  # 使用 openpyxl 保存为 .xlsx 格式
if __name__ == '__main__':
    '''
        对于模型Onefitall_16 compare with CCMC 
    '''
    model_type="Onefitall_16"
    pre_path_name,merge_path=duiqiOneandMPLus("TSS",model_type,"CCMC","mixed")
    max_one_index= drawTSSPlot(merge_path,"CV_average","TSS",model_type,pre_path_name)
    getBestTssMetrics(merge_path,max_one_index,"CV_average","TSS",model_type,pre_path_name)
    max_MPlus_index= drawTSSPlot(merge_path,"MPlus","TSS",model_type,pre_path_name)
    getBestTssMetrics(merge_path,max_MPlus_index,"MPlus","TSS",model_type,pre_path_name)


    model_type="Onefitall_16"
    pre_path_name,merge_path=duiqiOneandMPLus("TSS",model_type,"CCMC","single")
    max_one_index= drawTSSPlot(merge_path,"CV_average","TSS",model_type,pre_path_name)
    getBestTssMetrics(merge_path,max_one_index,"CV_average","TSS",model_type,pre_path_name)
    max_MPlus_index= drawTSSPlot(merge_path,"MPlus","TSS",model_type,pre_path_name)
    getBestTssMetrics(merge_path,max_MPlus_index,"MPlus","TSS",model_type,pre_path_name)
    #

    '''
        对于模型Onefitall_16 compare with SR 
    '''
    model_type="Onefitall_16"
    pre_path_name,merge_path=duiqiOneandMPLus("TSS",model_type,"SR","mixed")
    max_one_index= drawTSSPlot(merge_path,"CV_average","TSS",model_type,pre_path_name)
    getBestTssMetrics(merge_path,max_one_index,"CV_average","TSS",model_type,pre_path_name)
    max_MPlus_index= drawTSSPlot(merge_path,"MPlus","TSS",model_type,pre_path_name)
    getBestTssMetrics(merge_path,max_MPlus_index,"MPlus","TSS",model_type,pre_path_name)

    model_type="Onefitall_16"
    pre_path_name,merge_path=duiqiOneandMPLus("TSS",model_type,"SR","single")
    max_one_index= drawTSSPlot(merge_path,"CV_average","TSS",model_type,pre_path_name)
    getBestTssMetrics(merge_path,max_one_index,"CV_average","TSS",model_type,pre_path_name)
    max_MPlus_index= drawTSSPlot(merge_path,"MPlus","TSS",model_type,pre_path_name)
    getBestTssMetrics(merge_path,max_MPlus_index,"MPlus","TSS",model_type,pre_path_name)

