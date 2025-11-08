# https://nature.njit.edu/solardb/api/db/get_flares_predictions_tool.php
import pandas as pd
import requests


def get_historical_forecast_data():
    # 定义请求的URL和表单数据
    url = "https://nature.njit.edu/solardb/api/db/get_flares_predictions_tool.php"
    data = {
        "time_window": "24"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    # 发起POST请求
    history_response = requests.post(url, data=data, verify=False, headers=headers)
    history_json_records = history_response.json()["records"]  # list
    for history_json_subrecords in history_json_records:
        df = pd.DataFrame(history_json_subrecords)
        # 保存为CSV文件
        csv_filename = './scientific_reports_solar_flares_predictions.csv'
        df.to_csv(csv_filename, index=False, mode='a', header=False)


if __name__ == '__main__':
    get_historical_forecast_data()
