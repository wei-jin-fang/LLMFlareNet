import requests
from datetime import datetime, timedelta
import uuid
import pymysql


def download_cmcc_forecast_by_day(download_time: datetime):
    download_resp = None
    try:
        session = requests.Session()
        session.trust_env = False
        download_resp = requests.post(
            f"https://iswa.ccmc.gsfc.nasa.gov/IswaSystemWebApp/SolarFlareScoreboardServlet?issuetimestamp={download_time.strftime('%Y-%m-%d')}%2000:00:00&offset=0",
            timeout=300, )
        if download_resp.status_code != 200:
            print(f"下载错误，错误码：{download_resp.status_code}")
            return download_resp.status_code
        resp_json = download_resp.json()
        # 先处理全日面的预报数据
        # 检查是否已经存在相同记录的 SQL 语句
        check_query = """
            SELECT COUNT(*) FROM ccmc_solar_flare_prediction 
            WHERE model = %s AND window_start_times = %s
        """

        # 插入数据的 SQL 语句
        insert_query = """
            INSERT INTO ccmc_solar_flare_prediction (
                id, model, window_start_times, window_end_times, 
                cplus_probabilities, mplus_probabilities, 
                c_probabilities, m_probabilities, x_probabilities
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        # 数据库连接
        connection = pymysql.connect(
            host='localhost',  # 数据库地址
            user='root',  # 数据库用户名
            password='123456',  # 数据库密码
            database='ccmc_flare_prediction'  # 数据库名称
        )
        with connection.cursor() as cursor:
            full_disk_predictions = resp_json['full_disk_predictions']
            for full_disk_prediction in full_disk_predictions:
                for index, _ in enumerate(full_disk_prediction['window_start_times']):
                    # 提取model和start_time
                    model = full_disk_prediction['model']
                    window_start_time = full_disk_prediction['window_start_times'][index]

                    # 查询是否已存在相同的记录
                    cursor.execute(check_query, (model, window_start_time))
                    result = cursor.fetchone()

                    # 如果没有相同记录，才执行插入操作
                    if result[0] == 0:
                        data = [
                            str(uuid.uuid4()),  # id
                            model,  # model
                            window_start_time,  # window_start_times
                            full_disk_prediction['window_end_times'][index],  # window_end_times
                            full_disk_prediction.get('cplus_probabilities', [None])[index],  # cplus_probabilities
                            full_disk_prediction.get('mplus_probabilities', [None])[index],  # mplus_probabilities
                            full_disk_prediction.get('c_probabilities', [None])[index],  # c_probabilities
                            full_disk_prediction.get('m_probabilities', [None])[index],  # m_probabilities
                            full_disk_prediction.get('x_probabilities', [None])[index]  # x_probabilities
                        ]
                        # 执行插入操作
                        cursor.execute(insert_query, data)

        # 提交更改
        connection.commit()
        return 200
    except Exception as e:
        print(f"Error: {e}")
        connection.rollback()
        return -999
    finally:
        if download_resp is not None:
            download_resp.close()


def download_cmcc_AR_forecast_by_day(download_time: datetime):
    download_resp = None
    try:
        session = requests.Session()
        session.trust_env = False
        download_resp = requests.post(
            f"https://iswa.ccmc.gsfc.nasa.gov/IswaSystemWebApp/SolarFlareScoreboardServlet?issuetimestamp={download_time.strftime('%Y-%m-%d')}%2000:00:00&offset=0",
            timeout=300
        )
        if download_resp.status_code != 200:
            print(f"下载错误，错误码：{download_resp.status_code}")
            return download_resp.status_code
        resp_json = download_resp.json()
        # 处理活动区数据
        connection = pymysql.connect(
            host='localhost',  # 数据库地址
            user='root',  # 数据库用户名
            password='123456',  # 数据库密码
            database='ccmc_flare_prediction'  # 数据库名称
        )

        # 检查是否已经存在相同记录的 SQL 语句
        check_query = """
            SELECT COUNT(*) FROM ccmc_ar_solar_flare_forecast 
            WHERE hasNOAAMeta = %s 
            AND model = %s 
            AND issue_time = %s 
            AND prediction_window_start = %s
        """

        # 插入数据的 SQL 语句
        insert_query = """
            INSERT INTO ccmc_ar_solar_flare_forecast (
                id, noaaID, lat, lon, hasNOAAMeta, model, 
                 prediction_window_start, 
                prediction_window_end, issue_time, CPlus, 
                MPlus, C, M, X
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        with connection.cursor() as cursor:
            # 处理活动区数据
            smartActiveRegions = resp_json['smartActiveRegions']
            for smartActiveRegion in smartActiveRegions:
                if smartActiveRegion['hasNOAAMeta'] is True:
                    for prediction in smartActiveRegion['predictions']:
                        # 检查记录是否已存在
                        cursor.execute(check_query, (
                            str(smartActiveRegion['hasNOAAMeta']),  # hasNOAAMeta
                            prediction['model'],  # model
                            prediction['issue_time'],  # issue_time
                            prediction['prediction_window_start']  # prediction_window_start
                        ))
                        result = cursor.fetchone()

                        # 如果不存在相同的记录，则插入数据
                        if result[0] == 0:
                            data = [
                                str(uuid.uuid4()),  # id
                                smartActiveRegion['noaaID'],  # noaaID
                                smartActiveRegion['lat'],  # lat  # 负的代表S，正的代表N
                                smartActiveRegion['lon'],  # lon  # 负的代表E，正的代表W
                                str(smartActiveRegion['hasNOAAMeta']),  # hasNOAAMeta
                                prediction['model'],  # model
                                prediction['prediction_window_start'],  # prediction_window_start
                                prediction['prediction_window_end'],  # prediction_window_end
                                prediction['issue_time'],  # issue_time
                                None if prediction['CPlus'] == 'none' else prediction['CPlus'],  # CPlus
                                None if prediction['MPlus'] == 'none' else prediction['MPlus'],  # MPlus
                                None if prediction['C'] == 'none' else prediction['C'],  # C
                                None if prediction['M'] == 'none' else prediction['M'],  # M
                                None if prediction['X'] == 'none' else prediction['X'],  # X
                            ]
                            # print(data)
                            # 执行插入操作
                            cursor.execute(insert_query, data)
            connection.commit()

        return 200
    except Exception as e:
        print(f"Error: {e}")
        connection.rollback()
        print(e)
        return -999
    finally:
        if download_resp is not None:
            download_resp.close()


if __name__ == '__main__':
    # start_time = datetime.utcnow()
    start_time = datetime.strptime('2022-04-03', "%Y-%m-%d")
    while start_time >= datetime.strptime('2021-05-01', "%Y-%m-%d"):
        print(start_time)
        code = download_cmcc_forecast_by_day(start_time)
        if code != 200:
            continue

        # code_1 = download_cmcc_AR_forecast_by_day(start_time)
        # if code_1 != 200:
        #     continue

        start_time = start_time - timedelta(days=1)
        # break
