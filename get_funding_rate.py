import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def get_funding_rate_history(symbol='BTCUSDT', days=1200, limit=1000):
    """
    拉取币安合约资金费率历史数据（从指定天数前开始往后拉取）

    参数:
    symbol: 交易对，默认BTCUSDT
    days: 拉取天数，默认600天
    limit: 每次请求数量，默认1000（最大值）
    """
    base_url = "https://fapi.binance.com"
    endpoint = "/fapi/v1/fundingRate"

    all_data = []

    # 计算目标时间范围
    now = datetime.now()
    start_time = int((now - timedelta(days=days)).timestamp() * 1000)
    end_time = int(now.timestamp() * 1000)

    print(f"开始拉取 {symbol} 资金费率历史数据...")
    print(f"目标时间范围: 最近{days}天")
    print(f"起始时间: {datetime.fromtimestamp(start_time/1000)}")
    print(f"结束时间: {datetime.fromtimestamp(end_time/1000)}")
    print(f"策略: 从历史开始往后拉取\n")

    request_count = 0
    current_start_time = start_time

    while current_start_time < end_time:
        # 构建请求参数
        params = {
            'symbol': symbol,
            'startTime': current_start_time,
            'endTime': end_time,
            'limit': limit
        }

        try:
            # 发送请求
            response = requests.get(base_url + endpoint, params=params)
            response.raise_for_status()

            data = response.json()
            request_count += 1

            if not data:
                print("没有更多数据")
                break

            # 获取时间范围
            first_ts = data[0]['fundingTime']
            last_ts = data[-1]['fundingTime']
            first_time = datetime.fromtimestamp(first_ts/1000)
            last_time = datetime.fromtimestamp(last_ts/1000)

            print(f"第 {request_count} 次请求: 获取 {len(data)} 条记录 [{first_time} 至 {last_time}]")

            all_data.extend(data)

            # 如果返回的数据少于limit，说明已经获取完所有数据
            if len(data) < limit:
                print(f"\n已获取所有可用数据")
                break

            # 设置下一次请求的startTime为本次最后一条记录的时间戳 + 1
            current_start_time = last_ts + 1

            # 避免触发频率限制 (和GET /fapi/v1/fundingInfo共享500/5min/IP)
            time.sleep(0.65)

        except requests.exceptions.RequestException as e:
            print(f"请求错误: {e}")
            break

    print(f"\n总请求次数: {request_count}")
    print(f"总记录数: {len(all_data)}")

    return all_data

def save_to_csv(data, filename='BTC资金费率历史.csv'):
    """
    将数据保存为CSV文件

    参数:
    data: 数据列表
    filename: 保存的文件名
    """
    if not data:
        print("没有数据可保存")
        return

    # 转换为DataFrame
    df = pd.DataFrame(data)

    # 转换时间戳为可读格式
    df['datetime'] = pd.to_datetime(df['fundingTime'], unit='ms')

    # 转换数据类型
    df['fundingRate'] = df['fundingRate'].astype(float)
    df['markPrice'] = df['markPrice'].astype(float)

    # 重新排列列顺序
    columns = ['datetime', 'symbol', 'fundingRate', 'markPrice', 'fundingTime']
    df = df[columns]

    # 按时间排序
    df = df.sort_values('fundingTime')

    # 去重（以fundingTime为准）
    df = df.drop_duplicates(subset=['fundingTime'], keep='first')

    # 保存为CSV
    df.to_csv(filename, index=False, encoding='utf-8-sig')

    print(f"\n数据已保存到: {filename}")
    print(f"总记录数: {len(df)}")
    print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    print(f"\n数据预览:")
    print(df.head())
    print("\n统计信息:")
    print(df[['fundingRate', 'markPrice']].describe())
    print(f"\n平均资金费率: {df['fundingRate'].mean():.6%}")
    print(f"最大资金费率: {df['fundingRate'].max():.6%}")
    print(f"最小资金费率: {df['fundingRate'].min():.6%}")

if __name__ == "__main__":
    # 拉取数据（拉取最近600天，使用最大limit=1000）
    data = get_funding_rate_history(symbol='BTCUSDT', days=600, limit=1000)

    # 保存为CSV
    save_to_csv(data, 'BTC资金费率历史.csv')
