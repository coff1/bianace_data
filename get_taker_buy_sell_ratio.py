import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def get_taker_buy_sell_ratio(symbol='BTCUSDT', period='5m', limit=500):
    """
    拉取币安合约主动买卖量历史数据（从最新往前追溯30天）

    参数:
    symbol: 交易对，默认BTCUSDT
    period: 时间周期，默认5m（最大精确度）
           可选: "5m","15m","30m","1h","2h","4h","6h","12h","1d"
    limit: 每次请求数量，默认500（最大值）
    """
    base_url = "https://fapi.binance.com"
    endpoint = "/futures/data/takerlongshortRatio"

    all_data = []

    # 计算目标时间范围（30天）
    now = datetime.now()
    target_start_time = int((now - timedelta(days=30)).timestamp() * 1000)

    print(f"开始拉取 {symbol} 合约主动买卖量历史数据...")
    print(f"时间周期: {period}")
    print(f"目标时间范围: 最近30天")
    print(f"策略: 从最新数据往前追溯\n")

    request_count = 0
    current_end_time = None  # 第一次请求不设置endTime，获取最新数据

    while True:
        # 构建请求参数
        params = {
            'symbol': symbol,
            'period': period,
            'limit': limit
        }

        # 如果有endTime，则添加（用于往前追溯）
        if current_end_time:
            params['endTime'] = current_end_time

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
            first_ts = int(data[0]['timestamp'])
            last_ts = int(data[-1]['timestamp'])
            first_time = datetime.fromtimestamp(first_ts/1000)
            last_time = datetime.fromtimestamp(last_ts/1000)

            print(f"第 {request_count} 次请求: 获取 {len(data)} 条记录 [{first_time} 至 {last_time}]")

            all_data.extend(data)

            # 如果最早的数据已经达到或超过30天前，停止
            if first_ts <= target_start_time:
                print(f"\n已达到30天目标，停止获取")
                break

            # 如果返回的数据少于limit，说明已经没有更多数据了
            if len(data) < limit:
                print(f"\n已获取所有可用数据")
                break

            # 设置下一次请求的endTime为本次最早记录的时间戳 - 1
            current_end_time = first_ts - 1

            # 避免触发频率限制 (1000次/5分钟)
            time.sleep(0.35)

        except requests.exceptions.RequestException as e:
            print(f"请求错误: {e}")
            break

    print(f"\n总请求次数: {request_count}")
    print(f"总记录数: {len(all_data)}")

    return all_data

def save_to_csv(data, filename='BTC合约主动买卖量.csv'):
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
    df['timestamp'] = df['timestamp'].astype(int)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 转换数值列为float类型
    df['buySellRatio'] = df['buySellRatio'].astype(float)
    df['buyVol'] = df['buyVol'].astype(float)
    df['sellVol'] = df['sellVol'].astype(float)

    # 重新排列列顺序
    columns = ['datetime', 'buySellRatio', 'buyVol', 'sellVol', 'timestamp']
    df = df[columns]

    # 按时间排序
    df = df.sort_values('timestamp')

    # 去重（以timestamp为准）
    df = df.drop_duplicates(subset=['timestamp'], keep='first')

    # 保存为CSV
    df.to_csv(filename, index=False, encoding='utf-8-sig')

    print(f"\n数据已保存到: {filename}")
    print(f"总记录数: {len(df)}")
    print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    print(f"\n数据预览:")
    print(df.head())
    print("\n统计信息:")
    print(df[['buySellRatio', 'buyVol', 'sellVol']].describe())

if __name__ == "__main__":
    # 拉取数据（使用5分钟周期，最高精确度）
    data = get_taker_buy_sell_ratio(symbol='BTCUSDT', period='5m', limit=500)

    # 保存为CSV
    save_to_csv(data, 'BTC合约主动买卖量.csv')
