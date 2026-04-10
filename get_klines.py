import requests
import pandas as pd
from datetime import datetime
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_klines(symbol='BTCUSDT', interval='5m', start_time=None, end_time=None, limit=1500,
               timeout=30, max_retries=5, backoff_factor=1):
    """
    拉取币安合约K线数据

    参数:
    symbol: 交易对，默认BTCUSDT
    interval: 时间周期，如1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    start_time: 起始时间（datetime对象或时间戳毫秒）
    end_time: 结束时间（datetime对象或时间戳毫秒）
    limit: 每次请求数量，默认1500（最大值）
    timeout: 请求超时时间（秒），默认30秒
    max_retries: 最大重试次数，默认5次
    backoff_factor: 重试间隔倍数，默认1（重试间隔为 {backoff_factor} * (2 ** (重试次数 - 1)) 秒）
    """
    base_url = "https://fapi.binance.com"
    endpoint = "/fapi/v1/klines"

    # 配置重试策略
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],  # 对这些HTTP状态码进行重试
        allowed_methods=["GET"]
    )

    # 创建带有重试策略的session
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # 转换时间格式
    if isinstance(start_time, datetime):
        start_timestamp = int(start_time.timestamp() * 1000)
    else:
        start_timestamp = start_time

    if isinstance(end_time, datetime):
        end_timestamp = int(end_time.timestamp() * 1000)
    else:
        end_timestamp = end_time

    all_data = []

    print(f"开始拉取 {symbol} K线数据...")
    print(f"时间周期: {interval}")
    print(f"起始时间: {datetime.fromtimestamp(start_timestamp/1000)}")
    print(f"结束时间: {datetime.fromtimestamp(end_timestamp/1000)}\n")

    request_count = 0
    current_start = start_timestamp

    all_data_fetched = False  # 标志是否已获取所有数据

    while current_start < end_timestamp and not all_data_fetched:
        # 构建请求参数
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_timestamp,
            'limit': limit
        }

        retry_count = 0
        success = False

        while retry_count <= max_retries and not success:
            try:
                # 发送请求，使用session和timeout
                response = session.get(base_url + endpoint, params=params, timeout=timeout)
                response.raise_for_status()

                data = response.json()
                request_count += 1
                success = True

                if not data:
                    print("没有更多数据")
                    all_data_fetched = True
                    break

                # 获取时间范围
                first_time = datetime.fromtimestamp(data[0][0]/1000)
                last_time = datetime.fromtimestamp(data[-1][0]/1000)

                print(f"第 {request_count} 次请求: 获取 {len(data)} 条记录 [{first_time} 至 {last_time}]")

                all_data.extend(data)

                # 如果返回的数据少于limit，说明已经获取完所有数据
                if len(data) < limit:
                    print(f"\n已获取所有可用数据")
                    all_data_fetched = True
                    break

                # 设置下一次请求的起始时间为本次最后一根K线的收盘时间 + 1毫秒
                current_start = data[-1][6] + 1

                # 避免触发频率限制
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count <= max_retries:
                    wait_time = backoff_factor * (2 ** (retry_count - 1))
                    print(f"请求错误 (第 {retry_count}/{max_retries} 次重试): {e}")
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"请求失败，已达到最大重试次数: {e}")
                    all_data_fetched = True  # 失败时也退出主循环
                    break

        # 如果所有重试都失败，跳出主循环
        if not success:
            break

    print(f"\n总请求次数: {request_count}")
    print(f"总记录数: {len(all_data)}")

    return all_data

def save_klines_to_csv(data, symbol, interval, start_time, end_time):
    """
    将K线数据保存为CSV文件

    参数:
    data: K线数据列表
    symbol: 交易对
    interval: 时间周期
    start_time: 起始时间（datetime对象）
    end_time: 结束时间（datetime对象）
    """
    if not data:
        print("没有数据可保存")
        return

    # 定义列名
    columns = [
        'open_time',           # 开盘时间
        'open',                # 开盘价
        'high',                # 最高价
        'low',                 # 最低价
        'close',               # 收盘价
        'volume',              # 成交量
        'close_time',          # 收盘时间
        'quote_volume',        # 成交额
        'trades',              # 成交笔数
        'taker_buy_volume',    # 主动买入成交量
        'taker_buy_quote_volume',  # 主动买入成交额
        'ignore'               # 忽略
    ]

    # 转换为DataFrame
    df = pd.DataFrame(data, columns=columns)

    # 转换时间戳为可读格式
    df['open_datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_datetime'] = pd.to_datetime(df['close_time'], unit='ms')

    # 转换数值类型
    numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                       'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['trades'] = df['trades'].astype(int)

    # 重新排列列顺序（把时间列放在前面）
    columns_order = ['open_datetime', 'close_datetime', 'open', 'high', 'low', 'close',
                     'volume', 'quote_volume', 'trades',
                     'taker_buy_volume', 'taker_buy_quote_volume',
                     'open_time', 'close_time']
    df = df[columns_order]

    # 按开盘时间排序
    df = df.sort_values('open_time')

    # 去重（以open_time为准）
    df = df.drop_duplicates(subset=['open_time'], keep='first')

    # 生成文件名：标的_周期_起始时间_结束时间.csv
    start_str = start_time.strftime('%Y%m%d_%H%M%S')
    end_str = end_time.strftime('%Y%m%d_%H%M%S')
    filename = f"{symbol}_{interval}_{start_str}_{end_str}.csv"

    # 保存为CSV
    df.to_csv(filename, index=False, encoding='utf-8-sig')

    print(f"\n数据已保存到: {filename}")
    print(f"总记录数: {len(df)}")
    print(f"时间范围: {df['open_datetime'].min()} 到 {df['close_datetime'].max()}")
    print(f"\n数据预览:")
    print(df.head())
    print("\n价格统计:")
    print(df[['open', 'high', 'low', 'close', 'volume']].describe())

    return filename

if __name__ == "__main__":
    # 配置参数
    symbol = 'DOGEUSDT'          # 交易对
    interval = '5m'             # K线周期：1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

    # 设置时间范围（示例：最近7天）
    end_time = datetime.now()
    start_time = datetime(2024, 8, 7, 0, 0, 0)  # 可以自定义具体日期

    # 或者使用相对时间
    # from datetime import timedelta
    # start_time = end_time - timedelta(days=7)

    # 拉取K线数据
    kline_data = get_klines(
        symbol=symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
        limit=1500
    )

    # 保存为CSV
    if kline_data:
        save_klines_to_csv(kline_data, symbol, interval, start_time, end_time)
