import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

def get_basis_history(pair='BTCUSDT', contract_type='PERPETUAL', period='5m', limit=500):
    """
    拉取币安期货基差历史数据（从最新往前追溯30天）

    参数:
    pair: 交易对，默认BTCUSDT
    contract_type: 合约类型，PERPETUAL(永续), CURRENT_QUARTER(当季), NEXT_QUARTER(次季)
    period: 时间周期，默认5m（最高精度）
    limit: 每次请求数量，默认500（最大值）
    """
    base_url = "https://fapi.binance.com"
    endpoint = "/futures/data/basis"

    all_data = []

    # 计算目标时间范围（30天）
    now = datetime.now()
    target_start_time = int((now - timedelta(days=30)).timestamp() * 1000)

    print(f"开始拉取 {pair} 期货基差数据...")
    print(f"合约类型: {contract_type}")
    print(f"时间周期: {period}")
    print(f"目标时间范围: 最近30天")
    print(f"策略: 从最新数据往前追溯\n")

    request_count = 0

    # 第一次请求：不带时间参数，获取最新的数据
    print("步骤1: 获取最新数据...")
    params = {
        'pair': pair,
        'contractType': contract_type,
        'period': period,
        'limit': limit
    }

    try:
        response = requests.get(base_url + endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        request_count += 1

        if not data:
            print("没有可用数据")
            return []

        timestamps = [item['timestamp'] for item in data]
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        min_time = datetime.fromtimestamp(min_ts/1000)
        max_time = datetime.fromtimestamp(max_ts/1000)

        print(f"第 {request_count} 次请求: 获取 {len(data)} 条记录 [{min_time} 至 {max_time}]")
        all_data.extend(data)

        # 如果第一次请求就返回少于 limit 条，说明所有数据都获取了
        if len(data) < limit:
            print(f"\n所有数据已获取完毕（首次请求仅返回 {len(data)} 条）")
            return all_data

        # 否则，继续往前追溯历史数据
        print("\n步骤2: 向前追溯历史数据...")
        current_end_time = min_ts - 1
        print(f"设置 endTime = {current_end_time} ({datetime.fromtimestamp(current_end_time/1000)})")

        while True:
            params = {
                'pair': pair,
                'contractType': contract_type,
                'period': period,
                'limit': limit,
                'endTime': current_end_time
            }

            response = requests.get(base_url + endpoint, params=params)
            print(f"请求URL: {response.url}")
            response.raise_for_status()
            data = response.json()
            request_count += 1

            if not data:
                print("没有更多历史数据")
                break

            timestamps = [item['timestamp'] for item in data]
            min_ts_new = min(timestamps)
            max_ts_new = max(timestamps)
            min_time = datetime.fromtimestamp(min_ts_new/1000)
            max_time = datetime.fromtimestamp(max_ts_new/1000)

            print(f"第 {request_count} 次请求: 获取 {len(data)} 条记录 [{min_time} 至 {max_time}]")
            print(f"  对比: endTime={datetime.fromtimestamp(current_end_time/1000)}, 返回max_ts={max_time}")

            # 检查是否有新数据（防止死循环）
            # 如果返回的最大时间戳 >= endTime，说明API没有返回更早的数据
            if max_ts_new >= current_end_time:
                print(f"\n数据未向前推进（max_ts {max_ts_new} >= endTime {current_end_time}），已到达数据起点")
                break

            all_data.extend(data)

            # 如果已经达到30天目标
            if min_ts_new <= target_start_time:
                print(f"\n已达到30天目标")
                break

            # 如果返回数据少于 limit，说明已经到达数据起点
            if len(data) < limit:
                print(f"\n已到达数据起点（返回 {len(data)} < {limit}）")
                break

            # 继续向前追溯
            current_end_time = min_ts_new - 1
            print(f"设置下次 endTime = {current_end_time} ({datetime.fromtimestamp(current_end_time/1000)})\n")
            time.sleep(0.35)

    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return all_data

    print(f"\n总请求次数: {request_count}")
    print(f"总记录数: {len(all_data)}")

    return all_data

def save_to_csv(data, filename):
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
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 重新排列列顺序
    columns = ['datetime', 'pair', 'contractType', 'indexPrice', 'futuresPrice',
               'basis', 'basisRate', 'annualizedBasisRate', 'timestamp']
    df = df[columns]

    # 按时间排序
    df = df.sort_values('timestamp')

    # 去重（以timestamp为准）
    df = df.drop_duplicates(subset=['timestamp'], keep='first')

    # 转换数值列为浮点型
    numeric_columns = ['indexPrice', 'futuresPrice', 'basis', 'basisRate']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 保存为CSV
    df.to_csv(filename, index=False, encoding='utf-8-sig')

    print(f"\n数据已保存到: {filename}")
    print(f"总记录数: {len(df)}")
    print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    print(f"\n数据预览:")
    print(df.head())
    print("\n统计信息:")
    print(df[['indexPrice', 'futuresPrice', 'basis', 'basisRate']].describe())

def get_all_basis_data(pair='BTCUSDT'):
    """
    获取所有合约类型和周期的基差数据，分别保存到不同的CSV文件

    参数:
    pair: 交易对，默认BTCUSDT
    """
    # 定义所有合约类型
    contract_types = ['PERPETUAL', 'CURRENT_QUARTER', 'NEXT_QUARTER']

    # 定义所有周期（使用5m获得最高精度和最大数据量）
    periods = ['5m']

    # 创建输出目录
    output_dir = 'basis_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}\n")

    # 遍历所有组合
    for contract_type in contract_types:
        for period in periods:
            print("="*80)
            print(f"开始获取: {pair} - {contract_type} - {period}")
            print("="*80)

            # 获取数据
            data = get_basis_history(
                pair=pair,
                contract_type=contract_type,
                period=period,
                limit=500
            )

            # 生成文件名
            filename = f"{output_dir}/BTC期货基差_{contract_type}_{period}.csv"

            # 保存数据
            save_to_csv(data, filename)

            print("\n" + "="*80 + "\n")

            # 等待一段时间再请求下一个，避免频率限制
            time.sleep(2)

if __name__ == "__main__":
    # 获取所有合约类型的基差数据（最高精度：5分钟周期）
    get_all_basis_data(pair='BTCUSDT')

    print("\n" + "="*80)
    print("所有数据获取完成！")
    print("="*80)
