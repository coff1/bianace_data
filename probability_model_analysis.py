#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5分钟K线涨跌概率模型 - 基于随机游走假设
核心思想：在短期时间内，价格变动服从布朗运动，通过标准正态分布计算涨跌概率
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class RandomWalkProbabilityModel:
    """
    基于完全随机假设的涨跌概率模型

    核心公式：
    P(up) = Φ((current - open) / (σ * sqrt(remaining_time)))

    其中：
    - Φ 是标准正态分布的累积分布函数
    - current: 当前价格
    - open: 5分钟K线开盘价
    - σ: 实时波动率（秒级标准差）
    - remaining_time: 剩余时间（以分钟为单位）
    """

    def __init__(self, lookback_minutes=30):
        """
        初始化模型

        Args:
            lookback_minutes: 回溯时间窗口（用于计算实时波动率）
        """
        self.lookback_minutes = lookback_minutes
        self.results = []

    def calculate_volatility(self, prices):
        """
        计算实时波动率（基于对数收益率）

        Args:
            prices: 价格序列

        Returns:
            波动率（年化后转换为秒级）
        """
        if len(prices) < 2:
            return 0.001  # 默认最小波动率

        # 计算对数收益率
        log_returns = np.log(prices / prices.shift(1)).dropna()

        # 计算标准差（秒级）
        volatility = log_returns.std()

        # 防止波动率为0
        if volatility == 0 or np.isnan(volatility):
            volatility = 0.001

        return volatility

    def calculate_probability(self, open_price, current_price, remaining_minutes, volatility):
        """
        计算涨跌概率

        Args:
            open_price: 5分钟K线开盘价
            current_price: 当前价格
            remaining_minutes: 剩余时间（分钟）
            volatility: 实时波动率

        Returns:
            上涨概率
        """
        if remaining_minutes <= 0:
            # 时间已用完，直接看当前价格
            return 1.0 if current_price >= open_price else 0.0

        # 计算价格变动幅度（对数收益率）
        price_change = np.log(current_price / open_price)

        # 计算时间因子（转换为分钟的平方根）
        time_factor = np.sqrt(remaining_minutes)

        # 计算标准化得分（Z-score）
        if volatility == 0:
            z_score = 0
        else:
            z_score = price_change / (volatility * time_factor)

        # 使用标准正态分布累积函数计算概率
        probability = norm.cdf(z_score)

        return probability

    def analyze_5min_kline(self, kline_1min_data, kline_start_idx):
        """
        分析单根5分钟K线，在第2、3、4、5分钟计算概率

        Args:
            kline_1min_data: 完整的1分钟数据
            kline_start_idx: 5分钟K线的起始索引

        Returns:
            分析结果列表
        """
        results = []

        # 获取5分钟K线的5根1分钟数据
        if kline_start_idx + 5 > len(kline_1min_data):
            return results

        kline_5min = kline_1min_data.iloc[kline_start_idx:kline_start_idx + 5]

        # 5分钟K线的开盘价和收盘价
        open_price_5min = kline_5min.iloc[0]['open']
        close_price_5min = kline_5min.iloc[-1]['close']

        # 实际涨跌结果
        actual_up = 1 if close_price_5min >= open_price_5min else 0

        # 计算历史波动率（使用前30分钟数据）
        lookback_start = max(0, kline_start_idx - self.lookback_minutes)
        historical_prices = kline_1min_data.iloc[lookback_start:kline_start_idx]['close']
        volatility = self.calculate_volatility(historical_prices)

        # 在第2、3、4、5分钟计算概率
        for minute_idx in range(1, 5):  # 1,2,3,4 对应第2,3,4,5分钟
            current_row = kline_5min.iloc[minute_idx]
            current_price = current_row['close']
            remaining_minutes = 5 - (minute_idx + 1)

            # 计算概率
            prob_up = self.calculate_probability(
                open_price=open_price_5min,
                current_price=current_price,
                remaining_minutes=remaining_minutes,
                volatility=volatility
            )

            # 记录结果
            result = {
                'timestamp': current_row['open_datetime'],
                'kline_open': open_price_5min,
                'kline_close': close_price_5min,
                'current_price': current_price,
                'minute': minute_idx + 1,  # 第几分钟
                'remaining_minutes': remaining_minutes,
                'volatility': volatility,
                'prob_up': prob_up,
                'prob_down': 1 - prob_up,
                'predicted_up': 1 if prob_up > 0.5 else 0,
                'actual_up': actual_up,
                'price_change_pct': (close_price_5min - open_price_5min) / open_price_5min * 100,
                'current_change_pct': (current_price - open_price_5min) / open_price_5min * 100
            }

            results.append(result)

        return results

    def backtest(self, df_1min, train_ratio=0.7):
        """
        回测整个数据集

        Args:
            df_1min: 1分钟K线数据
            train_ratio: 训练集比例

        Returns:
            训练集结果、测试集结果
        """
        all_results = []

        # 每5根1分钟K线作为一根5分钟K线
        total_5min_klines = len(df_1min) // 5

        print(f"总共有 {len(df_1min)} 根1分钟K线")
        print(f"可以生成 {total_5min_klines} 根5分钟K线")
        print(f"开始回测...")

        for i in range(0, total_5min_klines * 5, 5):
            # 分析每根5分钟K线
            kline_results = self.analyze_5min_kline(df_1min, i)
            all_results.extend(kline_results)

            # 进度显示
            if (i // 5) % 1000 == 0:
                print(f"  已处理 {i // 5}/{total_5min_klines} 根5分钟K线...")

        # 转换为DataFrame
        df_results = pd.DataFrame(all_results)

        # 划分训练集和测试集
        split_idx = int(len(df_results) * train_ratio)
        df_train = df_results.iloc[:split_idx]
        df_test = df_results.iloc[split_idx:]

        print(f"\n数据划分完成:")
        print(f"  训练集: {len(df_train)} 条记录")
        print(f"  测试集: {len(df_test)} 条记录")

        return df_train, df_test


def evaluate_model(df, minute_filter=None, prob_threshold=0.5):
    """
    评估模型性能

    Args:
        df: 结果数据
        minute_filter: 只评估特定分钟（如 [4, 5] 只评估第4、5分钟）
        prob_threshold: 概率阈值
    """
    if minute_filter:
        df = df[df['minute'].isin(minute_filter)]

    if len(df) == 0:
        print("无数据")
        return

    y_true = df['actual_up'].values
    y_pred = (df['prob_up'] > prob_threshold).astype(int)

    # 基础指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 计算真实涨跌分布
    actual_up_rate = y_true.mean()

    # 计算平均概率与实际准确率的偏差
    prob_calibration = []
    for prob_bin in np.arange(0, 1.1, 0.1):
        bin_mask = (df['prob_up'] >= prob_bin) & (df['prob_up'] < prob_bin + 0.1)
        if bin_mask.sum() > 0:
            actual_rate = df.loc[bin_mask, 'actual_up'].mean()
            prob_calibration.append({
                'prob_range': f'{prob_bin:.1f}-{prob_bin+0.1:.1f}',
                'predicted_prob': prob_bin + 0.05,
                'actual_rate': actual_rate,
                'count': bin_mask.sum()
            })

    # 打印结果
    print(f"\n{'='*60}")
    print(f"模型评估结果 (阈值: {prob_threshold})")
    if minute_filter:
        print(f"筛选条件: 第 {minute_filter} 分钟")
    print(f"{'='*60}")
    print(f"样本数量: {len(df)}")
    print(f"实际上涨比例: {actual_up_rate:.2%}")
    print(f"\n准确率 (Accuracy): {accuracy:.4f} ({accuracy:.2%})")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"\n混淆矩阵:")
    print(f"  真负例(TN): {cm[0,0]:>6d}  |  假正例(FP): {cm[0,1]:>6d}")
    print(f"  假负例(FN): {cm[1,0]:>6d}  |  真正例(TP): {cm[1,1]:>6d}")

    # 打印概率校准
    print(f"\n概率校准分析:")
    print(f"{'预测概率区间':<15} {'实际上涨率':<12} {'样本数':<10} {'偏差':<10}")
    print(f"{'-'*50}")
    for item in prob_calibration:
        bias = item['actual_rate'] - item['predicted_prob']
        print(f"{item['prob_range']:<15} {item['actual_rate']:<12.2%} {item['count']:<10} {bias:+.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'actual_up_rate': actual_up_rate,
        'prob_calibration': prob_calibration
    }


def analyze_by_minute(df_train, df_test):
    """
    按分钟维度分析模型性能
    """
    print(f"\n{'='*60}")
    print("按分钟维度分析")
    print(f"{'='*60}")

    results_summary = []

    for minute in [2, 3, 4, 5]:
        print(f"\n【第 {minute} 分钟】")
        print(f"\n--- 训练集 ---")
        train_metrics = evaluate_model(df_train, minute_filter=[minute])

        print(f"\n--- 测试集 ---")
        test_metrics = evaluate_model(df_test, minute_filter=[minute])

        if train_metrics and test_metrics:
            results_summary.append({
                'minute': minute,
                'train_accuracy': train_metrics['accuracy'],
                'test_accuracy': test_metrics['accuracy'],
                'train_up_rate': train_metrics['actual_up_rate'],
                'test_up_rate': test_metrics['actual_up_rate']
            })

    # 打印汇总
    print(f"\n{'='*60}")
    print("汇总表")
    print(f"{'='*60}")
    df_summary = pd.DataFrame(results_summary)
    print(df_summary.to_string(index=False))

    return df_summary


def visualize_results(df_train, df_test):
    """
    可视化分析结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 不同分钟的准确率对比
    ax1 = axes[0, 0]
    minute_accuracy = []
    for minute in [2, 3, 4, 5]:
        train_acc = accuracy_score(
            df_train[df_train['minute'] == minute]['actual_up'],
            (df_train[df_train['minute'] == minute]['prob_up'] > 0.5).astype(int)
        )
        test_acc = accuracy_score(
            df_test[df_test['minute'] == minute]['actual_up'],
            (df_test[df_test['minute'] == minute]['prob_up'] > 0.5).astype(int)
        )
        minute_accuracy.append({'minute': minute, 'train': train_acc, 'test': test_acc})

    df_acc = pd.DataFrame(minute_accuracy)
    x = np.arange(len(df_acc))
    width = 0.35
    ax1.bar(x - width/2, df_acc['train'], width, label='训练集', alpha=0.8)
    ax1.bar(x + width/2, df_acc['test'], width, label='测试集', alpha=0.8)
    ax1.set_xlabel('第N分钟')
    ax1.set_ylabel('准确率')
    ax1.set_title('不同时间点的预测准确率')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'第{m}分钟' for m in df_acc['minute']])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0.5, color='r', linestyle='--', label='随机猜测基准')

    # 2. 概率校准曲线（测试集）
    ax2 = axes[0, 1]
    prob_bins = np.arange(0, 1.1, 0.1)
    predicted_probs = []
    actual_rates = []

    for i in range(len(prob_bins) - 1):
        mask = (df_test['prob_up'] >= prob_bins[i]) & (df_test['prob_up'] < prob_bins[i+1])
        if mask.sum() > 10:  # 至少10个样本
            predicted_probs.append(prob_bins[i] + 0.05)
            actual_rates.append(df_test.loc[mask, 'actual_up'].mean())

    ax2.plot(predicted_probs, actual_rates, 'o-', label='实际校准曲线', markersize=8)
    ax2.plot([0, 1], [0, 1], 'r--', label='完美校准线')
    ax2.set_xlabel('预测概率')
    ax2.set_ylabel('实际上涨率')
    ax2.set_title('概率校准曲线 (测试集)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. 概率分布直方图
    ax3 = axes[1, 0]
    ax3.hist(df_test[df_test['actual_up'] == 1]['prob_up'], bins=50, alpha=0.6, label='实际上涨', density=True)
    ax3.hist(df_test[df_test['actual_up'] == 0]['prob_up'], bins=50, alpha=0.6, label='实际下跌', density=True)
    ax3.set_xlabel('预测上涨概率')
    ax3.set_ylabel('密度')
    ax3.set_title('预测概率分布 (测试集)')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.axvline(x=0.5, color='r', linestyle='--', linewidth=2)

    # 4. 剩余时间与准确率的关系
    ax4 = axes[1, 1]
    remaining_accuracy = []
    for minute in [2, 3, 4, 5]:
        remaining = 5 - minute
        acc = accuracy_score(
            df_test[df_test['minute'] == minute]['actual_up'],
            (df_test[df_test['minute'] == minute]['prob_up'] > 0.5).astype(int)
        )
        remaining_accuracy.append({'remaining_minutes': remaining, 'accuracy': acc, 'minute': minute})

    df_remaining = pd.DataFrame(remaining_accuracy)
    ax4.plot(df_remaining['remaining_minutes'], df_remaining['accuracy'], 'o-', markersize=10, linewidth=2)
    ax4.set_xlabel('剩余时间 (分钟)')
    ax4.set_ylabel('准确率')
    ax4.set_title('剩余时间与准确率的关系')
    ax4.grid(alpha=0.3)
    ax4.axhline(y=0.5, color='r', linestyle='--', label='随机基准')
    for i, row in df_remaining.iterrows():
        ax4.annotate(f"第{row['minute']}分钟",
                    xy=(row['remaining_minutes'], row['accuracy']),
                    xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig('probability_model_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存: probability_model_analysis.png")

    return fig


def main():
    """
    主函数
    """
    print("="*60)
    print("BTC 5分钟K线涨跌概率模型 - 基于随机游走假设")
    print("="*60)

    # 1. 加载数据
    print("\n[1] 加载1分钟K线数据...")
    df = pd.read_csv('BTCUSDT_1m_20260107_000000_20260410_174117.csv')
    df['open_datetime'] = pd.to_datetime(df['open_datetime'])
    df['close_datetime'] = pd.to_datetime(df['close_datetime'])

    print(f"数据时间范围: {df['open_datetime'].min()} 至 {df['open_datetime'].max()}")
    print(f"总数据量: {len(df)} 根1分钟K线")

    # 2. 初始化模型并回测
    print("\n[2] 初始化模型并执行回测...")
    model = RandomWalkProbabilityModel(lookback_minutes=30)
    df_train, df_test = model.backtest(df, train_ratio=0.7)

    # 3. 整体评估
    print("\n[3] 整体模型评估...")
    print("\n【训练集 - 全部数据】")
    evaluate_model(df_train)

    print("\n【测试集 - 全部数据】")
    evaluate_model(df_test)

    # 4. 按分钟维度分析
    print("\n[4] 按分钟维度详细分析...")
    df_summary = analyze_by_minute(df_train, df_test)

    # 5. 可视化
    print("\n[5] 生成可视化图表...")
    visualize_results(df_train, df_test)

    # 6. 保存结果
    print("\n[6] 保存分析结果...")
    df_train.to_csv('train_results.csv', index=False)
    df_test.to_csv('test_results.csv', index=False)
    df_summary.to_csv('summary_by_minute.csv', index=False)

    print("\n结果文件已保存:")
    print("  - train_results.csv: 训练集详细结果")
    print("  - test_results.csv: 测试集详细结果")
    print("  - summary_by_minute.csv: 按分钟汇总结果")

    # 7. 最终结论
    print("\n" + "="*60)
    print("【最终结论】")
    print("="*60)

    # 计算第5分钟（剩余0分钟）的准确率
    test_minute5 = df_test[df_test['minute'] == 5]
    acc_minute5 = accuracy_score(
        test_minute5['actual_up'],
        (test_minute5['prob_up'] > 0.5).astype(int)
    )

    print(f"\n1. 模型基于完全随机假设（布朗运动），无需训练参数")
    print(f"2. 测试集整体准确率: {accuracy_score(df_test['actual_up'], (df_test['prob_up'] > 0.5).astype(int)):.2%}")
    print(f"3. 第5分钟（剩余0分钟）准确率: {acc_minute5:.2%}")

    if acc_minute5 > 0.70:
        print(f"\n✅ 第5分钟准确率 > 70%，符合预期，具备套利可行性")
    elif acc_minute5 > 0.58:
        print(f"\n⚠️  第5分钟准确率 58%-70%，有统计优势但需谨慎")
    else:
        print(f"\n❌ 第5分钟准确率 < 58%，接近随机，不建议套利")

    print(f"\n4. 建议策略:")
    print(f"   - 专注于第4、5分钟（剩余时间≤1分钟）进行套利")
    print(f"   - 设置概率阈值（如 >0.65 或 <0.35）过滤低信号强度交易")
    print(f"   - 实时监控波动率异常（>3σ）时停止交易")
    print(f"   - 考虑币安价格与Polymarket预言机的基差修正")

    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)


if __name__ == '__main__':
    main()
