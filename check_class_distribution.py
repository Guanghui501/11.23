#!/usr/bin/env python
"""
检查二分类数据集的类别分布
"""

import sys
import json
import numpy as np
from collections import Counter

def check_class_distribution(data_file):
    """
    检查数据集的类别分布

    Args:
        data_file: id_prop.csv 或类似的数据文件
    """
    print("="*60)
    print("  二分类数据集类别分布检查")
    print("="*60)
    print()

    # 读取数据
    try:
        import pandas as pd
        df = pd.read_csv(data_file)

        # 假设目标列名为 'target' 或第二列
        if 'target' in df.columns:
            labels = df['target'].values
        else:
            labels = df.iloc[:, 1].values

    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        print("请提供正确的CSV文件路径")
        return

    # 统计类别分布
    counter = Counter(labels)
    total = len(labels)

    print(f"📊 数据集总样本数: {total}")
    print()

    print("类别分布:")
    print("-"*60)
    for label in sorted(counter.keys()):
        count = counter[label]
        percentage = count / total * 100
        bar = "█" * int(percentage / 2)
        print(f"  类别 {label}: {count:6d} 样本 ({percentage:5.2f}%) {bar}")
    print()

    # 计算不平衡比率
    if len(counter) == 2:
        classes = sorted(counter.keys())
        majority_class = max(counter, key=counter.get)
        minority_class = min(counter, key=counter.get)

        imbalance_ratio = counter[majority_class] / counter[minority_class]

        print("不平衡分析:")
        print("-"*60)
        print(f"  多数类 (类别{majority_class}): {counter[majority_class]} 样本")
        print(f"  少数类 (类别{minority_class}): {counter[minority_class]} 样本")
        print(f"  不平衡比率: {imbalance_ratio:.2f}:1")
        print()

        # 评估不平衡程度
        if imbalance_ratio > 10:
            severity = "🔴 严重不平衡"
            recommendation = "强烈建议使用类别权重、过采样或欠采样"
        elif imbalance_ratio > 3:
            severity = "🟡 中度不平衡"
            recommendation = "建议使用类别权重或调整损失函数"
        else:
            severity = "🟢 轻度不平衡"
            recommendation = "可以考虑使用类别权重优化"

        print(f"严重程度: {severity}")
        print(f"建议: {recommendation}")
        print()

        # 计算建议的pos_weight（用于BCEWithLogitsLoss）
        pos_weight = counter[majority_class] / counter[minority_class]
        print("💡 推荐配置:")
        print("-"*60)
        print(f"  pos_weight (用于BCEWithLogitsLoss): {pos_weight:.4f}")
        print(f"  class_weight={{0: 1.0, 1: {pos_weight:.4f}}}")
        print()

    else:
        print(f"⚠️  检测到 {len(counter)} 个类别，不是二分类任务")
        print()

    # 建议的评估指标
    print("📈 推荐评估指标:")
    print("-"*60)
    print("  ✅ F1分数（macro/weighted）")
    print("  ✅ 精确率（Precision）")
    print("  ✅ 召回率（Recall）")
    print("  ✅ ROC-AUC")
    print("  ✅ PR-AUC（对不平衡数据更敏感）")
    print("  ⚠️  准确率（Accuracy）- 可能误导")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_class_distribution.py <data_file.csv>")
        print()
        print("示例:")
        print("  python check_class_distribution.py /path/to/id_prop.csv")
        sys.exit(1)

    data_file = sys.argv[1]
    check_class_distribution(data_file)
