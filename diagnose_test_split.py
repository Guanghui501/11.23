#!/usr/bin/env python
"""
诊断测试集划分问题

检查预处理数据的分割是否与训练时一致
"""

import pickle
import sys

# 预处理数据路径
preprocessed_dir = "/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/retu/preprocessed_data"

print("="*80)
print("测试集划分诊断")
print("="*80)
print()

# 加载三个split
for split in ['train', 'val', 'test']:
    pkl_file = f"{preprocessed_dir}/jarvis/mbj_bandgap/{split}.pkl"

    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        print(f"{split.upper()} 集:")
        print(f"  样本数: {len(data)}")

        # 显示前5个样本ID
        print(f"  前5个样本ID:")
        for i in range(min(5, len(data))):
            print(f"    {i+1}. {data[i]['id']}")

        print()

    except FileNotFoundError:
        print(f"✗ 找不到文件: {pkl_file}")
        print()

print("="*80)
print("问题诊断")
print("="*80)
print()

print("请回答以下问题：")
print()
print("1. 训练时使用的数据分割参数是什么？")
print("   - 随机种子 (seed): ?")
print("   - 训练集比例: ?")
print("   - 验证集比例: ?")
print("   - 测试集比例: ?")
print()

print("2. 预处理时使用的参数是什么？")
print("   - 查看 preprocess_mbj_bandgap.sh 或 preprocess_dataset.py 的运行日志")
print()

print("3. 训练时是否有保存测试集ID列表？")
print("   - 查找训练输出目录中的 id_test.json 或类似文件")
print()

print("="*80)
print("解决方案")
print("="*80)
print()

print("方案1: 使用训练时的相同参数重新预处理")
print("  - 找到训练时使用的 seed 和分割比例")
print("  - 使用相同参数运行 preprocess_dataset.py")
print()

print("方案2: 使用训练时保存的数据ID列表")
print("  - 如果训练目录中有 id_test.json")
print("  - 可以根据这些ID从预处理数据中筛选")
print()

print("方案3: 直接使用训练目录中的测试集数据")
print("  - 如果训练时保存了 test_loader.pkl")
print("  - 可以直接加载使用")
print()
