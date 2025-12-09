#!/usr/bin/env python
"""
修复测试集划分问题

这个脚本会：
1. 检查训练目录中是否有保存的测试集ID
2. 如果有，从预处理数据中筛选出正确的测试集
3. 如果没有，帮助用户诊断问题并提供解决方案

用法:
    python fix_test_split.py \
        --training_dir /path/to/training/output \
        --preprocessed_dir /path/to/preprocessed_data \
        --dataset jarvis \
        --property mbj_bandgap \
        --output_dir ./corrected_test_set
"""

import sys
import os
import pickle
import json
import argparse
from pathlib import Path

def find_test_ids_in_training_dir(training_dir):
    """在训练目录中查找测试集ID"""
    training_path = Path(training_dir)

    possible_files = [
        'id_test.json',
        'test_ids.json',
        'test_set_ids.json',
        'id_prop.csv',
        'test_data.pkl',
        'test_loader.pkl',
    ]

    found_files = {}

    for filename in possible_files:
        file_path = training_path / filename
        if file_path.exists():
            found_files[filename] = file_path

    return found_files


def load_test_ids(file_path):
    """从文件加载测试集ID"""
    file_path = Path(file_path)

    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'test' in data:
                return data['test']
            else:
                return list(data.keys())

    elif file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, list):
                # 如果是数据列表，提取ID
                if len(data) > 0 and isinstance(data[0], dict) and 'id' in data[0]:
                    return [item['id'] for item in data]
            return None

    elif file_path.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(file_path)
        if 'id' in df.columns:
            return df['id'].tolist()

    return None


def load_preprocessed_data(preprocessed_dir, dataset, property_name):
    """加载预处理数据"""
    data_dir = Path(preprocessed_dir) / dataset / property_name

    splits = {}
    for split in ['train', 'val', 'test']:
        pkl_file = data_dir / f"{split}.pkl"
        if pkl_file.exists():
            with open(pkl_file, 'rb') as f:
                splits[split] = pickle.load(f)
            print(f"✓ 加载 {split} 集: {len(splits[split])} 个样本")
        else:
            print(f"✗ 找不到 {split} 集: {pkl_file}")

    return splits


def filter_test_set(preprocessed_data, test_ids):
    """从预处理数据中筛选出正确的测试集"""
    # 合并所有split的数据
    all_data = []
    for split in ['train', 'val', 'test']:
        if split in preprocessed_data:
            all_data.extend(preprocessed_data[split])

    print(f"\n总数据量: {len(all_data)} 个样本")
    print(f"目标测试集ID数量: {len(test_ids)} 个")

    # 创建ID到数据的映射
    id_to_data = {item['id']: item for item in all_data}

    # 筛选测试集
    filtered_test_set = []
    missing_ids = []

    for test_id in test_ids:
        if test_id in id_to_data:
            filtered_test_set.append(id_to_data[test_id])
        else:
            missing_ids.append(test_id)

    print(f"\n✓ 成功匹配: {len(filtered_test_set)} 个样本")

    if missing_ids:
        print(f"⚠ 缺失ID: {len(missing_ids)} 个")
        print(f"  前5个缺失ID: {missing_ids[:5]}")

    return filtered_test_set, missing_ids


def check_test_set_consistency(training_test_ids, preprocessed_test_data):
    """检查测试集一致性"""
    preprocessed_ids = [item['id'] for item in preprocessed_test_data]

    training_set = set(training_test_ids)
    preprocessed_set = set(preprocessed_ids)

    in_both = training_set & preprocessed_set
    only_in_training = training_set - preprocessed_set
    only_in_preprocessed = preprocessed_set - training_set

    print("\n" + "="*80)
    print("测试集一致性检查")
    print("="*80)
    print(f"\n训练测试集ID数量: {len(training_test_ids)}")
    print(f"预处理测试集ID数量: {len(preprocessed_ids)}")
    print(f"\n共同ID数量: {len(in_both)} ({len(in_both)/len(training_test_ids)*100:.1f}%)")
    print(f"仅在训练中: {len(only_in_training)}")
    print(f"仅在预处理中: {len(only_in_preprocessed)}")

    if only_in_training:
        print(f"\n仅在训练中的ID (前5个): {list(only_in_training)[:5]}")

    if only_in_preprocessed:
        print(f"仅在预处理中的ID (前5个): {list(only_in_preprocessed)[:5]}")

    return len(in_both) == len(training_test_ids)


def main():
    parser = argparse.ArgumentParser(description='修复测试集划分')
    parser.add_argument('--training_dir', type=str, required=True,
                       help='训练输出目录')
    parser.add_argument('--preprocessed_dir', type=str, required=True,
                       help='预处理数据目录')
    parser.add_argument('--dataset', type=str, default='jarvis',
                       help='数据集名称')
    parser.add_argument('--property', type=str, default='mbj_bandgap',
                       help='属性名称')
    parser.add_argument('--output_dir', type=str, default='./corrected_test_set',
                       help='输出目录')

    args = parser.parse_args()

    print("="*80)
    print("测试集划分修复")
    print("="*80)
    print(f"\n训练目录: {args.training_dir}")
    print(f"预处理数据目录: {args.preprocessed_dir}")
    print(f"数据集: {args.dataset}")
    print(f"属性: {args.property}")
    print()

    # 步骤1: 查找训练目录中的测试集ID文件
    print("="*80)
    print("步骤1: 查找训练目录中的测试集ID")
    print("="*80)
    print()

    found_files = find_test_ids_in_training_dir(args.training_dir)

    if not found_files:
        print("✗ 训练目录中没有找到测试集ID文件")
        print()
        print("建议:")
        print("1. 检查训练脚本是否保存了测试集ID")
        print("2. 查看训练日志，找到使用的随机种子和分割比例")
        print("3. 使用相同参数重新预处理数据")
        print()
        return

    print(f"✓ 找到 {len(found_files)} 个可能的文件:")
    for filename, filepath in found_files.items():
        print(f"  - {filename}: {filepath}")
    print()

    # 步骤2: 尝试加载测试集ID
    print("="*80)
    print("步骤2: 加载测试集ID")
    print("="*80)
    print()

    test_ids = None
    used_file = None

    for filename, filepath in found_files.items():
        print(f"尝试从 {filename} 加载...")
        loaded_ids = load_test_ids(filepath)
        if loaded_ids:
            test_ids = loaded_ids
            used_file = filename
            print(f"✓ 成功加载 {len(test_ids)} 个测试集ID")
            print(f"  前5个ID: {test_ids[:5]}")
            break
        else:
            print(f"✗ 无法从 {filename} 提取ID")

    if not test_ids:
        print()
        print("✗ 无法从任何文件加载测试集ID")
        print("请手动检查训练目录中的文件")
        return

    # 步骤3: 加载预处理数据
    print()
    print("="*80)
    print("步骤3: 加载预处理数据")
    print("="*80)
    print()

    preprocessed_data = load_preprocessed_data(
        args.preprocessed_dir,
        args.dataset,
        args.property
    )

    if 'test' not in preprocessed_data:
        print()
        print("✗ 预处理数据中没有测试集")
        return

    # 步骤4: 检查一致性
    print()
    check_test_set_consistency(test_ids, preprocessed_data['test'])

    # 步骤5: 筛选正确的测试集
    print()
    print("="*80)
    print("步骤4: 筛选正确的测试集")
    print("="*80)
    print()

    filtered_test_set, missing_ids = filter_test_set(preprocessed_data, test_ids)

    if not filtered_test_set:
        print()
        print("✗ 无法筛选出测试集，没有匹配的ID")
        return

    # 步骤6: 保存正确的测试集
    print()
    print("="*80)
    print("步骤5: 保存正确的测试集")
    print("="*80)
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'test.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(filtered_test_set, f)

    print(f"✓ 正确的测试集已保存到: {output_file}")
    print(f"  样本数: {len(filtered_test_set)}")

    # 保存测试集ID列表
    ids_file = output_dir / 'test_ids.json'
    with open(ids_file, 'w') as f:
        json.dump([item['id'] for item in filtered_test_set], f, indent=2)
    print(f"✓ 测试集ID已保存到: {ids_file}")

    # 保存诊断信息
    info_file = output_dir / 'correction_info.json'
    info = {
        'source_training_dir': args.training_dir,
        'source_file': used_file,
        'original_test_ids_count': len(test_ids),
        'matched_samples': len(filtered_test_set),
        'missing_ids_count': len(missing_ids),
        'missing_ids': missing_ids[:10] if missing_ids else [],
    }
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"✓ 修复信息已保存到: {info_file}")

    print()
    print("="*80)
    print("下一步")
    print("="*80)
    print()
    print("现在你可以使用修复后的测试集运行评估：")
    print()
    print("python evaluate_text_masking.py \\")
    print(f"    --checkpoint {args.training_dir}/best_test_model.pt \\")
    print(f"    --test_data {output_file} \\")
    print(f"    --preprocessed_dir {args.preprocessed_dir} \\")
    print(f"    --dataset {args.dataset} \\")
    print(f"    --property {args.property} \\")
    print("    --masking_strategy random_token \\")
    print("    --masking_ratios 0.0 0.5 1.0 \\")
    print("    --output_dir ./test_output")
    print()


if __name__ == "__main__":
    main()
