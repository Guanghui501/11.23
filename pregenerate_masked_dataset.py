#!/usr/bin/env python3
"""
预先生成遮挡数据集

这个脚本解决随机遮挡不确定性的问题：
1. 预先生成各种遮挡率的数据集
2. 保存遮挡后的文本
3. 两个模型使用完全相同的遮挡数据进行公平对比

用法:
    # 生成遮挡数据集
    python pregenerate_masked_dataset.py \
        --input_data ./corrected_test_set/test.pkl \
        --output_dir ./masked_datasets \
        --strategies random_token sentence keep_keywords \
        --ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
        --seed 42

    # 然后使用相同的遮挡数据评估两个模型
    python evaluate_with_premasked_data.py \
        --model1 /path/to/model1.pt \
        --model2 /path/to/model2.pt \
        --masked_data ./masked_datasets/random_token_0.5.pkl
"""

import os
import sys
import pickle
import argparse
import random
import re
from typing import List, Dict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def set_seed(seed=42):
    """设置随机种子确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)


class DeterministicTextMasker:
    """确定性文本遮挡器 - 基于样本ID使用固定种子"""

    def __init__(self, tokenizer, strategy='random_token', base_seed=42):
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.base_seed = base_seed

        # 化学元素正则
        self.element_pattern = re.compile(
            r'\b(H|He|Li|Be|B|C|N|O|F|Ne|Na|Mg|Al|Si|P|S|Cl|Ar|K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|'
            r'Ga|Ge|As|Se|Br|Kr|Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|Xe|Cs|Ba|La|Ce|'
            r'Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|'
            r'Fr|Ra|Ac|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr)\b'
        )
        self.structure_keywords = [
            'cubic', 'tetragonal', 'orthorhombic', 'hexagonal', 'monoclinic',
            'triclinic', 'rhombohedral', 'space group', 'lattice', 'crystal'
        ]

    def mask_text_deterministic(self, text: str, masking_ratio: float, sample_id: str) -> str:
        """
        确定性遮挡：基于sample_id设置种子，保证每次遮挡结果相同

        Args:
            text: 原始文本
            masking_ratio: 遮挡率
            sample_id: 样本ID（用于生成确定性种子）

        Returns:
            遮挡后的文本
        """
        if masking_ratio == 0.0:
            return text

        if masking_ratio == 1.0:
            return ""

        # 基于sample_id和base_seed生成确定性种子
        seed = self.base_seed + hash(sample_id) % 1000000

        # 临时设置随机种子
        random_state = random.getstate()
        np_random_state = np.random.get_state()

        random.seed(seed)
        np.random.seed(seed)

        # 执行遮挡
        try:
            if self.strategy == 'random_token':
                masked = self._mask_random_tokens(text, masking_ratio)
            elif self.strategy == 'random_word':
                masked = self._mask_random_words(text, masking_ratio)
            elif self.strategy == 'random_chunk':
                masked = self._mask_random_chunks(text, masking_ratio)
            elif self.strategy == 'sentence':
                masked = self._mask_sentences(text, masking_ratio)
            elif self.strategy == 'keep_keywords':
                masked = self._keep_only_keywords(text, masking_ratio)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        finally:
            # 恢复原始随机状态
            random.setstate(random_state)
            np.random.set_state(np_random_state)

        return masked

    def _mask_random_tokens(self, text: str, ratio: float) -> str:
        """随机遮挡单个token"""
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) == 0:
            return text

        num_to_mask = int(len(tokens) * ratio)
        if num_to_mask == 0:
            return text

        mask_indices = random.sample(range(len(tokens)), min(num_to_mask, len(tokens)))

        for idx in mask_indices:
            tokens[idx] = '[MASK]'

        masked_text = self.tokenizer.convert_tokens_to_string(tokens)
        return masked_text

    def _mask_random_words(self, text: str, ratio: float) -> str:
        """随机遮挡完整单词"""
        words = text.split()
        if len(words) == 0:
            return text

        num_to_mask = int(len(words) * ratio)
        if num_to_mask == 0:
            return text

        mask_indices = random.sample(range(len(words)), min(num_to_mask, len(words)))

        for idx in mask_indices:
            words[idx] = '[MASK]'

        return ' '.join(words)

    def _mask_random_chunks(self, text: str, ratio: float) -> str:
        """随机遮挡连续文本块"""
        words = text.split()
        if len(words) == 0:
            return text

        num_to_mask = int(len(words) * ratio)
        if num_to_mask == 0:
            return text

        # 随机选择起始位置
        if len(words) <= num_to_mask:
            return '[MASK]'

        start_idx = random.randint(0, len(words) - num_to_mask)

        # 遮挡连续的chunk
        for i in range(start_idx, start_idx + num_to_mask):
            words[i] = '[MASK]'

        # 合并连续的[MASK]
        result = []
        prev_mask = False
        for word in words:
            if word == '[MASK]':
                if not prev_mask:
                    result.append('[MASK]')
                prev_mask = True
            else:
                result.append(word)
                prev_mask = False

        return ' '.join(result)

    def _mask_sentences(self, text: str, ratio: float) -> str:
        """随机遮挡完整句子"""
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) == 0:
            return text

        num_to_mask = max(1, int(len(sentences) * ratio))
        if num_to_mask == 0:
            return text

        mask_indices = set(random.sample(range(len(sentences)), min(num_to_mask, len(sentences))))

        masked_sentences = [
            '[MASK]' if i in mask_indices else s
            for i, s in enumerate(sentences)
        ]

        return '. '.join(masked_sentences)

    def _keep_only_keywords(self, text: str, ratio: float) -> str:
        """只保留关键词（化学元素和晶体结构术语）"""
        words = text.split()
        if len(words) == 0:
            return text

        # 找到所有关键词位置
        keyword_indices = []
        for i, word in enumerate(words):
            if self.element_pattern.search(word) or \
               any(kw in word.lower() for kw in self.structure_keywords):
                keyword_indices.append(i)

        if len(keyword_indices) == 0:
            return '[MASK]'

        # 根据ratio决定保留多少关键词
        num_keywords_to_keep = max(1, int(len(keyword_indices) * (1.0 - ratio)))
        keep_indices = set(random.sample(keyword_indices, min(num_keywords_to_keep, len(keyword_indices))))

        # 构建结果
        result = []
        for i, word in enumerate(words):
            if i in keep_indices:
                result.append(word)
            elif i in keyword_indices:
                result.append('[MASK]')

        return ' '.join(result) if result else '[MASK]'


def generate_masked_datasets(
    input_data_path: str,
    output_dir: str,
    strategies: List[str],
    ratios: List[float],
    tokenizer_name: str = 'm3rg-iitd/matscibert',
    seed: int = 42
):
    """
    生成所有遮挡策略和遮挡率的数据集

    Args:
        input_data_path: 输入数据路径（pickle文件）
        output_dir: 输出目录
        strategies: 遮挡策略列表
        ratios: 遮挡率列表
        tokenizer_name: tokenizer名称
        seed: 随机种子
    """
    # 设置全局种子
    set_seed(seed)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 加载数据
    print(f"Loading data from: {input_data_path}")
    with open(input_data_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} samples")

    # 为每个策略和遮挡率生成数据集
    total_combinations = len(strategies) * len(ratios)
    pbar = tqdm(total=total_combinations, desc="Generating masked datasets")

    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"Strategy: {strategy}")
        print('='*80)

        # 创建masker
        masker = DeterministicTextMasker(tokenizer, strategy=strategy, base_seed=seed)

        for ratio in ratios:
            # 创建遮挡数据集
            masked_data = []

            for sample in data:
                # 获取样本ID
                sample_id = str(sample.get('id', ''))

                # 获取原始文本
                original_text = sample.get('text', '')

                # 确定性遮挡
                masked_text = masker.mask_text_deterministic(original_text, ratio, sample_id)

                # 创建新样本（保留所有字段，只替换text）
                masked_sample = sample.copy()
                masked_sample['text'] = masked_text
                masked_sample['original_text'] = original_text  # 保存原始文本供参考
                masked_sample['masking_ratio'] = ratio
                masked_sample['masking_strategy'] = strategy

                masked_data.append(masked_sample)

            # 保存遮挡数据集
            output_filename = f"{strategy}_{ratio:.1f}.pkl"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'wb') as f:
                pickle.dump(masked_data, f)

            # 打印统计
            if ratio == 0.0:
                num_fully_masked = 0
            else:
                num_fully_masked = sum(1 for s in masked_data if not s['text'] or s['text'].strip() == '[MASK]')

            print(f"  Ratio {ratio*100:5.1f}%: Saved {len(masked_data)} samples to {output_filename}")
            print(f"             Fully masked: {num_fully_masked}/{len(masked_data)}")

            pbar.update(1)

    pbar.close()

    # 生成索引文件
    index = {
        'seed': seed,
        'tokenizer': tokenizer_name,
        'num_samples': len(data),
        'strategies': strategies,
        'ratios': ratios,
        'files': {}
    }

    for strategy in strategies:
        index['files'][strategy] = {}
        for ratio in ratios:
            filename = f"{strategy}_{ratio:.1f}.pkl"
            index['files'][strategy][ratio] = filename

    index_path = os.path.join(output_dir, 'index.json')
    import json
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\n{'='*80}")
    print("Generation completed!")
    print(f"Output directory: {output_dir}")
    print(f"Index file: {index_path}")
    print(f"Total files: {total_combinations}")
    print('='*80)

    # 打印使用说明
    print("\n使用说明:")
    print("="*80)
    print("现在你可以使用相同的遮挡数据评估不同的模型:")
    print()
    print("# 评估模型1")
    print(f"python evaluate_text_masking.py \\")
    print(f"    --checkpoint /path/to/model1.pt \\")
    print(f"    --test_data {output_dir}/random_token_0.5.pkl \\")
    print(f"    --use_premasked_text")
    print()
    print("# 评估模型2（使用相同的遮挡数据）")
    print(f"python evaluate_text_masking.py \\")
    print(f"    --checkpoint /path/to/model2.pt \\")
    print(f"    --test_data {output_dir}/random_token_0.5.pkl \\")
    print(f"    --use_premasked_text")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='预先生成遮挡数据集以确保公平对比'
    )

    parser.add_argument(
        '--input_data',
        type=str,
        required=True,
        help='输入数据路径（pickle文件）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./masked_datasets',
        help='输出目录'
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        default=['random_token', 'random_word', 'sentence', 'keep_keywords'],
        choices=['random_token', 'random_word', 'random_chunk', 'sentence', 'keep_keywords'],
        help='遮挡策略'
    )
    parser.add_argument(
        '--ratios',
        nargs='+',
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help='遮挡率列表'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='m3rg-iitd/matscibert',
        help='Tokenizer名称或路径'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（确保可重复性）'
    )

    args = parser.parse_args()

    generate_masked_datasets(
        input_data_path=args.input_data,
        output_dir=args.output_dir,
        strategies=args.strategies,
        ratios=args.ratios,
        tokenizer_name=args.tokenizer,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
