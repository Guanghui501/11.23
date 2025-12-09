#!/usr/bin/env python
"""
文本遮挡鲁棒性评估脚本

这个脚本测试模型对文本信息缺失的鲁棒性，通过逐步增加文本遮挡率来观察模型性能变化。

用法示例:
    # 评估已训练模型
    python evaluate_text_masking.py \
        --checkpoint ./results/jarvis/formation_energy/best_model.pt \
        --preprocessed_dir ./preprocessed_data \
        --dataset jarvis \
        --property formation_energy \
        --masking_strategy random_token \
        --masking_ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

遮挡策略:
    - random_token: 随机遮挡单个token
    - random_word: 随机遮挡完整单词
    - random_chunk: 随机遮挡连续文本块
    - sentence: 随机遮挡完整句子
    - keep_keywords: 只保留关键词（元素名称、晶体结构等）
"""

import os
import sys
import json
import pickle
import argparse
from typing import List, Dict, Tuple
import random
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))

from data import get_train_val_loaders
from models.alignn import ALIGNN, ALIGNNConfig
from transformers import AutoTokenizer

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextMasker:
    """文本遮挡器 - 支持多种遮挡策略"""

    def __init__(self, tokenizer, strategy='random_token'):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            strategy: 遮挡策略 ('random_token', 'random_word', 'random_chunk', 'sentence', 'keep_keywords')
        """
        self.tokenizer = tokenizer
        self.strategy = strategy

        # 化学元素和关键词正则表达式
        self.element_pattern = re.compile(
            r'\b(H|He|Li|Be|B|C|N|O|F|Ne|Na|Mg|Al|Si|P|S|Cl|Ar|K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|'
            r'Ga|Ge|As|Se|Br|Kr|Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|Xe|Cs|Ba|La|Ce|'
            r'Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|'
            r'Fr|Ra|Ac|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr)\b'
        )
        self.structure_keywords = ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal', 'monoclinic',
                                  'triclinic', 'rhombohedral', 'space group', 'lattice', 'crystal']

    def mask_text(self, text: str, masking_ratio: float) -> str:
        """
        根据策略遮挡文本

        Args:
            text: 原始文本
            masking_ratio: 遮挡率 (0.0 - 1.0)

        Returns:
            遮挡后的文本
        """
        if masking_ratio == 0.0:
            return text

        if masking_ratio == 1.0:
            return ""  # 完全遮挡

        if self.strategy == 'random_token':
            return self._mask_random_tokens(text, masking_ratio)
        elif self.strategy == 'random_word':
            return self._mask_random_words(text, masking_ratio)
        elif self.strategy == 'random_chunk':
            return self._mask_random_chunks(text, masking_ratio)
        elif self.strategy == 'sentence':
            return self._mask_sentences(text, masking_ratio)
        elif self.strategy == 'keep_keywords':
            return self._keep_only_keywords(text, masking_ratio)
        else:
            raise ValueError(f"Unknown masking strategy: {self.strategy}")

    def _mask_random_tokens(self, text: str, ratio: float) -> str:
        """随机遮挡单个token"""
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) == 0:
            return text

        # 计算要遮挡的token数量
        num_to_mask = int(len(tokens) * ratio)
        if num_to_mask == 0:
            return text

        # 随机选择要遮挡的token
        mask_indices = random.sample(range(len(tokens)), min(num_to_mask, len(tokens)))

        # 用[MASK]替换
        for idx in mask_indices:
            tokens[idx] = '[MASK]'

        # 转回文本
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

        total_to_mask = int(len(words) * ratio)
        if total_to_mask == 0:
            return text

        # 随机决定块的数量（1-5个块）
        num_chunks = random.randint(1, min(5, len(words)))
        chunk_size = total_to_mask // num_chunks

        masked_words = words.copy()
        masked_count = 0

        for _ in range(num_chunks):
            if masked_count >= total_to_mask:
                break

            # 随机选择起始位置
            if len(words) - chunk_size <= 0:
                break
            start_idx = random.randint(0, len(words) - chunk_size)

            # 遮挡这个块
            for i in range(start_idx, min(start_idx + chunk_size, len(words))):
                if masked_words[i] != '[MASK]':
                    masked_words[i] = '[MASK]'
                    masked_count += 1

        return ' '.join(masked_words)

    def _mask_sentences(self, text: str, ratio: float) -> str:
        """随机遮挡完整句子"""
        # 按句子分割
        sentences = re.split(r'[.!?]\s+', text)
        if len(sentences) <= 1:
            return text

        num_to_mask = int(len(sentences) * ratio)
        if num_to_mask == 0:
            return text

        mask_indices = random.sample(range(len(sentences)), min(num_to_mask, len(sentences)))

        for idx in mask_indices:
            sentences[idx] = '[MASK]'

        return '. '.join(sentences)

    def _keep_only_keywords(self, text: str, ratio: float) -> str:
        """
        只保留关键词（元素、晶体结构术语等）
        ratio越高，保留的关键词越少
        """
        words = text.split()

        # 找出所有关键词
        keywords = []
        keyword_indices = []
        for i, word in enumerate(words):
            if (self.element_pattern.search(word) or
                any(kw in word.lower() for kw in self.structure_keywords)):
                keywords.append(i)
                keyword_indices.append(i)

        # 决定保留多少关键词（ratio越高保留越少）
        num_keywords_to_keep = int(len(keywords) * (1.0 - ratio))

        if num_keywords_to_keep > 0 and len(keywords) > 0:
            keep_indices = set(random.sample(keywords, min(num_keywords_to_keep, len(keywords))))
        else:
            keep_indices = set()

        # 构建结果
        result = []
        for i, word in enumerate(words):
            if i in keep_indices:
                result.append(word)
            elif i in keyword_indices:
                result.append('[MASK]')  # 被遮挡的关键词

        return ' '.join(result) if result else '[MASK]'


def load_preprocessed_dataset(preprocessed_dir, dataset_name, property_name):
    """加载预处理的数据集"""
    splits = {}

    for split_name in ['train', 'val', 'test']:
        pkl_file = os.path.join(preprocessed_dir, dataset_name, property_name, f'{split_name}.pkl')

        if not os.path.exists(pkl_file):
            raise FileNotFoundError(f"找不到预处理数据: {pkl_file}")

        print(f"加载 {split_name} 集: {pkl_file}")
        with open(pkl_file, 'rb') as f:
            samples = pickle.load(f)

        data = []
        for sample in samples:
            info = {
                "graph": sample['graph'][0],
                "line_graph": sample['line_graph'],
                "jid": sample['id'],
                "text": sample['text'],
                "target": sample['target']
            }
            data.append(info)

        splits[split_name] = data
        print(f"  ✓ 加载了 {len(data)} 个样本")

    return splits['train'], splits['val'], splits['test']


def evaluate_with_masking(
    model: nn.Module,
    test_loader: DataLoader,
    prepare_batch,
    tokenizer,
    masking_strategy: str,
    masking_ratio: float,
    device: torch.device
) -> Dict[str, float]:
    """
    在给定遮挡率下评估模型

    Returns:
        metrics: {'mae': float, 'rmse': float, 'r2': float}
    """
    model.eval()
    masker = TextMasker(tokenizer, masking_strategy)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Masking ratio {masking_ratio:.1%}", leave=False):
            # 准备batch
            batch_data = prepare_batch(batch, device)
            g, lg, target, text_list = batch_data

            # 遮挡文本
            masked_text_list = [masker.mask_text(text, masking_ratio) for text in text_list]

            # 前向传播
            out_data = model([g, lg, masked_text_list])
            prediction = out_data.cpu().numpy()

            all_predictions.extend(prediction.flatten())
            all_targets.extend(target.cpu().numpy().flatten())

    # 计算指标
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'targets': targets
    }


def plot_results(results: Dict[str, List], output_dir: str, property_name: str, strategy: str):
    """绘制评估结果"""
    masking_ratios = results['masking_ratios']

    # 设置风格
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. MAE vs Masking Ratio
    ax1 = axes[0, 0]
    ax1.plot(masking_ratios, results['mae'], 'o-', linewidth=2, markersize=8, label='MAE')
    ax1.set_xlabel('Text Masking Ratio', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_title(f'MAE vs Text Masking Ratio\n({property_name}, {strategy})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. RMSE vs Masking Ratio
    ax2 = axes[0, 1]
    ax2.plot(masking_ratios, results['rmse'], 's-', linewidth=2, markersize=8, label='RMSE', color='orange')
    ax2.set_xlabel('Text Masking Ratio', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title(f'RMSE vs Text Masking Ratio\n({property_name}, {strategy})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. R² vs Masking Ratio
    ax3 = axes[1, 0]
    ax3.plot(masking_ratios, results['r2'], '^-', linewidth=2, markersize=8, label='R²', color='green')
    ax3.set_xlabel('Text Masking Ratio', fontsize=12)
    ax3.set_ylabel('R² Score', fontsize=12)
    ax3.set_title(f'R² Score vs Text Masking Ratio\n({property_name}, {strategy})', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 4. All metrics normalized
    ax4 = axes[1, 1]
    # 归一化到 [0, 1]
    mae_norm = (np.array(results['mae']) - np.min(results['mae'])) / (np.max(results['mae']) - np.min(results['mae']) + 1e-8)
    rmse_norm = (np.array(results['rmse']) - np.min(results['rmse'])) / (np.max(results['rmse']) - np.min(results['rmse']) + 1e-8)
    r2_norm = (np.array(results['r2']) - np.min(results['r2'])) / (np.max(results['r2']) - np.min(results['r2']) + 1e-8)

    ax4.plot(masking_ratios, mae_norm, 'o-', linewidth=2, markersize=6, label='MAE (normalized)')
    ax4.plot(masking_ratios, rmse_norm, 's-', linewidth=2, markersize=6, label='RMSE (normalized)')
    ax4.plot(masking_ratios, r2_norm, '^-', linewidth=2, markersize=6, label='R² (normalized)')
    ax4.set_xlabel('Text Masking Ratio', fontsize=12)
    ax4.set_ylabel('Normalized Score', fontsize=12)
    ax4.set_title('All Metrics (Normalized)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()

    # 保存图片
    plot_file = os.path.join(output_dir, f'text_masking_analysis_{strategy}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存到: {plot_file}")
    plt.close()


def save_results(results: Dict, output_dir: str, property_name: str, strategy: str):
    """保存评估结果"""
    # 保存详细结果
    results_file = os.path.join(output_dir, f'text_masking_results_{strategy}.json')

    # 转换为Python原生类型（JSON可序列化）
    results_to_save = {
        'masking_ratios': [float(x) for x in results['masking_ratios']],
        'mae': [float(x) for x in results['mae']],
        'rmse': [float(x) for x in results['rmse']],
        'r2': [float(x) for x in results['r2']],
        'property': property_name,
        'strategy': strategy
    }

    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=4)

    print(f"结果已保存到: {results_file}")

    # 生成文本报告
    report_file = os.path.join(output_dir, f'text_masking_report_{strategy}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"文本遮挡鲁棒性评估报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"属性: {property_name}\n")
        f.write(f"遮挡策略: {strategy}\n\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Masking Ratio':<15} {'MAE':<15} {'RMSE':<15} {'R²':<15}\n")
        f.write("-"*80 + "\n")

        for i, ratio in enumerate(results['masking_ratios']):
            f.write(f"{ratio:<15.1%} {results['mae'][i]:<15.4f} {results['rmse'][i]:<15.4f} {results['r2'][i]:<15.4f}\n")

        f.write("-"*80 + "\n\n")

        # 性能下降分析
        baseline_mae = results['mae'][0]  # 0% masking
        final_mae = results['mae'][-1]    # 100% masking
        mae_increase = ((final_mae - baseline_mae) / baseline_mae) * 100

        f.write("性能下降分析:\n")
        f.write(f"  基线 MAE (0% masking): {baseline_mae:.4f}\n")
        f.write(f"  完全遮挡 MAE (100% masking): {final_mae:.4f}\n")
        f.write(f"  MAE 增加: {mae_increase:.2f}%\n\n")

        # 找出性能下降最快的区间
        max_gradient = 0
        max_gradient_idx = 0
        for i in range(1, len(results['mae'])):
            gradient = (results['mae'][i] - results['mae'][i-1]) / (results['masking_ratios'][i] - results['masking_ratios'][i-1])
            if gradient > max_gradient:
                max_gradient = gradient
                max_gradient_idx = i

        f.write(f"性能下降最快的区间:\n")
        f.write(f"  {results['masking_ratios'][max_gradient_idx-1]:.1%} -> {results['masking_ratios'][max_gradient_idx]:.1%}\n")
        f.write(f"  MAE 变化: {results['mae'][max_gradient_idx-1]:.4f} -> {results['mae'][max_gradient_idx]:.4f}\n")
        f.write(f"  梯度: {max_gradient:.4f}\n\n")

        f.write("="*80 + "\n")

    print(f"报告已保存到: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='文本遮挡鲁棒性评估')

    # 模型和数据
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--config_file', type=str, default=None,
                       help='模型配置文件路径 (JSON格式，当checkpoint缺少model_config时使用)')
    parser.add_argument('--preprocessed_dir', type=str, default='./preprocessed_data',
                       help='预处理数据目录')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['jarvis', 'mp'],
                       help='数据集名称')
    parser.add_argument('--property', type=str, required=True,
                       help='目标属性')

    # 遮挡参数
    parser.add_argument('--masking_strategy', type=str, default='random_token',
                       choices=['random_token', 'random_word', 'random_chunk', 'sentence', 'keep_keywords'],
                       help='遮挡策略')
    parser.add_argument('--masking_ratios', type=float, nargs='+',
                       default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                       help='遮挡率列表')

    # 其他参数
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批大小')
    parser.add_argument('--output_dir', type=str, default='./masking_evaluation',
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}\n")

    # 加载checkpoint
    print(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 获取模型配置
    model_config = None

    # 策略1: 从checkpoint的model_config字段加载
    if 'model_config' in checkpoint:
        print("✓ 从checkpoint加载model_config")
        model_config = checkpoint['model_config']
        if isinstance(model_config, dict):
            model_config = ALIGNNConfig(**model_config)

    # 策略2: 从checkpoint的config字段加载（某些checkpoint使用这个名称）
    elif 'config' in checkpoint:
        print("✓ 从checkpoint加载config")
        config = checkpoint['config']
        if isinstance(config, dict):
            model_config = ALIGNNConfig(**config)
        elif hasattr(config, '__dict__'):
            model_config = ALIGNNConfig(**vars(config))

    # 策略3: 从外部配置文件加载
    elif args.config_file:
        print(f"✓ 从外部配置文件加载: {args.config_file}")
        import json
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        model_config = ALIGNNConfig(**config_dict)

    # 策略4: 都失败了，提供详细的错误信息
    else:
        print("\n" + "="*80)
        print("❌ 错误: 无法加载模型配置")
        print("="*80)
        print("\nCheckpoint中没有找到model_config或config字段。")
        print("\n可用的checkpoint字段:")
        for key in checkpoint.keys():
            print(f"  - {key}")
        print("\n" + "="*80)
        print("解决方案:")
        print("="*80)
        print("\n1. 使用inspect_checkpoint.py检查checkpoint内容:")
        print(f"   python inspect_checkpoint.py {args.checkpoint}")
        print("\n2. 使用create_config_from_checkpoint.py创建配置文件:")
        print(f"   python create_config_from_checkpoint.py \\")
        print(f"       --checkpoint {args.checkpoint} \\")
        print(f"       --output model_config.json")
        print("\n3. 然后使用配置文件重新运行评估:")
        print(f"   python evaluate_text_masking.py \\")
        print(f"       --checkpoint {args.checkpoint} \\")
        print(f"       --config_file model_config.json \\")
        print(f"       --preprocessed_dir {args.preprocessed_dir} \\")
        print(f"       --dataset {args.dataset} \\")
        print(f"       --property {args.property}")
        print()
        raise ValueError("无法加载模型配置，请参考上述解决方案")

    # 创建模型
    model = ALIGNN(model_config)

    # 加载模型权重（使用strict=False以处理可能的键不匹配）
    try:
        # 首先尝试严格加载
        model.load_state_dict(checkpoint['model'], strict=True)
        print("✓ 模型加载成功（严格匹配）\n")
    except RuntimeError as e:
        # 如果严格加载失败，使用宽松加载
        error_msg = str(e)
        print(f"⚠ 严格加载失败，尝试宽松加载...")
        print(f"   原因: {error_msg}\n")

        # 使用strict=False加载
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)

        # 显示详细信息
        if unexpected_keys:
            print(f"⚠ Checkpoint中有 {len(unexpected_keys)} 个额外的键（将被忽略）:")
            for key in unexpected_keys[:5]:  # 只显示前5个
                print(f"   - {key}")
            if len(unexpected_keys) > 5:
                print(f"   ... 还有 {len(unexpected_keys) - 5} 个")
            print()

        if missing_keys:
            print(f"⚠ 模型中有 {len(missing_keys)} 个键缺失（将使用随机初始化）:")
            for key in missing_keys[:5]:  # 只显示前5个
                print(f"   - {key}")
            if len(missing_keys) > 5:
                print(f"   ... 还有 {len(missing_keys) - 5} 个")
            print()

            # 如果缺失的键太多，发出警告
            if len(missing_keys) > 10:
                print(f"❌ 警告: 缺失的键过多 ({len(missing_keys)} 个)!")
                print(f"   这可能导致评估结果不准确。")
                print(f"   请检查模型代码是否与训练时一致。")
                raise RuntimeError("模型加载失败：缺失键过多")

        print("✓ 模型加载成功（宽松匹配）")
        print("  注意: 部分参数不匹配，但已成功加载大部分权重\n")

    model = model.to(device)
    model.eval()

    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')
    print("✓ Tokenizer加载成功\n")

    # 加载测试数据
    print("加载测试数据...")
    _, _, test_data = load_preprocessed_dataset(
        args.preprocessed_dir,
        args.dataset,
        args.property
    )

    # 创建数据加载器 - 使用预处理数据的简化版本
    import dgl
    from torch.utils.data import Dataset, DataLoader

    class PreprocessedDataset(Dataset):
        """预处理数据的Dataset"""
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    def collate_preprocessed(samples):
        """Collate function for preprocessed data"""
        # 提取batch数据
        graphs = []
        for s in samples:
            g = s['graph']
            # 处理两种可能的格式：元组或直接是图对象
            if isinstance(g, tuple):
                graphs.append(g[0])
            else:
                graphs.append(g)

        line_graphs = [s['line_graph'] for s in samples]
        texts = [s['text'] for s in samples]
        targets = torch.tensor([s['target'] for s in samples], dtype=torch.float32)

        # 批处理图
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)

        return batched_graph, batched_line_graph, targets, texts

    def prepare_batch_preprocessed(batch, device):
        """Prepare batch for model input"""
        g, lg, target, text_list = batch
        g = g.to(device)
        lg = lg.to(device)
        target = target.to(device)
        return g, lg, target, text_list

    # 创建dataset和dataloader
    test_dataset = PreprocessedDataset(test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_preprocessed,
        num_workers=0,
        pin_memory=True
    )
    prepare_batch = prepare_batch_preprocessed

    print(f"✓ 测试集大小: {len(test_dataset)}\n")

    # 评估不同遮挡率
    print("="*80)
    print(f"开始文本遮挡鲁棒性评估")
    print(f"  策略: {args.masking_strategy}")
    print(f"  遮挡率: {args.masking_ratios}")
    print("="*80 + "\n")

    results = {
        'masking_ratios': [],
        'mae': [],
        'rmse': [],
        'r2': []
    }

    for ratio in args.masking_ratios:
        print(f"\n评估遮挡率: {ratio:.1%}")
        metrics = evaluate_with_masking(
            model=model,
            test_loader=test_loader,
            prepare_batch=prepare_batch,
            tokenizer=tokenizer,
            masking_strategy=args.masking_strategy,
            masking_ratio=ratio,
            device=device
        )

        results['masking_ratios'].append(ratio)
        results['mae'].append(metrics['mae'])
        results['rmse'].append(metrics['rmse'])
        results['r2'].append(metrics['r2'])

        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")

    # 保存结果
    print("\n" + "="*80)
    print("保存结果...")
    save_results(results, args.output_dir, args.property, args.masking_strategy)

    # 绘制图表
    print("\n生成可视化...")
    plot_results(results, args.output_dir, args.property, args.masking_strategy)

    print("\n" + "="*80)
    print("评估完成！")
    print(f"结果保存在: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
