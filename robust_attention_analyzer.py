#!/usr/bin/env python3
"""
全新的细粒度注意力分析系统
==========================

特性：
1. 自动诊断注意力模式是否正常
2. 兼容不同版本的模型代码
3. 即使所有原子注意力相同也能提供有用分析
4. 提供多种可视化和统计分析
5. 自动检测和报告异常

作者: Enhanced Analysis System
日期: 2025-11
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings


class RobustAttentionAnalyzer:
    """健壮的注意力分析器 - 能处理各种边界情况"""

    def __init__(self, model=None, device='cuda'):
        """
        Args:
            model: 可选的模型（用于特征提取）
            device: 计算设备
        """
        self.model = model
        self.device = device

        # 停用词列表
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', '[cls]', '[sep]', '[pad]',
            '##s', '##ed', '##ing', '##ly'
        }

    def diagnose_attention_quality(self,
                                   attention_weights: Dict[str, torch.Tensor],
                                   elements: List[str],
                                   verbose: bool = True) -> Dict:
        """
        诊断注意力权重质量，自动检测问题

        Args:
            attention_weights: 包含 'atom_to_text' 和 'text_to_atom' 的字典
            elements: 原子元素列表
            verbose: 是否打印详细信息

        Returns:
            诊断结果字典
        """

        if verbose:
            print("\n" + "="*80)
            print("🔬 注意力权重质量诊断")
            print("="*80)

        diagnosis = {
            'quality': 'unknown',
            'issues': [],
            'recommendations': [],
            'atom_diversity': 0.0,
            'head_diversity': 0.0,
            'entropy': 0.0,
            'use_alternative_analysis': False
        }

        # 提取 atom_to_text 注意力
        atom_to_text = attention_weights.get('atom_to_text', None)
        if atom_to_text is None:
            diagnosis['quality'] = 'missing'
            diagnosis['issues'].append("缺少 atom_to_text 注意力权重")
            return diagnosis

        # 转换为 numpy
        if isinstance(atom_to_text, torch.Tensor):
            atom_to_text = atom_to_text.cpu().numpy()

        # [batch, heads, num_atoms, seq_len]
        if len(atom_to_text.shape) == 4:
            atom_to_text = atom_to_text[0]  # 取第一个batch

        num_heads, num_atoms, seq_len = atom_to_text.shape

        if verbose:
            print(f"\n1️⃣ 基本信息:")
            print(f"   - Attention heads: {num_heads}")
            print(f"   - Atoms: {num_atoms}")
            print(f"   - Sequence length: {seq_len}")

        # 检查1: 不同head是否有差异
        head_correlations = []
        for i in range(num_heads - 1):
            corr = np.corrcoef(
                atom_to_text[i].flatten(),
                atom_to_text[i + 1].flatten()
            )[0, 1]
            head_correlations.append(corr)

        avg_head_corr = np.mean(head_correlations)
        diagnosis['head_diversity'] = 1.0 - avg_head_corr

        if verbose:
            print(f"\n2️⃣ 多头注意力分析:")
            print(f"   - 平均头间相关性: {avg_head_corr:.4f}")
            print(f"   - 头多样性分数: {diagnosis['head_diversity']:.4f}")

        if avg_head_corr > 0.99:
            diagnosis['issues'].append("所有attention heads几乎相同（多头退化）")

        # 检查2: 不同原子是否有差异
        atom_to_text_avg = atom_to_text.mean(axis=0)  # [num_atoms, seq_len]

        atom_correlations = []
        for i in range(num_atoms - 1):
            if num_atoms > 1:
                corr = np.corrcoef(
                    atom_to_text_avg[i],
                    atom_to_text_avg[i + 1]
                )[0, 1]
                atom_correlations.append(corr)

        if atom_correlations:
            avg_atom_corr = np.mean(atom_correlations)
            diagnosis['atom_diversity'] = 1.0 - avg_atom_corr

            if verbose:
                print(f"\n3️⃣ 原子特异性分析:")
                print(f"   - 平均原子间相关性: {avg_atom_corr:.4f}")
                print(f"   - 原子多样性分数: {diagnosis['atom_diversity']:.4f}")

            if avg_atom_corr > 0.99:
                diagnosis['issues'].append("所有原子的注意力模式几乎相同")
                diagnosis['use_alternative_analysis'] = True

        # 检查3: 注意力分布的熵（是否过于集中）
        # 计算每个原子的注意力熵
        entropies = []
        for i in range(num_atoms):
            p = atom_to_text_avg[i] + 1e-10  # 避免log(0)
            entropy = -np.sum(p * np.log(p))
            entropies.append(entropy)

        diagnosis['entropy'] = np.mean(entropies)

        if verbose:
            print(f"\n4️⃣ 注意力分布分析:")
            print(f"   - 平均熵: {diagnosis['entropy']:.4f}")
            print(f"   - 最大可能熵: {np.log(seq_len):.4f}")

        if diagnosis['entropy'] < 1.0:
            diagnosis['issues'].append("注意力分布过于集中（低熵）")

        # 综合评估
        if len(diagnosis['issues']) == 0:
            diagnosis['quality'] = 'good'
        elif len(diagnosis['issues']) <= 2:
            diagnosis['quality'] = 'acceptable'
        else:
            diagnosis['quality'] = 'poor'

        # 生成建议
        if diagnosis['use_alternative_analysis']:
            diagnosis['recommendations'].append(
                "建议使用全局分析而非逐原子分析"
            )
            diagnosis['recommendations'].append(
                "检查GNN层输出的节点特征是否过于相似"
            )
            diagnosis['recommendations'].append(
                "考虑减少GNN层数或添加残差连接"
            )

        if diagnosis['head_diversity'] < 0.1:
            diagnosis['recommendations'].append(
                "多头注意力退化，考虑增加head diversity正则化"
            )

        if verbose:
            print(f"\n5️⃣ 诊断结论:")
            print(f"   - 质量评估: {diagnosis['quality'].upper()}")
            if diagnosis['issues']:
                print(f"   - 发现问题:")
                for issue in diagnosis['issues']:
                    print(f"      • {issue}")
            if diagnosis['recommendations']:
                print(f"   - 建议:")
                for rec in diagnosis['recommendations']:
                    print(f"      • {rec}")
            print("="*80 + "\n")

        return diagnosis

    def analyze_with_fallback(self,
                              attention_weights: Dict[str, torch.Tensor],
                              atoms_object,
                              text_tokens: List[str],
                              save_dir: Optional[Path] = None,
                              top_k: int = 15) -> Dict:
        """
        带降级策略的分析：如果逐原子分析失败，自动切换到全局分析

        Args:
            attention_weights: 注意力权重字典
            atoms_object: JARVIS Atoms对象
            text_tokens: 文本token列表
            save_dir: 保存目录
            top_k: 显示top-k结果

        Returns:
            分析结果字典
        """
        elements = [str(atoms_object.elements[i]) for i in range(atoms_object.num_atoms)]

        # 首先诊断质量
        diagnosis = self.diagnose_attention_quality(
            attention_weights, elements, verbose=True
        )

        results = {'diagnosis': diagnosis}

        # 根据诊断结果选择分析策略
        if diagnosis['use_alternative_analysis']:
            print("⚠️  检测到原子注意力模式相同，使用全局分析策略...\n")
            results['global_analysis'] = self._analyze_global_patterns(
                attention_weights, atoms_object, text_tokens, save_dir, top_k
            )
        else:
            print("✅ 原子注意力模式正常，使用标准分析...\n")
            results['per_atom_analysis'] = self._analyze_per_atom(
                attention_weights, atoms_object, text_tokens, save_dir, top_k
            )

        # 无论如何都做统计分析
        results['statistics'] = self._compute_statistics(
            attention_weights, elements, text_tokens
        )

        return results

    def _analyze_global_patterns(self,
                                 attention_weights: Dict[str, torch.Tensor],
                                 atoms_object,
                                 text_tokens: List[str],
                                 save_dir: Optional[Path],
                                 top_k: int) -> Dict:
        """
        全局分析：当所有原子注意力相同时，分析整体模式
        """
        print("="*80)
        print("📊 全局注意力模式分析")
        print("="*80)

        atom_to_text = attention_weights['atom_to_text']
        if isinstance(atom_to_text, torch.Tensor):
            atom_to_text = atom_to_text.cpu().numpy()

        if len(atom_to_text.shape) == 4:
            atom_to_text = atom_to_text[0]

        # 对所有原子和所有头取平均
        global_attention = atom_to_text.mean(axis=(0, 1))  # [seq_len]

        # 获取top-k最重要的tokens
        top_indices = global_attention.argsort()[-top_k:][::-1]

        results = {
            'top_tokens': [],
            'token_categories': {},
            'visualization_path': None
        }

        print(f"\n🔤 全局最重要的 {top_k} 个 Tokens:")
        print(f"{'Rank':<6} {'Token':<20} {'Importance':<12} {'Category'}")
        print("-" * 60)

        for rank, idx in enumerate(top_indices, 1):
            if idx < len(text_tokens):
                token = text_tokens[idx]
                importance = global_attention[idx]
                category = self._categorize_token(token)

                results['top_tokens'].append({
                    'token': token,
                    'importance': float(importance),
                    'category': category
                })

                # 统计类别
                if category not in results['token_categories']:
                    results['token_categories'][category] = 0
                results['token_categories'][category] += 1

                print(f"{rank:<6} {token:<20} {importance:<12.6f} {category}")

        # 可视化
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 1. Top tokens柱状图
            tokens_display = [text_tokens[i] if i < len(text_tokens) else f"[{i}]"
                            for i in top_indices[:10]]
            importances = [global_attention[i] for i in top_indices[:10]]

            axes[0, 0].barh(range(10), importances[::-1], color='steelblue')
            axes[0, 0].set_yticks(range(10))
            axes[0, 0].set_yticklabels(tokens_display[::-1])
            axes[0, 0].set_xlabel('Attention Weight')
            axes[0, 0].set_title('Top 10 Most Important Tokens (Global)', fontweight='bold')
            axes[0, 0].grid(axis='x', alpha=0.3)

            # 2. Token类别分布
            categories = list(results['token_categories'].keys())
            counts = list(results['token_categories'].values())
            axes[0, 1].pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Token Category Distribution', fontweight='bold')

            # 3. 注意力分布热图（所有heads平均）
            avg_per_head = atom_to_text.mean(axis=1)  # [heads, seq_len]
            top_head_idx = avg_per_head.max(axis=1).argmax()

            # 显示最活跃的head的注意力
            sns.heatmap(
                atom_to_text[top_head_idx, :, :min(50, len(text_tokens))],
                xticklabels=text_tokens[:min(50, len(text_tokens))],
                yticklabels=[f"{atoms_object.elements[i]}_{i}" for i in range(atom_to_text.shape[1])],
                cmap='YlOrRd',
                ax=axes[1, 0],
                cbar_kws={'label': 'Attention Weight'}
            )
            axes[1, 0].set_title(f'Most Active Head (Head {top_head_idx})', fontweight='bold')
            axes[1, 0].set_xlabel('Text Tokens (first 50)')
            axes[1, 0].set_ylabel('Atoms')
            plt.setp(axes[1, 0].get_xticklabels(), rotation=90, ha='right', fontsize=7)

            # 4. 注意力权重分布直方图
            all_weights = global_attention.flatten()
            axes[1, 1].hist(all_weights, bins=50, color='coral', alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Attention Weight')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Attention Weight Distribution', fontweight='bold')
            axes[1, 1].axvline(all_weights.mean(), color='red', linestyle='--',
                              label=f'Mean: {all_weights.mean():.4f}')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)

            plt.suptitle('Global Attention Pattern Analysis\n(All atoms show similar patterns)',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()

            viz_path = save_dir / 'global_attention_analysis.png'
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()

            results['visualization_path'] = str(viz_path)
            print(f"\n✅ 可视化已保存: {viz_path}")

        print("="*80 + "\n")
        return results

    def _analyze_per_atom(self,
                         attention_weights: Dict[str, torch.Tensor],
                         atoms_object,
                         text_tokens: List[str],
                         save_dir: Optional[Path],
                         top_k: int) -> Dict:
        """
        逐原子分析：当原子注意力模式不同时使用
        """
        print("="*80)
        print("⚛️  逐原子注意力分析")
        print("="*80)

        atom_to_text = attention_weights['atom_to_text']
        if isinstance(atom_to_text, torch.Tensor):
            atom_to_text = atom_to_text.cpu().numpy()

        if len(atom_to_text.shape) == 4:
            atom_to_text = atom_to_text[0]

        # 对heads取平均
        atom_to_text_avg = atom_to_text.mean(axis=0)  # [num_atoms, seq_len]

        results = {'atoms': {}}

        elements = [str(atoms_object.elements[i]) for i in range(atoms_object.num_atoms)]

        # 分析每个原子
        for i, element in enumerate(elements):
            atom_attention = atom_to_text_avg[i]
            top_indices = atom_attention.argsort()[-top_k:][::-1]

            atom_results = {
                'element': element,
                'index': i,
                'top_tokens': []
            }

            for idx in top_indices:
                if idx < len(text_tokens):
                    token = text_tokens[idx]
                    if token.lower() not in self.stopwords:
                        atom_results['top_tokens'].append({
                            'token': token,
                            'weight': float(atom_attention[idx])
                        })

            results['atoms'][f"{element}_{i}"] = atom_results

            # 打印
            print(f"\n{element}_{i}:")
            for item in atom_results['top_tokens'][:5]:
                print(f"  - {item['token']:<20} {item['weight']:.6f}")

        # 可视化
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            num_atoms = len(elements)
            fig_height = max(8, num_atoms * 0.8)

            fig, ax = plt.subplots(1, 1, figsize=(14, fig_height))

            # 创建热图数据：每个原子的top-10 tokens
            max_display = min(10, top_k)
            heatmap_data = np.zeros((num_atoms, max_display))
            token_labels = []

            for i in range(num_atoms):
                atom_key = f"{elements[i]}_{i}"
                top_tokens = results['atoms'][atom_key]['top_tokens'][:max_display]

                for j, item in enumerate(top_tokens):
                    heatmap_data[i, j] = item['weight']
                    if i == 0:  # 只在第一行记录token名称
                        token_labels.append(item['token'])

            # 填充token标签
            while len(token_labels) < max_display:
                token_labels.append('')

            sns.heatmap(
                heatmap_data,
                xticklabels=token_labels,
                yticklabels=[f"{elements[i]}_{i}" for i in range(num_atoms)],
                cmap='YlOrRd',
                ax=ax,
                annot=True,
                fmt='.4f',
                cbar_kws={'label': 'Attention Weight'}
            )

            ax.set_title(f'Per-Atom Top {max_display} Attended Tokens', fontweight='bold', fontsize=14)
            ax.set_xlabel('Top Tokens', fontsize=11)
            ax.set_ylabel('Atoms', fontsize=11)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            plt.tight_layout()

            viz_path = save_dir / 'per_atom_attention.png'
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()

            results['visualization_path'] = str(viz_path)
            print(f"\n✅ 可视化已保存: {viz_path}")

        print("="*80 + "\n")
        return results

    def _compute_statistics(self,
                           attention_weights: Dict[str, torch.Tensor],
                           elements: List[str],
                           text_tokens: List[str]) -> Dict:
        """计算统计信息"""

        atom_to_text = attention_weights['atom_to_text']
        if isinstance(atom_to_text, torch.Tensor):
            atom_to_text = atom_to_text.cpu().numpy()

        if len(atom_to_text.shape) == 4:
            atom_to_text = atom_to_text[0]

        stats = {
            'num_heads': atom_to_text.shape[0],
            'num_atoms': atom_to_text.shape[1],
            'seq_len': atom_to_text.shape[2],
            'mean_attention': float(atom_to_text.mean()),
            'std_attention': float(atom_to_text.std()),
            'max_attention': float(atom_to_text.max()),
            'min_attention': float(atom_to_text.min()),
            'sparsity': float((atom_to_text < 0.01).sum() / atom_to_text.size)
        }

        return stats

    def _categorize_token(self, token: str) -> str:
        """将token分类"""
        token_lower = token.lower().replace('##', '')

        # 元素符号
        elements = {'h', 'he', 'li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne',
                   'na', 'mg', 'al', 'si', 'p', 's', 'cl', 'ar', 'k', 'ca',
                   'ba', 'hf', 'ti', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd'}

        # 数字
        if token_lower.isdigit():
            return 'Number'

        # 元素
        if token_lower in elements or any(token_lower.startswith(e) for e in elements):
            return 'Element'

        # 晶体学术语
        crystal_terms = {'cubic', 'tetragonal', 'orthorhombic', 'monoclinic',
                        'triclinic', 'hexagonal', 'rhombohedral', 'space', 'group',
                        'coordinate', 'symmetry', 'lattice', 'framework'}
        if any(term in token_lower for term in crystal_terms):
            return 'Crystallography'

        # 化学术语
        chem_terms = {'bond', 'atom', 'molecule', 'cluster', 'ion', 'electron',
                     'oxidation', 'valence', 'coordination'}
        if any(term in token_lower for term in chem_terms):
            return 'Chemistry'

        # 停用词
        if token_lower in self.stopwords:
            return 'Stopword'

        return 'Other'


def run_complete_analysis(model, g, lg, text, atoms_object, save_dir=None):
    """
    运行完整的注意力分析

    Args:
        model: 训练好的模型
        g, lg: DGL图
        text: 文本描述
        atoms_object: JARVIS Atoms对象
        save_dir: 保存目录
    """
    from transformers import BertTokenizer

    device = next(model.parameters()).device
    g = g.to(device)
    lg = lg.to(device)

    # 获取预测和注意力
    with torch.no_grad():
        output = model(
            [g, lg, [text]],
            return_features=True,
            return_attention=True
        )

    prediction = output['predictions'].cpu().item()
    fg_attn = output.get('fine_grained_attention_weights', None)

    if fg_attn is None:
        print("❌ 模型未返回 fine-grained attention weights")
        return None

    print(f"\n✅ 预测值: {prediction:.4f}")

    # Tokenize文本
    tokenizer = BertTokenizer.from_pretrained('m3rg-iitd/matscibert')
    tokens = tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    # 对齐长度
    seq_len = fg_attn['atom_to_text'].shape[-1]
    if len(tokens) > seq_len:
        tokens = tokens[:seq_len]
    elif len(tokens) < seq_len:
        tokens = tokens + ['[PAD]'] * (seq_len - len(tokens))

    # 创建分析器并运行
    analyzer = RobustAttentionAnalyzer(model, device)
    results = analyzer.analyze_with_fallback(
        fg_attn,
        atoms_object,
        tokens,
        save_dir=save_dir,
        top_k=15
    )

    return results


if __name__ == '__main__':
    print("Robust Attention Analyzer - 可独立运行或作为模块导入")
    print("使用 run_complete_analysis() 函数进行完整分析")
