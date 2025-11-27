#!/usr/bin/env python
"""
双模型CKA对比脚本
用于计算两个不同模型（如baseline vs SGANet）在相同特征阶段的CKA相似度
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

# 导入模型和数据加载器
import sys
sys.path.insert(0, os.path.dirname(__file__))
from models.alignn import ALIGNN
from train_with_cross_modal_attention import load_dataset, get_dataset_paths
from data import get_train_val_loaders


def centered_kernel_alignment(X, Y):
    """
    计算 CKA (Centered Kernel Alignment) 相似度

    Args:
        X: 特征矩阵1 [N, D1]
        Y: 特征矩阵2 [N, D2]

    Returns:
        CKA score (0-1之间，越高越相似)
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    K = X @ X.T
    L = Y @ Y.T
    hsic = np.sum(K * L)
    denom = np.sqrt(np.sum(K * K) * np.sum(L * L))
    return hsic / denom if denom > 0 else 0.0


def load_model(path, device):
    """加载模型"""
    print(f"📦 加载模型: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get('config') or ckpt.get('model_config')
    model = ALIGNN(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    # 打印模型配置
    print(f"   配置: 中期融合={model.use_middle_fusion}, "
          f"细粒度注意力={model.use_fine_grained_attention}, "
          f"全局注意力={model.use_cross_modal_attention}")

    return model


def extract_all_stage_features(model, loader, device, max_samples=None):
    """
    提取所有阶段的特征

    Returns:
        features_dict: {
            'graph_base': [...],
            'graph_middle': [...],
            'graph_fine': [...],
            'graph_final': [...],
            'text_base': [...],
            'text_fine': [...],
            'text_final': [...],
            'fused': [...]
        }
        targets: 目标值
    """
    print("🔄 提取所有阶段的特征...")

    features_dict = {
        'graph_base': [],
        'graph_middle': [],
        'graph_fine': [],
        'graph_final': [],
        'text_base': [],
        'text_fine': [],
        'text_final': [],
        'fused': []
    }
    targets = []

    sample_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="提取特征"):
            if len(batch) == 3:
                g, text, y = batch
                lg = None
            elif len(batch) == 4:
                g, lg, text, y = batch
            else:
                raise ValueError(f"不支持的batch格式: {len(batch)}个元素")

            g = g.to(device)
            if lg is not None:
                lg = lg.to(device)

            # 处理text
            if isinstance(text, dict):
                text = {k: v.to(device) for k, v in text.items()}
            elif isinstance(text, (list, tuple)):
                pass
            elif torch.is_tensor(text):
                text = text.to(device)

            # 构建模型输入
            if lg is not None:
                model_input = (g, lg, text)
            else:
                model_input = (g, text)

            # 提取所有中间特征
            out = model(model_input, return_intermediate_features=True)

            # 提取各阶段特征
            if 'graph_base' in out:
                features_dict['graph_base'].append(out['graph_base'].cpu().numpy())

            if 'graph_middle' in out:
                features_dict['graph_middle'].append(out['graph_middle'].cpu().numpy())

            if 'graph_fine' in out:
                features_dict['graph_fine'].append(out['graph_fine'].cpu().numpy())

            if 'graph_features' in out:
                features_dict['graph_final'].append(out['graph_features'].cpu().numpy())

            if 'text_base' in out:
                features_dict['text_base'].append(out['text_base'].cpu().numpy())

            if 'text_fine' in out:
                features_dict['text_fine'].append(out['text_fine'].cpu().numpy())

            if 'text_features' in out:
                features_dict['text_final'].append(out['text_features'].cpu().numpy())

            # 融合特征
            graph_feat = out.get('graph_features')
            text_feat = out.get('text_features')
            if graph_feat is not None and text_feat is not None:
                fused = torch.cat([graph_feat, text_feat], dim=1)
                features_dict['fused'].append(fused.cpu().numpy())

            targets.append(y.cpu().numpy())

            sample_count += y.size(0)
            if max_samples and sample_count >= max_samples:
                break

    # 转换为numpy数组
    for key in list(features_dict.keys()):
        if len(features_dict[key]) > 0:
            features_dict[key] = np.vstack(features_dict[key])
        else:
            del features_dict[key]  # 删除空特征

    targets = np.concatenate(targets)

    print(f"✅ 提取完成! 样本数: {len(targets)}, 特征阶段: {list(features_dict.keys())}")

    return features_dict, targets


def compute_twin_model_cka(features_model1, features_model2, model1_name='Model 1', model2_name='Model 2'):
    """
    计算两个模型在相同阶段的CKA相似度

    Args:
        features_model1: 模型1的特征字典
        features_model2: 模型2的特征字典
        model1_name: 模型1的名称
        model2_name: 模型2的名称

    Returns:
        cka_scores: {stage: cka_score}
    """
    print(f"\n🔍 计算 {model1_name} vs {model2_name} 的CKA相似度...")

    # 找到两个模型共有的特征阶段
    common_stages = set(features_model1.keys()) & set(features_model2.keys())

    if not common_stages:
        print("⚠️  两个模型没有共同的特征阶段!")
        return {}

    print(f"   共同阶段: {sorted(common_stages)}")

    cka_scores = {}

    for stage in sorted(common_stages):
        print(f"   计算阶段: {stage}")
        feat1 = features_model1[stage]
        feat2 = features_model2[stage]

        # 确保样本数一致
        min_samples = min(len(feat1), len(feat2))
        feat1 = feat1[:min_samples]
        feat2 = feat2[:min_samples]

        # 计算CKA
        cka = centered_kernel_alignment(feat1, feat2)
        cka_scores[stage] = cka

        print(f"      {stage}: {cka:.4f}")

    return cka_scores


def visualize_cka_scores(cka_scores, model1_name, model2_name, save_dir):
    """
    可视化CKA分数

    Args:
        cka_scores: {stage: cka_score}
        model1_name: 模型1名称
        model2_name: 模型2名称
        save_dir: 保存目录
    """
    print("\n📊 生成CKA分数可视化...")

    if not cka_scores:
        print("⚠️  没有CKA分数可视化")
        return

    # 准备数据
    stages = list(cka_scores.keys())
    scores = list(cka_scores.values())

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 子图1: 柱状图
    colors = plt.cm.RdYlGn([s for s in scores])
    bars = ax1.bar(range(len(stages)), scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(stages)))
    ax1.set_xticklabels(stages, rotation=45, ha='right')
    ax1.set_ylabel('CKA Similarity Score', fontweight='bold', fontsize=12)
    ax1.set_title(f'CKA Similarity: {model1_name} vs {model2_name}',
                 fontweight='bold', fontsize=14, pad=15)
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0.9, color='green', linestyle='--', linewidth=1, alpha=0.5, label='High (0.9)')
    ax1.axhline(y=0.7, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Moderate (0.7)')
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Low (0.5)')
    ax1.legend(loc='lower right', fontsize=10)

    # 标注数值
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 子图2: 阶段演变线图
    stage_order = ['graph_base', 'graph_middle', 'graph_fine', 'graph_final',
                   'text_base', 'text_fine', 'text_final', 'fused']
    ordered_stages = [s for s in stage_order if s in stages]
    ordered_scores = [cka_scores[s] for s in ordered_stages]

    ax2.plot(range(len(ordered_stages)), ordered_scores, 'o-',
            linewidth=2, markersize=10, color='steelblue',
            markeredgecolor='black', markeredgewidth=1.5)
    ax2.set_xticks(range(len(ordered_stages)))
    ax2.set_xticklabels(ordered_stages, rotation=45, ha='right')
    ax2.set_ylabel('CKA Similarity Score', fontweight='bold', fontsize=12)
    ax2.set_title('CKA Evolution Across Processing Stages',
                 fontweight='bold', fontsize=14, pad=15)
    ax2.set_ylim([0, 1.0])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=0.9, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=0.7, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # 标注数值
    for i, score in enumerate(ordered_scores):
        ax2.text(i, score + 0.02, f'{score:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'twin_models_cka_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ CKA可视化已保存: {save_path}")
    plt.close()


def generate_cka_report(cka_scores, model1_name, model2_name, save_dir):
    """
    生成CKA对比报告

    Args:
        cka_scores: {stage: cka_score}
        model1_name: 模型1名称
        model2_name: 模型2名称
        save_dir: 保存目录
    """
    print("\n📝 生成CKA对比报告...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"Twin Model CKA Similarity Report")
    report_lines.append(f"Model 1: {model1_name}")
    report_lines.append(f"Model 2: {model2_name}")
    report_lines.append("=" * 80)
    report_lines.append("")

    # 1. 整体统计
    scores = list(cka_scores.values())
    report_lines.append("📊 Overall Statistics:")
    report_lines.append(f"  • Mean CKA Score: {np.mean(scores):.4f}")
    report_lines.append(f"  • Median CKA Score: {np.median(scores):.4f}")
    report_lines.append(f"  • Min CKA Score: {np.min(scores):.4f} (Stage: {min(cka_scores, key=cka_scores.get)})")
    report_lines.append(f"  • Max CKA Score: {np.max(scores):.4f} (Stage: {max(cka_scores, key=cka_scores.get)})")
    report_lines.append(f"  • Std CKA Score: {np.std(scores):.4f}")
    report_lines.append("")

    # 2. 各阶段详细分数
    report_lines.append("🔬 Stage-by-Stage CKA Scores:")
    report_lines.append("")

    stage_descriptions = {
        'graph_base': 'Graph Base (GCN后，注意力前)',
        'graph_middle': 'Graph Middle (中期融合后)',
        'graph_fine': 'Graph Fine (细粒度注意力后)',
        'graph_final': 'Graph Final (最终图特征)',
        'text_base': 'Text Base (初始文本特征)',
        'text_fine': 'Text Fine (细粒度注意力后)',
        'text_final': 'Text Final (最终文本特征)',
        'fused': 'Fused (图+文本融合特征)'
    }

    for stage in sorted(cka_scores.keys()):
        score = cka_scores[stage]
        desc = stage_descriptions.get(stage, stage)

        # 解释分数
        if score > 0.9:
            interpretation = "极高相似度 - 两个模型学到了几乎相同的表示"
        elif score > 0.7:
            interpretation = "高相似度 - 两个模型学到了相似的主要模式"
        elif score > 0.5:
            interpretation = "中等相似度 - 两个模型有显著差异"
        else:
            interpretation = "低相似度 - 两个模型学到了非常不同的表示"

        report_lines.append(f"  • {desc}")
        report_lines.append(f"    Stage: {stage}")
        report_lines.append(f"    CKA Score: {score:.4f}")
        report_lines.append(f"    解释: {interpretation}")
        report_lines.append("")

    # 3. 关键发现
    report_lines.append("🔍 Key Findings:")

    # 找出最相似和最不相似的阶段
    max_stage = max(cka_scores, key=cka_scores.get)
    min_stage = min(cka_scores, key=cka_scores.get)

    report_lines.append(f"  • 最相似阶段: {max_stage} (CKA = {cka_scores[max_stage]:.4f})")
    report_lines.append(f"    → 两个模型在此阶段的表示最接近")
    report_lines.append("")
    report_lines.append(f"  • 最不相似阶段: {min_stage} (CKA = {cka_scores[min_stage]:.4f})")
    report_lines.append(f"    → 两个模型在此阶段的差异最大")
    report_lines.append("")

    # 融合效果分析
    if 'graph_base' in cka_scores and 'graph_final' in cka_scores:
        base_cka = cka_scores['graph_base']
        final_cka = cka_scores['graph_final']
        delta = final_cka - base_cka

        report_lines.append("  • 融合过程的影响:")
        report_lines.append(f"    Graph Base CKA: {base_cka:.4f}")
        report_lines.append(f"    Graph Final CKA: {final_cka:.4f}")
        report_lines.append(f"    变化: {delta:+.4f}")

        if delta > 0.05:
            report_lines.append(f"    → 融合过程使两个模型的表示更加相似")
        elif delta < -0.05:
            report_lines.append(f"    → 融合过程增加了两个模型的差异")
        else:
            report_lines.append(f"    → 融合过程对相似度影响较小")
        report_lines.append("")

    # 4. 建议
    report_lines.append("💡 Insights and Recommendations:")
    avg_cka = np.mean(scores)

    if avg_cka > 0.85:
        report_lines.append("  • 两个模型整体相似度很高")
        report_lines.append("  • 可能原因: 模型架构接近，训练数据相同，融合机制影响有限")
        report_lines.append("  • 建议: 如果希望增加多样性，可以尝试更强的融合机制")
    elif avg_cka > 0.65:
        report_lines.append("  • 两个模型保持了适度的相似性和差异性")
        report_lines.append("  • 融合机制带来了可观察的变化，但保留了基础表示")
        report_lines.append("  • 建议: 当前配置较为合理，可以分析具体阶段的差异来优化")
    else:
        report_lines.append("  • 两个模型的表示差异较大")
        report_lines.append("  • 可能原因: 融合机制大幅改变了特征空间，或训练不稳定")
        report_lines.append("  • 建议: 检查训练过程，确保模型收敛；分析是否有信息损失")

    report_lines.append("")
    report_lines.append("=" * 80)

    # 保存报告
    report_text = "\n".join(report_lines)
    save_path = os.path.join(save_dir, 'twin_models_cka_report.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"✅ CKA报告已保存: {save_path}")
    print("\n" + report_text)

    # 保存CSV
    df = pd.DataFrame([
        {'Stage': stage, 'CKA_Score': score}
        for stage, score in sorted(cka_scores.items())
    ])
    csv_path = os.path.join(save_dir, 'twin_models_cka_scores.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ CKA分数CSV已保存: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='双模型CKA相似度对比')
    parser.add_argument('--ckpt_model1', type=str, required=True,
                       help='模型1的checkpoint路径 (如baseline)')
    parser.add_argument('--ckpt_model2', type=str, required=True,
                       help='模型2的checkpoint路径 (如SGANet)')
    parser.add_argument('--model1_name', type=str, default='Model 1',
                       help='模型1的名称')
    parser.add_argument('--model2_name', type=str, default='Model 2',
                       help='模型2的名称')
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集类型')
    parser.add_argument('--property', type=str, required=True,
                       help='目标属性')
    parser.add_argument('--root_dir', type=str,
                       default='/public/home/ghzhang/crysmmnet-main/dataset',
                       help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='最大样本数')
    parser.add_argument('--save_dir', type=str, default='./twin_cka_comparison',
                       help='结果保存目录')
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}\n")

    # 加载模型1
    print("=" * 80)
    model1 = load_model(args.ckpt_model1, device)

    # 加载模型2
    print("=" * 80)
    model2 = load_model(args.ckpt_model2, device)
    print("=" * 80)

    # 加载数据集
    print(f"\n🔄 加载数据集: {args.dataset} - {args.property}")

    try:
        # 获取数据集路径
        cif_dir, id_prop_file = get_dataset_paths(args.root_dir, args.dataset, args.property)

        # 加载数据集
        df = load_dataset(cif_dir, id_prop_file, args.dataset, args.property)
        print(f"✅ 加载数据集: {len(df)} 样本")

        # 采样
        if args.max_samples and len(df) > args.max_samples:
            print(f"⚠️  数据集过大，随机采样 {args.max_samples} 样本")
            import random
            random.seed(42)
            df = random.sample(df, args.max_samples)

        # 获取模型1的配置
        config = model1.config if hasattr(model1, 'config') else None
        if config is None:
            # 尝试从checkpoint加载
            ckpt = torch.load(args.ckpt_model1, map_location='cpu', weights_only=False)
            config = ckpt.get('config') or ckpt.get('model_config')

        # 创建数据加载器
        train_loader, val_loader, test_loader, _ = get_train_val_loaders(
            dataset='user_data',
            dataset_array=df,
            target='target',
            n_train=None,
            n_val=None,
            n_test=None,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            batch_size=args.batch_size,
            atom_features=config.atom_features if hasattr(config, 'atom_features') else 'cgcnn',
            neighbor_strategy='k-nearest',
            line_graph=config.line_graph if hasattr(config, 'line_graph') else True,
            split_seed=42,
            workers=0,
            pin_memory=False,
            save_dataloader=False,
            filename='temp_twin_cka',
            id_tag='jid',
            use_canonize=True,
            cutoff=8.0,
            max_neighbors=12,
            output_dir=args.save_dir
        )

        print(f"✅ 测试集样本数: {len(test_loader.dataset)}")

    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        raise

    # 提取模型1的特征
    print("\n" + "=" * 80)
    print(f"提取 {args.model1_name} 的特征")
    print("=" * 80)
    features_model1, targets1 = extract_all_stage_features(
        model1, test_loader, device, max_samples=args.max_samples
    )

    # 提取模型2的特征
    print("\n" + "=" * 80)
    print(f"提取 {args.model2_name} 的特征")
    print("=" * 80)
    features_model2, targets2 = extract_all_stage_features(
        model2, test_loader, device, max_samples=args.max_samples
    )

    # 计算CKA相似度
    print("\n" + "=" * 80)
    cka_scores = compute_twin_model_cka(
        features_model1, features_model2,
        args.model1_name, args.model2_name
    )

    # 可视化
    visualize_cka_scores(cka_scores, args.model1_name, args.model2_name, args.save_dir)

    # 生成报告
    generate_cka_report(cka_scores, args.model1_name, args.model2_name, args.save_dir)

    print(f"\n🎉 分析完成! 结果保存在: {args.save_dir}")


if __name__ == '__main__':
    main()
