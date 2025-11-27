#!/usr/bin/env python
"""
模型性能对比脚本
用于分析两个模型的预测性能差异，结合CKA相似度判断融合机制的有效性
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# 导入模型和数据加载器
import sys
sys.path.insert(0, os.path.dirname(__file__))
from models.alignn import ALIGNN
from train_with_cross_modal_attention import load_dataset, get_dataset_paths
from data import get_train_val_loaders


def load_model(path, device):
    """加载模型"""
    print(f"📦 加载模型: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get('config') or ckpt.get('model_config')
    model = ALIGNN(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    print(f"   配置: 中期融合={model.use_middle_fusion}, "
          f"细粒度注意力={model.use_fine_grained_attention}, "
          f"全局注意力={model.use_cross_modal_attention}")
    return model


def get_predictions(model, loader, device, max_samples=None):
    """
    获取模型预测

    Returns:
        predictions: 预测值
        targets: 真实值
    """
    print("🔄 获取模型预测...")

    predictions = []
    targets = []

    sample_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="预测中"):
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

            # 获取预测
            out = model(model_input)
            pred = out if torch.is_tensor(out) else out.get('prediction', out.get('out'))

            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())

            sample_count += y.size(0)
            if max_samples and sample_count >= max_samples:
                break

    predictions = np.concatenate(predictions).flatten()
    targets = np.concatenate(targets).flatten()

    print(f"✅ 预测完成! 样本数: {len(targets)}")

    return predictions, targets


def compute_metrics(predictions, targets):
    """计算性能指标"""
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    pearson_corr, _ = pearsonr(targets, predictions)
    spearman_corr, _ = spearmanr(targets, predictions)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Pearson': pearson_corr,
        'Spearman': spearman_corr
    }


def visualize_predictions(pred1, pred2, targets, model1_name, model2_name, save_dir):
    """可视化预测结果对比"""
    print("\n📊 生成预测对比可视化...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 计算指标
    metrics1 = compute_metrics(pred1, targets)
    metrics2 = compute_metrics(pred2, targets)

    # 1. Model 1 预测散点图
    ax = axes[0, 0]
    ax.scatter(targets, pred1, alpha=0.5, s=20)
    ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()],
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('True Values', fontweight='bold')
    ax.set_ylabel('Predictions', fontweight='bold')
    ax.set_title(f'{model1_name}\nMAE={metrics1["MAE"]:.4f}, R²={metrics1["R2"]:.4f}',
                fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Model 2 预测散点图
    ax = axes[0, 1]
    ax.scatter(targets, pred2, alpha=0.5, s=20, color='orange')
    ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()],
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('True Values', fontweight='bold')
    ax.set_ylabel('Predictions', fontweight='bold')
    ax.set_title(f'{model2_name}\nMAE={metrics2["MAE"]:.4f}, R²={metrics2["R2"]:.4f}',
                fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 预测差异散点图
    ax = axes[0, 2]
    pred_diff = np.abs(pred1 - pred2)
    scatter = ax.scatter(targets, pred_diff, c=pred_diff, cmap='coolwarm',
                        alpha=0.6, s=20)
    ax.set_xlabel('True Values', fontweight='bold')
    ax.set_ylabel('|Pred1 - Pred2|', fontweight='bold')
    ax.set_title(f'Prediction Difference\nMean={pred_diff.mean():.4f}, Max={pred_diff.max():.4f}',
                fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Difference')
    ax.grid(True, alpha=0.3)

    # 4. 误差分布对比
    ax = axes[1, 0]
    error1 = pred1 - targets
    error2 = pred2 - targets
    ax.hist(error1, bins=50, alpha=0.5, label=model1_name, color='blue')
    ax.hist(error2, bins=50, alpha=0.5, label=model2_name, color='orange')
    ax.set_xlabel('Prediction Error', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Error Distribution Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 指标对比柱状图
    ax = axes[1, 1]
    metrics_names = ['MAE', 'RMSE', 'R2', 'Pearson', 'Spearman']
    x = np.arange(len(metrics_names))
    width = 0.35

    values1 = [metrics1[m] for m in metrics_names]
    values2 = [metrics2[m] for m in metrics_names]

    bars1 = ax.bar(x - width/2, values1, width, label=model1_name, alpha=0.8)
    bars2 = ax.bar(x + width/2, values2, width, label=model2_name, alpha=0.8)

    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('Metrics Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 标注数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # 6. 预测一致性散点图
    ax = axes[1, 2]
    ax.scatter(pred1, pred2, alpha=0.5, s=20)
    ax.plot([pred1.min(), pred1.max()], [pred1.min(), pred1.max()],
            'r--', lw=2, label='Perfect Agreement')
    ax.set_xlabel(f'{model1_name} Predictions', fontweight='bold')
    ax.set_ylabel(f'{model2_name} Predictions', fontweight='bold')
    corr = np.corrcoef(pred1, pred2)[0, 1]
    ax.set_title(f'Prediction Agreement\nCorrelation={corr:.4f}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'performance_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化已保存: {save_path}")
    plt.close()

    return metrics1, metrics2


def generate_performance_report(metrics1, metrics2, pred1, pred2, targets,
                                model1_name, model2_name, save_dir):
    """生成性能对比报告"""
    print("\n📝 生成性能对比报告...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Model Performance Comparison Report")
    report_lines.append(f"Model 1: {model1_name}")
    report_lines.append(f"Model 2: {model2_name}")
    report_lines.append("=" * 80)
    report_lines.append("")

    # 1. 性能指标对比
    report_lines.append("📊 Performance Metrics Comparison:")
    report_lines.append("")
    report_lines.append(f"{'Metric':<20} {model1_name:<15} {model2_name:<15} {'Difference':<15} {'Change %':<15}")
    report_lines.append("-" * 80)

    for metric in ['MAE', 'RMSE', 'R2', 'Pearson', 'Spearman']:
        val1 = metrics1[metric]
        val2 = metrics2[metric]
        diff = val2 - val1

        # 对于MAE和RMSE，越小越好；对于其他指标，越大越好
        if metric in ['MAE', 'RMSE']:
            change_pct = (diff / val1) * 100 if val1 != 0 else 0
            better = "✓" if diff < 0 else "✗"
        else:
            change_pct = (diff / val1) * 100 if val1 != 0 else 0
            better = "✓" if diff > 0 else "✗"

        report_lines.append(f"{metric:<20} {val1:<15.4f} {val2:<15.4f} "
                          f"{diff:+.4f} ({change_pct:+.2f}%) {better}")

    report_lines.append("")

    # 2. 预测差异分析
    pred_diff = np.abs(pred1 - pred2)
    pred_corr = np.corrcoef(pred1, pred2)[0, 1]

    report_lines.append("🔍 Prediction Difference Analysis:")
    report_lines.append(f"  • Prediction Correlation: {pred_corr:.4f}")
    report_lines.append(f"  • Mean Absolute Difference: {pred_diff.mean():.4f}")
    report_lines.append(f"  • Max Absolute Difference: {pred_diff.max():.4f}")
    report_lines.append(f"  • Std Absolute Difference: {pred_diff.std():.4f}")
    report_lines.append("")

    # 计算预测差异的百分位数
    percentiles = [50, 75, 90, 95, 99]
    report_lines.append("  Prediction Difference Percentiles:")
    for p in percentiles:
        val = np.percentile(pred_diff, p)
        report_lines.append(f"    {p}th percentile: {val:.4f}")
    report_lines.append("")

    # 3. 关键发现和解释
    report_lines.append("💡 Key Findings:")
    report_lines.append("")

    # 判断性能差异是否显著
    mae_diff_pct = abs((metrics2['MAE'] - metrics1['MAE']) / metrics1['MAE']) * 100
    r2_diff_pct = abs((metrics2['R2'] - metrics1['R2']) / metrics1['R2']) * 100

    if mae_diff_pct < 1 and r2_diff_pct < 1:
        report_lines.append("  🎯 结论: 两个模型的预测性能几乎相同")
        report_lines.append("")
        report_lines.append("  可能的解释:")
        report_lines.append("  1. 融合机制改变了中间表示，但最终收敛到相似的预测")
        report_lines.append("  2. 模型架构的其他部分（如输出层）主导了最终预测")
        report_lines.append("  3. 数据集可能不需要复杂的融合机制即可达到性能上限")
        report_lines.append("")
        report_lines.append("  ⚠️  警告: CKA相似度高 + 性能相同 = 融合机制可能未充分利用")
        report_lines.append("")
        report_lines.append("  建议:")
        report_lines.append("  • 检查融合机制是否真的在起作用（可能被后续层抵消了）")
        report_lines.append("  • 尝试更强的融合机制或更早的融合位置")
        report_lines.append("  • 考虑使用更具挑战性的数据集来验证融合效果")
        report_lines.append("  • 分析text_fine阶段的差异是否被后续层"抹平"了")

    elif mae_diff_pct < 5 and r2_diff_pct < 5:
        report_lines.append("  🎯 结论: 两个模型的预测性能有轻微差异")
        report_lines.append("")

        if metrics2['MAE'] < metrics1['MAE']:
            report_lines.append(f"  ✓ {model2_name} 性能更好 (MAE降低 {mae_diff_pct:.2f}%)")
            report_lines.append("")
            report_lines.append("  解释: 融合机制带来了小幅但一致的性能提升")
            report_lines.append("  • CKA相似度高说明最终表示相近")
            report_lines.append("  • 但微小的差异足以改善预测准确度")
            report_lines.append("  • 这是一个合理的优化结果")
        else:
            report_lines.append(f"  ✗ {model2_name} 性能反而下降 (MAE增加 {mae_diff_pct:.2f}%)")
            report_lines.append("")
            report_lines.append("  ⚠️  警告: 融合机制未能带来性能提升")
            report_lines.append("  可能原因:")
            report_lines.append("  • 融合引入了噪声或过拟合")
            report_lines.append("  • 融合位置或强度不合适")
            report_lines.append("  • 需要调整超参数或训练策略")

    else:
        report_lines.append("  🎯 结论: 两个模型的预测性能有显著差异")
        report_lines.append("")

        if metrics2['MAE'] < metrics1['MAE']:
            report_lines.append(f"  ✓✓ {model2_name} 性能明显更好 (MAE降低 {mae_diff_pct:.2f}%)")
            report_lines.append("")
            report_lines.append("  解释: 融合机制有效改善了模型性能")
            report_lines.append("  • 尽管CKA相似度高，但关键的差异足以产生显著效果")
            report_lines.append("  • 融合机制成功捕获了有用的跨模态信息")
            report_lines.append("  • 这是理想的融合效果")
        else:
            report_lines.append(f"  ✗✗ {model2_name} 性能明显下降 (MAE增加 {mae_diff_pct:.2f}%)")
            report_lines.append("")
            report_lines.append("  ⚠️⚠️  严重警告: 融合机制严重损害了性能")
            report_lines.append("  需要立即检查:")
            report_lines.append("  • 模型训练是否收敛")
            report_lines.append("  • 融合机制是否有bug")
            report_lines.append("  • 超参数是否合理")

    report_lines.append("")
    report_lines.append("=" * 80)

    # 保存报告
    report_text = "\n".join(report_lines)
    save_path = os.path.join(save_dir, 'performance_report.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"✅ 性能报告已保存: {save_path}")
    print("\n" + report_text)

    # 保存CSV
    df = pd.DataFrame({
        'Metric': list(metrics1.keys()),
        model1_name: list(metrics1.values()),
        model2_name: list(metrics2.values()),
        'Difference': [metrics2[k] - metrics1[k] for k in metrics1.keys()],
        'Change_%': [((metrics2[k] - metrics1[k]) / metrics1[k] * 100) if metrics1[k] != 0 else 0
                     for k in metrics1.keys()]
    })
    csv_path = os.path.join(save_dir, 'performance_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ 性能指标CSV已保存: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='模型性能对比分析')
    parser.add_argument('--ckpt_model1', type=str, required=True,
                       help='模型1的checkpoint路径')
    parser.add_argument('--ckpt_model2', type=str, required=True,
                       help='模型2的checkpoint路径')
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
    parser.add_argument('--save_dir', type=str, default='./performance_comparison',
                       help='结果保存目录')
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}\n")

    # 加载模型
    print("=" * 80)
    model1 = load_model(args.ckpt_model1, device)
    print("=" * 80)
    model2 = load_model(args.ckpt_model2, device)
    print("=" * 80)

    # 加载数据集
    print(f"\n🔄 加载数据集: {args.dataset} - {args.property}")

    try:
        cif_dir, id_prop_file = get_dataset_paths(args.root_dir, args.dataset, args.property)
        df = load_dataset(cif_dir, id_prop_file, args.dataset, args.property)
        print(f"✅ 加载数据集: {len(df)} 样本")

        if args.max_samples and len(df) > args.max_samples:
            print(f"⚠️  数据集过大，随机采样 {args.max_samples} 样本")
            import random
            random.seed(42)
            df = random.sample(df, args.max_samples)

        # 获取配置
        config = model1.config if hasattr(model1, 'config') else None
        if config is None:
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
            filename='temp_performance',
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

    # 获取模型1的预测
    print("\n" + "=" * 80)
    print(f"获取 {args.model1_name} 的预测")
    print("=" * 80)
    pred1, targets = get_predictions(model1, test_loader, device, max_samples=args.max_samples)

    # 获取模型2的预测
    print("\n" + "=" * 80)
    print(f"获取 {args.model2_name} 的预测")
    print("=" * 80)
    pred2, _ = get_predictions(model2, test_loader, device, max_samples=args.max_samples)

    # 可视化和分析
    print("\n" + "=" * 80)
    metrics1, metrics2 = visualize_predictions(
        pred1, pred2, targets,
        args.model1_name, args.model2_name,
        args.save_dir
    )

    # 生成报告
    generate_performance_report(
        metrics1, metrics2, pred1, pred2, targets,
        args.model1_name, args.model2_name,
        args.save_dir
    )

    print(f"\n🎉 分析完成! 结果保存在: {args.save_dir}")


if __name__ == '__main__':
    main()
