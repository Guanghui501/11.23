#!/usr/bin/env python
"""
简化的 Retrieval 评估示例
演示如何使用 RetrievalEvaluator
"""

import torch
from evaluate_retrieval import RetrievalEvaluator
from models.alignn import ALIGNN, ALIGNNConfig
from data import get_train_val_loaders
from utils_retrieval import load_model_checkpoint


def simple_retrieval_demo():
    """简单的检索评估示例"""

    # ========== 1. 初始化模型 ==========
    print("🔧 初始化模型...")
    model_config = ALIGNNConfig(
        name="alignn",
        classification=True,
        use_cross_modal_attention=True,    # 使用跨模态注意力
        use_fine_grained_attention=False,  # 可选：细粒度注意力
        use_middle_fusion=True,            # 使用中期融合（提高对齐）
        middle_fusion_layers="2",          # 在第2层融合
        graph_dropout=0.0                  # 评估时不用 dropout
    )
    model = ALIGNN(model_config)

    # 加载训练好的权重
    checkpoint_path = "checkpoints/best_model.pt"

    # 使用智能加载函数（自动处理不同的检查点格式）
    model, checkpoint_info = load_model_checkpoint(
        model, checkpoint_path, device='cuda', verbose=True
    )

    # ========== 2. 加载数据 ==========
    print("📊 加载数据...")
    train_loader, val_loader, test_loader = get_train_val_loaders(
        dataset="your_dataset_path",
        target="target_property",
        batch_size=32,
        workers=4
    )

    # ========== 3. 创建评估器 ==========
    evaluator = RetrievalEvaluator(model, device='cuda')

    # ========== 4. 运行评估 ==========
    print("\n" + "="*80)
    print("🎯 开始评估检索性能...")
    print("="*80 + "\n")

    metrics = evaluator.evaluate(
        dataloader=val_loader,
        max_samples=1000,           # 评估 1000 个样本（更快）
        k_values=[1, 5, 10, 20],   # 计算 R@1, R@5, R@10, R@20
        visualize=True,             # 生成可视化
        output_dir='./retrieval_results'
    )

    # ========== 5. 解读结果 ==========
    print("\n" + "="*80)
    print("📊 结果解读:")
    print("="*80)

    g2t_r1 = metrics['g2t_R@1'] * 100
    t2g_r1 = metrics['t2g_R@1'] * 100
    avg_r1 = metrics['avg_R@1'] * 100

    print(f"\n✨ R@1 性能:")
    print(f"   - Graph→Text: {g2t_r1:.2f}%")
    print(f"   - Text→Graph: {t2g_r1:.2f}%")
    print(f"   - 平均: {avg_r1:.2f}%")

    # 性能评级
    if avg_r1 >= 80:
        grade = "🏆 优秀！模型的图-文本对齐能力非常强"
    elif avg_r1 >= 60:
        grade = "👍 良好！中期融合起作用了"
    elif avg_r1 >= 40:
        grade = "😐 一般，还有提升空间"
    else:
        grade = "❌ 较差，建议检查融合策略"

    print(f"\n评级: {grade}")

    print(f"\n💡 建议:")
    if avg_r1 < 60:
        print("   - 检查 use_middle_fusion 是否开启")
        print("   - 尝试增加 contrastive_loss_weight")
        print("   - 考虑使用 use_fine_grained_attention")
    else:
        print("   - 模型对齐能力已经很好！")
        print("   - 可以考虑增加 graph_dropout 进行正则化")

    return metrics


def quick_retrieval_check(model, dataloader, num_samples=100):
    """
    快速检查检索性能（不保存结果）

    用于训练过程中快速评估
    """
    evaluator = RetrievalEvaluator(model, device='cuda')

    # 提取特征
    graph_features, text_features, _ = evaluator.extract_features(
        dataloader, max_samples=num_samples
    )

    # 计算相似度
    similarity_matrix = evaluator.compute_similarity_matrix(
        graph_features, text_features
    )

    # 只计算 R@1
    N = similarity_matrix.size(0)
    correct_indices = torch.arange(N)

    # Graph-to-Text R@1
    _, sorted_indices = torch.sort(similarity_matrix, dim=1, descending=True)
    top_1_indices = sorted_indices[:, 0]
    g2t_r1 = (top_1_indices == correct_indices).float().mean().item()

    # Text-to-Graph R@1
    _, sorted_indices = torch.sort(similarity_matrix, dim=0, descending=True)
    top_1_indices = sorted_indices[0, :]
    t2g_r1 = (top_1_indices == correct_indices).float().mean().item()

    avg_r1 = (g2t_r1 + t2g_r1) / 2

    return {
        'g2t_R@1': g2t_r1,
        't2g_R@1': t2g_r1,
        'avg_R@1': avg_r1
    }


def compare_models_retrieval(model_paths, dataloader, labels):
    """
    比较多个模型的检索性能

    Args:
        model_paths: 模型路径列表
        dataloader: 数据加载器
        labels: 模型标签列表
    """
    import matplotlib.pyplot as plt

    results = []

    for model_path, label in zip(model_paths, labels):
        print(f"\n评估模型: {label}")
        print("-" * 60)

        # 加载模型
        model_config = ALIGNNConfig(name="alignn", classification=True)
        model = ALIGNN(model_config)

        # 使用智能加载函数
        model, _ = load_model_checkpoint(model, model_path, device='cuda', verbose=False)

        # 快速评估
        metrics = quick_retrieval_check(model, dataloader, num_samples=500)

        results.append({
            'label': label,
            'metrics': metrics
        })

        print(f"R@1 = {metrics['avg_R@1']*100:.2f}%")

    # 可视化比较
    labels_list = [r['label'] for r in results]
    g2t_values = [r['metrics']['g2t_R@1'] * 100 for r in results]
    t2g_values = [r['metrics']['t2g_R@1'] * 100 for r in results]
    avg_values = [r['metrics']['avg_R@1'] * 100 for r in results]

    x = range(len(labels_list))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width for i in x], g2t_values, width, label='Graph→Text', alpha=0.8)
    ax.bar(x, t2g_values, width, label='Text→Graph', alpha=0.8)
    ax.bar([i + width for i in x], avg_values, width, label='Average', alpha=0.8)

    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('R@1 (%)', fontsize=12)
    ax.set_title('模型检索性能对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison_retrieval.png', dpi=300)
    print("\n💾 对比图已保存: model_comparison_retrieval.png")

    return results


if __name__ == '__main__':
    # 运行简单示例
    simple_retrieval_demo()

    # 或者比较不同模型
    # compare_models_retrieval(
    #     model_paths=[
    #         'checkpoints/no_fusion.pt',
    #         'checkpoints/middle_fusion.pt',
    #         'checkpoints/fine_grained.pt'
    #     ],
    #     dataloader=val_loader,
    #     labels=['No Fusion', 'Middle Fusion', 'Fine-Grained']
    # )
