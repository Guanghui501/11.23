#!/usr/bin/env python
"""
Retrieval Evaluation Script for Graph-Text Alignment
评估图-文本对齐能力：R@1, R@5, R@10

这个脚本实现了"连连看"游戏：
- 给定 N 个图和 N 段文本
- 对于每个图，能否在所有文本中找到正确匹配？
- R@1: 第一名就是正确答案的比例
- R@5: 前5名包含正确答案的比例
- R@10: 前10名包含正确答案的比例
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 导入你的模型和数据加载器
from models.alignn import ALIGNN, ALIGNNConfig
from data import get_train_val_loaders
from config import TrainingConfig


class RetrievalEvaluator:
    """图-文本检索评估器"""

    def __init__(self, model, device='cuda'):
        """
        初始化评估器

        Args:
            model: 训练好的 ALIGNN 模型
            device: 'cuda' 或 'cpu'
        """
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, dataloader, max_samples=None):
        """
        从数据集提取图特征和文本特征

        Args:
            dataloader: 数据加载器
            max_samples: 最多提取多少样本（None = 全部）

        Returns:
            graph_features: [N, feature_dim] 图特征矩阵
            text_features: [N, feature_dim] 文本特征矩阵
            labels: [N] 标签（用于后续分析）
        """
        print("🔍 正在提取特征...")

        graph_features_list = []
        text_features_list = []
        labels_list = []

        total_samples = 0

        for batch_idx, (g, lg, text, labels) in enumerate(tqdm(dataloader)):
            # 移动到设备
            g = g.to(self.device)
            lg = lg.to(self.device)

            # 前向传播，获取特征
            output_dict = self.model((g, lg, text), return_features=True)

            # 提取图和文本特征
            graph_feat = output_dict['graph_features']  # [batch, 64]
            text_feat = output_dict['text_features']    # [batch, 64]

            # L2 归一化（用于余弦相似度）
            graph_feat = F.normalize(graph_feat, dim=1)
            text_feat = F.normalize(text_feat, dim=1)

            graph_features_list.append(graph_feat.cpu())
            text_features_list.append(text_feat.cpu())
            labels_list.append(labels.cpu())

            total_samples += graph_feat.size(0)

            # 达到最大样本数则停止
            if max_samples and total_samples >= max_samples:
                break

        # 拼接所有批次
        graph_features = torch.cat(graph_features_list, dim=0)
        text_features = torch.cat(text_features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        # 截断到指定样本数
        if max_samples:
            graph_features = graph_features[:max_samples]
            text_features = text_features[:max_samples]
            labels = labels[:max_samples]

        print(f"✅ 提取完成: {graph_features.size(0)} 个样本")
        print(f"   - 图特征维度: {graph_features.shape}")
        print(f"   - 文本特征维度: {text_features.shape}")

        return graph_features, text_features, labels

    def compute_similarity_matrix(self, graph_features, text_features):
        """
        计算图-文本相似度矩阵

        Args:
            graph_features: [N, D] 图特征（已归一化）
            text_features: [N, D] 文本特征（已归一化）

        Returns:
            similarity_matrix: [N, N] 相似度矩阵
                similarity[i, j] = cosine_similarity(graph_i, text_j)
        """
        print("📊 计算相似度矩阵...")

        # 余弦相似度 = 归一化向量的点积
        similarity_matrix = torch.matmul(graph_features, text_features.T)

        print(f"✅ 相似度矩阵: {similarity_matrix.shape}")
        return similarity_matrix

    def compute_retrieval_metrics(self, similarity_matrix, k_values=[1, 5, 10]):
        """
        计算检索指标 R@K

        Args:
            similarity_matrix: [N, N] 相似度矩阵
            k_values: 要计算的 K 值列表

        Returns:
            metrics: 字典，包含 Graph-to-Text 和 Text-to-Graph 的 R@K
        """
        N = similarity_matrix.size(0)

        # 正确答案的索引（对角线）
        # 第 i 个图对应第 i 个文本
        correct_indices = torch.arange(N)

        metrics = {}

        # ========== Graph-to-Text 检索 ==========
        print("\n🔎 Graph-to-Text 检索（给定图，找文本）:")

        # 对每一行排序（每个图在所有文本中的相似度排名）
        # sorted_indices[i] = 第 i 个图的文本排名列表
        _, sorted_indices = torch.sort(similarity_matrix, dim=1, descending=True)

        for k in k_values:
            # 检查正确答案是否在前 K 名
            top_k_indices = sorted_indices[:, :k]  # [N, K]

            # 对于每个样本，检查正确索引是否在 top-K 中
            correct_in_top_k = (top_k_indices == correct_indices.unsqueeze(1)).any(dim=1)

            recall_at_k = correct_in_top_k.float().mean().item()
            metrics[f'g2t_R@{k}'] = recall_at_k

            print(f"   R@{k:2d} = {recall_at_k*100:.2f}%  "
                  f"({correct_in_top_k.sum().item()}/{N} 样本成功)")

        # ========== Text-to-Graph 检索 ==========
        print("\n🔎 Text-to-Graph 检索（给定文本，找图）:")

        # 对每一列排序（每个文本在所有图中的相似度排名）
        _, sorted_indices = torch.sort(similarity_matrix, dim=0, descending=True)

        for k in k_values:
            # 检查正确答案是否在前 K 名
            top_k_indices = sorted_indices[:k, :]  # [K, N]

            # 对于每个样本，检查正确索引是否在 top-K 中
            correct_in_top_k = (top_k_indices == correct_indices.unsqueeze(0)).any(dim=0)

            recall_at_k = correct_in_top_k.float().mean().item()
            metrics[f't2g_R@{k}'] = recall_at_k

            print(f"   R@{k:2d} = {recall_at_k*100:.2f}%  "
                  f"({correct_in_top_k.sum().item()}/{N} 样本成功)")

        # ========== 平均检索性能 ==========
        print("\n📈 平均检索性能:")
        for k in k_values:
            avg_recall = (metrics[f'g2t_R@{k}'] + metrics[f't2g_R@{k}']) / 2
            metrics[f'avg_R@{k}'] = avg_recall
            print(f"   Avg R@{k:2d} = {avg_recall*100:.2f}%")

        return metrics

    def analyze_failure_cases(self, similarity_matrix, graph_features, text_features,
                             labels, top_k=5):
        """
        分析检索失败案例

        Args:
            similarity_matrix: [N, N] 相似度矩阵
            graph_features, text_features: 特征矩阵
            labels: 标签
            top_k: 显示前 K 个最差案例
        """
        N = similarity_matrix.size(0)
        correct_indices = torch.arange(N)

        print(f"\n❌ 分析检索失败案例（最差 {top_k} 个）:")
        print("=" * 80)

        # Graph-to-Text 检索
        _, sorted_indices = torch.sort(similarity_matrix, dim=1, descending=True)

        # 找到正确答案的排名
        ranks = []
        for i in range(N):
            correct_idx = correct_indices[i]
            rank = (sorted_indices[i] == correct_idx).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

        ranks = torch.tensor(ranks)

        # 找出最差的案例（排名最靠后）
        worst_indices = torch.argsort(ranks, descending=True)[:top_k]

        for idx in worst_indices:
            idx = idx.item()
            rank = ranks[idx].item()
            correct_sim = similarity_matrix[idx, idx].item()

            # 找出排在前面的错误匹配
            top_wrong = sorted_indices[idx, 0].item()
            wrong_sim = similarity_matrix[idx, top_wrong].item()

            print(f"\n样本 {idx}:")
            print(f"  - 真实标签: {labels[idx].item():.4f}")
            print(f"  - 正确匹配排名: {rank} / {N}")
            print(f"  - 正确匹配相似度: {correct_sim:.4f}")
            print(f"  - 最高错误匹配 (索引 {top_wrong}): 相似度 {wrong_sim:.4f}")
            print(f"  - 相似度差距: {wrong_sim - correct_sim:.4f}")

        # 统计排名分布
        print("\n📊 排名分布:")
        rank_bins = [1, 5, 10, 50, 100, N]
        for i in range(len(rank_bins) - 1):
            count = ((ranks > rank_bins[i]) & (ranks <= rank_bins[i+1])).sum().item()
            print(f"   排名 {rank_bins[i]+1:4d}-{rank_bins[i+1]:4d}: {count:5d} 样本 "
                  f"({count/N*100:.1f}%)")

    def visualize_similarity_matrix(self, similarity_matrix, save_path=None,
                                   max_display=100):
        """
        可视化相似度矩阵

        Args:
            similarity_matrix: [N, N] 相似度矩阵
            save_path: 保存路径
            max_display: 最多显示多少个样本（避免图太大）
        """
        N = min(similarity_matrix.size(0), max_display)
        sim_matrix = similarity_matrix[:N, :N].numpy()

        plt.figure(figsize=(12, 10))

        # 绘制热力图
        sns.heatmap(sim_matrix, cmap='RdYlGn', center=0,
                   vmin=-1, vmax=1, square=True,
                   cbar_kws={'label': 'Cosine Similarity'},
                   xticklabels=False, yticklabels=False)

        plt.title(f'Graph-Text Similarity Matrix (前 {N} 个样本)\n'
                 f'对角线 = 正确匹配', fontsize=14, pad=20)
        plt.xlabel('Text Index', fontsize=12)
        plt.ylabel('Graph Index', fontsize=12)

        # 添加对角线标记
        plt.plot([0, N], [0, N], 'b--', linewidth=2, alpha=0.5, label='Perfect Alignment')
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 相似度矩阵已保存: {save_path}")

        plt.show()

    def visualize_retrieval_metrics(self, metrics, save_path=None):
        """
        可视化检索指标

        Args:
            metrics: 指标字典
            save_path: 保存路径
        """
        k_values = [1, 5, 10]

        g2t_values = [metrics[f'g2t_R@{k}'] * 100 for k in k_values]
        t2g_values = [metrics[f't2g_R@{k}'] * 100 for k in k_values]
        avg_values = [metrics[f'avg_R@{k}'] * 100 for k in k_values]

        x = np.arange(len(k_values))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width, g2t_values, width, label='Graph→Text',
                      color='steelblue', alpha=0.8)
        bars2 = ax.bar(x, t2g_values, width, label='Text→Graph',
                      color='coral', alpha=0.8)
        bars3 = ax.bar(x + width, avg_values, width, label='Average',
                      color='mediumseagreen', alpha=0.8)

        ax.set_xlabel('K (Rank)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Recall@K (%)', fontsize=12, fontweight='bold')
        ax.set_title('图-文本检索性能 (Retrieval Performance)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([f'R@{k}' for k in k_values])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 100)

        # 在柱子上添加数值
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=9)

        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 检索指标图已保存: {save_path}")

        plt.show()

    def evaluate(self, dataloader, max_samples=None, k_values=[1, 5, 10],
                visualize=True, output_dir='./retrieval_results'):
        """
        完整的检索评估流程

        Args:
            dataloader: 数据加载器
            max_samples: 最多评估多少样本
            k_values: 要计算的 K 值
            visualize: 是否可视化
            output_dir: 结果输出目录

        Returns:
            metrics: 评估指标字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print("=" * 80)
        print("🎯 开始图-文本检索评估")
        print("=" * 80)

        # Step 1: 提取特征
        graph_features, text_features, labels = self.extract_features(
            dataloader, max_samples
        )

        # Step 2: 计算相似度矩阵
        similarity_matrix = self.compute_similarity_matrix(
            graph_features, text_features
        )

        # Step 3: 计算检索指标
        metrics = self.compute_retrieval_metrics(similarity_matrix, k_values)

        # Step 4: 分析失败案例
        self.analyze_failure_cases(
            similarity_matrix, graph_features, text_features, labels, top_k=5
        )

        # Step 5: 可视化
        if visualize:
            print("\n📊 生成可视化...")

            # 相似度矩阵
            self.visualize_similarity_matrix(
                similarity_matrix,
                save_path=output_dir / 'similarity_matrix.png'
            )

            # 检索指标
            self.visualize_retrieval_metrics(
                metrics,
                save_path=output_dir / 'retrieval_metrics.png'
            )

        # Step 6: 保存结果
        results = {
            'metrics': metrics,
            'num_samples': len(graph_features),
            'feature_dim': graph_features.size(1)
        }

        with open(output_dir / 'retrieval_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 结果已保存到: {output_dir}")
        print("=" * 80)
        print("✅ 评估完成!")
        print("=" * 80)

        return metrics


def main():
    """主函数：命令行接口"""
    parser = argparse.ArgumentParser(
        description='图-文本检索评估 (Retrieval Evaluation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 评估验证集
  python evaluate_retrieval.py --checkpoint best_model.pt --split val

  # 评估测试集，最多 500 个样本
  python evaluate_retrieval.py --checkpoint best_model.pt --split test --max_samples 500

  # 计算 R@1, R@5, R@10, R@20
  python evaluate_retrieval.py --checkpoint best_model.pt --k_values 1 5 10 20
        """
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='评估哪个数据集')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最多评估多少样本（None = 全部）')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 5, 10],
                       help='计算 R@K 的 K 值列表')
    parser.add_argument('--output_dir', type=str, default='./retrieval_results',
                       help='结果输出目录')
    parser.add_argument('--no_visualize', action='store_true',
                       help='不生成可视化图表')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='运行设备')

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")

    # 加载配置
    print(f"📖 加载配置: {args.config}")
    # 这里假设你有 TrainingConfig 类
    # config = TrainingConfig.from_json(args.config)

    # 初始化模型
    print(f"🔧 初始化模型...")
    model_config = ALIGNNConfig(
        name="alignn",
        classification=True,
        use_cross_modal_attention=True,
        use_fine_grained_attention=False,
        use_middle_fusion=True,
        graph_dropout=0.0  # 评估时不用 dropout
    )
    model = ALIGNN(model_config).to(device)

    # 加载权重
    print(f"📥 加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # 加载数据
    print(f"📊 加载 {args.split} 数据集...")
    train_loader, val_loader, test_loader = get_train_val_loaders(
        dataset="your_dataset_path",  # 根据你的数据集修改
        target="target_property",
        batch_size=32,
        workers=4
    )

    if args.split == 'train':
        dataloader = train_loader
    elif args.split == 'val':
        dataloader = val_loader
    else:
        dataloader = test_loader

    # 创建评估器
    evaluator = RetrievalEvaluator(model, device)

    # 运行评估
    metrics = evaluator.evaluate(
        dataloader=dataloader,
        max_samples=args.max_samples,
        k_values=args.k_values,
        visualize=not args.no_visualize,
        output_dir=args.output_dir
    )

    # 打印最终结果
    print("\n" + "=" * 80)
    print("📊 最终检索性能:")
    print("=" * 80)
    for key, value in sorted(metrics.items()):
        print(f"  {key:15s}: {value*100:6.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    main()
