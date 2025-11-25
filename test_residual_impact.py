"""
测试双重残差连接的影响
"""
import torch
import torch.nn.functional as F

# 模拟场景
node_feat = torch.randn(100, 64)  # 100个节点
text_feat = torch.randn(100, 64)  # 广播后的文本

# 模拟路由器输出不同的权重分配
scenarios = [
    ("文本主导", 0.2, 0.8),  # 希望文本占主导
    ("图主导", 0.8, 0.2),    # 希望图占主导
    ("均衡", 0.5, 0.5),      # 希望平等融合
]

print("=" * 70)
print("双重残差连接影响分析")
print("=" * 70)

for name, w_g, w_t in scenarios:
    print(f"\n场景: {name}")
    print(f"  期望权重 - 图:{w_g:.1f}, 文本:{w_t:.1f}")

    # 方案1: 双重残差（当前代码）
    fused = w_g * node_feat + w_t * text_feat
    out_double = node_feat + fused
    actual_w_g = 1 + w_g
    actual_w_t = w_t
    print(f"  实际权重 - 图:{actual_w_g:.1f}, 文本:{actual_w_t:.1f}")

    # 计算特征贡献比例
    graph_contribution = torch.norm(actual_w_g * node_feat)
    text_contribution = torch.norm(actual_w_t * text_feat)
    total = graph_contribution + text_contribution

    print(f"  实际贡献 - 图:{graph_contribution/total*100:.1f}%, "
          f"文本:{text_contribution/total*100:.1f}%")

    # 权重失衡度
    imbalance = max(actual_w_g, actual_w_t) / min(actual_w_g, actual_w_t)
    print(f"  权重失衡度: {imbalance:.2f}x")

    # 检查是否符合预期
    if name == "文本主导" and graph_contribution > text_contribution:
        print(f"  ⚠️  问题: 期望文本主导，但图特征贡献更大！")
    elif name == "图主导" and graph_contribution < text_contribution:
        print(f"  ⚠️  问题: 期望图主导，但文本特征贡献更大！")
    elif name == "均衡":
        ratio = graph_contribution / text_contribution
        if ratio < 0.8 or ratio > 1.2:
            print(f"  ⚠️  问题: 期望均衡，但比例为 {ratio:.2f}！")

print("\n" + "=" * 70)
print("修复方案对比")
print("=" * 70)

# 测试修复方案
node_feat = torch.randn(100, 64)
text_feat = torch.randn(100, 64)
w_g, w_t = 0.2, 0.8  # 期望文本主导

print("\n方案A: 完全替换（无外部残差）")
out_A = w_g * node_feat + w_t * text_feat
graph_contrib_A = torch.norm(w_g * node_feat)
text_contrib_A = torch.norm(w_t * text_feat)
total_A = graph_contrib_A + text_contrib_A
print(f"  图贡献: {graph_contrib_A/total_A*100:.1f}%")
print(f"  文本贡献: {text_contrib_A/total_A*100:.1f}%")
print(f"  ✅ 符合预期: 文本主导")

print("\n方案B: 当前代码（双重残差）")
out_B = node_feat + (w_g * node_feat + w_t * text_feat)
actual_w_g_B = 1 + w_g
graph_contrib_B = torch.norm(actual_w_g_B * node_feat)
text_contrib_B = torch.norm(w_t * text_feat)
total_B = graph_contrib_B + text_contrib_B
print(f"  图贡献: {graph_contrib_B/total_B*100:.1f}%")
print(f"  文本贡献: {text_contrib_B/total_B*100:.1f}%")
print(f"  ❌ 不符合预期: 图仍然主导")
