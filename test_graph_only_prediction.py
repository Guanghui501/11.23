#!/usr/bin/env python
"""
测试图特征单独预测功能
验证模型架构和前向传播是否正常
"""

import torch
import dgl
from models.alignn import ALIGNN, ALIGNNConfig

def create_dummy_graph(num_nodes=10, num_edges=20):
    """创建虚拟图用于测试"""
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    g = dgl.graph((src, dst))

    # 节点特征 (原子特征)
    g.ndata['atom_features'] = torch.randn(num_nodes, 92)

    # 边特征 (距离)
    g.edata['r'] = torch.rand(num_edges) * 5.0 + 1.0

    return g

def create_dummy_line_graph(g):
    """创建虚拟线图"""
    lg = dgl.line_graph(g, backtracking=False)

    # 边的边特征 (键角余弦)
    num_lg_edges = lg.num_edges()
    lg.edata['h'] = torch.randn(num_lg_edges, 1)

    return lg

def create_dummy_text():
    """创建虚拟文本输入"""
    return ["This is a test crystal structure with high stability"]

def test_graph_only_prediction():
    """测试图特征单独预测模式"""

    print("="*60)
    print("  测试图特征单独预测功能")
    print("="*60)
    print()

    # 测试1: use_only_graph_for_prediction = False (标准模式)
    print("📊 测试1: 标准融合模式")
    print("-"*60)

    config1 = ALIGNNConfig(
        name="alignn",
        alignn_layers=2,
        gcn_layers=2,
        hidden_features=128,
        use_fine_grained_attention=True,
        fine_grained_hidden_dim=128,
        fine_grained_num_heads=4,
        use_cross_modal_attention=True,
        cross_modal_hidden_dim=128,
        cross_modal_num_heads=2,
        use_only_graph_for_prediction=False,  # 标准模式
        output_features=1
    )

    model1 = ALIGNN(config1)
    print(f"✅ 模型创建成功")
    print(f"   - use_only_graph_for_prediction: {model1.use_only_graph_for_prediction}")
    print(f"   - FC1层输入维度: {model1.fc1.in_features}")
    print(f"   - FC1层输出维度: {model1.fc1.out_features}")

    # 创建测试数据
    g = create_dummy_graph(num_nodes=10, num_edges=20)
    lg = create_dummy_line_graph(g)
    text = create_dummy_text()

    # 批处理
    batch_g = dgl.batch([g])
    batch_lg = dgl.batch([lg])

    # 前向传播
    model1.eval()
    with torch.no_grad():
        try:
            output = model1((batch_g, batch_lg, text), return_intermediate_features=True)
            print(f"✅ 前向传播成功")
            print(f"   - 预测形状: {output['predictions'].shape}")
            print(f"   - 预测值: {output['predictions'].item():.4f}")
            print(f"   - 图特征形状: {output['graph_features'].shape}")
            print(f"   - 文本特征形状: {output['text_features'].shape}")
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            return False

    print()

    # 测试2: use_only_graph_for_prediction = True (图特征单独预测)
    print("📊 测试2: 图特征单独预测模式")
    print("-"*60)

    config2 = ALIGNNConfig(
        name="alignn",
        alignn_layers=2,
        gcn_layers=2,
        hidden_features=128,
        use_fine_grained_attention=True,
        fine_grained_hidden_dim=128,
        fine_grained_num_heads=4,
        use_cross_modal_attention=True,
        cross_modal_hidden_dim=128,
        cross_modal_num_heads=2,
        use_only_graph_for_prediction=True,  # 图特征单独预测
        output_features=1
    )

    model2 = ALIGNN(config2)
    print(f"✅ 模型创建成功")
    print(f"   - use_only_graph_for_prediction: {model2.use_only_graph_for_prediction}")
    print(f"   - FC1层输入维度: {model2.fc1.in_features}")
    print(f"   - FC1层输出维度: {model2.fc1.out_features}")

    # 前向传播
    model2.eval()
    with torch.no_grad():
        try:
            output = model2((batch_g, batch_lg, text), return_intermediate_features=True)
            print(f"✅ 前向传播成功")
            print(f"   - 预测形状: {output['predictions'].shape}")
            print(f"   - 预测值: {output['predictions'].item():.4f}")
            print(f"   - 图特征形状: {output['graph_features'].shape}")
            print(f"   - 文本特征形状: {output['text_features'].shape}")
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            return False

    print()

    # 测试3: 对比两种模式
    print("📊 测试3: 对比两种模式")
    print("-"*60)

    # 参数数量对比
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())

    print(f"参数数量:")
    print(f"   - 标准模式: {params1:,}")
    print(f"   - 图特征预测: {params2:,}")
    print(f"   - 差异: {abs(params1 - params2):,}")

    if params1 == params2:
        print(f"✅ 参数数量相同 (预期行为)")
    else:
        print(f"⚠️  参数数量不同 (可能是配置导致)")

    print()

    # 测试4: 不使用cross-modal attention的情况
    print("📊 测试4: 无全局注意力 + 图特征预测")
    print("-"*60)

    config3 = ALIGNNConfig(
        name="alignn",
        alignn_layers=2,
        gcn_layers=2,
        hidden_features=128,
        use_fine_grained_attention=True,
        fine_grained_hidden_dim=128,
        fine_grained_num_heads=4,
        use_cross_modal_attention=False,  # 不使用全局注意力
        use_only_graph_for_prediction=True,
        output_features=1
    )

    model3 = ALIGNN(config3)
    print(f"✅ 模型创建成功")
    print(f"   - use_cross_modal_attention: {model3.use_cross_modal_attention}")
    print(f"   - use_only_graph_for_prediction: {model3.use_only_graph_for_prediction}")
    print(f"   - FC1层输入维度: {model3.fc1.in_features}")

    model3.eval()
    with torch.no_grad():
        try:
            output = model3((batch_g, batch_lg, text))
            print(f"✅ 前向传播成功")
            print(f"   - 预测形状: {output.shape}")
            print(f"   - 预测值: {output.item():.4f}")
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            return False

    print()

    # 测试5: 批量数据测试
    print("📊 测试5: 批量数据测试 (batch_size=3)")
    print("-"*60)

    # 创建批量数据
    graphs = [create_dummy_graph(num_nodes=8+i*2, num_edges=15+i*5) for i in range(3)]
    line_graphs = [create_dummy_line_graph(g) for g in graphs]
    texts = [
        "Crystal structure A with high conductivity",
        "Material B showing excellent stability",
        "Compound C with unique magnetic properties"
    ]

    batch_g_multi = dgl.batch(graphs)
    batch_lg_multi = dgl.batch(line_graphs)

    model2.eval()
    with torch.no_grad():
        try:
            output = model2((batch_g_multi, batch_lg_multi, texts), return_intermediate_features=True)
            print(f"✅ 批量前向传播成功")
            print(f"   - 预测形状: {output['predictions'].shape}")
            print(f"   - 预测值: {output['predictions'].tolist()}")
            print(f"   - 图特征形状: {output['graph_features'].shape}")
            print(f"   - 文本特征形状: {output['text_features'].shape}")
        except Exception as e:
            print(f"❌ 批量前向传播失败: {e}")
            return False

    print()
    print("="*60)
    print("  ✅ 所有测试通过!")
    print("="*60)
    print()
    print("📝 总结:")
    print("   1. ✅ 图特征单独预测模式正常工作")
    print("   2. ✅ 模型架构正确初始化")
    print("   3. ✅ 前向传播无错误")
    print("   4. ✅ 批量处理正常")
    print("   5. ✅ 兼容不同配置组合")
    print()

    return True


if __name__ == "__main__":
    success = test_graph_only_prediction()
    exit(0 if success else 1)
