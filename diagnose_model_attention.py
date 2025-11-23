#!/usr/bin/env python3
"""
诊断脚本：检查模型是否输出不同原子的不同注意力模式

将此代码添加到您的 demo_fine_grained_attention.py 中，
在第 166 行 (提取 fine-grained attention 之后) 添加。
"""

import numpy as np
import torch

def diagnose_fine_grained_attention(fg_attn, elements=None):
    """
    诊断 fine-grained attention 是否为所有原子输出相同的模式

    Args:
        fg_attn: fine-grained attention weights dict from model
        elements: list of element symbols (optional)
    """

    print("\n" + "="*80)
    print("🔬 Fine-Grained Attention 诊断")
    print("="*80)

    if fg_attn is None or 'atom_to_text' not in fg_attn:
        print("❌ 没有找到 fine-grained attention weights!")
        return

    atom_to_text = fg_attn['atom_to_text']  # [batch, heads, num_atoms, seq_len]

    print(f"\n1️⃣ 原始形状检查:")
    print(f"   atom_to_text shape: {atom_to_text.shape}")
    batch, num_heads, num_atoms, seq_len = atom_to_text.shape
    print(f"   - Batch size: {batch}")
    print(f"   - Number of heads: {num_heads}")
    print(f"   - Number of atoms: {num_atoms}")
    print(f"   - Sequence length: {seq_len}")

    # Extract first batch
    atom_to_text = atom_to_text[0]  # [heads, num_atoms, seq_len]

    print(f"\n2️⃣ 检查不同 Attention Head 是否有差异:")
    for head in range(num_heads):
        head_data = atom_to_text[head].cpu().numpy()  # [num_atoms, seq_len]
        entropy = -np.sum(head_data * np.log(head_data + 1e-10)) / (num_atoms * seq_len)
        print(f"   Head {head}: Entropy = {entropy:.4f}")

        # Check if all atoms identical in this head
        if num_atoms > 1:
            identical = np.allclose(head_data[0], head_data[1], atol=1e-6)
            if identical:
                print(f"      ⚠️  Atom 0 和 Atom 1 在此 head 中完全相同!")

    print(f"\n3️⃣ 对所有 heads 取平均后检查:")
    atom_to_text_avg = atom_to_text.mean(dim=0).cpu().numpy()  # [num_atoms, seq_len]
    print(f"   平均后形状: {atom_to_text_avg.shape}")

    # Check each atom's top 5 tokens
    print(f"\n   每个原子的 Top 5 tokens (平均后，合并前):")
    for i in range(min(5, num_atoms)):
        top_5_indices = atom_to_text_avg[i].argsort()[-5:][::-1]
        top_5_weights = atom_to_text_avg[i, top_5_indices]
        element_name = elements[i] if elements else f"Atom_{i}"
        print(f"   {element_name:8s}: indices={top_5_indices}, weights={top_5_weights}")

    # Statistical comparison between atoms
    print(f"\n4️⃣ 原子间统计比较:")
    if num_atoms > 1:
        # Compare first two atoms
        correlation = np.corrcoef(atom_to_text_avg[0], atom_to_text_avg[1])[0, 1]
        print(f"   Atom 0 和 Atom 1 相关系数: {correlation:.6f}")

        if correlation > 0.99:
            print(f"   ⚠️  警告：相关系数 > 0.99，两个原子的注意力模式几乎相同!")

        # Check if completely identical
        identical = np.allclose(atom_to_text_avg[0], atom_to_text_avg[1], atol=1e-6)
        if identical:
            print(f"   ❌ 错误：Atom 0 和 Atom 1 完全相同 (allclose with atol=1e-6)")
            print(f"      这说明模型没有学到区分不同原子的能力!")
        else:
            print(f"   ✅ Atom 0 和 Atom 1 不完全相同")

        # Check variance across atoms
        atom_means = atom_to_text_avg.mean(axis=1)  # [num_atoms]
        atom_variance = atom_means.var()
        print(f"   原子间平均注意力的方差: {atom_variance:.6f}")

        if atom_variance < 1e-6:
            print(f"   ⚠️  警告：方差极小，所有原子可能有相同的平均注意力")

    # Check specific patterns
    print(f"\n5️⃣ 检查是否所有原子关注相同的 tokens:")
    top_tokens_per_atom = [atom_to_text_avg[i].argmax() for i in range(min(5, num_atoms))]
    unique_top_tokens = len(set(top_tokens_per_atom))
    print(f"   前5个原子的 top token: {top_tokens_per_atom}")
    print(f"   独特的 top tokens 数量: {unique_top_tokens} / {min(5, num_atoms)}")

    if unique_top_tokens == 1:
        print(f"   ❌ 所有原子都关注同一个 token!")
    elif unique_top_tokens < min(3, num_atoms):
        print(f"   ⚠️  大部分原子关注相同的 tokens")
    else:
        print(f"   ✅ 不同原子关注不同的 tokens")

    print(f"\n6️⃣ 诊断结论:")

    # Determine the issue
    issues = []

    # Check if all heads are identical
    all_heads_identical = True
    for head in range(num_heads - 1):
        if not np.allclose(atom_to_text[head].cpu().numpy(),
                          atom_to_text[head + 1].cpu().numpy(), atol=1e-6):
            all_heads_identical = False
            break

    if all_heads_identical:
        issues.append("所有 attention heads 完全相同 (多头注意力退化)")

    # Check if all atoms identical within averaged attention
    all_atoms_identical = True
    for i in range(num_atoms - 1):
        if not np.allclose(atom_to_text_avg[i], atom_to_text_avg[i + 1], atol=1e-6):
            all_atoms_identical = False
            break

    if all_atoms_identical:
        issues.append("所有原子的注意力模式完全相同 (fine-grained attention 失效)")

    if len(issues) > 0:
        print(f"   ❌ 发现问题:")
        for issue in issues:
            print(f"      - {issue}")
        print(f"\n   💡 建议:")
        print(f"      1. 检查模型训练是否正确 (use_fine_grained_attention=True)")
        print(f"      2. 检查 checkpoint 是否正确加载")
        print(f"      3. 检查模型是否在fine-grained attention任务上训练过")
        print(f"      4. 尝试可视化不同的样本，看是否都有此问题")
    else:
        print(f"   ✅ 未发现明显问题")
        print(f"      模型输出的 fine-grained attention 看起来正常")
        print(f"      如果仍然看到相同的 top words，问题可能在后续处理步骤")

    print("="*80 + "\n")

    return {
        'num_heads': num_heads,
        'num_atoms': num_atoms,
        'all_heads_identical': all_heads_identical,
        'all_atoms_identical': all_atoms_identical,
        'correlation_0_1': correlation if num_atoms > 1 else None,
        'issues': issues
    }


# 使用示例 (添加到 demo_fine_grained_attention.py 中):
"""
# 在第 166 行之后添加:

    # 诊断模型输出
    from diagnose_model_attention import diagnose_fine_grained_attention

    diagnosis = diagnose_fine_grained_attention(
        fg_attn,
        elements=[str(atoms_object.elements[i]) for i in range(atoms_object.num_atoms)]
    )
"""
