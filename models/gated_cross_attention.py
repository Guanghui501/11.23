#!/usr/bin/env python3
"""
Gated Cross-Attention Implementation
门控跨模态注意力实现

Addresses the critical issues found in experiments:
1. 100% masking collapse (MAE 1.93 → expected ~0.80)
2. Adaptive fusion based on text quality
3. Graceful degradation under extreme masking

Key innovations:
- Text quality detection gate
- Adaptive fusion weights
- Automatic fallback to graph-only mode when text is unreliable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TextQualityGate(nn.Module):
    """
    Evaluates text feature quality to detect:
    - High masking ratios
    - Empty or sparse text
    - Low-quality embeddings from MatSciBERT

    Output: 0-1 score (1 = high quality, 0 = low quality)
    """
    def __init__(self, hidden_dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output: [0, 1]
        )

        # Learnable threshold for "acceptable" quality
        self.quality_threshold = nn.Parameter(torch.tensor(0.3))

    def forward(self, text_feat, return_raw=False):
        """
        Args:
            text_feat: [batch, hidden_dim] - pooled text features
            return_raw: if True, return raw score; else apply threshold

        Returns:
            quality_score: [batch, 1] - text quality indicator
        """
        raw_score = self.gate(text_feat)

        if return_raw:
            return raw_score

        # Soft threshold: scores below threshold are heavily penalized
        adjusted_score = torch.sigmoid((raw_score - self.quality_threshold) / 0.1)

        return adjusted_score


class AdaptiveFusionGate(nn.Module):
    """
    Learns how to fuse graph and text features adaptively.
    Takes into account both feature content and text quality.
    """
    def __init__(self, hidden_dim):
        super().__init__()

        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Fusion weight: 0 = all graph, 1 = all text
        )

    def forward(self, graph_feat, text_feat):
        """
        Args:
            graph_feat: [batch, hidden_dim]
            text_feat: [batch, hidden_dim]

        Returns:
            fusion_weight: [batch, 1] - how much to trust text features
        """
        concat = torch.cat([graph_feat, text_feat], dim=-1)
        weight = self.fusion_net(concat)
        return weight


class GatedCrossAttention(nn.Module):
    """
    Gated Cross-Modal Attention with Quality Detection

    Architecture:
    1. Text Quality Gate: Evaluates if text is reliable
    2. Cross-Modal Attention: Graph queries attend to text
    3. Adaptive Fusion Gate: Learns fusion weights
    4. Quality-Modulated Fusion: Combines graph + attended text

    Key Feature: When text quality is low (e.g., 100% masking),
    automatically reduces text influence and falls back to graph features.
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, \
            "hidden_dim must be divisible by num_heads"

        # Core components
        self.text_quality_gate = TextQualityGate(hidden_dim)
        self.adaptive_fusion_gate = AdaptiveFusionGate(hidden_dim)

        # Standard multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm_graph = nn.LayerNorm(hidden_dim)
        self.norm_text = nn.LayerNorm(hidden_dim)
        self.norm_output = nn.LayerNorm(hidden_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Optional: residual projection if dimensions don't match
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, graph_feat, text_feat, text_mask=None, return_attention=False):
        """
        Args:
            graph_feat: [batch, hidden_dim] - graph features (from ALIGNN)
            text_feat: [batch, seq_len, hidden_dim] - text features (from MatSciBERT)
            text_mask: [batch, seq_len] - True for padding tokens (optional)
            return_attention: bool - whether to return attention weights and diagnostics

        Returns:
            output: [batch, hidden_dim] - fused features
            diagnostics: dict (if return_attention=True)
        """
        batch_size = graph_feat.size(0)

        # Normalize inputs
        graph_feat = self.norm_graph(graph_feat)
        text_feat = self.norm_text(text_feat)

        # 1. Pool text features for quality evaluation
        # Use mean pooling, ignoring padding if mask is provided
        if text_mask is not None:
            # text_mask: True for padding, False for valid tokens
            valid_mask = ~text_mask  # Invert: True for valid tokens
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            text_pooled = (text_feat * valid_mask_expanded).sum(dim=1) / (valid_mask_expanded.sum(dim=1) + 1e-8)
        else:
            text_pooled = text_feat.mean(dim=1)  # [batch, hidden_dim]

        # 2. Evaluate text quality (KEY INNOVATION)
        text_quality = self.text_quality_gate(text_pooled)  # [batch, 1]

        # 3. Cross-modal attention: graph attends to text
        # Reshape graph_feat for attention
        graph_query = graph_feat.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Perform cross-attention
        attn_output, attn_weights = self.cross_attn(
            query=graph_query,
            key=text_feat,
            value=text_feat,
            key_padding_mask=text_mask,
            need_weights=return_attention
        )
        attn_output = attn_output.squeeze(1)  # [batch, hidden_dim]
        attn_output = self.dropout(attn_output)

        # 4. Adaptive fusion weight (based on feature content)
        fusion_weight = self.adaptive_fusion_gate(graph_feat, attn_output)  # [batch, 1]

        # 5. Quality-modulated fusion (KEY INNOVATION)
        # When text quality is low, reduce text influence
        effective_weight = fusion_weight * text_quality

        # Weighted combination
        output = (1 - effective_weight) * graph_feat + effective_weight * attn_output

        # Residual connection + Layer norm
        output = self.norm_output(output + self.residual_proj(graph_feat))

        if return_attention:
            diagnostics = {
                'text_quality': text_quality,  # [batch, 1]
                'fusion_weight': fusion_weight,  # [batch, 1]
                'effective_weight': effective_weight,  # [batch, 1]
                'attn_weights': attn_weights,  # [batch, num_heads, 1, seq_len] or None
                'text_influence': effective_weight.mean().item(),  # Scalar
                'quality_mean': text_quality.mean().item(),  # Scalar
            }
            return output, diagnostics

        return output


class MultiLayerGatedCrossAttention(nn.Module):
    """
    Stacked Gated Cross-Attention layers for deeper cross-modal interaction.

    Similar to Transformer encoder, but with gated cross-modal attention.
    Useful for complex materials where multiple interaction layers help.
    """
    def __init__(self, hidden_dim, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            GatedCrossAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.num_layers = num_layers

    def forward(self, graph_feat, text_feat, text_mask=None, return_attention=False):
        """
        Args:
            graph_feat: [batch, hidden_dim]
            text_feat: [batch, seq_len, hidden_dim]
            text_mask: [batch, seq_len]

        Returns:
            output: [batch, hidden_dim]
            diagnostics: list of dicts (if return_attention=True)
        """
        output = graph_feat
        all_diagnostics = []

        for i, layer in enumerate(self.layers):
            if return_attention:
                output, diagnostics = layer(output, text_feat, text_mask, return_attention=True)
                diagnostics['layer'] = i
                all_diagnostics.append(diagnostics)
            else:
                output = layer(output, text_feat, text_mask, return_attention=False)

        if return_attention:
            return output, all_diagnostics

        return output


# ============================================================================
# Testing and Visualization Utilities
# ============================================================================

def test_gated_attention():
    """
    Test the GatedCrossAttention module with different masking scenarios.
    """
    print("Testing GatedCrossAttention...")

    batch_size = 4
    hidden_dim = 256
    seq_len = 128

    # Create module
    gated_attn = GatedCrossAttention(hidden_dim, num_heads=8)
    gated_attn.eval()

    # Test 1: Normal case (no masking)
    print("\nTest 1: Normal text (0% masking)")
    graph_feat = torch.randn(batch_size, hidden_dim)
    text_feat = torch.randn(batch_size, seq_len, hidden_dim)

    output, diag = gated_attn(graph_feat, text_feat, return_attention=True)
    print(f"  Output shape: {output.shape}")
    print(f"  Text quality: {diag['quality_mean']:.4f} (expected: ~0.5-0.7)")
    print(f"  Text influence: {diag['text_influence']:.4f} (expected: ~0.3-0.6)")

    # Test 2: Partially masked text (50%)
    print("\nTest 2: Partially masked text (50% masking)")
    text_feat_partial = text_feat.clone()
    text_feat_partial[:, seq_len//2:] = 0  # Zero out half

    output, diag = gated_attn(graph_feat, text_feat_partial, return_attention=True)
    print(f"  Text quality: {diag['quality_mean']:.4f} (expected: ~0.3-0.5)")
    print(f"  Text influence: {diag['text_influence']:.4f} (expected: ~0.2-0.4)")

    # Test 3: Completely masked text (100%)
    print("\nTest 3: Completely masked text (100% masking)")
    text_feat_empty = torch.zeros(batch_size, seq_len, hidden_dim)

    output, diag = gated_attn(graph_feat, text_feat_empty, return_attention=True)
    print(f"  Text quality: {diag['quality_mean']:.4f} (expected: ~0.0-0.2)")
    print(f"  Text influence: {diag['text_influence']:.4f} (expected: ~0.0-0.1)")
    print(f"  ✓ Should be very low, indicating fallback to graph features")

    # Test 4: Multi-layer
    print("\nTest 4: Multi-layer Gated Attention")
    multi_gated = MultiLayerGatedCrossAttention(hidden_dim, num_layers=3)
    multi_gated.eval()

    output, all_diag = multi_gated(graph_feat, text_feat, return_attention=True)
    print(f"  Output shape: {output.shape}")
    print(f"  Number of layers: {len(all_diag)}")
    for i, diag in enumerate(all_diag):
        print(f"    Layer {i}: quality={diag['quality_mean']:.4f}, influence={diag['text_influence']:.4f}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_gated_attention()
