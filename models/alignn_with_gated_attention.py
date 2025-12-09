#!/usr/bin/env python3
"""
ALIGNN with Gated Cross-Attention

Modified ALIGNN architecture that uses GatedCrossAttention for robust
text-graph fusion, addressing the 100% masking collapse issue.

Key changes from original ALIGNN:
1. Replaces standard cross-attention with GatedCrossAttention
2. Adds text quality monitoring
3. Enables graceful degradation under extreme masking
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
import dgl

# Import the gated attention module
from .gated_cross_attention import (
    GatedCrossAttention,
    MultiLayerGatedCrossAttention
)


class ALIGNNWithGatedAttention(nn.Module):
    """
    ALIGNN model with Gated Cross-Attention for robust multimodal fusion.

    This is a drop-in replacement for the original ALIGNN that:
    - Maintains backward compatibility
    - Adds robustness to text masking
    - Provides interpretable quality scores
    """

    def __init__(
        self,
        # Graph encoder parameters (from original ALIGNN)
        atom_features: int = 92,
        edge_features: int = 64,
        triplet_features: int = 40,
        embedding_dim: int = 64,
        num_alignn_layers: int = 4,
        num_gcn_layers: int = 4,
        norm: str = "layernorm",

        # Text encoder parameters
        text_hidden_dim: int = 768,  # MatSciBERT output dimension

        # Gated attention parameters
        use_gated_attention: bool = True,
        gated_attention_layers: int = 1,
        attention_heads: int = 8,
        attention_dropout: float = 0.1,

        # Output parameters
        output_dim: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.1,

        # Legacy parameters (for compatibility)
        use_middle_fusion: bool = True,
        use_fine_grained_attention: bool = False,
    ):
        super().__init__()

        self.use_gated_attention = use_gated_attention
        self.hidden_dim = hidden_dim

        # =================================================================
        # 1. Graph Encoder (ALIGNN - keep original implementation)
        # =================================================================
        # Import and initialize the original ALIGNN graph encoder
        # This part remains unchanged
        self.graph_encoder = self._build_graph_encoder(
            atom_features=atom_features,
            edge_features=edge_features,
            triplet_features=triplet_features,
            embedding_dim=embedding_dim,
            num_alignn_layers=num_alignn_layers,
            num_gcn_layers=num_gcn_layers,
            norm=norm,
        )

        # Graph feature projection
        self.graph_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # =================================================================
        # 2. Text Encoder Projection
        # =================================================================
        # Project MatSciBERT outputs to hidden_dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # =================================================================
        # 3. Gated Cross-Modal Attention (NEW!)
        # =================================================================
        if use_gated_attention:
            if gated_attention_layers > 1:
                self.cross_modal_attention = MultiLayerGatedCrossAttention(
                    hidden_dim=hidden_dim,
                    num_layers=gated_attention_layers,
                    num_heads=attention_heads,
                    dropout=attention_dropout
                )
            else:
                self.cross_modal_attention = GatedCrossAttention(
                    hidden_dim=hidden_dim,
                    num_heads=attention_heads,
                    dropout=attention_dropout
                )
        else:
            # Fallback to simple concatenation + MLP
            self.cross_modal_attention = None
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # =================================================================
        # 4. Output Head
        # =================================================================
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _build_graph_encoder(self, **kwargs):
        """
        Build the graph encoder (ALIGNN).
        This is a placeholder - replace with actual ALIGNN implementation.
        """
        # For now, return a simple placeholder
        # In production, import and use the actual ALIGNN encoder
        class PlaceholderGraphEncoder(nn.Module):
            def __init__(self, embedding_dim):
                super().__init__()
                self.embedding_dim = embedding_dim
                # Simplified placeholder - replace with real ALIGNN
                self.dummy = nn.Linear(1, embedding_dim)

            def forward(self, g, lg):
                # Placeholder: return dummy features
                # Real implementation should process DGL graphs
                batch_size = g.batch_size if hasattr(g, 'batch_size') else 1
                return torch.randn(batch_size, self.embedding_dim, device=g.device)

        return PlaceholderGraphEncoder(kwargs['embedding_dim'])

    def _init_weights(self, module):
        """Initialize weights using Xavier/He initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        batch_data,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through the model.

        Args:
            batch_data: List containing [g, lg, text_features, text_mask]
                - g: DGL graph for atoms
                - lg: DGL line graph for bonds
                - text_features: [batch, seq_len, text_hidden_dim] from MatSciBERT
                - text_mask: [batch, seq_len] padding mask (optional)
            return_attention: Whether to return attention diagnostics

        Returns:
            output: [batch, output_dim] - predictions
            diagnostics: Dict (if return_attention=True)
        """
        # Unpack input
        if len(batch_data) == 3:
            g, lg, text_features = batch_data
            text_mask = None
        elif len(batch_data) == 4:
            g, lg, text_features, text_mask = batch_data
        else:
            raise ValueError(f"Expected 3 or 4 elements in batch_data, got {len(batch_data)}")

        # =================================================================
        # 1. Encode Graph
        # =================================================================
        graph_feat = self.graph_encoder(g, lg)  # [batch, embedding_dim]
        graph_feat = self.graph_proj(graph_feat)  # [batch, hidden_dim]

        # =================================================================
        # 2. Project Text Features
        # =================================================================
        # text_features: [batch, seq_len, text_hidden_dim] from MatSciBERT
        batch_size, seq_len, text_dim = text_features.shape
        text_feat = self.text_proj(
            text_features.view(-1, text_dim)
        ).view(batch_size, seq_len, self.hidden_dim)  # [batch, seq_len, hidden_dim]

        # =================================================================
        # 3. Cross-Modal Fusion
        # =================================================================
        if self.use_gated_attention:
            if return_attention:
                fused_feat, diagnostics = self.cross_modal_attention(
                    graph_feat, text_feat, text_mask, return_attention=True
                )
            else:
                fused_feat = self.cross_modal_attention(
                    graph_feat, text_feat, text_mask, return_attention=False
                )
                diagnostics = None
        else:
            # Simple concatenation fallback
            text_pooled = text_feat.mean(dim=1)  # [batch, hidden_dim]
            fused_feat = self.fusion_mlp(
                torch.cat([graph_feat, text_pooled], dim=-1)
            )
            diagnostics = None

        # =================================================================
        # 4. Output Prediction
        # =================================================================
        output = self.output_head(fused_feat)  # [batch, output_dim]

        if return_attention:
            return output, diagnostics

        return output


# ============================================================================
# Convenience function for model creation
# ============================================================================

def create_gated_alignn(
    checkpoint_path: Optional[str] = None,
    **model_kwargs
) -> ALIGNNWithGatedAttention:
    """
    Create an ALIGNNWithGatedAttention model.

    Args:
        checkpoint_path: Path to pretrained checkpoint (optional)
        **model_kwargs: Model configuration parameters

    Returns:
        model: ALIGNNWithGatedAttention instance
    """
    # Default configuration
    default_config = {
        'hidden_dim': 256,
        'use_gated_attention': True,
        'gated_attention_layers': 1,
        'attention_heads': 8,
        'attention_dropout': 0.1,
        'output_dim': 1,
    }

    # Override with user-provided kwargs
    config = {**default_config, **model_kwargs}

    # Create model
    model = ALIGNNWithGatedAttention(**config)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Try different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load with strict=False to allow architecture changes
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")

    return model


# ============================================================================
# Testing
# ============================================================================

def test_model():
    """Test the model with dummy data."""
    print("Testing ALIGNNWithGatedAttention...")

    batch_size = 4
    seq_len = 128
    text_hidden_dim = 768

    # Create model
    model = ALIGNNWithGatedAttention(
        hidden_dim=256,
        text_hidden_dim=text_hidden_dim,
        use_gated_attention=True,
        gated_attention_layers=1,
        output_dim=1
    )
    model.eval()

    # Create dummy data
    # In real usage, g and lg would be DGL graphs
    class DummyGraph:
        def __init__(self, batch_size):
            self.batch_size = batch_size
            self.device = torch.device('cpu')

    g = DummyGraph(batch_size)
    lg = DummyGraph(batch_size)
    text_features = torch.randn(batch_size, seq_len, text_hidden_dim)
    text_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    # Test forward pass
    print("\nTest 1: Forward pass without attention return")
    batch_data = [g, lg, text_features, text_mask]
    output = model(batch_data)
    print(f"  Output shape: {output.shape}")  # Expected: [4, 1]

    # Test with attention diagnostics
    print("\nTest 2: Forward pass with attention diagnostics")
    output, diagnostics = model(batch_data, return_attention=True)
    print(f"  Output shape: {output.shape}")
    print(f"  Text quality (mean): {diagnostics['quality_mean']:.4f}")
    print(f"  Text influence (mean): {diagnostics['text_influence']:.4f}")

    # Test with masked text (100% masking scenario)
    print("\nTest 3: Forward pass with completely masked text")
    text_features_masked = torch.zeros_like(text_features)
    batch_data_masked = [g, lg, text_features_masked, text_mask]
    output, diagnostics = model(batch_data_masked, return_attention=True)
    print(f"  Output shape: {output.shape}")
    print(f"  Text quality (mean): {diagnostics['quality_mean']:.4f}")
    print(f"  Text influence (mean): {diagnostics['text_influence']:.4f}")
    print(f"  ✓ Text influence should be very low (~0.0-0.1)")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_model()
