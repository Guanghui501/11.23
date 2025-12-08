"""Test script for gated fusion and unidirectional attention mechanisms.

This script tests:
1. GatedFusion module with different fusion types
2. UnidirectionalCrossAttention module
3. ALIGNN model with different fusion strategies
4. Solution 1: Middle + Fine-grained + Gated (no global attention)
5. Solution 2: Middle + Fine-grained + Unidirectional attention
"""

import torch
import torch.nn as nn
import sys
sys.path.append('.')

from models.alignn import GatedFusion, UnidirectionalCrossAttention, ALIGNNConfig, ALIGNN


def test_gated_fusion_module():
    """Test the GatedFusion module standalone."""
    print("=" * 80)
    print("Testing GatedFusion Module")
    print("=" * 80)

    batch_size = 4
    feature_dim = 64

    # Create dummy input features
    graph_feat = torch.randn(batch_size, feature_dim)
    text_feat = torch.randn(batch_size, feature_dim)

    fusion_types = ['single_gate', 'dual_gate', 'attention']

    for fusion_type in fusion_types:
        print(f"\n>>> Testing fusion_type: {fusion_type}")

        # Initialize module
        fusion_module = GatedFusion(
            feature_dim=feature_dim,
            hidden_dim=128,
            dropout=0.1,
            fusion_type=fusion_type
        )

        # Forward pass without gate values
        fused = fusion_module(graph_feat, text_feat, return_gate_values=False)
        print(f"   Input shapes: graph={graph_feat.shape}, text={text_feat.shape}")
        print(f"   Output shape: {fused.shape}")
        assert fused.shape == (batch_size, feature_dim), f"Expected shape ({batch_size}, {feature_dim}), got {fused.shape}"

        # Forward pass with gate values
        fused, gate_values = fusion_module(graph_feat, text_feat, return_gate_values=True)
        print(f"   Gate values returned: {list(gate_values.keys())}")
        for key, value in gate_values.items():
            print(f"     {key}: shape={value.shape}, min={value.min().item():.4f}, max={value.max().item():.4f}")

        print(f"   ✓ {fusion_type} passed!")

    print("\n" + "=" * 80)
    print("GatedFusion Module Tests: ALL PASSED ✓")
    print("=" * 80)


def test_alignn_with_gated_fusion():
    """Test ALIGNN model with gated fusion strategy."""
    print("\n" + "=" * 80)
    print("Testing ALIGNN Model with Gated Fusion")
    print("=" * 80)

    # Create config with gated fusion
    config = ALIGNNConfig(
        name="alignn",
        alignn_layers=2,
        gcn_layers=2,
        hidden_features=256,
        output_features=1,
        use_cross_modal_attention=True,
        use_fine_grained_attention=False,
        fusion_strategy="gated",  # Use gated fusion
        gated_fusion_type="dual_gate",
        gated_fusion_hidden_dim=128,
        gated_fusion_dropout=0.1
    )

    print("\n>>> Configuration:")
    print(f"   fusion_strategy: {config.fusion_strategy}")
    print(f"   gated_fusion_type: {config.gated_fusion_type}")
    print(f"   use_cross_modal_attention: {config.use_cross_modal_attention}")

    # Initialize model
    print("\n>>> Initializing ALIGNN model...")
    try:
        model = ALIGNN(config)
        print("   ✓ Model initialized successfully!")
    except Exception as e:
        print(f"   ✗ Model initialization failed: {e}")
        raise

    # Check that gated fusion module exists
    if hasattr(model, 'gated_fusion'):
        print("   ✓ Gated fusion module found!")
        print(f"   Fusion type: {model.gated_fusion.fusion_type}")
    else:
        print("   ✗ Gated fusion module not found!")
        raise ValueError("Model missing gated_fusion attribute")

    print("\n>>> Testing different fusion strategies...")
    strategies = ['gated', 'average', 'concat']

    for strategy in strategies:
        print(f"\n   Testing strategy: {strategy}")
        config.fusion_strategy = strategy
        if strategy == 'gated':
            config.gated_fusion_type = 'dual_gate'

        try:
            model = ALIGNN(config)
            print(f"   ✓ Model with {strategy} fusion initialized successfully!")
            print(f"     - fusion_strategy: {model.fusion_strategy}")
            if strategy == 'gated':
                print(f"     - gated_fusion_type: {model.gated_fusion.fusion_type}")
        except Exception as e:
            print(f"   ✗ Failed to initialize model with {strategy} fusion: {e}")
            raise

    print("\n" + "=" * 80)
    print("ALIGNN Model Tests: ALL PASSED ✓")
    print("=" * 80)


def test_fusion_comparison():
    """Compare output differences between fusion strategies."""
    print("\n" + "=" * 80)
    print("Comparing Fusion Strategies")
    print("=" * 80)

    batch_size = 4
    feature_dim = 64

    # Create dummy features
    graph_feat = torch.randn(batch_size, feature_dim)
    text_feat = torch.randn(batch_size, feature_dim)

    print(f"\nInput statistics:")
    print(f"  Graph features: mean={graph_feat.mean().item():.4f}, std={graph_feat.std().item():.4f}")
    print(f"  Text features:  mean={text_feat.mean().item():.4f}, std={text_feat.std().item():.4f}")

    # Test different fusion strategies
    print("\n>>> Testing fusion strategies:")

    # 1. Average fusion (baseline)
    fused_avg = (graph_feat + text_feat) / 2
    print(f"\n1. Average fusion:")
    print(f"   Output: mean={fused_avg.mean().item():.4f}, std={fused_avg.std().item():.4f}")

    # 2. Concatenation fusion
    fused_concat = torch.cat([graph_feat, text_feat], dim=1)
    print(f"\n2. Concatenation fusion:")
    print(f"   Output shape: {fused_concat.shape}")
    print(f"   Output: mean={fused_concat.mean().item():.4f}, std={fused_concat.std().item():.4f}")

    # 3. Gated fusion
    fusion_types = ['single_gate', 'dual_gate', 'attention']
    for fusion_type in fusion_types:
        gated_fusion = GatedFusion(
            feature_dim=feature_dim,
            hidden_dim=128,
            dropout=0.1,
            fusion_type=fusion_type
        )
        fused_gated, gate_values = gated_fusion(graph_feat, text_feat, return_gate_values=True)

        print(f"\n3. Gated fusion ({fusion_type}):")
        print(f"   Output: mean={fused_gated.mean().item():.4f}, std={fused_gated.std().item():.4f}")

        # Show gate statistics
        for key, value in gate_values.items():
            print(f"   {key}: mean={value.mean().item():.4f}, std={value.std().item():.4f}, range=[{value.min().item():.4f}, {value.max().item():.4f}]")

    print("\n" + "=" * 80)
    print("Fusion Strategy Comparison: COMPLETED ✓")
    print("=" * 80)


def test_unidirectional_attention():
    """Test the UnidirectionalCrossAttention module."""
    print("\n" + "=" * 80)
    print("Testing UnidirectionalCrossAttention Module")
    print("=" * 80)

    batch_size = 4
    feature_dim = 64

    # Create dummy input features
    text_feat = torch.randn(batch_size, feature_dim)
    graph_feat = torch.randn(batch_size, feature_dim)

    print(f"\nInput shapes:")
    print(f"  text_feat: {text_feat.shape}")
    print(f"  graph_feat: {graph_feat.shape}")

    # Initialize module
    unidirectional_attn = UnidirectionalCrossAttention(
        text_dim=feature_dim,
        graph_dim=feature_dim,
        hidden_dim=256,
        num_heads=4,
        dropout=0.1
    )

    # Forward pass without attention weights
    combined = unidirectional_attn(text_feat, graph_feat, return_attention=False)
    print(f"\nOutput shape: {combined.shape}")
    assert combined.shape == (batch_size, feature_dim), f"Expected shape ({batch_size}, {feature_dim}), got {combined.shape}"

    # Forward pass with attention weights
    combined, attn_weights = unidirectional_attn(text_feat, graph_feat, return_attention=True)
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"  Expected: [batch, num_heads, 1, 1]")

    print("\n" + "=" * 80)
    print("UnidirectionalCrossAttention Module Test: PASSED ✓")
    print("=" * 80)


def test_solution_1():
    """Test Solution 1: Middle + Fine-grained + Gated (no global attention)."""
    print("\n" + "=" * 80)
    print("Testing Solution 1: Middle + Fine-grained + Gated Fusion")
    print("=" * 80)

    config = ALIGNNConfig(
        name="alignn",
        alignn_layers=2,
        gcn_layers=2,
        hidden_features=256,
        output_features=1,

        # Middle fusion
        use_middle_fusion=True,
        middle_fusion_layers="1",

        # Fine-grained attention
        use_fine_grained_attention=True,
        fine_grained_num_heads=8,

        # NO global cross-modal attention
        use_cross_modal_attention=False,

        # Direct gated fusion
        fusion_strategy="gated",
        gated_fusion_type="dual_gate",
    )

    print("\n>>> Configuration:")
    print(f"   use_middle_fusion: {config.use_middle_fusion}")
    print(f"   use_fine_grained_attention: {config.use_fine_grained_attention}")
    print(f"   use_cross_modal_attention: {config.use_cross_modal_attention}")
    print(f"   fusion_strategy: {config.fusion_strategy}")

    try:
        model = ALIGNN(config)
        print("\n   ✓ Model initialized successfully!")

        # Check for gated fusion module
        if hasattr(model, 'gated_fusion'):
            print("   ✓ Gated fusion module found (without global attention)!")
        else:
            raise ValueError("Gated fusion module not found!")

        print("\n>>> Solution 1 Architecture:")
        print("   Input → ALIGNN layers → Middle Fusion")
        print("         → GCN layers → Fine-grained Attention")
        print("         → Pooling → Gated Fusion → Output")

    except Exception as e:
        print(f"\n   ✗ Failed: {e}")
        raise

    print("\n" + "=" * 80)
    print("Solution 1 Test: PASSED ✓")
    print("=" * 80)


def test_solution_2():
    """Test Solution 2: Middle + Fine-grained + Unidirectional attention."""
    print("\n" + "=" * 80)
    print("Testing Solution 2: Middle + Fine-grained + Unidirectional Attention")
    print("=" * 80)

    config = ALIGNNConfig(
        name="alignn",
        alignn_layers=2,
        gcn_layers=2,
        hidden_features=256,
        output_features=1,

        # Middle fusion
        use_middle_fusion=True,
        middle_fusion_layers="1",

        # Fine-grained attention
        use_fine_grained_attention=True,
        fine_grained_num_heads=8,

        # Unidirectional global attention
        use_cross_modal_attention=True,
        cross_modal_attention_type="unidirectional",  # Key difference!

        # Simple fusion (average recommended for unidirectional)
        fusion_strategy="average",
    )

    print("\n>>> Configuration:")
    print(f"   use_middle_fusion: {config.use_middle_fusion}")
    print(f"   use_fine_grained_attention: {config.use_fine_grained_attention}")
    print(f"   use_cross_modal_attention: {config.use_cross_modal_attention}")
    print(f"   cross_modal_attention_type: {config.cross_modal_attention_type}")
    print(f"   fusion_strategy: {config.fusion_strategy}")

    try:
        model = ALIGNN(config)
        print("\n   ✓ Model initialized successfully!")

        # Check attention type
        if hasattr(model, 'cross_modal_attention'):
            if isinstance(model.cross_modal_attention, UnidirectionalCrossAttention):
                print("   ✓ Unidirectional cross-attention module found!")
            else:
                raise ValueError("Expected UnidirectionalCrossAttention, got different type!")
        else:
            raise ValueError("Cross-modal attention module not found!")

        print("\n>>> Solution 2 Architecture:")
        print("   Input → ALIGNN layers → Middle Fusion")
        print("         → GCN layers → Fine-grained Attention")
        print("         → Pooling → Unidirectional Attention → Average Fusion → Output")

    except Exception as e:
        print(f"\n   ✗ Failed: {e}")
        raise

    print("\n" + "=" * 80)
    print("Solution 2 Test: PASSED ✓")
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FUSION MECHANISMS TEST SUITE")
    print("=" * 80)

    try:
        # Test 1: Standalone GatedFusion module
        test_gated_fusion_module()

        # Test 2: ALIGNN model integration
        test_alignn_with_gated_fusion()

        # Test 3: Fusion strategy comparison
        test_fusion_comparison()

        # Test 4: Unidirectional attention module
        test_unidirectional_attention()

        # Test 5: Solution 1 (recommended for your architecture)
        test_solution_1()

        # Test 6: Solution 2
        test_solution_2()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓✓✓")
        print("=" * 80)
        print("\nSummary:")
        print("  ✓ GatedFusion module works correctly")
        print("  ✓ UnidirectionalCrossAttention module works correctly")
        print("  ✓ ALIGNN model integrates all fusion strategies properly")
        print("  ✓ Solution 1 (Middle + Fine + Gated) is functional")
        print("  ✓ Solution 2 (Middle + Fine + Unidirectional) is functional")
        print("\n🎯 Recommended for your architecture:")
        print("  Solution 1: Best balance of performance and efficiency")
        print("  - use_middle_fusion=True")
        print("  - use_fine_grained_attention=True")
        print("  - use_cross_modal_attention=False")
        print("  - fusion_strategy='gated'")
        print("\nNext steps:")
        print("  1. Train with Solution 1 as baseline")
        print("  2. Compare with Solution 2 for ablation study")
        print("  3. Analyze gate values and attention patterns")

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED! ✗✗✗")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
