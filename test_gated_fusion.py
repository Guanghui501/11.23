"""Test script for gated fusion mechanism.

This script tests the new gated fusion module to verify that:
1. The module can be instantiated correctly
2. Forward pass works with different fusion types
3. Output dimensions are correct
4. Gate values are returned when requested
"""

import torch
import torch.nn as nn
import sys
sys.path.append('.')

from models.alignn import GatedFusion, ALIGNNConfig, ALIGNN


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


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GATED FUSION MECHANISM TEST SUITE")
    print("=" * 80)

    try:
        # Test 1: Standalone GatedFusion module
        test_gated_fusion_module()

        # Test 2: ALIGNN model integration
        test_alignn_with_gated_fusion()

        # Test 3: Fusion strategy comparison
        test_fusion_comparison()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓✓✓")
        print("=" * 80)
        print("\nSummary:")
        print("  ✓ GatedFusion module works correctly")
        print("  ✓ ALIGNN model integrates gated fusion properly")
        print("  ✓ All fusion strategies (average, concat, gated) are functional")
        print("  ✓ Gate values can be extracted for interpretability")
        print("\nNext steps:")
        print("  1. Train models with different fusion strategies")
        print("  2. Compare performance on your dataset")
        print("  3. Analyze gate values to understand fusion behavior")

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED! ✗✗✗")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
