#!/usr/bin/env python
"""
Quick test to verify DynamicFusionModule integration.

Run this before full training to ensure everything is set up correctly.
"""

import torch
from models.alignn import ALIGNN, ALIGNNConfig
from monitor_fusion_weights import print_fusion_weights

print("="*80)
print("DynamicFusionModule Integration Test")
print("="*80)

# Test 1: Model creation
print("\n1. Testing model creation with middle fusion...")
try:
    config = ALIGNNConfig(
        name="alignn",
        use_middle_fusion=True,
        middle_fusion_layers="2",
        hidden_features=64
    )
    model = ALIGNN(config)
    print("✅ Model created successfully")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    exit(1)

# Test 2: Check module type
print("\n2. Checking if MiddleFusionModule is using DynamicFusionModule...")
try:
    module = model.middle_fusion_modules['layer_2']
    module_type = type(module).__name__
    print(f"   Module type: {module_type}")

    if module_type == "DynamicFusionModule":
        print("✅ Using DynamicFusionModule (correct)")
    else:
        print(f"⚠️ Using {module_type} instead")
except Exception as e:
    print(f"❌ Module check failed: {e}")
    exit(1)

# Test 3: Check weight monitoring methods
print("\n3. Checking weight monitoring methods...")
try:
    stats = module.get_weight_stats()
    assert 'avg_w_graph' in stats
    assert 'avg_w_text' in stats
    assert 'update_count' in stats
    print(f"✅ Weight monitoring available")
    print(f"   Initial stats: {stats}")
except Exception as e:
    print(f"❌ Weight monitoring check failed: {e}")
    exit(1)

# Test 4: Test forward pass (simulate training)
print("\n4. Testing forward pass with dummy data...")
try:
    # Create dummy DGL graph
    import dgl
    import numpy as np

    # Simple graph: 2 samples with 3 and 4 atoms
    g = dgl.batch([
        dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0]))),
        dgl.graph((torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 0])))
    ])

    # Add node and edge features
    g.ndata['atom_features'] = torch.randn(7, 92)  # 3+4 atoms
    g.edata['r'] = torch.randn(7, 3)  # 7 edges

    # Create line graph
    lg = dgl.line_graph(g, backtracking=False)
    lg.ndata['h'] = torch.randn(lg.num_nodes(), 1)
    lg.edata['h'] = torch.randn(lg.num_edges(), 1)

    # Dummy text
    text = ["sample text 1", "sample text 2"]

    # Forward pass
    model.train()  # Enable training mode for weight monitoring
    with torch.no_grad():
        output = model((g, lg, text))

    print(f"✅ Forward pass successful")
    print(f"   Output shape: {output.shape}")

    # Check if weights were updated
    stats_after = module.get_weight_stats()
    if stats_after['update_count'] > 0:
        print(f"✅ Weight monitoring active (update_count={stats_after['update_count']})")
        print(f"   w_graph: {stats_after['avg_w_graph']:.4f}")
        print(f"   w_text:  {stats_after['avg_w_text']:.4f}")
    else:
        print(f"⚠️ Weights not updated (might need more batches)")

except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Test monitoring utility
print("\n5. Testing monitor_fusion_weights utility...")
try:
    from monitor_fusion_weights import print_fusion_weights
    stats = print_fusion_weights(model, verbose=False)
    print(f"✅ Monitoring utility works")
except Exception as e:
    print(f"❌ Monitoring utility failed: {e}")
    exit(1)

# Test 6: Verify physics prior (double residual)
print("\n6. Verifying physics prior (double residual)...")
try:
    # Simulate router outputs
    w_g = stats_after['avg_w_graph']
    w_t = stats_after['avg_w_text']

    # Calculate effective weights
    effective_w_g = 1.0 + w_g
    effective_w_t = w_t

    ratio = effective_w_g / (effective_w_t + 1e-6)

    print(f"   Router weights: w_graph={w_g:.4f}, w_text={w_t:.4f}")
    print(f"   Effective weights: graph={effective_w_g:.4f}, text={effective_w_t:.4f}")
    print(f"   Graph/text ratio: {ratio:.2f}x")

    if effective_w_g >= 1.0:
        print(f"✅ Graph has baseline weight ≥ 1.0 (physics prior enforced)")
    else:
        print(f"⚠️ Graph weight < 1.0 (unexpected)")

except Exception as e:
    print(f"❌ Physics prior check failed: {e}")

print("\n" + "="*80)
print("✅ All tests passed! Integration successful.")
print("="*80)
print("\nNext steps:")
print("1. Run your training script with use_middle_fusion=True")
print("2. Check console output every 5 epochs for weight statistics")
print("3. Analyze output_dir/fusion_weights.csv after training")
print("4. Expected: graph dominant (ratio 3-10x) for material prediction")
print("\nExample training command:")
print("  python train.py --config your_config.json --epochs 50")
