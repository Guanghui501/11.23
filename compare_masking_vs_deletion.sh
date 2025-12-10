#!/bin/bash
#
# Compare MASKING vs DELETION methods
#
# Key difference:
# - Masking:  100% → "[MASK] [MASK] [MASK]" (text still exists)
# - Deletion: 100% → "" (empty string, true graph-only)
#

set -e

# Configuration
TEST_DATA="./corrected_test_set/test.pkl"
MODEL1="./checkpoints/model1_checkpoint.pt"
MODEL2="./checkpoints/sage_net_checkpoint.pt"
MODEL1_NAME="model1+2 (no Middle Fusion)"
MODEL2_NAME="SAGE-Net (with Middle Fusion)"

STRATEGIES="random_token random_word keep_keywords"
RATIOS="0.0 0.5 1.0"
SEED=42

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                           ║"
echo "║  Comparing MASKING vs DELETION                                           ║"
echo "║                                                                           ║"
echo "║  Masking:  Tokens replaced with [MASK]                                  ║"
echo "║            100% → '[MASK] [MASK] [MASK] ...'                            ║"
echo "║            Text encoder still produces embeddings!                       ║"
echo "║                                                                           ║"
echo "║  Deletion: Tokens removed completely                                     ║"
echo "║            100% → '' (empty string)                                     ║"
echo "║            True graph-only performance!                                  ║"
echo "║                                                                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo

# ============================================================================
# Step 1: Generate MASKING datasets
# ============================================================================
echo "Step 1: Generating MASKING datasets..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python pregenerate_masked_dataset.py \
    --input_data "$TEST_DATA" \
    --output_dir "./masked_datasets_comparison" \
    --strategies $STRATEGIES \
    --ratios $RATIOS \
    --seed $SEED

echo
echo "✓ Masking datasets generated in ./masked_datasets_comparison"
echo

# ============================================================================
# Step 2: Generate DELETION datasets
# ============================================================================
echo "Step 2: Generating DELETION datasets..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python pregenerate_deletion_dataset.py \
    --input_data "$TEST_DATA" \
    --output_dir "./deletion_datasets_comparison" \
    --strategies $STRATEGIES \
    --ratios $RATIOS \
    --seed $SEED

echo
echo "✓ Deletion datasets generated in ./deletion_datasets_comparison"
echo

# ============================================================================
# Step 3: Evaluate with MASKING datasets
# ============================================================================
echo "Step 3: Evaluating models with MASKING datasets..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Evaluating $MODEL1_NAME with MASKING..."
for strategy in $STRATEGIES; do
    for ratio in $RATIOS; do
        python evaluate_with_premasked_data.py \
            --checkpoint "$MODEL1" \
            --masked_data "./masked_datasets_comparison/${strategy}_${ratio}.pkl" \
            --output "./results_masking_model1_${strategy}_${ratio}.json"
    done
done

echo
echo "Evaluating $MODEL2_NAME with MASKING..."
for strategy in $STRATEGIES; do
    for ratio in $RATIOS; do
        python evaluate_with_premasked_data.py \
            --checkpoint "$MODEL2" \
            --masked_data "./masked_datasets_comparison/${strategy}_${ratio}.pkl" \
            --output "./results_masking_model2_${strategy}_${ratio}.json"
    done
done

echo
echo "✓ MASKING evaluation complete"
echo

# ============================================================================
# Step 4: Evaluate with DELETION datasets
# ============================================================================
echo "Step 4: Evaluating models with DELETION datasets..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Evaluating $MODEL1_NAME with DELETION..."
for strategy in $STRATEGIES; do
    for ratio in $RATIOS; do
        python evaluate_with_premasked_data.py \
            --checkpoint "$MODEL1" \
            --masked_data "./deletion_datasets_comparison/${strategy}_${ratio}.pkl" \
            --output "./results_deletion_model1_${strategy}_${ratio}.json"
    done
done

echo
echo "Evaluating $MODEL2_NAME with DELETION..."
for strategy in $STRATEGIES; do
    for ratio in $RATIOS; do
        python evaluate_with_premasked_data.py \
            --checkpoint "$MODEL2" \
            --masked_data "./deletion_datasets_comparison/${strategy}_${ratio}.pkl" \
            --output "./results_deletion_model2_${strategy}_${ratio}.json"
    done
done

echo
echo "✓ DELETION evaluation complete"
echo

# ============================================================================
# Step 5: Compare results
# ============================================================================
echo "Step 5: Comparing MASKING vs DELETION results..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python compare_masking_deletion_results.py \
    --masking_model1 "./results_masking_model1_*.json" \
    --masking_model2 "./results_masking_model2_*.json" \
    --deletion_model1 "./results_deletion_model1_*.json" \
    --deletion_model2 "./results_deletion_model2_*.json" \
    --model1_name "$MODEL1_NAME" \
    --model2_name "$MODEL2_NAME" \
    --output_dir "./masking_vs_deletion_comparison"

echo
echo "✓ Comparison complete"
echo

# ============================================================================
# Summary
# ============================================================================
echo
echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                           ║"
echo "║  All Done!                                                               ║"
echo "║                                                                           ║"
echo "║  Results:                                                                ║"
echo "║    - MASKING results:  ./results_masking_*.json                         ║"
echo "║    - DELETION results: ./results_deletion_*.json                        ║"
echo "║    - Comparison plots: ./masking_vs_deletion_comparison/                ║"
echo "║                                                                           ║"
echo "║  Key comparison:                                                         ║"
echo "║    At 100% ratio:                                                        ║"
echo "║      - MASKING:  Text = '[MASK] [MASK] ...' (has embeddings)           ║"
echo "║      - DELETION: Text = '' (truly empty, graph-only)                    ║"
echo "║                                                                           ║"
echo "║  Expected:                                                               ║"
echo "║    DELETION should give BETTER performance at 100% because it's         ║"
echo "║    true graph-only (no noisy [MASK] embeddings mixed in)!              ║"
echo "║                                                                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo
