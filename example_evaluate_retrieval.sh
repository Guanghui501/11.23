#!/bin/bash

# Example script for running retrieval evaluation
# This script demonstrates different use cases of evaluate_retrieval.py

echo "========================================"
echo "Multimodal Retrieval Evaluation Examples"
echo "========================================"

# ====================
# Example 1: Basic evaluation on test set
# ====================
echo ""
echo "Example 1: Basic evaluation on test set with visualization"
echo "-----------------------------------------------------------"

python evaluate_retrieval.py \
    --model_path outputs/best_val_model.pt \
    --split test \
    --visualize

# ====================
# Example 2: Evaluation with custom output directory
# ====================
echo ""
echo "Example 2: Custom output directory"
echo "-----------------------------------"

python evaluate_retrieval.py \
    --model_path outputs/best_val_model.pt \
    --config_path outputs/config.json \
    --split test \
    --output_dir outputs/retrieval_evaluation_test \
    --batch_size 32 \
    --visualize

# ====================
# Example 3: Evaluate on validation set (for debugging/analysis)
# ====================
echo ""
echo "Example 3: Validation set evaluation"
echo "-------------------------------------"

python evaluate_retrieval.py \
    --model_path outputs/best_val_model.pt \
    --split val \
    --output_dir outputs/retrieval_evaluation_val \
    --visualize

# ====================
# Example 4: Evaluation on CPU (for systems without GPU)
# ====================
echo ""
echo "Example 4: CPU evaluation"
echo "-------------------------"

python evaluate_retrieval.py \
    --model_path outputs/best_val_model.pt \
    --split test \
    --device cpu \
    --batch_size 16

# ====================
# Example 5: Compare multiple models
# ====================
echo ""
echo "Example 5: Compare multiple models"
echo "-----------------------------------"

# Baseline model
python evaluate_retrieval.py \
    --model_path outputs/baseline/best_val_model.pt \
    --split test \
    --output_dir outputs/baseline/retrieval \
    --visualize

# Model with contrastive learning
python evaluate_retrieval.py \
    --model_path outputs/with_contrastive/best_val_model.pt \
    --split test \
    --output_dir outputs/with_contrastive/retrieval \
    --visualize

# Model with cross-modal attention
python evaluate_retrieval.py \
    --model_path outputs/with_attention/best_val_model.pt \
    --split test \
    --output_dir outputs/with_attention/retrieval \
    --visualize

echo ""
echo "Compare results by checking retrieval_metrics_test.json in each output directory"

# ====================
# Example 6: Save detailed top-20 retrievals for analysis
# ====================
echo ""
echo "Example 6: Detailed retrieval results (top-20)"
echo "----------------------------------------------"

python evaluate_retrieval.py \
    --model_path outputs/best_val_model.pt \
    --split test \
    --top_k 20 \
    --output_dir outputs/detailed_retrieval \
    --visualize

echo ""
echo "========================================"
echo "All examples completed!"
echo "========================================"
