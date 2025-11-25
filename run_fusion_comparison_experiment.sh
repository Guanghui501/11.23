#!/bin/bash

# 融合位置对比实验
# 对比ALIGNN层融合、GCN层融合和层次化融合的效果

DATASET="jarvis"
PROPERTY="formation_energy_peratom"
ROOT_DIR="/public/home/ghzhang/crysmmnet-main/dataset"
EPOCHS=300
BATCH_SIZE=64

echo "======================================"
echo "  融合位置对比实验"
echo "======================================"
echo ""

# 实验1: ALIGNN层融合 (Middle Fusion)
echo "[实验1/3] 在ALIGNN层融合文本信息..."
python train_with_cross_modal_attention.py \
    --config configs/fusion_at_alignn.json \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --save_dir results/fusion_at_alignn \
    --description "ALIGNN层融合(中间融合)" \
    2>&1 | tee logs/fusion_at_alignn.log

echo ""
echo "✅ 实验1完成"
echo ""

# 实验2: GCN层融合 (Fine-grained Attention)
echo "[实验2/3] 在GCN层之后融合文本信息..."
python train_with_cross_modal_attention.py \
    --config configs/fusion_at_gcn.json \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --save_dir results/fusion_at_gcn \
    --description "GCN层融合(细粒度注意力)" \
    2>&1 | tee logs/fusion_at_gcn.log

echo ""
echo "✅ 实验2完成"
echo ""

# 实验3: 层次化融合 (Hierarchical Fusion)
echo "[实验3/3] 层次化融合(ALIGNN+GCN+全局)..."
python train_with_cross_modal_attention.py \
    --config configs/fusion_hierarchical.json \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --save_dir results/fusion_hierarchical \
    --description "层次化融合(ALIGNN+GCN+全局)" \
    2>&1 | tee logs/fusion_hierarchical.log

echo ""
echo "✅ 实验3完成"
echo ""

# 对比分析
echo "======================================"
echo "  开始对比分析..."
echo "======================================"

# 分析实验1
echo "[分析1/3] 分析ALIGNN层融合..."
python compare_fusion_mechanisms.py \
    --checkpoint results/fusion_at_alignn/best_test_model.pt \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --batch_size 32 \
    --max_samples 500 \
    --save_dir analysis/fusion_at_alignn

# 分析实验2
echo "[分析2/3] 分析GCN层融合..."
python compare_fusion_mechanisms.py \
    --checkpoint results/fusion_at_gcn/best_test_model.pt \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --batch_size 32 \
    --max_samples 500 \
    --save_dir analysis/fusion_at_gcn

# 分析实验3
echo "[分析3/3] 分析层次化融合..."
python compare_fusion_mechanisms.py \
    --checkpoint results/fusion_hierarchical/best_test_model.pt \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --batch_size 32 \
    --max_samples 500 \
    --save_dir analysis/fusion_hierarchical

echo ""
echo "======================================"
echo "  实验完成!"
echo "======================================"
echo ""
echo "结果位置:"
echo "  - ALIGNN层融合: results/fusion_at_alignn/"
echo "  - GCN层融合: results/fusion_at_gcn/"
echo "  - 层次化融合: results/fusion_hierarchical/"
echo ""
echo "分析结果:"
echo "  - ALIGNN层融合: analysis/fusion_at_alignn/"
echo "  - GCN层融合: analysis/fusion_at_gcn/"
echo "  - 层次化融合: analysis/fusion_hierarchical/"
echo ""
echo "请查看各个目录下的可视化结果和指标文件"
