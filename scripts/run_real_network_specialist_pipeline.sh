#!/bin/bash
# BiT-HyRL 真实网络专家模型完整训练流程
# 1. 数据增强 2. 多阶段训练 3. 可视化 4. 评估

set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo "Step 1: Generate augmented data"
echo "========================================"
python scripts/generate_real_network_augmentation.py \
    --data-dir dataset/all/real \
    --output dataset/all/real_augmented \
    --per-graph 2 \
    --types BA,ER,Bimodal

echo ""
echo "========================================"
echo "Step 2: Multi-stage training (multi-seed)"
echo "========================================"
python scripts/train_real_network_specialist.py \
    --use-augmentation \
    --epochs 400 \
    --seeds 42,123,456,789,2024 \
    --k-ratios 0.08,0.10,0.12

echo ""
echo "========================================"
echo "Step 3: Evaluate on Colt, Chinanet, UsCarrier"
echo "========================================"
python scripts/evaluate_real_network_specialist.py \
    -m models/real_network_specialist_seed42.pth \
    --run-main \
    --datasets Colt,Chinanet,UsCarrier

echo ""
echo "Done! Check results/ and results/training_real_specialist/"
