MODEL_PATH=$1

python -m training.train \
    --config configs.vit_small \
    --pretrained $MODEL_PATH \
    --dataset cifar10 \
    --batch_size 32 \
    --epochs 20 \
    --lr 1e-4 \
    --mixed_precision