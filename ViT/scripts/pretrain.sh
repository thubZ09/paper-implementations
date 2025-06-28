python -m training.train \
    --config configs.vit_small \
    --dataset cifar10 \
    --batch_size 64 \
    --epochs 100 \
    --lr 3e-4 \
    --mixed_precision