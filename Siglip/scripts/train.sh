python -m training.train \
    --batch_size 64 \
    --epochs 20 \
    --learning_rate 3e-4 \
    --image_encoder_lr 1e-4 \
    --text_encoder_lr 1e-5 \
    --head_lr 1e-3 \
    --mixed_precision \
    --output_dir ./checkpoints