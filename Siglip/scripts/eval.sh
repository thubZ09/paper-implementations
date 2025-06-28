MODEL_PATH=$1

python -m evaluation.retrieval \
    --model_path $MODEL_PATH \
    --split validation

python -m evaluation.zero_shot \
    --model_path $MODEL_PATH \
    --dataset_path ./imagenet