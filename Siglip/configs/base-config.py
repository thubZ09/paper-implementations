class BaseConfig:
    #parameters
    image_encoder = "vit_small_patch16_224"
    text_encoder = "distilroberta-base"
    projection_dim = 256
    temperature = 10.0
    bias = -10.0
    
    #training 
    batch_size = 64
    learning_rate = 3e-4
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    head_lr = 1e-3
    weight_decay = 0.01
    grad_clip = 1.0
    epochs = 20
    warmup_steps = 1000
    
    #dataset parameters
    image_size = (224, 224)
    max_seq_length = 64
    dataset_name = "conceptual_captions"
    dataset_subset = 5  
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = True
    num_workers = 2
    seed = 42
    
    #eval
    eval_batch_size = 128
    k_vals = [1, 5, 10]