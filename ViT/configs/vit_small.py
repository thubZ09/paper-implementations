class ViT_Small:
    image_size = 224
    patch_size = 16
    num_classes = 10  
    dim = 384
    depth = 12
    heads = 6
    mlp_dim = 1536
    dropout = 0.1
    emb_dropout = 0.1
    pool = 'cls'
    channels = 3
    
    #traing
    batch_size = 32
    epochs = 100
    lr = 3e-4
    weight_decay = 0.03
    warmup_steps = 1000
    grad_clip = 1.0
    
    #sys
    device = 'cuda'
    mixed_precision = True
    num_workers = 2