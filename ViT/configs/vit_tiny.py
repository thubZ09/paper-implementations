class ViT_Tiny:
    image_size = 224
    patch_size = 16
    num_classes = 10  
    dim = 192
    depth = 12
    heads = 3
    mlp_dim = 768
    dropout = 0.1
    emb_dropout = 0.1
    pool = 'cls'
    channels = 3
    
    #training
    batch_size = 64
    epochs = 100
    lr = 3e-4
    weight_decay = 0.03
    warmup_steps = 1000
    grad_clip = 1.0
    
    #system
    device = 'cuda'
    mixed_precision = True
    num_workers = 2