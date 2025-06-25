import torch
import torch.nn as nn

class PatchEmbeddings(nn.Module):
    def __init__(self, in_channels: int=3, embeddings_dimensions: int=768, patch_size: int=16):
        super().__init__()
        self.patch_size = patch_size
        self.patched_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embeddings_dimensions,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        self.flatten_embeddings = nn.Flatten(start_dim=2, end_dim=3)
        
    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, \
            f"Image size {image_resolution} divisible by patch size {self.patch_size}"
        
        x_patched = self.patched_embeddings(x)
        x_flatten = self.flatten_embeddings(x_patched)
        return x_flatten.permute(0, 2, 1)
    
class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, num_heads: int=12, embeddings_dimension: int=768, attn_dropout: float=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embeddings_dimension)
        self.multihead_attn_layer = nn.MultiheadAttention(
            embed_dim=embeddings_dimension,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn_layer(
            query=x, key=x, value=x, need_weights=False
        )
        return attn_output
        
class MLPBlock(nn.Module):
    def __init__(self, embeddings_dimension: int=768, mlp_size: int=3072, dropout: float=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embeddings_dimension)
        self.mlp = nn.Sequential(
            nn.Linear(embeddings_dimension, mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_size, embeddings_dimension),
            nn.Dropout(p=dropout)
        )
  
    def forward(self, x):
        return self.mlp(self.layer_norm(x))
        
class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_heads: int=12, embeddings_dimension: int=768, 
                 dropout: float=0.1, mlp_size: int=3072, attn_dropout: float=0.1):
        super().__init__()
        self.msa_layer = MultiHeadSelfAttentionBlock(
            num_heads=num_heads,
            embeddings_dimension=embeddings_dimension,
            attn_dropout=attn_dropout
        )
        self.mlp_block = MLPBlock(
            embeddings_dimension=embeddings_dimension,
            mlp_size=mlp_size,
            dropout=dropout
        )
        
    def forward(self, x):
        x = self.msa_layer(x) + x 
        x = self.mlp_block(x) + x 
        return x

class ViT(nn.Module):
    def __init__(self, num_heads: int=12, embeddings_dimension: int=768, dropout: float=0.1,
                 mlp_size: int=3072, attn_dropout: float=0.1, num_of_encoder_layers: int=12,
                 patch_size: int=16, image_width: int=224, img_height: int=224, 
                 no_channels: int=3, positional_embedding_dropout: float=0.1, 
                 projection_dims: int=256):
        super().__init__()
        #number of patches
        self.number_of_patches = (image_width * img_height) // (patch_size ** 2)
        
        #their embeddings
        self.patch_embeddings = PatchEmbeddings(
            in_channels=no_channels,
            embeddings_dimensions=embeddings_dimension,
            patch_size=patch_size
        )
        
        #ositional embeddings and tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, embeddings_dimension))
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, self.number_of_patches + 1, embeddings_dimension)
        )
        
        #transformer encoder
        self.encoder_block = nn.Sequential(*[
            TransformerEncoderBlock(
                num_heads=num_heads,
                embeddings_dimension=embeddings_dimension,
                dropout=dropout,
                mlp_size=mlp_size,
                attn_dropout=attn_dropout
            ) for _ in range(num_of_encoder_layers)
        ])
        
        #layer norms | dropout
        self.layer_norm = nn.LayerNorm(embeddings_dimension)
        self.dropout_after_positional_embeddings = nn.Dropout(p=positional_embedding_dropout)
        
        self.projection = nn.Sequential(
            nn.LayerNorm(embeddings_dimension),
            nn.Linear(embeddings_dimension, projection_dims)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.patch_embeddings(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.positional_embeddings
        x = self.dropout_after_positional_embeddings(x)
        x = self.layer_norm(x)

        x = self.encoder_block(x)
        cls_token = x[:, 0]
        return self.projection(cls_token)