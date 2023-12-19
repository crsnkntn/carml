from dataclasses import dataclass

'''
Configuration for a Vanilla Transformer

Defaulted to the parameters of "Attention is All You Need"
'''
@dataclass
class VanillaTransformerConfig:
    debug: bool = False         # Flag for debug output
    init_range: float = 2e-3    # The weight init range [-init_range, init_range]
    d_model: int = 512          # The model's embedding dimension
    n_decoder_layers: int = 4   # Number of decoder layers
    n_encoder_layers: int = 4   # Number of encoder layers
    n_decoder_heads: int = 8    # Number of attention heads
    n_encoder_heads: int = 8    # Number of attention heads
    d_head: int = 64            # Hidden dimension of Q, K, V matrices    
    d_mlp: int = 2048           # MLP hidden dimension
    ln_eps: float = 1e-5        # Base value to prevent division by zero
    dropout_rate: float = 1e-1  # Dropout rate for all model components


'''
Configuration for a Vision Transformer.

Defaulted to the parameters from "An Image is Worth 16x16 Words"
'''
@dataclass
class VisionTransformerConfig:
    debug: bool = False         # Flag for debug output

    Height: int = 512           # Input image height    
    Width: int = 512            # Input image width
    Patch_dim: int = 16         # Height and width of a patch
    n_classes: int = 27         # Number of classes

    init_range: float = 2e-3    # The weight init range [-init_range, init_range]
    d_model: int = Patch_dim**2 # The model's embedding dimension
    n_decoder_layers: int = 12  # Number of decoder layers
    n_encoder_layers: int = 12  # Number of encoder layers
    n_decoder_heads: int = 12   # Number of attention heads
    n_encoder_heads: int = 12   # Number of attention heads
    d_head: int = 768           # Hidden dimension of Q, K, V matrices    
    d_mlp: int = 3072           # MLP hidden dimension
    ln_eps: float = 1e-5        # Base value to prevent division by zero
    dropout_rate: float = 1e-1  # Dropout rate for all model components
