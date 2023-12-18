from dataclasses import dataclass


@dataclass
class VanillaTransformerConfig:
    debug: bool = False

    # The Height and Width of a patch
    P: int = 4

    # The Dimension of the Model (size of a flattened patch)
    d_model: int = 784

    # The number of transformer layers
    n_layers: int = 4

    # Base value to prevent division by zero
    ln_eps: float = 1e-5

    # The weight initialization range [-init_range, init_range]
    init_range: float = 0.02

    # The number of attention heads in each layer
    n_heads: int = 12

    # The hidden dimension of Q, K, V matrices
    d_head: int = 64

    # MLP hidden dimension
    d_mlp: int = 248

    # The number of classes, or the output size of the unembedder
    n_classes: int = 27



'''
Configuration for a Vision Transformer.

Defaulted to the parameters from "An Image is Worth 16x16 Words."

The value of n_classes is the unembedder's output size
'''
@dataclass
class VisionTransformerConfig:
    debug: bool = False

    # Images are standardized to Height x Width
    Height: int = 512
    Width: int = 512

    # The Height and Width of a patch
    Patch_dim: int = 16

    # The Dimension of the Model (size of a flattened patch)
    d_model: int = Patch_dim * Patch_dim

    # The weight initialization range [-init_range, init_range]
    init_range: float = 0.02

    # The number of transformer layers
    n_layers: int = 4

    # Base value to prevent division by zero
    ln_eps: float = 1e-5

    # The number of attention heads in each layer
    n_heads: int = 12

    # The hidden dimension of Q, K, V matrices
    d_head: int = 64

    # MLP hidden dimension
    d_mlp: int = 248

    # The number of classes, or the output size of the unembedder
    n_classes: int = 27