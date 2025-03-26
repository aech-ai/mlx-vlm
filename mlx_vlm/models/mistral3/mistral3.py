import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..pixtral import LanguageModel as PixtralLanguageModel
from ..pixtral import Model as PixtralModel
from ..pixtral import TextConfig, VisionConfig, VisionModel as PixtralVisionModel


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = 10
    vision_feature_select_strategy: str = "full"
    vision_feature_layer: int = -1
    vocab_size: int = 32000
    spatial_merge_size: int = 2
    multimodal_projector_bias: bool = False

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def _pair(x) -> Tuple[int, int]:
    """Convert input to a pair of values."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def unfold(
    input: mx.array,
    kernel_size: Union[int, Tuple[int, int], List[int]],
    dilation: Union[int, Tuple[int, int], List[int]] = 1,
    padding: Union[int, Tuple[int, int], List[int]] = 0,
    stride: Union[int, Tuple[int, int], List[int]] = 1,
) -> mx.array:
    """
    Extract sliding local blocks from a batched input tensor (MLX implementation).

    This is equivalent to PyTorch's nn.functional.unfold or im2col operation.

    Args:
        input: Input tensor of shape (B, C, H, W)
        kernel_size: Size of the sliding blocks
        dilation: Controls the spacing between kernel elements
        padding: Controls the amount of implicit padding
        stride: Controls the stride between blocks

    Returns:
        Unfolded tensor of shape (B, C*kernel_height*kernel_width, L)
        where L is the number of blocks
    """
    # Convert to pairs
    kernel_size = _pair(kernel_size)
    dilation = _pair(dilation)
    padding = _pair(padding)
    stride = _pair(stride)

    # Input shape
    batch_size, channels, height, width = input.shape

    # Add padding if needed
    if padding[0] > 0 or padding[1] > 0:
        padding_shape = (
            (0, 0),
            (0, 0),
            (padding[0], padding[0]),
            (padding[1], padding[1]),
        )
        input = mx.pad(input, padding_shape)

    # Calculate output dimensions
    height_out = (
        height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1
    width_out = (
        width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1

    # Initialize output arrays
    blocks = []

    # Extract blocks
    for i in range(
        0, height + 2 * padding[0] - kernel_size[0] * dilation[0] + 1, stride[0]
    ):
        for j in range(
            0, width + 2 * padding[1] - kernel_size[1] * dilation[1] + 1, stride[1]
        ):
            # Extract the block for all channels
            block = []
            for di in range(kernel_size[0]):
                for dj in range(kernel_size[1]):
                    h_idx = i + di * dilation[0]
                    w_idx = j + dj * dilation[1]
                    # Get the block for all channels and add to our list
                    block.append(input[:, :, h_idx, w_idx])

            # Stack the channel-blocks
            block = mx.stack(block, axis=1)  # Shape: (B, k*k, C)
            block = mx.transpose(block, [0, 2, 1])  # Shape: (B, C, k*k)
            blocks.append(block)

    # Stack all blocks together
    result = mx.stack(blocks, axis=-1)  # Shape: (B, C, k*k, L)

    # Reshape to match PyTorch's unfold output format: (B, C*k*k, L)
    result = mx.reshape(
        result,
        (
            batch_size,
            channels * kernel_size[0] * kernel_size[1],
            height_out * width_out,
        ),
    )

    return result


class Mistral3PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = self.config.vision_config.patch_size
        self.merging_layer = nn.Linear(
            hidden_size * self.spatial_merge_size**2, hidden_size, bias=False
        )

    def __call__(self, image_features: mx.array, image_sizes: mx.array) -> mx.array:

        image_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size)
            for image_size in image_sizes
        ]

        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]
        image_features = image_features.astype(mx.bfloat16)
        image_sizes = mx.array(image_sizes)

        # Split the image features into chunks based on tokens_per_image
        split_indices = []
        current_index = 0
        for tokens in tokens_per_image:
            split_indices.append(current_index + tokens)
            current_index += tokens

        # Perform the split
        chunks = mx.split(image_features, split_indices[:-1], axis=1)

        permuted_tensor = []
        for image_index, image_tokens in enumerate(chunks):

            # Reshape image_tokens into a 2D grid
            if image_tokens.shape[1] > 0:
                h, w = image_sizes[image_index].tolist()

                image_grid = image_tokens.reshape(h, w, d).transpose(2, 0, 1)[None, ...]

                grid = unfold(
                    image_grid,
                    kernel_size=self.spatial_merge_size,
                    stride=self.spatial_merge_size,
                )
                grid = grid.reshape(d * self.spatial_merge_size**2, -1).T
                permuted_tensor.append(grid)

        image_features = mx.concatenate(permuted_tensor, axis=0)
        image_features = self.merging_layer(image_features)
        return image_features[None, ...]


class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.norm = nn.RMSNorm(config.vision_config.hidden_size)
        self.patch_merger = Mistral3PatchMerger(config)

        num_feature_layers = (
            1
            if isinstance(config.vision_feature_layer, int)
            else len(config.vision_feature_layer)
        )
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )

    def __call__(self, x: mx.array, image_sizes: mx.array) -> mx.array:
        x = self.norm(x)

        x = self.patch_merger(x, image_sizes)
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


class LanguageModel(PixtralLanguageModel):
    @staticmethod
    def sanitize(weights):
        # First do the standard sanitization from parent class
        sanitized_weights = {}
        
        for k, v in weights.items():
            # Skip removing freqs as in the original sanitize method
            if "self_attn.rotary_emb.inv_freq" in k:
                continue
                
            # Handle language model weights without 'language_model.' prefix
            if k.startswith("layers."):
                # Convert the layer weight format if needed
                if ".feed_forward.w1.weight" in k:
                    new_key = k.replace(".feed_forward.w1.weight", ".mlp.gate_proj.weight")
                    sanitized_weights[new_key] = v
                elif ".feed_forward.w2.weight" in k:
                    new_key = k.replace(".feed_forward.w2.weight", ".mlp.down_proj.weight")
                    sanitized_weights[new_key] = v
                elif ".feed_forward.w3.weight" in k:
                    new_key = k.replace(".feed_forward.w3.weight", ".mlp.up_proj.weight")
                    sanitized_weights[new_key] = v
                # Convert attention weights if needed
                elif ".attention.wq.weight" in k:
                    new_key = k.replace(".attention.wq.weight", ".self_attn.q_proj.weight")
                    sanitized_weights[new_key] = v
                elif ".attention.wk.weight" in k:
                    new_key = k.replace(".attention.wk.weight", ".self_attn.k_proj.weight")
                    sanitized_weights[new_key] = v
                elif ".attention.wv.weight" in k:
                    new_key = k.replace(".attention.wv.weight", ".self_attn.v_proj.weight")
                    sanitized_weights[new_key] = v
                elif ".attention.wo.weight" in k:
                    new_key = k.replace(".attention.wo.weight", ".self_attn.o_proj.weight")
                    sanitized_weights[new_key] = v
                # Handle norm weights
                elif ".attention_norm.weight" in k:
                    new_key = k.replace(".attention_norm.weight", ".input_layernorm.weight")
                    sanitized_weights[new_key] = v
                elif ".ffn_norm.weight" in k:
                    new_key = k.replace(".ffn_norm.weight", ".post_attention_layernorm.weight")
                    sanitized_weights[new_key] = v
                else:
                    # For other layer weights, prepend 'model.' to match Pixtral's structure
                    sanitized_weights[f"model.{k}"] = v
            elif k == "norm.weight":
                # Handle the final norm layer
                sanitized_weights["model.norm.weight"] = v
            elif k == "tok_embeddings.weight":
                # Handle token embeddings
                sanitized_weights["model.embed_tokens.weight"] = v
            elif k == "output.weight":
                # Handle output projection
                sanitized_weights["lm_head.weight"] = v
            else:
                # Keep other weights as they are
                sanitized_weights[k] = v
                
        return sanitized_weights


class VisionModel(PixtralVisionModel):
    @staticmethod
    def sanitize(weights):
        # First do the standard sanitization
        sanitized_weights = {}
        mistral_vision_weights = {}
        
        # Prefixes that might appear in the weights
        vision_prefixes = ["vision_encoder.", "vision_model."]
        
        # Process weight mappings for Mistral3 vision encoder
        for k, v in weights.items():
            new_key = k
            
            # Handle various vision encoder prefixes
            for prefix in vision_prefixes:
                if k.startswith(prefix):
                    # Standardize on vision_model prefix
                    new_key = k.replace(prefix, "vision_model.")
                    
                    # Handle transformer layers
                    if "transformer.layers" in new_key:
                        # Convert feed_forward weights to MLX format
                        if ".feed_forward.w1.weight" in new_key:
                            new_key = new_key.replace(".feed_forward.w1.weight", ".feed_forward.gate_proj.weight")
                        elif ".feed_forward.w2.weight" in new_key:
                            new_key = new_key.replace(".feed_forward.w2.weight", ".feed_forward.down_proj.weight")
                        elif ".feed_forward.w3.weight" in new_key:
                            new_key = new_key.replace(".feed_forward.w3.weight", ".feed_forward.up_proj.weight")
                        
                        # Convert attention weights
                        elif ".attention.wq.weight" in new_key:
                            new_key = new_key.replace(".attention.wq.weight", ".attention.q_proj.weight")
                        elif ".attention.wk.weight" in new_key:
                            new_key = new_key.replace(".attention.wk.weight", ".attention.k_proj.weight")
                        elif ".attention.wv.weight" in new_key:
                            new_key = new_key.replace(".attention.wv.weight", ".attention.v_proj.weight")
                        elif ".attention.wo.weight" in new_key:
                            new_key = new_key.replace(".attention.wo.weight", ".attention.o_proj.weight")
                            
                        # Handle norm weights
                        elif ".attention_norm.weight" in new_key:
                            new_key = new_key.replace(".attention_norm.weight", ".attention_norm.weight")
                        elif ".ffn_norm.weight" in new_key:
                            new_key = new_key.replace(".ffn_norm.weight", ".ffn_norm.weight")
                    
                    # Handle other vision encoder weights
                    if "patch_merger.merging_layer.weight" in new_key:
                        # Keep as is, this is specific to Mistral3
                        pass
                    elif "pre_mm_projector_norm.weight" in new_key:
                        # Handle special norm layers
                        pass
                    
                    mistral_vision_weights[new_key] = v
                    break
            else:
                # If no prefix matched, keep the original key
                sanitized_weights[k] = v
        
        # Add sanitized Mistral vision weights to the result
        for k, v in mistral_vision_weights.items():
            # Handle the patch_conv weight which needs to be transposed
            if "patch_conv.weight" in k:
                if len(v.shape) == 4:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
                else:
                    sanitized_weights[k] = v
            else:
                sanitized_weights[k] = v
        
        # Handle non-vision weights that need special processing
        for k, v in weights.items():
            # Handle language model weights, multimodal weights, etc.
            if k.startswith("vision_language_adapter."):
                sanitized_weights[k] = v
            elif k == "pre_mm_projector_norm.weight":
                sanitized_weights["vision_model.ln_pre.weight"] = v
        
        return sanitized_weights


class Model(PixtralModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

        self.multi_modal_projector = Mistral3MultiModalProjector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        image_sizes = kwargs.get("image_sizes", None)

        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the output hidden states from the vision model
        if isinstance(pixel_values, list):
            pixel_values = mx.concatenate(
                [mx.array(pv)[None, ...] for pv in pixel_values], axis=0
            )
        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None, ...]

        # Pass pixel_values as list of images, as each image is individually run through conv2d and position encoding
        # Reference code from transformers: https://github.com/huggingface/transformers/blob/main/src/transformers/models/pixtral/modeling_pixtral.py#L479C9-L479C21
        # and mistral_inference: https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/vision_encoder.py#L85
        *_, hidden_states = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1),
            output_hidden_states=True,
        )
        # Select the hidden states from the desired layer
        selected_image_feature = hidden_states[self.vision_feature_layer]

        # Pass image features through the multi-modal projector
        if image_sizes is None:
            # Default to a square image size based on the number of patches
            h, w = pixel_values.shape[2] // self.vision_tower.vision_model.config.patch_size, pixel_values.shape[3] // self.vision_tower.vision_model.config.patch_size
            image_sizes = [(h, w)] * pixel_values.shape[0]
            
        image_features = self.multi_modal_projector(selected_image_feature, image_sizes)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        image_token_index = self.config.image_token_index
        num_images, num_image_patches, embed_dim = image_features.shape

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = np.where(input_ids == image_token_index)[1].tolist()

        text_segments = []
        start_idx = 0

        for position in image_positions:
            text_segments.append(inputs_embeds[:, start_idx:position])
            start_idx = position + 1

        # Split image features into separate embeddings for each image
        image_embeddings = mx.split(image_features, num_image_patches, axis=1)
        final_embeddings = [v for p in zip(text_segments, image_embeddings) for v in p]
        final_embeddings += [inputs_embeds[:, start_idx:]]

        # Create a final embedding of shape
        # (1, num_image_patches*num_images + sequence_len, embed_dim)
        return mx.concatenate(final_embeddings, axis=1)

    @staticmethod
    def sanitize(weights):
        # Handle model-specific weights 
        sanitized_weights = {}
        
        for k, v in weights.items():
            if k.startswith("vision_language_adapter."):
                # Handle vision-language adapter weights
                sanitized_weights[k] = v
            elif k == "pre_mm_projector_norm.weight":
                # This is a special norm layer in Mistral-Small-3.1
                sanitized_weights[k] = v
            else:
                sanitized_weights[k] = v
        
        return sanitized_weights
