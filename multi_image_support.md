# Multi-Image Support for Mistral Small 3.1

This document summarizes the changes made to add multi-image support for Mistral Small 3.1 and other models.

## Overview

Mistral Small 3.1 supports multiple images in a single conversation. We made several updates to ensure proper handling of multiple image inputs while preserving high resolution for document analysis.

## Changes Made

### 1. Updated `prompt_utils.py`

1. Added model name variants to the `model_to_format` mapping in the `get_message_json` function:
   ```python
   "mistral-small": "message_list_with_image_first",
   "mistral-small-3.1": "message_list_with_image_first",
   "small_3.1": "message_list_with_image_first",
   "mistral_small": "message_list_with_image_first",
   ```

2. Enhanced the `get_chat_template` function to better handle Mistral models:
   ```python
   # For mistral models, directly use the processed messages since the tokenizer handles image tokens correctly
   if model_type == "mistral3" or model_type.startswith("mistral"):
       try:
           if hasattr(processor, "apply_chat_template"):
               return processor.apply_chat_template(
                   messages,
                   tokenize=False,
                   add_generation_prompt=add_generation_prompt,
                   **kwargs,
               )
       except Exception as e:
           logging.warning(f"Direct application of chat template failed: {e}. Falling back to manual processing.")
   ```

3. Fixed string handling to properly process nested content structures.

### 2. Updated `server.py`

1. Enhanced the `_prepare_image_for_generation` method to maintain original resolution while ensuring consistent dimensions across images:
   ```python
   # Find maximum dimensions across all images
   max_width = max(width for width, _ in image_sizes)
   max_height = max(height for _, height in image_sizes)
   
   # Pad smaller images to match the largest dimensions
   if needs_padding:
       padded_image = Image.new("RGB", (max_width, max_height), color=(0, 0, 0))
       padded_image.paste(pil_image, (0, 0))  # Paste at top-left corner
   ```

2. Modified the chat completions handler to pass model_type and num_images to get_chat_template:
   ```python
   # Determine if this is a mistral model from config or model name
   model_type = ""
   if hasattr(model, "config") and hasattr(model.config, "model_type"):
       model_type = model.config.model_type
   elif "mistral" in self.requested_model.lower():
       model_type = "mistral3"
       
   # Include model_type for special handling
   prompt_text = get_chat_template(
       processor, 
       processed_messages_for_template, 
       add_generation_prompt=True,
       model_type=model_type,
       num_images=len(image_urls) if image_urls else 0
   )
   ```

3. Updated the generation functions to pass image file paths directly to the model.

### 3. Created Test Script

Created a test script (`test_multi_image.py`) to verify multi-image support:
- Takes model path and image paths as parameters
- Maintains full resolution of all images
- Pads smaller images to match largest dimensions for consistent tensor shapes
- Uses the correct model_type ("mistral3") for message formatting
- Handles multiple images in a single prompt

## How It Works

1. When a request with multiple images comes in:
   - All images are loaded first to determine maximum dimensions
   - Each image is padded (if necessary) to match the largest dimensions
   - Original resolution is preserved (no downscaling)
   - Images are saved to temporary files with consistent dimensions

2. The message JSON is formatted with image tokens for each image:
   ```json
   {
     "role": "user",
     "content": [
       {"type": "image"},
       {"type": "image"},
       {"type": "text", "text": "Compare these two images..."}
     ]
   }
   ```

3. The chat template is applied with model-specific handling:
   - For Mistral models, we pass the messages directly to apply_chat_template
   - For other models, we format the messages according to their conventions

4. The generation function is called with the list of image paths, all having consistent dimensions while preserving original resolution

## Detail Preservation

A key feature of this implementation is that it **preserves the original image resolution** while ensuring consistent dimensions through padding. This is crucial for:
- Document analysis (invoices, receipts, forms)
- Medical imaging
- High-resolution photos with important details
- Text-heavy images where downscaling would make text illegible

### Technical Details on Padding

Vision transformers typically divide images into patches (e.g., 16x16 pixels) and process each patch. For multiple images to be processed together, they must have consistent dimensions to create properly-shaped tensors. Our solution:

1. Finds the maximum width and height across all images
2. Pads smaller images with black pixels to match these dimensions
3. This ensures all images have the same number of patches without losing resolution
4. The padding is neutral (black) and placed at the bottom/right to minimize impact on content

## Testing

The implementation was tested with the Mistral Small 3.1 (24B, 8-bit) model and successfully processed multiple images with varying dimensions. The padding approach allows the model to handle the images as a consistent batch while preserving all detail in the original images.

## Next Steps

- Add additional model testing
- Consider optimizations for very large images (optional tiling approaches)
- Update documentation to illustrate multi-image capabilities 