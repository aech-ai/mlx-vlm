# Multi-Image Support for Mistral Small 3.1

This document summarizes the changes made to add multi-image support for Mistral Small 3.1 and other models.

## Overview

Mistral Small 3.1 supports multiple images in a single conversation. We made several updates to ensure proper handling of multiple image inputs.

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

1. Enhanced the `_prepare_image_for_generation` method to resize images to a consistent size (224x224):
   ```python
   # Resize all images to the same dimensions to avoid reshape errors
   pil_image = pil_image.resize(target_size, Image.LANCZOS)
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

3. Updated the generation functions to convert image paths to PIL Image objects and pass image sizes.

### 3. Created Test Script

Created a test script (`test_multi_image.py`) to verify multi-image support:
- Takes model path and image paths as parameters
- Resizes images to a consistent size to avoid reshape errors
- Uses the correct model_type ("mistral3") for message formatting
- Handles multiple images in a single prompt

## How It Works

1. When a request with multiple images comes in, each image is:
   - Loaded from the URL
   - Resized to a consistent size (224x224 pixels)
   - Saved to a temporary file

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

4. The generation function is called with the list of image paths and their dimensions

## Testing

The implementation was tested with the Mistral Small 3.1 (24B, 8-bit) model and successfully processed two different images, comparing a mountain landscape and a beach scene.

## Next Steps

- Add additional model testing
- Consider further optimizations for image processing
- Update documentation to illustrate multi-image capabilities 