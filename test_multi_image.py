import argparse
import os
import sys
from typing import List, Optional
import json
import logging
import tempfile

import mlx.core as mx
from PIL import Image

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template, get_message_json
from mlx_vlm.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_multi_image_generation(
    model_path: str,
    image_paths: List[str],
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    adapter_path: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """
    Test generating text from multiple images using Mistral Small 3.1
    
    Args:
        model_path: Path to the model
        image_paths: List of paths to image files
        prompt: Text prompt to use
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation
        adapter_path: Optional path to adapter weights
        trust_remote_code: Whether to trust remote code
    """
    logging.info(f"Loading model from {model_path}")
    model, processor = load(
        model_path,
        adapter_path=adapter_path,
        trust_remote_code=trust_remote_code,
    )
    config = load_config(model_path)
    
    # Load the images
    images = []
    processed_image_paths = []
    
    # First, load all images and find the max dimensions
    loaded_images = []
    max_width = 0
    max_height = 0
    
    for path in image_paths:
        logging.info(f"Loading image from {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
            
        # Load the image without resizing
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        # Keep track of maximum dimensions
        width, height = img.size
        max_width = max(max_width, width)
        max_height = max(max_height, height)
        
        loaded_images.append(img)
    
    logging.info(f"All images loaded. Maximum dimensions: {max_width}x{max_height}")
    
    # Process each image - padding to consistent dimensions if needed
    for i, img in enumerate(loaded_images):
        width, height = img.size
        needs_padding = width < max_width or height < max_height
        
        if needs_padding:
            logging.info(f"Padding image {i+1} from {width}x{height} to {max_width}x{max_height}")
            # Create a new image with the max dimensions and paste the original
            padded_image = Image.new("RGB", (max_width, max_height), color=(0, 0, 0))
            padded_image.paste(img, (0, 0))  # Paste at top-left corner
            process_image = padded_image
        else:
            process_image = img
        
        # Determine appropriate file extension based on image format
        if hasattr(img, 'format') and img.format:
            ext = f".{img.format.lower()}"
        else:
            ext = ".jpg"  # Default to jpg if format is unknown
            
        # Save the image to a temporary file with consistent dimensions
        temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        temp_path = temp_file.name
        temp_file.close()
        process_image.save(temp_path, format=img.format or "JPEG")
        logging.info(f"Saved image to {temp_path} with dimensions {process_image.size}")
        
        images.append(process_image)
        processed_image_paths.append(temp_path)
    
    # Get image sizes for the model if needed
    image_sizes = []
    for img in images:
        image_sizes.append([img.height, img.width])
    
    logging.info(f"Processing {len(images)} images with prompt: {prompt}")
    
    # Apply chat template with multi-image support
    # For Mistral models, we need to properly format the message to include image tokens
    model_type = getattr(model.config, "model_type", "")
    logging.info(f"Model type: {model_type}")
    
    # For Mistral models, use proper message formatting with image tokens
    messages = []
    formatted_content = get_message_json(
        model_type="mistral3",
        model_name="mistral3",
        prompt=prompt,
        role="user",
        skip_image_token=False,
        num_images=len(images)
    )
    
    if isinstance(formatted_content, dict):
        messages.append(formatted_content)
    else:
        messages.append({"role": "user", "content": formatted_content})
    
    logging.info(f"Formatted messages: {json.dumps(messages, indent=2)}")
    
    # Apply chat template
    formatted_prompt = apply_chat_template(
        processor, 
        config, 
        messages, 
        add_generation_prompt=True, 
        num_images=len(images),
        model_type="mistral3"  # Pass model_type explicitly
    )
    
    logging.info(f"Final formatted prompt: {formatted_prompt}")
    
    # Set model to eval mode
    model.eval()
    
    # Generate
    logging.info("Generating response...")
    output = generate(
        model=model,
        processor=processor,
        prompt=formatted_prompt,
        image=processed_image_paths,  # Pass the list of processed image paths
        max_tokens=max_tokens,
        temperature=temperature,
        image_sizes=image_sizes,  # Pass image sizes if needed
        verbose=True,
    )
    
    logging.info(f"Generated output: {output}")
    return output

def main():
    parser = argparse.ArgumentParser(description="Test multi-image generation with Mistral Small 3.1")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--images", type=str, nargs="+", required=True, help="Paths to image files")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--adapter-path", type=str, help="Path to adapter weights")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    
    args = parser.parse_args()
    
    test_multi_image_generation(
        model_path=args.model,
        image_paths=args.images,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        adapter_path=args.adapter_path,
        trust_remote_code=args.trust_remote_code,
    )

if __name__ == "__main__":
    main() 