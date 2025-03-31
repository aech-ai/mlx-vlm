"""
Server implementation for mlx-vlm that provides OpenAI API compatibility.
Adapted from mlx-lm server.py (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py)
"""

import argparse
import base64
import json
import logging
import os
import socket
import tempfile
import time
import uuid
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import urllib.request
import copy
import re

import mlx.core as mx
import numpy as np
from PIL import Image

from . import load, apply_chat_template
from .prompt_utils import get_chat_template, get_message_json
from .utils import GenerationResult, stream_generate
from .version import __version__


def get_system_fingerprint():
    return f"mlx-vlm-{__version__}"


class PromptCache:
    """
    A simple cache to avoid re-tokenizing prompts.
    """

    def __init__(self, max_entries=100):
        self.cache = {}
        self.max_entries = max_entries

    def get(self, key):
        return self.cache.get(key, None)

    def put(self, key, value):
        if len(self.cache) >= self.max_entries:
            # Simple LRU: just clear half the cache when it gets full
            for k in list(self.cache.keys())[: self.max_entries // 2]:
                del self.cache[k]
        self.cache[key] = value


class ModelProvider:
    """
    A class that provides vision-language models for the server.
    """

    def __init__(self, args):
        self.args = args
        self.model = None
        self.processor = None
        self.config = None
        self.lock = None
        self.default_chat_template = args.default_chat_template if hasattr(args, "default_chat_template") else ""

    def load_model(self):
        """
        Load the model if it hasn't been loaded already.
        """
        if self.model is None or self.processor is None:
            logging.info(
                f"Loading model from {self.args.model}"
                + (f" with adapter {self.args.adapter_path}" if self.args.adapter_path else "")
            )
            self.model, self.processor = load(
                self.args.model,
                adapter_path=self.args.adapter_path,
                trust_remote_code=self.args.trust_remote_code,
            )
            # Set custom chat template if provided
            if hasattr(self.processor, "tokenizer") and self.args.chat_template:
                self.processor.tokenizer.chat_template = self.args.chat_template
                
            # Set model to eval mode
            self.model.eval()
            
            logging.info("Model loaded successfully")

    def get_model(self):
        """
        Get the model, loading it if necessary.
        """
        self.load_model()
        return self.model, self.processor


def process_message_content(messages) -> List[str]:
    """Process message content to handle multimodal inputs (images).
    Extracts all image URLs found but leaves the original content structure intact.
    Returns a list of image URLs.
    """
    image_urls_found = [] # Store all image URLs found
    
    for message_index, message in enumerate(messages):
        if isinstance(message.get("content"), list):
            has_image = False
            text_parts = [] # Collect text parts for logging/debugging if needed
            
            for content_part in message["content"]:
                part_type = content_part.get("type")
                if part_type == "text":
                    text_parts.append(content_part.get("text", ""))
                elif part_type == "image_url":
                    has_image = True
                    # Extract the URL (either base64 or regular)
                    image_data = content_part.get("image_url", {})
                    url = image_data.get("url")
                    if url: # Store every valid image URL we find
                        image_urls_found.append(url)
                        logging.debug(f"Found image URL in message {message_index}: {url[:100] if url else 'None'}...")
            
            # Log what we found, but DON'T modify message["content"]
            text_content_for_log = " ".join(text_parts)
            logging.debug(f"Processed message {message_index}: content remains list, found image: {has_image}, text parts: '{text_content_for_log}', total images found so far: {len(image_urls_found)}")
            
        elif isinstance(message.get("content"), str):
             # If content is already a string, just log it
             logging.debug(f"Processed message: content is already a string: '{message['content'][:100]}...'")

    # Return the list of image URLs found (can be empty)
    # Limit to 10 images as requested
    if len(image_urls_found) > 10:
        logging.warning(f"Found {len(image_urls_found)} images, but limiting to the first 10.")
        image_urls_found = image_urls_found[:10]
        
    return image_urls_found


def default_chat_template(messages, template=""):
    """
    A simple default chat template that works when the tokenizer doesn't have one.
    
    Args:
        messages: List of message dictionaries
        template: Optional custom template format string with {role} and {content} placeholders
    """
    if template:
        # Use the custom template for each message
        result = []
        for message in messages:
            role = message["role"].upper()
            content = message["content"]
            result.append(template.format(role=role, content=content))
        return "\n".join(result) + "\nASSISTANT:"
    else:
        # Use the default format
        result = []
        for message in messages:
            role = message["role"].upper()
            content = message["content"]
            result.append(f"{role}: {content}")
        return "\n".join(result) + "\nASSISTANT:"


def load_image_from_url(url):
    """Load an image from a URL or base64 data URL."""
    try:
        if url.startswith("data:image/"):
            # Handle base64 encoded images
            logging.debug(f"Loading base64 encoded image, length: {len(url)}")
            base64_data = url.split(",")[1]
            image_data = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_data))
            logging.debug(f"Successfully loaded base64 image: {image.size}, mode: {image.mode}")
        else:
            # Handle regular URLs
            logging.debug(f"Loading image from URL: {url}")
            with urllib.request.urlopen(url) as response:
                image_data = response.read()
                content_type = response.getheader('Content-Type', '')
                logging.debug(f"Image Content-Type from URL: {content_type}")
            image = Image.open(BytesIO(image_data))
            logging.debug(f"Successfully loaded URL image: {image.size}, mode: {image.mode}")
            
        # Ensure the image is in RGB mode for compatibility
        if image.mode != "RGB":
            image = image.convert("RGB")
            logging.debug(f"Converted image to RGB mode: {image.size}")
            
        return image
    except Exception as e:
        logging.error(f"Error loading image: {str(e)}")
        raise ValueError(f"Failed to load image: {str(e)}")


class APIHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for the API server.
    """

    def __init__(
        self,
        model_provider,
        prompt_cache=None,
        system_fingerprint="",
        *args,
        **kwargs,
    ):
        self.model_provider = model_provider
        self.prompt_cache = prompt_cache or PromptCache()
        self.system_fingerprint = system_fingerprint
        self.function_calling_system_prompt = (
            "You have access to the following functions:\n"
            "{function_descriptions}\n\n"
            "To call a function, respond with a JSON object with the following structure:\n"
            "{{\"function_name\": \"<function_name>\", \"arguments\": {{<arguments>}}}}\n\n"
            "If multiple functions are needed, respond with a list of JSON objects, each with a function_name and arguments."
        )
        BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

    def _set_completion_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, openai-ephemeral-user-id, openai-conversation-id",
        )

    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight."""
        self._set_completion_headers()
        self.end_headers()

    def do_POST(self):
        """
        Handle POST requests from clients.
        """
        content_length = int(self.headers["Content-Length"])
        body_bytes = self.rfile.read(content_length)
        body_str = body_bytes.decode("utf-8")
        try:
            self.body = json.loads(body_str)
        except json.JSONDecodeError:
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON body"}')
            return

        indent = "\\t"  # Backslashes can't be inside of f-strings
        logging.debug(f"Incoming Request: {json.dumps(self.body, indent=indent)}")

        # Extract common parameters
        self.stream = self.body.get("stream", False)
        self.created = int(time.time())
        self.requested_model = self.body.get("model", "mlx_vlm")

        # Configure streaming options
        self.stream_options = None
        if "stream_options" in self.body:
            self.stream_options = self.body["stream_options"]

        # Parse logprobs and top_logprobs parameters
        self.logprobs = self.body.get("logprobs", False)
        self.top_logprobs = None
        if self.logprobs:
            self.top_logprobs = self.body.get("top_logprobs", 0)

        # Extract function calling parameters
        self.functions = self.body.get("functions", [])
        self.tools = self.body.get("tools", [])
        self.function_call = self.body.get("function_call", None)
        self.tool_choice = self.body.get("tool_choice", None)
        
        # Normalize tools to functions format for internal processing
        # This handles both the older 'functions' and newer 'tools' formats
        self.normalized_functions = []
        
        # Convert tools to normalized function format
        if self.tools:
            for tool in self.tools:
                if tool.get("type") == "function":
                    self.normalized_functions.append(tool.get("function", {}))
        
        # Add regular functions
        if self.functions:
            self.normalized_functions.extend(self.functions)
            
        # Process tool_choice/function_call
        self.selected_function = None
        if self.tool_choice:
            if isinstance(self.tool_choice, dict) and self.tool_choice.get("type") == "function":
                self.selected_function = self.tool_choice.get("function", {}).get("name")
            elif self.tool_choice == "auto":
                self.selected_function = "auto"
            elif self.tool_choice == "none":
                self.selected_function = "none"
        elif self.function_call:
            if isinstance(self.function_call, dict):
                self.selected_function = self.function_call.get("name")
            elif self.function_call == "auto":
                self.selected_function = "auto"
            elif self.function_call == "none":
                self.selected_function = "none"

        # Determine endpoint and handle request
        if self.path == "/v1/chat/completions":
            self._set_completion_headers()
            if self.stream:
                self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            
            # Get the model and processor
            try:
                model, processor = self.model_provider.get_model()
                tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            except Exception as e:
                logging.error(f"Failed to load model/processor: {str(e)}")
                self._set_completion_headers(500) # Internal Server Error
                self.end_headers()
                error_response = {
                    "error": {
                        "message": f"Failed to load model: {str(e)}",
                        "type": "server_error",
                        "code": 500
                    }
                }
                self.wfile.write(json.dumps(error_response).encode())
                return # Stop processing the request
            
            # Proceed with handling the completion if model loaded successfully
            try:
                self.handle_chat_completions(model, processor, tokenizer)
            except Exception as e:
                logging.error(f"Error processing completion: {str(e)}")
                error_response = {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": 500
                    }
                }
                self.wfile.write(json.dumps(error_response).encode())
                
        elif self.path == "/v1/completions":
            self._set_completion_headers(501)
            self.end_headers()
            self.wfile.write(b'{"error": "Text completions not supported for VLM models"}')
        else:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not Found"}')

    def _prepare_image_for_generation(self, image_urls: List[str]) -> Tuple[List[str], List[tempfile._TemporaryFileWrapper]]:
        """Loads images from a list of URLs, pads to consistent dimensions, returns lists of paths and file objects."""
        temp_files = []
        temp_paths = []
        created_files = [] # Keep track of successfully created file objects for cleanup
        
        try:
            # First, load all images and determine the max dimensions
            loaded_images = []
            max_width = 0
            max_height = 0
            
            for i, url in enumerate(image_urls):
                try:
                    logging.debug(f"Loading image {i+1}/{len(image_urls)} from URL: {url[:100]}...")
                    pil_image = load_image_from_url(url)
                    logging.debug(f"Loaded image {i+1} successfully: {pil_image.size}, mode: {pil_image.mode}")
                    
                    # Convert to RGB for consistency
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                        
                    # Track maximum dimensions
                    width, height = pil_image.size
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)
                    
                    loaded_images.append(pil_image)
                    
                except Exception as e_inner:
                    logging.error(f"Error loading image {i+1} from URL {url[:100]}...: {str(e_inner)}")
                    # Continue to the next image URL
            
            if not loaded_images:
                raise ValueError("Failed to load any images from the provided URLs.")
                
            logging.debug(f"All images loaded. Maximum dimensions: {max_width}x{max_height}")
            
            # Now process each image - padding to the same dimensions if needed
            for i, pil_image in enumerate(loaded_images):
                try:
                    # Determine if we need to pad
                    width, height = pil_image.size
                    needs_padding = width < max_width or height < max_height
                    
                    if needs_padding:
                        logging.debug(f"Padding image {i+1} from {width}x{height} to {max_width}x{max_height}")
                        # Create a new image with the max dimensions and paste the original
                        padded_image = Image.new("RGB", (max_width, max_height), color=(0, 0, 0))
                        padded_image.paste(pil_image, (0, 0))  # Paste at top-left corner
                        process_image = padded_image
                    else:
                        process_image = pil_image
                    
                    # Determine the appropriate file extension based on image format
                    if hasattr(pil_image, 'format') and pil_image.format:
                        ext = f".{pil_image.format.lower()}"
                    else:
                        ext = ".jpg"  # Default to jpg if format is unknown
                    
                    # Create a temporary file with the appropriate extension
                    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                    temp_path = temp_file.name
                    temp_file.close() # Close for saving
                    logging.debug(f"Temporary file path created for image {i+1}: {temp_path}")

                    # Save with consistent dimensions through padding, not resizing
                    process_image.save(temp_path, format=pil_image.format or "JPEG")
                    logging.debug(f"Saved image {i+1} to temporary file: {temp_path} with dimensions {process_image.size}")

                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        logging.debug(f"Temporary file for image {i+1} verified: {temp_path}")
                        temp_paths.append(temp_path)
                        created_files.append(temp_file) # Add successfully created file obj for later cleanup
                    else:
                        logging.error(f"Failed to save or verify temporary file for image {i+1}: {temp_path}")
                        # Clean up this specific failed file if it exists
                        if temp_path and os.path.exists(temp_path):
                            try: os.unlink(temp_path)
                            except OSError as e: logging.warning(f"Error deleting failed temp file {temp_path}: {e}")
                        # Do not add to temp_paths or created_files, effectively skipping this image

                except Exception as e_inner:
                    logging.error(f"Error processing image {i+1}: {str(e_inner)}")
                    # Continue to the next image
            
            # Check if at least one image was processed successfully
            if not temp_paths:
                 raise ValueError("Failed to prepare any images from the provided URLs.")
                 
            return temp_paths, created_files

        except Exception as e_outer: # Catch broader errors, though inner loop handles most image specifics
            logging.error(f"Outer error during image preparation loop: {str(e_outer)}")
            # Attempt cleanup of any files created before the outer error
            for f in created_files:
                try:
                    if os.path.exists(f.name):
                         os.unlink(f.name)
                except Exception as e_cleanup:
                    logging.warning(f"Error during cleanup after outer exception for {f.name}: {e_cleanup}")
            raise # Re-raise the exception that caused the failure

    def handle_chat_completions(self, model, processor, tokenizer):
        """
        Handle a chat completion request.
        """
        if "messages" not in self.body:
            raise ValueError("Request did not contain messages")

        # Determine response type
        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = "chat.completion.chunk" if self.stream else "chat.completion"

        # Extract generation parameters
        temperature = float(self.body.get("temperature", 0.0))
        top_p = float(self.body.get("top_p", 1.0))
        max_tokens = int(self.body.get("max_tokens", 100))
        stop = self.body.get("stop", None)
        frequency_penalty = float(self.body.get("frequency_penalty", 0.0))
        presence_penalty = float(self.body.get("presence_penalty", 0.0))

        # Process messages for multimodal content - returns a LIST of image URLs
        messages = self.body["messages"]
        image_urls = process_message_content(messages) # Now returns List[str]
        logging.debug(f"Original messages structure: {json.dumps(messages)}")
        if image_urls:
             logging.debug(f"Extracted {len(image_urls)} image URLs for loading.")

        # --- ADDED: Prepare messages for template using get_message_json --- 
        processed_messages_for_template = []
        try:
            model_config = model.config # Assuming model has a config attribute
            model_type = model_config.model_type
            logging.debug(f"Detected model_type for get_message_json: {model_type}")
            
            # This logic might need adjustment for multiple images within a single message turn
            # Currently identifies the *first* message containing *any* of the extracted image URLs
            image_message_indices = set()
            if image_urls:
                for i, msg in enumerate(messages):
                    if isinstance(msg.get("content"), list):
                        for part in msg["content"]:
                            if part.get("type") == "image_url" and part.get("image_url", {}).get("url") in image_urls:
                                image_message_indices.add(i)
                                # Don't break, allow finding multiple images in the same message if needed later
            
            logging.debug(f"Message indices containing target image URLs: {sorted(list(image_message_indices))}")

            for i, msg in enumerate(messages):
                role = msg["role"]
                original_content = msg["content"]
                text_content_for_json = ""
                num_images_in_msg = 0
                
                # Extract text and count images from the original content list/string
                if isinstance(original_content, list):
                    text_parts = []
                    msg_image_urls = []
                    for part in original_content:
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            url = part.get("image_url", {}).get("url")
                            if url in image_urls: # Count only the images we are processing
                                num_images_in_msg += 1
                                msg_image_urls.append(url)
                    text_content_for_json = " ".join(text_parts)
                    logging.debug(f"Message {i} contains {num_images_in_msg} target images.")
                elif isinstance(original_content, str):
                    text_content_for_json = original_content
                    num_images_in_msg = 0 # Assume no image if content is just string
                
                # Determine if we should skip the image token for this message
                # This simple check might need refinement. Assumes image tokens needed only in messages that originally contained images.
                skip_image = (i not in image_message_indices)
                
                logging.debug(f"Calling get_message_json for msg {i}: role={role}, text='{text_content_for_json[:50]}...', skip_image={skip_image}, num_images={num_images_in_msg}")
                
                # Call get_message_json to format content (potentially adding <image> tokens)
                formatted_content_or_dict = get_message_json(
                    model_type=model_type,
                    model_name=self.requested_model,
                    prompt=text_content_for_json,
                    role=role, 
                    skip_image_token=skip_image, # Still potentially simplistic
                    num_images=num_images_in_msg # Pass the count for the current msg
                )
                
                # get_message_json might return a dict or just the content string
                if isinstance(formatted_content_or_dict, dict):
                    processed_messages_for_template.append(formatted_content_or_dict)
                else: # Assume it's just the content string
                    processed_messages_for_template.append({"role": role, "content": formatted_content_or_dict})
                
                logging.debug(f"Result from get_message_json for msg {i}: {json.dumps(processed_messages_for_template[-1])}")

        except Exception as e:
            logging.error(f"Error during get_message_json pre-processing: {str(e)}. Falling back to original messages for template.")
            processed_messages_for_template = messages # Fallback
        # --- END ADDED --- 
        
        # --- Use the pre-processed messages for get_chat_template --- 
        try:
            # Determine if this is a mistral model from config or model name
            model_type = ""
            if hasattr(model, "config") and hasattr(model.config, "model_type"):
                model_type = model.config.model_type
            elif "mistral" in self.requested_model.lower():
                model_type = "mistral3"
                
            # Pass the processor and the *potentially modified* messages list
            # Include model_type for special handling in get_chat_template
            prompt_text = get_chat_template(
                processor, 
                processed_messages_for_template, 
                add_generation_prompt=True,
                model_type=model_type,
                num_images=len(image_urls) if image_urls else 0
            )
            logging.debug(f"Using get_chat_template with pre-processed messages and model_type={model_type}: {prompt_text[:200]}...")
        except Exception as e:
            logging.error(f"get_chat_template failed even with pre-processing: {str(e)}. Falling back to basic default.")
            # Fallback using the *original* messages might be safer here if pre-processing failed
            prompt_text = default_chat_template(messages, template=self.model_provider.default_chat_template)
            logging.debug(f"Fell back to default_chat_template after error: {prompt_text[:100]}...")
        
        # Process function calling
        function_calling_enabled = False
        function_descriptions = ""
        
        if self.normalized_functions and self.selected_function != "none":
            function_calling_enabled = True
            # Format functions for inclusion in prompt
            function_descriptions = self._format_functions_for_model(self.normalized_functions)
            
            # Add function calling system prompt before the main prompt if using functions
            if function_descriptions:
                # Add a system message to the beginning of the conversation if none exists
                if not messages or messages[0]["role"] != "system":
                    system_content = "You are a helpful assistant that can use functions when needed."
                    if function_descriptions:
                        system_content += "\n\n" + function_descriptions
                    messages.insert(0, {"role": "system", "content": system_content})
                else:
                    # Append to existing system message
                    messages[0]["content"] += "\n\n" + function_descriptions

        # Generation parameters
        generation_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": 1.0,  # Default
            "stop": stop,
        }
        
        # Log the final prompt with image(s)
        image_tag = f"with {len(image_urls)} image(s)" if image_urls else "without image"
        logging.info(f"===== FINAL PROMPT ({image_tag}) =====")
        logging.info(prompt_text)
        logging.info("=================================")
        
        logging.debug(f"Calling generation with prompt: {prompt_text[:100]}... and {len(image_urls) if image_urls else 0} image(s)")
        
        # Stream or complete generation - pass image_urls list
        if self.stream:
            self._stream_chat_completion(model, processor, prompt_text, image_urls, generation_kwargs)
        else:
            self._complete_chat(model, processor, prompt_text, image_urls, generation_kwargs)
            
    def _stream_chat_completion(self, model, processor, prompt_text, image_urls: List[str], generation_kwargs):
        """Stream chat completion using the built-in stream_generate function."""
        first_token = True
        full_text = ""
        temp_files = [] # Store list of temp file objects for cleanup
        
        try:
            # Process the images separately if needed
            image_paths = []
            image_sizes = []
            if image_urls:
                # Use the helper method which now returns lists
                temp_paths, temp_files_objs = self._prepare_image_for_generation(image_urls)
                if temp_paths:
                    # Collect paths and image sizes without converting to PIL objects
                    for img_path in temp_paths:
                        # Just get the dimensions for sizing without holding the image in memory
                        with Image.open(img_path) as img:
                            image_sizes.append([img.height, img.width])
                        image_paths.append(img_path)
                    temp_files = temp_files_objs # Store file objs list for cleanup in finally
                else:
                     # Error already logged in helper, raise specific error
                     raise ValueError("Image preparation failed, check logs.")
            
            # Determine generator based on whether images were processed
            if image_paths:
                logging.debug(f"Calling stream_generate with {len(image_paths)} image path(s): {', '.join(image_paths)}")
                generator = stream_generate(
                    model=model,
                    processor=processor,
                    prompt=prompt_text,
                    image=image_paths,  # Pass the LIST of file paths directly
                    **generation_kwargs
                )
            else:
                # Text-only case
                logging.debug("Calling stream_generate without image paths")
                generator = stream_generate(
                    model=model,
                    processor=processor,
                    prompt=prompt_text,
                    image=None,
                    **generation_kwargs
                )
            
            for gen_result in generator:
                # Extract the generation result parts
                segment = gen_result.text
                full_text += segment
                
                # Build the response for this chunk
                response = {
                    "id": self.request_id,
                    "model": self.requested_model,
                    "created": self.created,
                    "object": self.object_type,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": segment},
                            "finish_reason": None
                        }
                    ]
                }
                
                # First chunk includes role
                if first_token:
                    response["choices"][0]["delta"]["role"] = "assistant"
                    first_token = False
                    
                # Send the chunk
                self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                self.wfile.flush()
            
            # Check if the response is a function call
            function_name, arguments, parsed_successfully = self._parse_function_call(full_text)
            finish_reason = "stop"
            
            if self.normalized_functions and parsed_successfully and function_name:
                logging.info(f"Detected function call in streaming mode: {function_name} with arguments: {arguments}")
                finish_reason = "function_call"
                
                # For streaming, send the function call as a special final chunk
                function_call = {
                    "name": function_name,
                    "arguments": json.dumps(arguments)
                }
                
                tool_calls = [{
                    "id": f"call_{uuid.uuid4()}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(arguments)
                    }
                }]
                
                # Create special response with the function call
                if self.stream:
                    # For streaming, we first send an empty content delta
                    function_response = {
                        "id": self.request_id,
                        "model": self.requested_model,
                        "created": self.created,
                        "object": self.object_type,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": None},
                                "finish_reason": None
                            }
                        ]
                    }
                    
                    self.wfile.write(f"data: {json.dumps(function_response)}\n\n".encode())
                    self.wfile.flush()
                    
                    # Then send the function_call delta
                    function_response = {
                        "id": self.request_id,
                        "model": self.requested_model,
                        "created": self.created,
                        "object": self.object_type,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"function_call": function_call},
                                "finish_reason": None
                            }
                        ]
                    }
                    
                    self.wfile.write(f"data: {json.dumps(function_response)}\n\n".encode())
                    self.wfile.flush()
            
            # Send the final chunk with finish_reason
            response = {
                "id": self.request_id,
                "model": self.requested_model,
                "created": self.created,
                "object": self.object_type,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason
                    }
                ]
            }
            
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()
            
            # Signal end of stream
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            
        except Exception as e:
            logging.error(f"Error during streaming: {str(e)}")
            raise # Re-raise after logging
        finally:
            # Clean up the temporary files using the stored object names
            logging.debug(f"Cleaning up {len(temp_files)} temporary image files.")
            for temp_file in temp_files:
                try:
                    if temp_file and hasattr(temp_file, 'name') and os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                        logging.debug(f"Deleted temporary file: {temp_file.name}")
                    elif temp_file and hasattr(temp_file, 'name'):
                         logging.debug(f"Temporary file already deleted or never existed: {temp_file.name}")
                except Exception as e:
                    logging.warning(f"Error deleting temporary file {getattr(temp_file, 'name', '[unknown]')}: {str(e)}")
    
    def _complete_chat(self, model, processor, prompt_text, image_urls: List[str], generation_kwargs):
        """Generate complete chat response using the built-in stream_generate function."""
        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        temp_files = [] # Store list of temp file objects for cleanup
        
        try:
            # Process the images separately if needed
            image_paths = []
            image_sizes = []
            if image_urls:
                 # Use the helper method which now returns lists
                temp_paths, temp_files_objs = self._prepare_image_for_generation(image_urls)
                if temp_paths:
                    # Collect paths and image sizes without converting to PIL objects
                    for img_path in temp_paths:
                        # Just get the dimensions for sizing without holding the image in memory
                        with Image.open(img_path) as img:
                            image_sizes.append([img.height, img.width])
                        image_paths.append(img_path)
                    temp_files = temp_files_objs # Store file objs list for cleanup in finally
                else:
                    # Error already logged in helper, raise specific error
                    raise ValueError("Image preparation failed, check logs.")
            
            # Determine generator based on whether images were processed
            if image_paths:
                logging.debug(f"Calling stream_generate with {len(image_paths)} image path(s): {', '.join(image_paths)}")
                generator = stream_generate(
                    model=model,
                    processor=processor,
                    prompt=prompt_text,
                    image=image_paths,  # Pass the LIST of file paths directly
                    **generation_kwargs
                )
            else:
                 # Text-only case
                logging.debug("Calling stream_generate without image paths")
                generator = stream_generate(
                    model=model,
                    processor=processor,
                    prompt=prompt_text,
                    image=None,
                    **generation_kwargs
                )
            
            # Collect all generation results
            for gen_result in generator:
                full_text += gen_result.text
                prompt_tokens = gen_result.prompt_tokens
                completion_tokens = gen_result.generation_tokens
            
            # Check if the response is a function call
            finish_reason = "stop"
            function_call = None
            tool_calls = None
            
            if self.normalized_functions:
                function_name, arguments, parsed_successfully = self._parse_function_call(full_text)
                if parsed_successfully and function_name:
                    logging.info(f"Detected function call: {function_name} with arguments: {arguments}")
                    finish_reason = "function_call"
                    
                    # Format as function_call for backwards compatibility
                    function_call = {
                        "name": function_name,
                        "arguments": json.dumps(arguments)
                    }
                    
                    # Format as tool_calls for newer format
                    tool_calls = [{
                        "id": f"call_{uuid.uuid4()}",
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": json.dumps(arguments)
                        }
                    }]
                    
                    # For XML-style function calls, replace the raw output in content
                    if "<function_call" in full_text and "</function_call>" in full_text:
                        full_text = f"Function call: {function_name}"
            
            # Update the response object to include function_call or tool_calls if present
            message = {
                "role": "assistant",
                "content": full_text
            }
            
            if function_call:
                message["function_call"] = function_call
                
            if tool_calls:
                message["tool_calls"] = tool_calls
            
            response = {
                "id": self.request_id,
                "object": "chat.completion",
                "created": self.created,
                "model": self.requested_model,
                "system_fingerprint": self.system_fingerprint,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            
            self.wfile.write(json.dumps(response).encode())
            self.wfile.flush()
            
        except Exception as e:
            logging.error(f"Error during chat completion: {str(e)}")
            raise # Re-raise after logging
        finally:
            # Clean up the temporary files using the stored object names
            logging.debug(f"Cleaning up {len(temp_files)} temporary image files.")
            for temp_file in temp_files:
                try:
                    if temp_file and hasattr(temp_file, 'name') and os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                        logging.debug(f"Deleted temporary file: {temp_file.name}")
                    elif temp_file and hasattr(temp_file, 'name'):
                         logging.debug(f"Temporary file already deleted or never existed: {temp_file.name}")
                except Exception as e:
                    logging.warning(f"Error deleting temporary file {getattr(temp_file, 'name', '[unknown]')}: {str(e)}")

    def do_GET(self):
        """
        Respond to a GET request from a client.
        """
        if self.path == "/v1/models":
            self.handle_models_request()
        else:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not Found"}')

    def handle_models_request(self):
        """
        Handle a GET request for the /v1/models endpoint.
        """
        self._set_completion_headers(200)
        self.end_headers()
        
        # Simply return the configured model as available
        model_name = "mlx_vlm"
        if hasattr(self.model_provider, "args") and hasattr(self.model_provider.args, "model"):
            model_name = self.model_provider.args.model
            
        # Remove path components and just use the model name
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        
        models = [
            {
                "id": model_name,
                "object": "model",
                "created": self.created,
                "owned_by": "mlx-vlm"
            }
        ]
        
        response = {"object": "list", "data": models}
        response_json = json.dumps(response).encode()
        
        self.wfile.write(response_json)
        self.wfile.flush()

    def _parse_function_call(self, text):
        """
        Parse function calls from generated text.
        Handles various JSON formats following the OpenAI API specification.
        Returns a tuple of (function_name, arguments, parsed_successfully)
        """
        # Extract JSON objects that might contain function calls
        potential_json_objects = self._extract_json_objects(text)
        
        for json_obj in potential_json_objects:
            try:
                obj = json.loads(json_obj)
                
                # Format 1: {"name": "function_name", "arguments": {...}}
                # This is the standard OpenAI format
                if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                    function_name = obj["name"]
                    arguments = obj["arguments"]
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            pass  # Keep arguments as string if not valid JSON
                    return function_name, arguments, True
                
                # Format 2: {"function_name": "name", "arguments": {...}}
                # Alternative format for backward compatibility
                elif isinstance(obj, dict) and "function_name" in obj and "arguments" in obj:
                    function_name = obj["function_name"]
                    arguments = obj["arguments"]
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            pass  # Keep arguments as string if not valid JSON
                    return function_name, arguments, True
                
                # Format 3: {"function": {"name": "function_name", "arguments": {...}}}
                # Used in some nested contexts
                elif isinstance(obj, dict) and "function" in obj and isinstance(obj["function"], dict):
                    func_obj = obj["function"]
                    if "name" in func_obj and "arguments" in func_obj:
                        function_name = func_obj["name"]
                        arguments = func_obj["arguments"]
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                pass  # Keep arguments as string if not valid JSON
                        return function_name, arguments, True
                
                # Format 4: Nested JSON with function_call data
                # Example: {"type": "function_call", "data": {"function_name": "name", "arguments": {...}}}
                elif isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
                    data_obj = obj["data"]
                    if "function_name" in data_obj and "arguments" in data_obj:
                        function_name = data_obj["function_name"]
                        arguments = data_obj["arguments"]
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                pass
                        return function_name, arguments, True
                
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logging.debug(f"Failed to parse potential JSON object: {str(e)}")
                # Continue to the next potential JSON object
        
        # If we couldn't parse anything, return None
        return None, None, False

    def _extract_json_objects(self, text):
        """
        Extract potential JSON objects from text.
        This handles multiple JSON objects and even nested ones.
        """
        potential_objects = []
        
        # Simple case: try to find objects enclosed in curly braces
        brace_pattern = r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})'
        matches = re.finditer(brace_pattern, text, re.DOTALL)
        for match in matches:
            potential_objects.append(match.group(1))
        
        # Look for code blocks that might contain JSON (markdown format)
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        code_matches = re.finditer(code_block_pattern, text, re.DOTALL)
        for match in code_matches:
            code_content = match.group(1).strip()
            # If it starts with a curly brace, it might be JSON
            if code_content.startswith('{') and code_content.endswith('}'):
                potential_objects.append(code_content)
        
        return potential_objects

    def _format_functions_for_model(self, functions):
        """
        Format the functions for inclusion in the prompt.
        Uses standard OpenAI API-compatible JSON format.
        """
        # Standard OpenAI-style JSON format
        formatted_instructions = "You have access to the following functions:\n\n"
        
        for i, func in enumerate(functions):
            formatted_instructions += f"{i+1}. {func.get('name')}: {func.get('description', '')}\n"
            if "parameters" in func:
                formatted_instructions += f"   Parameters: {json.dumps(func.get('parameters', {}), indent=2)}\n\n"
        
        # Add instructions on how to call functions
        formatted_instructions += "\nTo call a function, respond with a JSON object formatted as:\n"
        formatted_instructions += "{\n"
        formatted_instructions += '  "name": "function_name",\n'
        formatted_instructions += '  "arguments": {\n'
        formatted_instructions += '    "param1": "value1",\n'
        formatted_instructions += '    "param2": "value2"\n'
        formatted_instructions += "  }\n"
        formatted_instructions += "}\n"
        
        return formatted_instructions


def run(
    host: str,
    port: int,
    model_provider: ModelProvider,
    server_class=HTTPServer,
    handler_class=APIHandler,
):
    server_address = (host, port)
    prompt_cache = PromptCache()
    
    infos = socket.getaddrinfo(
        *server_address, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )
    
    server_class.address_family, _, _, _, server_address = next(iter(infos))
    
    httpd = server_class(
        server_address,
        lambda request, client_address, server: handler_class(
            model_provider,
            prompt_cache,
            get_system_fingerprint(),
            request, 
            client_address, 
            server
        ),
    )
    
    warnings.warn(
        "mlx_vlm.server is not recommended for production as "
        "it only implements basic security checks."
    )
    
    logging.info(f"Starting httpd at {host} on port {port}...")
    httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="MLX VLM HTTP Server.")
    
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the MLX VLM model weights, tokenizer, and config",
    )
    
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the HTTP server (default: 127.0.0.1)",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Specify a chat template for the tokenizer",
        required=False,
    )
    
    parser.add_argument(
        "--default-chat-template",
        type=str,
        default="",
        help="Specify a default chat template format string to use when the tokenizer doesn't have one. Use placeholders {role} and {content}.",
        required=False,
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    run(args.host, args.port, ModelProvider(args))


if __name__ == "__main__":
    main() 