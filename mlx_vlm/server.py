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


def process_message_content(messages):
    """Process message content to handle multimodal inputs (images).
    Extracts the first image URL found but leaves the original content structure intact.
    """
    image_url_found = None # Store the first image URL found across all messages
    
    for message in messages:
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
                    if url and not image_url_found: # Store the *first* image URL we find
                        image_url_found = url
                        logging.debug(f"Found image URL in message content: {url[:100] if url else 'None'}...")
            
            # Log what we found, but DON'T modify message["content"]
            text_content_for_log = " ".join(text_parts)
            logging.debug(f"Processed message: content remains list, found image: {has_image}, text parts: '{text_content_for_log}', first image URL stored: {'Yes' if image_url_found == url else 'No (already found one earlier)'}")
            
        elif isinstance(message.get("content"), str):
             # If content is already a string, just log it
             logging.debug(f"Processed message: content is already a string: '{message['content'][:100]}...'")

    # Return the first image URL found (or None)
    return image_url_found


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

    def _prepare_image_for_generation(self, image_url: str) -> Tuple[Optional[str], Optional[tempfile._TemporaryFileWrapper]]:
        """Loads image from URL, saves to temp file, returns path and file object."""
        temp_file = None
        temp_path = None
        try:
            pil_image = load_image_from_url(image_url)
            logging.debug(f"Loaded image successfully: {pil_image.size}, mode: {pil_image.mode}")

            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            temp_path = temp_file.name
            temp_file.close() # Close for saving

            logging.debug(f"Temporary file path created: {temp_path}")

            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            pil_image.save(temp_path, format="JPEG")
            logging.debug(f"Saved image to temporary file: {temp_path}")

            # Verify save
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                logging.debug(f"Temporary file verified (exists, size > 0): {temp_path}")
                return temp_path, temp_file # Return path AND the file object wrapper for cleanup
            else:
                logging.error(f"Failed to save or verify temporary image file: {temp_path}")
                # Clean up if save failed
                if temp_file:
                    try:
                        os.unlink(temp_path)
                    except OSError as e:
                         logging.warning(f"Error deleting failed temp file {temp_path}: {e}")
                return None, None

        except Exception as e:
            logging.error(f"Error preparing image: {str(e)}")
            # Clean up if any exception occurred
            if temp_path and os.path.exists(temp_path):
                 try:
                     os.unlink(temp_path)
                 except OSError as e_unlink:
                     logging.warning(f"Error deleting temp file {temp_path} after exception: {e_unlink}")
            elif temp_file:
                 # If temp_file object exists but path wasn't set or file wasn't created
                 pass # Nothing to unlink
            raise ValueError(f"Failed to prepare image: {str(e)}")

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

        # Process messages for multimodal content - extracts first image URL, leaves structure
        messages = self.body["messages"]
        image_url = process_message_content(messages) # Returns first image URL or None
        logging.debug(f"Original messages structure: {json.dumps(messages)}")
        if image_url:
             logging.debug(f"Extracted image URL for loading: {image_url[:100] if image_url else 'None'}...")

        # --- ADDED: Prepare messages for template using get_message_json --- 
        processed_messages_for_template = []
        try:
            model_config = model.config # Assuming model has a config attribute
            model_type = model_config.model_type
            logging.debug(f"Detected model_type for get_message_json: {model_type}")
            
            # Identify which message contains the image URL we are using
            image_message_index = -1
            if image_url:
                for i, msg in enumerate(messages):
                    if isinstance(msg.get("content"), list):
                        for part in msg["content"]:
                            if part.get("type") == "image_url" and part.get("image_url", {}).get("url") == image_url:
                                image_message_index = i
                                break
                    if image_message_index != -1:
                        break
            
            logging.debug(f"Message index containing the target image_url: {image_message_index}")

            for i, msg in enumerate(messages):
                role = msg["role"]
                original_content = msg["content"]
                text_content_for_json = ""
                num_images_in_msg = 0
                
                # Extract text and count images from the original content list/string
                if isinstance(original_content, list):
                    text_parts = []
                    for part in original_content:
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            num_images_in_msg += 1
                    text_content_for_json = " ".join(text_parts)
                elif isinstance(original_content, str):
                    text_content_for_json = original_content
                    num_images_in_msg = 0 # Assume no image if content is just string
                
                # Determine if we should skip the image token for this message
                # Skip if it's not the message containing the *specific* image_url we are processing
                skip_image = (i != image_message_index)
                
                logging.debug(f"Calling get_message_json for msg {i}: role={role}, text='{text_content_for_json[:50]}...', skip_image={skip_image}, num_images={num_images_in_msg}")
                
                # Call get_message_json to format content (potentially adding <image>)
                formatted_content_or_dict = get_message_json(
                    model_type=model_type,
                    model_name=self.requested_model,
                    prompt=text_content_for_json,
                    role=role, 
                    skip_image_token=skip_image,
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
            # Pass the processor and the *potentially modified* messages list
            prompt_text = get_chat_template(processor, processed_messages_for_template, add_generation_prompt=True)
            logging.debug(f"Using get_chat_template with pre-processed messages: {prompt_text[:200]}...")
        except Exception as e:
            logging.error(f"get_chat_template failed even with pre-processing: {str(e)}. Falling back to basic default.")
            # Fallback using the *original* messages might be safer here if pre-processing failed
            prompt_text = default_chat_template(messages, template=self.model_provider.default_chat_template)
            logging.debug(f"Fell back to default_chat_template after error: {prompt_text[:100]}...")
        
        # Generation parameters
        generation_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": 1.0,  # Default
            "stop": stop,
        }
        
        # Log the final prompt with image
        image_tag = "with image" if image_url else "without image"
        logging.info(f"===== FINAL PROMPT ({image_tag}) =====")
        logging.info(prompt_text)
        logging.info("=================================")
        
        logging.debug(f"Calling generation with prompt: {prompt_text[:100]}... and image: {'Yes' if image_url else 'No'}")
        
        # Stream or complete generation
        if self.stream:
            self._stream_chat_completion(model, processor, prompt_text, image_url, generation_kwargs)
        else:
            self._complete_chat(model, processor, prompt_text, image_url, generation_kwargs)
            
    def _stream_chat_completion(self, model, processor, prompt_text, image_url, generation_kwargs):
        """Stream chat completion using the built-in stream_generate function."""
        first_token = True
        full_text = ""
        temp_file = None
        
        try:
            # Process the image separately if needed
            image_path_for_generate = None
            if image_url:
                # Use the helper method
                temp_path, temp_file_obj = self._prepare_image_for_generation(image_url)
                if temp_path:
                    image_path_for_generate = temp_path
                    temp_file = temp_file_obj # Store file obj for cleanup in finally
                else:
                     # Error already logged in helper, raise specific error
                     raise ValueError("Image preparation failed, check logs.")
            
            # Determine generator based on whether image was processed
            if image_path_for_generate:
                logging.debug(f"Calling stream_generate with image path: {image_path_for_generate}")
                generator = stream_generate(
                    model=model,
                    processor=processor,
                    prompt=prompt_text,
                    image=image_path_for_generate,  # Pass the file path
                    **generation_kwargs
                )
            else:
                # Text-only case
                logging.debug("Calling stream_generate without image path")
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
                        "finish_reason": "stop"
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
            raise
        finally:
            # Clean up the temporary file using the stored object name
            if temp_file is not None:
                try:
                    os.unlink(temp_file.name)
                    logging.debug(f"Deleted temporary file: {temp_file.name}")
                except Exception as e:
                    logging.warning(f"Error deleting temporary file: {str(e)}")
    
    def _complete_chat(self, model, processor, prompt_text, image_url, generation_kwargs):
        """Generate complete chat response using the built-in stream_generate function."""
        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        temp_file = None
        
        try:
            # Process the image separately if needed
            image_path_for_generate = None
            if image_url:
                 # Use the helper method
                temp_path, temp_file_obj = self._prepare_image_for_generation(image_url)
                if temp_path:
                    image_path_for_generate = temp_path
                    temp_file = temp_file_obj # Store file obj for cleanup in finally
                else:
                    # Error already logged in helper, raise specific error
                    raise ValueError("Image preparation failed, check logs.")
            
            # Determine generator based on whether image was processed
            if image_path_for_generate:
                logging.debug(f"Calling stream_generate with image path: {image_path_for_generate}")
                generator = stream_generate(
                    model=model,
                    processor=processor,
                    prompt=prompt_text,
                    image=image_path_for_generate,  # Pass the file path
                    **generation_kwargs
                )
            else:
                 # Text-only case
                logging.debug("Calling stream_generate without image path")
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
            
            # Build the complete response
            response = {
                "id": self.request_id,
                "object": "chat.completion",
                "created": self.created,
                "model": self.requested_model,
                "system_fingerprint": self.system_fingerprint,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_text
                        },
                        "finish_reason": "stop"
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
            raise
        finally:
            # Clean up the temporary file using the stored object name
            if temp_file is not None:
                try:
                    os.unlink(temp_file.name)
                    logging.debug(f"Deleted temporary file: {temp_file.name}")
                except Exception as e:
                    logging.warning(f"Error deleting temporary file: {str(e)}")

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