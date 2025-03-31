#!/usr/bin/env python3
"""
MLX-VLM Server Stream Client

This script provides a user-friendly client for the MLX-VLM streaming API.
It handles both single and multiple images, formatting the streamed output
in a readable way while preserving the performance benefits of streaming.

Features:
- Support for single or multiple images
- Well-formatted streaming output
- Optional verbose mode for debugging
- Customizable model parameters
- Support for custom server URL and port

Usage:
    Single image:
        python stream_client.py "Describe this image." "https://example.com/image.jpg"
    
    Multiple images:
        python stream_client.py "Compare these images." "https://example.com/image1.jpg" "https://example.com/image2.jpg"
    
    With options:
        python stream_client.py --model mistral3 --max-tokens 512 --temp 0.7 --server http://localhost:8000 \
            "What objects do you see?" "https://example.com/image.jpg"

    Verbose mode:
        python stream_client.py --verbose "Describe this image." "https://example.com/image.jpg"
"""

import argparse
import json
import requests
import sys
import time


def stream_chat(
    prompt: str,
    image_urls: list,
    model: str = "mistral3",
    max_tokens: int = 256,
    temperature: float = 0.0,
    server_url: str = "http://127.0.0.1:8000",
    verbose: bool = False
) -> str:
    """
    Stream a chat completion from the MLX-VLM server.
    
    Args:
        prompt: The text prompt to send to the model
        image_urls: List of image URLs to process
        model: Model ID to use (default: mistral3)
        max_tokens: Maximum tokens to generate (default: 256)
        temperature: Sampling temperature (default: 0.0)
        server_url: Base URL of the server (default: http://127.0.0.1:8000)
        verbose: Whether to print verbose debug information (default: False)
        
    Returns:
        The complete generated text
    """
    # Ensure server_url doesn't end with a slash
    if server_url.endswith('/'):
        server_url = server_url[:-1]
        
    url = f"{server_url}/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Build the message content with images and text
    message_content = []
    for image_url in image_urls:
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        })
    
    # Add the text prompt last
    message_content.append({
        "type": "text",
        "text": prompt
    })
    
    data = {
        "model": model,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": message_content
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # Print request details if verbose
    if verbose:
        print(f"Request URL: {url}")
        print(f"Request headers: {headers}")
        print(f"Request data: {json.dumps(data, indent=2)}")
        print(f"Processing {len(image_urls)} image(s)...")
    
    # Time the request
    start_time = time.time()
    
    # Make the request
    response = requests.post(url, headers=headers, json=data, stream=True)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return ""
    
    # For accumulating the full response
    full_text = ""
    
    # Process the streaming response
    print("\n--- Response ---")
    
    for line in response.iter_lines():
        if line:
            # Skip the "data: " prefix
            if line.startswith(b'data: '):
                line = line[6:]
                
            # Check for the [DONE] message
            if line == b'[DONE]':
                break
                
            try:
                # Parse the JSON
                chunk = json.loads(line)
                
                # Extract the content if available
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        content = delta['content']
                        full_text += content
                        # Print the content without a newline to simulate streaming
                        print(content, end='', flush=True)
                    elif 'role' in delta and verbose:
                        print(f"\n[Role: {delta['role']}]", end='', flush=True)
            except json.JSONDecodeError:
                if verbose:
                    print(f"\nError parsing JSON: {line.decode('utf-8', errors='replace')}")
    
    # Calculate time elapsed
    elapsed_time = time.time() - start_time
    
    # Print additional information if verbose
    if verbose:
        print(f"\n\nRequest completed in {elapsed_time:.2f} seconds")
    
    print("\n\n--- Full Text ---")
    print(full_text)
    
    return full_text


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MLX-VLM Streaming Client")
    
    # Required arguments
    parser.add_argument("prompt", type=str, help="Text prompt to send to the model")
    parser.add_argument("images", nargs='+', type=str, help="One or more image URLs")
    
    # Optional arguments
    parser.add_argument("--model", type=str, default="mistral3", 
                        help="Model ID to use (default: mistral3)")
    parser.add_argument("--max-tokens", type=int, default=256, 
                        help="Maximum tokens to generate (default: 256)")
    parser.add_argument("--temp", type=float, default=0.0, 
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000", 
                        help="Server URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--verbose", action="store_true", 
                        help="Print verbose debug information")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Call the stream_chat function with the parsed arguments
    stream_chat(
        prompt=args.prompt,
        image_urls=args.images,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        server_url=args.server,
        verbose=args.verbose
    ) 