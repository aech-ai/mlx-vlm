[![Upload Python Package](https://github.com/Blaizzy/mlx-vlm/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Blaizzy/mlx-vlm/actions/workflows/python-publish.yml)
# MLX-VLM

MLX-VLM is a package for inference and fine-tuning of Vision Language Models (VLMs) on your Mac using MLX.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
  - [Chat UI with Gradio](#chat-ui-with-gradio)
  - [Python Script](#python-script)
- [Multi-Image Chat Support](#multi-image-chat-support)
  - [Supported Models](#supported-models)
  - [Usage Examples](#usage-examples)
- [Video Understanding](#video-understanding)
  - [Supported Models](#supported-models-1)
  - [Usage Examples](#usage-examples-1)
- [OpenAI-Compatible Server](#openai-compatible-server)
  - [Starting the Server](#starting-the-server)
  - [Using the Server](#using-the-server)
  - [Example Scripts](#example-scripts)
- [Fine-tuning](#fine-tuning)

## Installation

The easiest way to get started is to install the `mlx-vlm` package using pip:

```sh
pip install mlx-vlm
```

## Usage

### Command Line Interface (CLI)

Generate output from a model using the CLI:

```sh
python -m mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --temperature 0.0 --image http://images.cocodataset.org/val2017/000000039769.jpg
```

### Chat UI with Gradio

Launch a chat interface using Gradio:

```sh
python -m mlx_vlm.chat_ui --model mlx-community/Qwen2-VL-2B-Instruct-4bit
```

### Python Script

Here's an example of how to use MLX-VLM in a Python script:

```python
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Load the model
model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

# Prepare input
image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
# image = [Image.open("...")] can also be used with PIL.Image.Image objects
prompt = "Describe this image."

# Apply chat template
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=len(image)
)

# Generate output
output = generate(model, processor, formatted_prompt, image, verbose=False)
print(output)
```

## Multi-Image Chat Support

MLX-VLM supports analyzing multiple images simultaneously with select models. This feature enables more complex visual reasoning tasks and comprehensive analysis across multiple images in a single conversation.

### Supported Models

The following models support multi-image chat:

1. Idefics 2
2. LLaVA (Interleave)
3. Qwen2-VL
4. Phi3-Vision
5. Pixtral

### Usage Examples

#### Python Script

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

images = ["path/to/image1.jpg", "path/to/image2.jpg"]
prompt = "Compare these two images."

formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=len(images)
)

output = generate(model, processor, formatted_prompt, images, verbose=False)
print(output)
```

#### Command Line

```sh
python -m mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --prompt "Compare these images" --image path/to/image1.jpg path/to/image2.jpg
```

## Video Understanding

MLX-VLM also supports video analysis such as captioning, summarization, and more, with select models.

### Supported Models

The following models support video chat:

1. Qwen2-VL
2. Qwen2.5-VL
3. Idefics3
4. LLaVA

With more coming soon.

### Usage Examples

#### Command Line
```sh
python -m mlx_vlm.video_generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --prompt "Describe this video" --video path/to/video.mp4 --max-pixels 224 224 --fps 1.0
```

## OpenAI-Compatible Server

MLX-VLM includes an OpenAI-compatible server that allows you to use your MLX models with any OpenAI API client. The server supports:

- Chat completions endpoint
- Image analysis (base64 or URL)
- Streaming responses
- Function calling (tools API)
- Multiple image inputs

### Starting the Server

Start the server with:

```sh
python -m mlx_vlm.server --model mlx-community/Qwen2-VL-2B-Instruct-4bit --port 8000
```

Server options:
- `--model`: The model to use (default: "mlx-community/Qwen2-VL-2B-Instruct-4bit")
- `--port`: The port to listen on (default: 8000)
- `--host`: The host to bind to (default: "127.0.0.1")
- `--max-tokens`: Maximum new tokens to generate (default: 1024)

### Using the Server

Once the server is running, you can use it with any OpenAI API client. The server implements the `/v1/chat/completions` endpoint.

#### Python Example with OpenAI SDK

```python
from openai import OpenAI

# Create a client with a custom base URL
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",  # Local MLX-VLM server
    api_key="not-needed"  # API key not required
)

# Create a chat completion
response = client.chat.completions.create(
    model="any-model-name",  # Model name is ignored, using the loaded model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's in this image?"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

#### Image Analysis

To analyze images, include them in the user message:

```python
response = client.chat.completions.create(
    model="any-model-name",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]}
    ]
)
```

#### Function Calling

The server supports OpenAI's function calling (tools API):

```python
response = client.chat.completions.create(
    model="any-model-name",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Paris?"}
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }],
    tool_choice="auto"
)
```

### Example Scripts

The repository includes several example scripts that demonstrate how to use the server:

- `examples/server/openai_function_call_test.py`: Demonstrates the complete function calling workflow
- `examples/server/stream_client.py`: Shows how to use streaming with image inputs

Run the examples:

```sh
# Install dependencies
pip install -r examples/server/requirements.txt

# Run the function calling example
python examples/server/openai_function_call_test.py

# Run the streaming client
python examples/server/stream_client.py "Describe this image" "https://example.com/image.jpg"
```

For more details, see the documentation in the `examples/server` directory.

# Fine-tuning

MLX-VLM supports fine-tuning models with LoRA and QLoRA.

## LoRA & QLoRA

To learn more about LoRA, please refer to the [LoRA.md](./mlx_vlm/LORA.MD) file.
