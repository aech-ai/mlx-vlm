# MLX-VLM Server Tools

This directory contains tools and examples for working with the MLX-VLM server, which provides an OpenAI-compatible API for vision-language models.

## Requirements

To run the examples in this directory, install the required dependencies:

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

Or install the dependencies individually:

```bash
pip install openai>=1.12.0 pydantic>=2.0.0 requests>=2.31.0
```

## Stream Client

The `stream_client.py` script allows you to interact with the MLX-VLM server's streaming API in a user-friendly way. It processes the stream format and displays the generated text as it's being produced, making it easy to see the model's output.

### Features

- Support for single or multiple images
- Clean, readable streaming output
- Customizable model parameters (temperature, max tokens)
- Optional verbose mode for debugging
- Detailed timing information

### Requirements

```
pip install requests
```

### Usage

#### Basic Usage

To use the stream client with a single image:

```bash
python stream_client.py "Describe this image." "https://example.com/image.jpg"
```

#### Multiple Images

To process multiple images in one request:

```bash
python stream_client.py "Compare these two images." "https://example.com/image1.jpg" "https://example.com/image2.jpg"
```

#### Advanced Options

You can customize the model and generation parameters:

```bash
python stream_client.py --model mistral3 --max-tokens 512 --temp 0.7 \
    --server http://localhost:8080 \
    "What objects do you see?" "https://example.com/image.jpg"
```

#### Debugging

For troubleshooting or to see detailed information about the request:

```bash
python stream_client.py --verbose "Describe this image." "https://example.com/image.jpg"
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `prompt` | Text prompt to send to the model | (required) |
| `images` | One or more image URLs | (required) |
| `--model` | Model ID to use | mistral3 |
| `--max-tokens` | Maximum tokens to generate | 256 |
| `--temp` | Sampling temperature | 0.0 |
| `--server` | Server URL | http://127.0.0.1:8000 |
| `--verbose` | Print verbose debug information | False |

## OpenAI-Compatible Function Calling

The `openai_function_call_test.py` script demonstrates the complete function calling workflow with MLX-VLM server following the exact OpenAI API specification. It illustrates the full round-trip process as documented in the OpenAI documentation:

1. Developer sends tool definitions and messages to the model
2. Model returns tool calls
3. Execute function code locally
4. Send results back to the model
5. Get final response

### Features

- Implements the complete OpenAI function calling workflow
- Uses the standard OpenAI tools API format
- Demonstrates how to process tool call responses
- Shows how to execute local functions and send results back
- Handles the full conversation context with function results

### Usage

Basic usage:

```bash
python openai_function_call_test.py
```

This will run a complete function calling flow, asking about the weather in Paris, executing the simulated weather function, and returning the formatted result to the model.

Custom server or model:

```bash
python openai_function_call_test.py --server http://localhost:8080 --model mistral3
```

Verbose mode for debugging:

```bash
python openai_function_call_test.py --verbose
```

### Function Calling in MLX-VLM

The MLX-VLM server fully supports OpenAI-compatible function calling, allowing the model to:

1. Identify when a function should be called based on user input
2. Generate structured JSON arguments for the function
3. Return a well-formatted response indicating the function call
4. Process function results and provide a final response

The server implements the standard OpenAI API specification for function calling, using JSON for all function definitions and function call responses.

#### Function Call Format

Function calls in MLX-VLM follow the standard OpenAI API format:

```json
{
  "name": "function_name",
  "arguments": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

For the newer tools API format:

```json
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "function_name",
        "arguments": "{\"param1\":\"value1\",\"param2\":\"value2\"}"
      }
    }
  ]
}
```

This ensures compatibility with client libraries that work with the OpenAI API.

## Examples

### Single Image Analysis

```