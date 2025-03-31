#!/usr/bin/env python3
"""
OpenAI-compatible Function Calling Test

This script demonstrates the complete function calling workflow with MLX-VLM server
following the exact OpenAI API specification. It illustrates the full round-trip process:
1. Send tool definitions and messages to the model
2. Receive tool calls from the model
3. Execute function code locally
4. Send results back to the model
5. Get the final response

The example implements a weather lookup function that demonstrates the complete flow.
"""

import json
import argparse
from datetime import datetime
from typing import Dict, Optional, List, Union, Literal
import os

from openai import OpenAI
from pydantic import BaseModel, Field

# Define Pydantic models for function parameters and responses
class WeatherParams(BaseModel):
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
    unit: Optional[Literal["celsius", "fahrenheit"]] = Field("celsius", description="The temperature unit to use")

class WeatherInfo(BaseModel):
    temperature: float = Field(..., description="Current temperature")
    unit: Literal["celsius", "fahrenheit"] = Field(..., description="Temperature unit")
    description: str = Field(..., description="Weather description")
    humidity: int = Field(..., description="Humidity percentage")
    wind_speed: float = Field(..., description="Wind speed")
    wind_direction: str = Field(..., description="Wind direction")

class WeatherResponse(BaseModel):
    location: str = Field(..., description="The requested location")
    date: str = Field(..., description="Current date (YYYY-MM-DD)")
    time: str = Field(..., description="Current time (HH:MM:SS)")
    weather: WeatherInfo = Field(..., description="Weather information")

def get_weather(params: WeatherParams) -> WeatherResponse:
    """
    Simulate getting weather data for a location.
    
    In a real application, this would call an actual weather API.
    
    Args:
        params: Pydantic model containing location and unit
        
    Returns:
        A WeatherResponse object with the weather information
    """
    location = params.location
    unit = params.unit or "celsius"
    
    # Get current date/time for the simulation
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Simulate different weather based on location
    weather_data = {
        "San Francisco": WeatherInfo(
            temperature=18 if unit == "celsius" else 64,
            unit=unit,
            description="Partly cloudy with fog",
            humidity=72,
            wind_speed=10,
            wind_direction="West",
        ),
        "New York": WeatherInfo(
            temperature=22 if unit == "celsius" else 72,
            unit=unit,
            description="Sunny with clear skies",
            humidity=45,
            wind_speed=5,
            wind_direction="South",
        ),
        "London": WeatherInfo(
            temperature=15 if unit == "celsius" else 59,
            unit=unit,
            description="Light rain",
            humidity=85,
            wind_speed=12,
            wind_direction="Southwest",
        ),
        "Paris": WeatherInfo(
            temperature=14 if unit == "celsius" else 57,
            unit=unit,
            description="Clear skies",
            humidity=65,
            wind_speed=8,
            wind_direction="East",
        )
    }
    
    # Default weather if location not found
    default_weather = WeatherInfo(
        temperature=20 if unit == "celsius" else 68,
        unit=unit,
        description="Clear skies",
        humidity=60,
        wind_speed=5,
        wind_direction="North",
    )
    
    # Find the best match for the location
    best_match = None
    for known_location in weather_data.keys():
        if known_location.lower() in location.lower():
            best_match = known_location
            break
    
    result = WeatherResponse(
        location=location,
        date=date_str,
        time=time_str,
        weather=weather_data.get(best_match, default_weather)
    )
    
    return result

def run_function_calling_test(server_url="http://127.0.0.1:8000", model="mistral3", verbose=False):
    """
    Run a complete function calling test with the OpenAI API format.
    
    This follows the 5-step process shown in the OpenAI documentation:
    1. Developer sends tool definitions + messages
    2. Model returns tool calls
    3. Execute function code
    4. Send results back to the model
    5. Get final response
    
    Args:
        server_url: URL of the MLX-VLM server
        model: Model name to use
        verbose: Whether to print detailed request/response information
    """
    # Ensure server_url doesn't end with a slash
    if server_url.endswith('/'):
        server_url = server_url[:-1]
    
    # Create OpenAI client with custom base URL pointing to our MLX-VLM server
    client = OpenAI(
        base_url=f"{server_url}/v1",
        api_key="not-needed"  # MLX-VLM server doesn't require an API key
    )
    
    print("\n--- STEP 1: Send Tool Definitions and Messages ---")
    
    # Define the weather function using OpenAI's functions and tools formats
    weather_function = {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "parameters": WeatherParams.model_json_schema()
    }
    
    # Create explicit system prompt with function definition included
    system_prompt = """You are a helpful assistant that can provide weather information.

You have access to the following function:
get_weather(location: string, unit: string = "celsius")
- location: The city and state, e.g. San Francisco, CA (required)
- unit: The temperature unit to use, either "celsius" or "fahrenheit" (optional, default: "celsius")

ALWAYS use the get_weather function when asked about weather.
DO NOT make up weather information or refuse to use the function.

To call the function, respond with:
{
  "name": "get_weather",
  "arguments": {
    "location": "location name",
    "unit": "celsius"
  }
}
"""
    
    print("Sending initial request with user question...")
    
    # Try both the functions and tools format to see which one works
    try:
        # First attempt: Use the tools format (OpenAI standard)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "What's the weather in Paris?"}
            ],
            tools=[{
                "type": "function",
                "function": weather_function
            }],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            temperature=0
        )
    except Exception as e:
        print(f"Tools format failed, trying functions format: {e}")
        
        # Second attempt: Use the legacy functions format
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "What's the weather in Paris?"}
            ],
            functions=[weather_function],
            function_call={"name": "get_weather"},  # Legacy format
            temperature=0
        )
    
    if verbose:
        print(f"Response: {response.model_dump_json(indent=2)}")
    
    # Check if the model returned tool calls or function_call
    assistant_message = response.choices[0].message
    
    # Check for function calls (in either format)
    tool_calls = assistant_message.tool_calls if hasattr(assistant_message, 'tool_calls') else None
    function_call = assistant_message.function_call if hasattr(assistant_message, 'function_call') else None
    
    if not tool_calls and not function_call:
        print("\nModel did not return any function calls. Exiting test.")
        print(f"Model response: {assistant_message.content}")
        return
    
    print("\n--- STEP 2: Model Returns Function/Tool Calls ---")
    if tool_calls:
        print(f"Tool calls received: {len(tool_calls)}")
    elif function_call:
        print(f"Function call received: {function_call.name}")
    
    # Build the conversation history
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What's the weather in Paris?"},
        assistant_message.model_dump()  # Convert to dict for appending to messages
    ]
    
    # Process the function/tool call
    if tool_calls:
        # Process tool calls (new format)
        for tool_call in tool_calls:
            if tool_call.type != "function":
                continue
                
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"\nFunction: {function_name}")
            print(f"Arguments: {json.dumps(function_args, indent=2)}")
            
            # STEP 3: Execute Function Code
            print("\n--- STEP 3: Execute Function Code ---")
            if function_name == "get_weather":
                # Parse the arguments with Pydantic for validation
                params = WeatherParams(**function_args)
                
                print(f"Executing get_weather(location={params.location}, unit={params.unit})...")
                weather_result = get_weather(params)
                
                # Convert to dict for JSON serialization
                result_dict = weather_result.model_dump()
                print(f"Function result: {json.dumps(result_dict, indent=2)}")
                
                # STEP 4: Send Results Back to the Model
                print("\n--- STEP 4: Send Results Back to the Model ---")
                
                # Add the tool response to our conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(result_dict)
                })
    
    elif function_call:
        # Process function call (legacy format)
        function_name = function_call.name
        function_args = json.loads(function_call.arguments)
        
        print(f"\nFunction: {function_name}")
        print(f"Arguments: {json.dumps(function_args, indent=2)}")
        
        # STEP 3: Execute Function Code
        print("\n--- STEP 3: Execute Function Code ---")
        if function_name == "get_weather":
            # Parse the arguments with Pydantic for validation
            params = WeatherParams(**function_args)
            
            print(f"Executing get_weather(location={params.location}, unit={params.unit})...")
            weather_result = get_weather(params)
            
            # Convert to dict for JSON serialization
            result_dict = weather_result.model_dump()
            print(f"Function result: {json.dumps(result_dict, indent=2)}")
            
            # STEP 4: Send Results Back to the Model
            print("\n--- STEP 4: Send Results Back to the Model ---")
            
            # Add the function result to our conversation
            messages.append({
                "role": "function",
                "name": function_name,
                "content": json.dumps(result_dict)
            })
    
    print("Sending function results to the model...")
    
    # Make the second request with the tool results
    second_response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    
    if verbose:
        print(f"Second response: {second_response.model_dump_json(indent=2)}")
    
    # STEP 5: Model Provides Final Response
    print("\n--- STEP 5: Model Provides Final Response ---")
    final_message = second_response.choices[0].message
    
    print(f"Final response: {final_message.content}")
    
    # Display the full conversation flow for clarity
    print("\n=== Complete Conversation Flow ===")
    
    # System message
    print("\n[SYSTEM]")
    print(messages[0]["content"])
    
    # User message
    print("\n[USER]")
    print(messages[1]["content"])
    
    # Assistant message with function calls
    print("\n[ASSISTANT]")
    print(assistant_message.content or "")
    if tool_calls:
        for tc in tool_calls:
            if tc.type == "function":
                print(f"Called: {tc.function.name}({tc.function.arguments})")
    elif function_call:
        print(f"Called: {function_call.name}({function_call.arguments})")
    
    # Function/Tool response
    print("\n[FUNCTION/TOOL RESPONSE]")
    print(f"Result: {messages[3]['content']}")
    
    # Final assistant response
    print("\n[ASSISTANT]")
    print(final_message.content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OpenAI-compatible function calling")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000", 
                        help="Server URL")
    parser.add_argument("--model", type=str, default="mistral3",
                        help="Model name")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    args = parser.parse_args()
    
    run_function_calling_test(args.server, args.model, args.verbose) 