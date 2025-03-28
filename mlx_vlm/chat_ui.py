import argparse
import pprint
from typing import Union, List, Dict

import gradio as gr

from mlx_vlm import load

from .prompt_utils import get_chat_template, get_message_json
from .utils import load, load_config, load_image_processor, stream_generate, GenerationResult, load_image


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate text from an image using a model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qnguyen3/nanoLLaVA",
        help="The path to the local model directory or Hugging Face repo.",
    )
    return parser.parse_args()


args = parse_arguments()
config = load_config(args.model)
model, processor = load(args.model, processor_kwargs={"trust_remote_code": True})
image_processor = load_image_processor(args.model)


def create_message_content(text_input: str, files: list = None, model_type: str = None, num_images: int = 1, **kwargs) -> Union[str, List[Dict[str, str]]]:
    """Creates the content block for a message, handling multimodal input."""
    if files and len(files) > 0:
        # Assume multimodal if files are present
        # Use get_message_json to create the list structure for the current turn
        # We only need the 'content' part here. Role will be added later.
        # Set skip_image_token=False as this is the turn with the image.
        message_structure = get_message_json(
            model_type,
            text_input,
            role="user", # Role is temporary here, needed by get_message_json
            skip_image_token=False,
            num_images=len(files), # Use actual number of files attached
            **kwargs
        )
        # Extract the content part (which should be the list)
        # Handle cases where get_message_json might return just the string (e.g., paligemma)
        if isinstance(message_structure, dict):
             return message_structure.get("content", text_input) # Default to text if content missing
        else:
             # Handle models like paligemma that return only the formatted string
             # We need to reconstruct a pseudo-list for consistency if needed downstream,
             # but for template processing, the string might be okay. Check get_chat_template.
             # For now, assume get_chat_template handles string content correctly.
             return message_structure # Return the string directly
    else:
        # Text-only message
        return text_input


def chat(message: Dict, history: List[List[Union[str, None]]], temperature: float, max_tokens: int):
    # message is now a Dict: {"text": "...", "files": ["..."]}
    # history is List[List[str, str]]

    # ---- VERY EARLY DEBUG ----
    # Keep this temporarily to confirm if Gradio sends stale files
    print("--- Raw Gradio message input ---")
    pprint.pprint(message)
    # ---- END DEBUG ----


    if config["model_type"] != "paligemma":
        messages = [] # Rebuild the full history in the expected format

        # Process past history
        for turn_index, turn in enumerate(history):
            user_input_raw, assistant_text = turn # user_input_raw might be str or tuple

            # --- MODIFIED USER MESSAGE RECONSTRUCTION ---
            user_text = None
            if isinstance(user_input_raw, str):
                # If it's already a string, use it directly
                user_text = user_input_raw
            elif isinstance(user_input_raw, tuple):
                # If it's a tuple, try to find a string element (heuristic for tuples format)
                # This handles cases like (('/path/img.jpg',), 'prompt') or ('prompt', ('/path/img.jpg',)) etc.
                found_str = False
                for item in user_input_raw:
                    if isinstance(item, str):
                        user_text = item
                        found_str = True
                        break # Take the first string found
                if not found_str:
                     print(f"\033[33mWarning\033[0m: History turn {turn_index} user input was tuple without string: {user_input_raw}. Skipping.")
                # else: user_text contains the extracted string

            # If we successfully extracted user_text as a string
            if user_text:
                 # Check for and skip potential duplicates resulting from tuple format issues
                 # If the previous message added was also a user message with the same text, skip this one.
                 if messages and messages[-1].get("role") == "user" and messages[-1].get("content") == user_text:
                     print(f"\033[33mWarning\033[0m: Skipping likely duplicate user message from history: {user_text}")
                     pass # Skip adding this duplicate
                 else:
                     user_msg_struct = get_message_json(
                         config["model_type"], user_text, role="user", skip_image_token=True # Keep skip_image_token=True for history
                     )
                     # Ensure the structure is valid before appending
                     # Convert string output from get_message_json (e.g. paligemma) to dict for consistency
                     if isinstance(user_msg_struct, str):
                          messages.append({'role': 'user', 'content': user_msg_struct})
                     elif isinstance(user_msg_struct, dict) and 'content' in user_msg_struct and 'role' in user_msg_struct:
                          messages.append(user_msg_struct)
                     else:
                          print(f"\033[33mWarning\033[0m: Invalid structure from get_message_json for user history: {user_msg_struct}")


            # --- Assistant Message Reconstruction (Error handling added) ---
            if assistant_text:
                 # Ensure assistant_text is a string before processing
                 if isinstance(assistant_text, str):
                     assistant_msg_struct = get_message_json(
                         config["model_type"], assistant_text, role="assistant", skip_image_token=True
                     )
                     # Convert string output from get_message_json to dict for consistency
                     if isinstance(assistant_msg_struct, str):
                          messages.append({'role': 'assistant', 'content': assistant_msg_struct})
                     elif isinstance(assistant_msg_struct, dict) and 'content' in assistant_msg_struct and 'role' in assistant_msg_struct:
                          messages.append(assistant_msg_struct)
                     else:
                          print(f"\033[33mWarning\033[0m: Invalid structure from get_message_json for assistant history: {assistant_msg_struct}")
                 else:
                     print(f"\033[33mWarning\033[0m: Assistant history item was not a string: {assistant_text}")


        # --- Process Current User Message ---
        current_text = message["text"]
        # Get files submitted THIS TURN. Default to empty list.
        current_files = message.get("files", [])

        # --- MODIFIED LOGIC ---
        # Determine if an image was *actually* submitted this turn
        image_submitted_this_turn = bool(current_files)

        # Create the content based ONLY on whether an image was submitted *this turn*
        if image_submitted_this_turn:
            # Create multimodal content list if files are present this turn
            current_content = create_message_content(
                current_text,
                current_files, # Pass the actual files for this turn
                model_type=config["model_type"],
                num_images=len(current_files)
            )
        else:
            # Create text-only content string if no files this turn
            # Explicitly pass None for files to create_message_content's else branch
            current_content = create_message_content(
                current_text,
                None,
                model_type=config["model_type"]
            )
            # Defensive check: Ensure it's a string if no image submitted
            if not isinstance(current_content, str):
                 print(f"\033[33mWarning\033[0m: create_message_content returned non-string ({type(current_content)}) even without files. Falling back to raw text.")
                 current_content = current_text
        # --- END MODIFIED LOGIC ---

        # Add the current user message with the correctly determined content
        current_message_json = {"role": "user", "content": current_content}

        # Append the fully formed current message
        messages.append(current_message_json)

        # ---- DEBUGGING START ----
        print("--- Messages before get_chat_template ---")
        pprint.pprint(messages)
        # ---- DEBUGGING END ----

        # Apply the chat template (this handles list/str content conversion)
        prompt_string = get_chat_template(processor, messages, add_generation_prompt=True)

    else: # Handle paligemma separately
        # PaliGemma specific logic if different from above
        # This might involve just taking message["text"] and using prompt_with_image_token format
        prompt_string = get_message_json(
            config["model_type"],
            message["text"],
            role="user",
            skip_image_token=(not message.get("files")), # Skip if no files this turn
            num_images=len(message.get("files", []))
        )
        # Note: PaliGemma might directly return the final prompt string here


    # Pass only the LATEST image file(s) if present THIS TURN
    latest_image_path = None
    if message.get("files"): # Check if files list exists and is not empty
        latest_image_path = message["files"][-1]

    response = ""
    # Variables to store the latest stats
    latest_prompt_tokens = 0
    latest_generation_tokens = 0
    latest_prompt_tps = 0.0
    latest_generation_tps = 0.0
    latest_peak_memory = 0.0

    # Ensure stream_generate receives the string prompt and optional latest image path
    for chunk in stream_generate(
        model,
        processor,
        prompt_string, # Pass the final string prompt
        image=latest_image_path, # Pass path of the current image (if any), or None
        max_tokens=max_tokens,
        temperature=temperature,
    ):
        # Check if chunk is GenerationResult and update stats
        if isinstance(chunk, GenerationResult):
             if chunk.text: # Check if text exists in the chunk
                response += chunk.text
             # Update stats with the latest values from the chunk
             latest_prompt_tokens = chunk.prompt_tokens
             latest_generation_tokens = chunk.generation_tokens
             latest_prompt_tps = chunk.prompt_tps
             latest_generation_tps = chunk.generation_tps
             latest_peak_memory = chunk.peak_memory
        elif isinstance(chunk, str): # Handle case where generate_step yields strings directly
             response += chunk
        # else: handle other potential types or errors

        # Yield partial response for streaming effect
        yield response

    # --- After the loop finishes ---
    # Format the final statistics in a more compact plain text format
    stats_str = (
        f"\n<div style='font-size: 0.65em; color: #666; margin-top: 0px;'>"
        f"Prompt: {latest_prompt_tokens} tokens ({latest_prompt_tps:.1f} t/s) | "
        f"Generation: {latest_generation_tokens} tokens ({latest_generation_tps:.1f} t/s) | "
        f"Peak memory: {latest_peak_memory:.2f} GB"
        f"</div>"
    )

    # Append stats to the final response
    final_response_with_stats = response + stats_str

    # Yield the complete response with stats one last time
    yield final_response_with_stats


demo = gr.ChatInterface(
    fn=chat,
    title="MLX-VLM Chat UI",
    additional_inputs_accordion=gr.Accordion(
        label="⚙️ Parameters", open=False, render=False
    ),
    additional_inputs=[
        gr.Slider(
            minimum=0, maximum=1, step=0.1, value=0.1, label="Temperature", render=False
        ),
        gr.Slider(
            minimum=128,
            maximum=4096,
            step=1,
            value=200,
            label="Max new tokens",
            render=False,
        ),
    ],
    description=f"Now Running {args.model}",
    multimodal=True,
)

demo.launch(inbrowser=True)
