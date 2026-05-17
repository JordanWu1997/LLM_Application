#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Ollama server REST API Documentation
A comprehensive CLI for managing models and interacting with Ollama servers.
- https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import base64
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterator, List, Optional, Union

import ollama
import requests
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from ollama_utils import get_input_from_editor, print_context_warning

PERSONAS = {
    "coder":
    "You are an expert senior software engineer. Reply only with clean, well-commented, and highly optimized code. Avoid unnecessary conversational filler.",
    "concise":
    "You are an AI assistant optimized for speed and efficiency. Answer as briefly as possible. Use bullet points, bold text for emphasis, and skip all pleasantries.",
    "teacher":
    "You are a patient and encouraging professor. Explain complex topics using simple terms, relatable analogies, and step-by-step breakdowns.",
    "critic":
    "You are a harsh but mathematically logical critic. Analyze the user's prompt, point out logical flaws, edge cases, and areas for improvement. Do not sugarcoat your response.",
    "writer":
    "You are a creative author and copywriter. Use highly descriptive, engaging, evocative, and persuasive language.",
    "json":
    "You are a data-formatting engine. Output your response STRICTLY as valid JSON. Do not include markdown formatting like ```json, just the raw JSON object.",
    "pirate":
    "You are a swashbuckling pirate. Speak strictly in nautical terms, use a pirate accent, and be slightly aggressive. Arrr!",
    "terminal":
    "You are a Linux terminal. Output only the raw terminal output that would result from the user's command. Do not explain the commands."
}


class OllamaTokenizer:
    """
    A lightweight, automated tokenizer handler for Ollama models.
    Matches Ollama models to their HuggingFace equivalents to provide accurate token counts.
    """

    # Mapping Ollama families to lightweight HF tokenizer repos
    _MAP = {
        "llama": "meta-llama/Llama-3.1-8B",
        "qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "mistral": "mistralai/Mistral-7B-v0.1",
        "phi": "microsoft/phi-2",
        "gemma": "google/gemma-2b"
    }

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = self._load_auto_tokenizer(model_name)

    def _load_auto_tokenizer(self, model_name: str) -> Tokenizer:
        """Determines the correct tokenizer and loads it."""
        try:
            # Get model metadata from Ollama
            info = ollama.show(model_name)
            family = info.get('details', {}).get('family', '').lower()
        except Exception:
            family = model_name.lower()
        self.model_family = family

        # Match family or model name to our map
        hf_repo = "gpt2"  # Global fallback
        for key, repo in self._MAP.items():
            if key in family or key in model_name.lower():
                hf_repo = repo
                break

        try:
            # Only download the tiny configuration file (~2MB)
            file_path = hf_hub_download(repo_id=hf_repo,
                                        filename="tokenizer.json",
                                        local_files_only=False)
            return Tokenizer.from_file(file_path)
        except Exception as e:
            # Fallback to a generic BPE tokenizer if download fails
            return Tokenizer.from_pretrained("gpt2")

    def count(self, text: str) -> int:
        """Returns the number of tokens in a string."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text).ids)

    def get_stats(self, thinking_txt: str, content_txt: str):
        """Returns a dictionary with split token counts."""
        t_count = self.count(thinking_txt)
        c_count = self.count(content_txt)
        return {
            "thinking_tokens": t_count,
            "content_tokens": c_count,
            "total_tokens": t_count + c_count
        }


class OllamaStats():

    def __init__(self, context_window):
        self.context_window = context_window
        self.TTFT = 0
        self.prompt_TPS = 0
        self.generation_TPS = 0
        self.total_input_token = 0
        self.system_prompt_token = 0
        self.user_prompt_token = 0
        self.total_output_token = 0
        self.thinking_token = 0
        self.content_token = 0

    def reset_stats(self, context_window=None):
        if context_window is not None:
            self.context_window = context_window
        self.TTFT = 0
        self.prompt_TPS = 0
        self.generation_TPS = 0
        self.total_input_token = 0
        self.system_prompt_token = 0
        self.user_prompt_token = 0
        self.total_output_token = 0
        self.thinking_token = 0
        self.content_token = 0

    def update_context_window(self, context_window):
        self.context_window = context_window

    def display(self, session_stats):
        # Stats
        print("\n[Statistics]")
        print(f'- Performance')
        # TTFT
        print(f"  - First token latency (TTFT): "
              f"{self.TTFT:.2f} sec")
        # TPS
        print(f"  - Prompt prefilling: "
              f"{self.prompt_TPS:.1f} tokens/sec")
        print(f"  - Token generation: "
              f"{self.generation_TPS:.1f} tokens/sec")
        # Token stats
        print(f"- Token Usage")
        print(f"  - Input Tokens: {self.total_input_token}")
        print(f"    - Estimate System Prompt: {self.system_prompt_token}")
        print(f"    - Estimate User Prompt: {self.user_prompt_token}")
        print(f"  - Output Tokens: {self.total_output_token}")
        print(f"    - Estimate Thinking: {self.thinking_token}")
        print(f"    - Estimate Content: {self.content_token}")
        print(f"  - Total (Input + Output) Tokens: "
              f"{self.total_input_token + self.total_output_token}")
        print(f"- Context Window Usage")
        print(
            f"  - Used: "
            f"{self.total_input_token + self.total_output_token}/{self.context_window} "
            f"({(self.total_input_token + self.total_output_token)/self.context_window:.1%})"
        )


class OllamaClient:
    """Client for interacting with Ollama REST API."""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 11434,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 40,
                 num_ctx: int = 4096,
                 think: bool = False,
                 stream: bool = False,
                 keep_alive: str = "10m"):
        """ Initialize the Ollama client. """
        self.host = host
        self.port = port
        self.update_base_url()
        # General settings
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_ctx = num_ctx
        self.think = think
        self.stream = stream
        self.keep_alive = keep_alive
        self.stdout_lock = threading.Lock()
        self._tokenizer_cache: Dict[str, OllamaTokenizer] = {}

    def update_base_url(self):
        self.base_url = f"http://{self.host}:{self.port}/api"

    def get_tokenizer(self, model_name: str) -> OllamaTokenizer:
        """Returns a cached tokenizer or creates a new one."""
        if model_name not in self._tokenizer_cache:
            self._tokenizer_cache[model_name] = OllamaTokenizer(model_name)
        return self._tokenizer_cache[model_name]

    def list_models(self) -> List[Dict]:
        """
        List all available models.

        Returns:
            List of model information dictionaries
        """
        response = requests.get(f"{self.base_url}/tags")
        response.raise_for_status()
        return response.json()["models"]

    def list_running_models(self) -> List[Dict]:
        """
        List currently running models.

        Returns:
            List of running model information dictionaries
        """
        try:
            response = requests.get(f"{self.base_url}/ps")
            response.raise_for_status()
            return response.json()["models"]
        except requests.exceptions.HTTPError as e:
            # If the API doesn't support this endpoint directly
            # This is a fallback implementation
            print(
                "Warning: running models endpoint not supported, approximating from generate status"
            )
            # Return empty list as a fallback
            return []

    def show_model_info(self, model_name: str) -> Dict:
        """
        Fetches information about a specific model from the server.

        Args:
            model_name (str): The name of the model for which to fetch information.

        Returns:
            Dict: A dictionary containing details about the specified model.
        """
        response = requests.post(f"{self.base_url}/show",
                                 json={
                                     "model": model_name,
                                 })
        response.raise_for_status()
        return response.json()

    def load_model(self, model_name: str) -> Dict:
        """
        Load a model.

        Args:
            model_name: Name of the model to load

        Returns:
            Response dictionary with load status
        """
        try:
            stop_spinner = threading.Event()
            spinner_thread = threading.Thread(target=self._spinner_task,
                                              args=(stop_spinner,
                                                    self.stdout_lock))
            spinner_thread.start()
            response = requests.post(f"{self.base_url}/generate",
                                     json={
                                         "model": model_name,
                                         "keep_alive": self.keep_alive
                                     })
            response.raise_for_status()
            return response.json()
        except KeyboardInterrupt:
            # This is critical for stopping the request mid-load
            print(f"\n[INTERRUPT] Loading {model_name} aborted.")
            return {"status": "aborted"}
        finally:
            stop_spinner.set()
            spinner_thread.join(timeout=1)

    def unload_model(self, model_name: str) -> Dict:
        """
        Unload a running model.

        Args:
            model_name: Name of the model to unload

        Returns:
            Response dictionary with unload status
        """
        response = requests.post(f"{self.base_url}/generate",
                                 json={
                                     "model": model_name,
                                     "keep_alive": 0,
                                 })
        response.raise_for_status()
        return response.json()

    def pull_model(self, model_name: str):
        """ Pull a model from the Ollama registry """
        print(f"Pulling {model_name}... (This may take a while)")
        response = requests.post(f"{self.base_url}/pull",
                                 json={"model": model_name},
                                 stream=True)
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get("status", "")
                print(f"\r\033[K{status}", end="")  # \033[K clears the line
        print("\nDone!")

    def delete_model(self, model_name: str):
        """ Delete a model from disk """
        response = requests.delete(f"{self.base_url}/delete",
                                   json={"model": model_name})
        response.raise_for_status()
        print(f"Deleted {model_name}")

    def chat(
        self,
        model_name: str,
        prompt: str = '',
        image_paths: List[str] = [],
        file_paths: List[str] = [],
        messages: List[Dict[str, str]] = [],
        num_ctx: int = -1,
        think: bool = True,
    ) -> tuple[Union[Dict, Iterator], List[Dict[str, str]]]:
        """Sends a chat request to the model, supporting text, images, and file attachments.

        This method processes provided images and text files, appends the new user
        message to the existing conversation history, and communicates with the
        model API. It supports both streaming and non-streaming responses.

        Args:
            model_name (str): The name of the model to use for the chat session.
            prompt (str, optional): The initial text prompt. Defaults to empty string.
            image_paths (List[str], optional): A list of file paths to images.
                Images are automatically encoded to base64. Defaults to an empty list.
            file_paths (List[str], optional): A list of file paths to text files.
                The contents of these files will be appended to the prompt.
                Defaults to an empty list.
            messages (List[Dict[str, str]], optional): The existing conversation
                history. Note: This list is modified in-place to include the
                new user message. Defaults to an empty list.
            num_ctx (int, optional): The size of the context window. If -1,
                the class's default `num_ctx` is used. Defaults to -1.
            think (bool, optional): Whether to enable the model's reasoning/thinking
                process (useful for models like DeepSeek-R1). Defaults to True.

        Returns:
            tuple[Union[Dict, Iterator], List[Dict[str, str]]]: A tuple containing:
                - The model's response. This is a dictionary if `self.stream` is
                  False, or an iterator if `self.stream` is True.
                - The updated list of messages, including the newly appended
                  user message with encoded images.

        Side Effects:
            Modifies the `messages` list passed as an argument by appending
            the new user role message.
        """

        # Read image and encode to base64
        encoded_images = []
        for image_path in image_paths:
            with open(image_path, "rb") as img_file:
                encoded_images.append(
                    base64.b64encode(img_file.read()).decode("utf-8"))

        # Read text files and combine with prompt
        prompt = self._append_files_to_prompt(prompt, file_paths)

        # Add the message with image to messages list
        messages.append({
            "role": "user",
            "content": prompt,
            "images": encoded_images
        })

        # Generate payload
        payload = {
            "model": model_name,
            "messages": messages,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "num_ctx": num_ctx if num_ctx > 0 else self.num_ctx,
            },
            "think": think,
            "stream": self.stream,
        }
        if encoded_images != []:
            payload["images"] = encoded_images

        endpoint = "chat"
        if self.stream:
            return self._stream_response(endpoint, payload), messages
        else:
            response = requests.post(f"{self.base_url}/{endpoint}",
                                     json=payload)
            response.raise_for_status()
            return response.json(), messages

    def generate(self,
                 model_name: str,
                 prompt: str = '',
                 system: str = '',
                 image_paths: List[str] = [],
                 file_paths: List[str] = [],
                 num_ctx: int = -1) -> Union[Dict, Iterator]:
        """Generates a single completion response from the specified model.

        This method processes text files and images, constructs a payload including
        system instructions, and sends a request to the generation endpoint.
        It supports both streaming and non-streaming responses based on the
        class configuration.

        Args:
            model_name (str): The name of the model to use for generation.
            prompt (str, optional): The primary text prompt. Defaults to an empty string.
            system (str, optional): System-level instructions to guide the model's
                behavior. Defaults to an empty string.
            image_paths (List[str], optional): A list of file paths to images.
                Images are automatically encoded to base64. Defaults to an empty list.
            file_paths (List[str], optional): A list of file paths to text files.
                The contents of these files will be appended to the prompt.
                Defaults to an empty list.
            num_ctx (int, optional): The size of the context window. If -1,
                the class's default `num_ctx` is used. Defaults to -1.

        Returns:
            Union[Dict, Iterator]: The model's response. Returns a dictionary
                if `self.stream` is False, or an iterator if `self.stream` is True.

        Raises:
            requests.exceptions.HTTPError: If the API request returns an error status.

        """

        # Read image and encode to base64
        encoded_images = []
        for image_path in image_paths:
            with open(image_path, "rb") as img_file:
                encoded_images.append(
                    base64.b64encode(img_file.read()).decode("utf-8"))

        # Read text files and combine with prompt
        prompt = self._append_files_to_prompt(prompt, file_paths)

        # Prepare payload
        payload = {
            "model": model_name,
            "prompt": prompt,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "num_ctx": num_ctx if num_ctx > 0 else self.num_ctx,
            },
            "stream": self.stream
        }

        # Inject system prompt into payload if it exists
        if system:
            payload["system"] = system

        # Add images if they exist
        if encoded_images:
            payload["images"] = encoded_images

        # Send request
        endpoint = "generate"
        if self.stream:
            return self._stream_response(endpoint, payload)
        else:
            response = requests.post(f"{self.base_url}/{endpoint}",
                                     json=payload)
            response.raise_for_status()
            return response.json()

    def _stream_response(self, endpoint, payload: Dict):
        """
        Stream chat responses.

        Args:
            payload: Request payload dictionary

        Yields:
            Streamed response chunks
        """
        with requests.post(f"{self.base_url}/{endpoint}",
                           json=payload,
                           stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield {"error": "Failed to parse streaming response"}

    @staticmethod
    def _append_files_to_prompt(prompt, file_paths):
        """Appends the contents of text files to a prompt using XML-style delimiters.

        This method reads the content of each file in `file_paths`, wraps the content
        in `<document>` tags with metadata, and encapsulates all documents within
        a `<context>` block appended to the original prompt. It uses `os.path.expanduser`
        to support tilde (`~`) paths.

        Args:
            prompt (str): The original base prompt.
            file_paths (List[str]): A list of file paths to be appended.
                Supports absolute paths and paths with '~'.

        Returns:
            str: The augmented prompt. If no files are provided or all files
                fail to load, the original prompt is returned.

        Note:
            The resulting prompt structure follows this pattern:
            <original_prompt>

            <context>
            <document index="1" path="path/to/file.txt">
            [file_content]
            </document>
            ...
            </context>
            </context>

        Error Handling:
            - Skips files that are not UTF-8 encoded (binary files).
            - Skips files where permission is denied.
            - Skips files that do not exist or encounter other I/O errors.
            - Prints warnings/errors to the console for skipped files.
        """

        # If no files, return the original prompt
        if not file_paths:
            return prompt

        context_blocks = []

        # Read files and format them in a single pass
        for index, file_path in enumerate(file_paths, start=1):
            expanded_path = os.path.expanduser(file_path)
            try:
                with open(expanded_path, "r", encoding='utf-8') as file:
                    content = file.read()

                # Use XML tags for clear LLM parsing
                block = (f'<document index="{index}" path="{file_path}">\n'
                         f'{content}\n'
                         f'</document>')
                context_blocks.append(block)

            except UnicodeDecodeError:
                print(
                    f"\033[91m[WARNING] Skipped {file_path}: File is binary or not UTF-8 text.\033[0m"
                )
                continue

            except PermissionError:
                print(f"\033[91m[ERROR] Permission denied: {file_path}\033[0m")

            except Exception as e:
                # Good practice: handle files that don't exist or have encoding errors
                print(f"Warning: Skipped file {file_path} due to error: {e}")

        # Combine all blocks and append to the prompt
        if context_blocks:
            all_documents = "\n\n".join(context_blocks)

            # Structure: Base Prompt -> Context Wrapper -> Documents
            prompt += f"\n\n<context>\n{all_documents}\n</context>"

        return prompt

    @staticmethod
    def _calculate_tokens_per_second(response: Dict) -> Optional[float]:
        """
        Calculate tokens per second from the response metadata.

        Args:
            response: Response dictionary from Ollama API

        Returns:
            Tokens per second (TPS) as a float, or None if data is not available
        """
        try:
            # Extract timing info from the response
            eval_count = response.get("eval_count")
            eval_duration = response.get("eval_duration")

            # Check if we have the necessary information
            if eval_count is None or eval_duration is None or eval_duration == 0:
                return None

            # Convert nanoseconds to seconds if necessary
            # Ollama typically returns eval_duration in nanoseconds
            duration_in_seconds = eval_duration / 1_000_000_000

            # Calculate tokens per second
            tokens_per_second = eval_count / duration_in_seconds

            return tokens_per_second

        except (KeyError, TypeError, ZeroDivisionError):
            return None

    @staticmethod
    def _spinner_task(stop_event, lock: threading.Lock):
        """A background thread task that displays a visual loading spinner.

        This method runs in a loop, cycling through a set of characters ('\\', '|', '/')
        to indicate an ongoing process. It is designed to be run in a separate thread.
        When the `stop_event` is set, the loop terminates and the method cleans up
        the terminal line to prevent visual artifacts.

        Args:
            stop_event (threading.Event): An event object used to signal the
                task to stop. The loop continues as long as `is_set()` is False.
            lock (threading.Lock): A thread lock used to synchronize access to
                `sys.stdout`. This prevents the spinner from overwriting text
                being printed by other threads (e.g., the typing effect).

        Note:
            Upon termination, this method clears the current line in the terminal
            by overwriting the spinner with spaces.
        """

        spinner = ['\\', '|', '/']
        i = 0
        while not stop_event.is_set():
            with lock:
                sys.stdout.write(f'\r{spinner[i % len(spinner)]}')
                sys.stdout.flush()
            time.sleep(0.1)
            i += 1

        # sys.stdout.write('\rDone')
        with lock:
            sys.stdout.write('\r' + ' ' * 10 + '\r')
            sys.stdout.flush()

    @staticmethod
    def _print_typing_effect(text, lock: threading.Lock):
        """Prints text to the console with a typewriter-style animation.

        Iterates through the provided string character by character, printing
        each one with a slight delay to simulate a typing effect.

        Args:
            text (str): The string of text to be printed to the console.
            lock (threading.Lock): A thread lock used to synchronize access to
                `sys.stdout`. This ensures that the typing effect does not
                interleave with or get corrupted by the `_spinner_task`
                overwriting the same line.

        Side Effects:
            Modifies `sys.stdout` and introduces a small delay (0.005s) per character.
        """
        for char in text:
            with lock:
                sys.stdout.write(char)
                sys.stdout.flush()
            time.sleep(0.005)


def list_all_models(client, with_index=True, with_info=True):
    """
    List all models faster using parallel threads for metadata fetching.
    """
    models = []

    try:
        models = client.list_models()
        if not models:
            print("\nNo models found.")
            return []

        print(f"\nFetching info for {len(models)} models...")

        # This list will store our results in the correct order
        results = [None] * len(models)

        def fetch_info(index, model):
            """Worker function to fetch info for a single model."""
            name = model['name']
            size_raw = model.get('size', 0)
            size_gb = f"{float(size_raw) / (1024**3):.2f}" if size_raw else "N/A"

            info_str = ""
            if with_info:
                # Parallel API call
                model_info = client.show_model_info(name)
                capability = model_info.get('capabilities', 'N/A')
                params = model_info['details'].get('parameter_size', 'N/A')
                info_str = f"{capability}, Param: {params}, "

            prefix = f"{index + 1:02d}. " if with_index else "- "
            return index, f"{prefix}{name} ({info_str}Size: {size_gb} GB)"

        # Use ThreadPoolExecutor to fetch info in parallel
        # max_workers=10 is usually a sweet spot for local Ollama servers
        try:
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_model = {
                    executor.submit(fetch_info, i, m): i
                    for i, m in enumerate(models)
                }

                for future in as_completed(future_to_model):
                    idx, text = future.result()
                    results[idx] = text
        except KeyboardInterrupt:
            # Handle Ctrl+C during parallel execution
            print("\n[INTERRUPT] Cancelling data fetch...")
            executor.shutdown(wait=False, cancel_futures=True)
            return models

        # Print final ordered list
        print("\nAvailable models:")
        for line in results:
            if line:
                print(line)
                time.sleep(0.05)

    except Exception as e:
        print(f"Error: {e}")

    return models


def list_running_models(client, with_index=True):
    """
    List all the running models on a specified platform.

    Args:
        with_index (bool): If True, include model indices in the output. Default is True.

    Returns:
        list: A list of dictionaries containing information about each running model.
        Each dictionary includes 'name', 'capability', and 'size' fields.
    """
    models = []

    try:
        models = client.list_running_models()
        if models:
            print("\nRunning models:")
            for i, model in enumerate(models):

                model_size = model.get('size', 'N/A')
                if model_size != 'N/A':
                    model_size = float(model_size) / 1024 / 1024 / 1024
                    model_size = f'{model_size:.2f}'
                size_vram = model.get('size_vram', 'N/A')
                if size_vram != 'N/A':
                    size_vram = float(size_vram) / 1024 / 1024 / 1024
                    size_vram = f'{size_vram:.2f}'
                num_ctx = model.get('context_length', 'N/A')
                if num_ctx != 'N/A':
                    num_ctx = int(num_ctx)
                expires_at = model.get('expires_at', 'N/A')

                # Model info
                model_info = client.show_model_info(model['name'])
                model_capability = model_info.get('capabilities', 'N/A')
                model_params = model_info['details'].get(
                    'parameter_size', 'N/A')
                model_quant = model_info['details'].get(
                    'quantization_level', 'N/A')

                # Print
                if with_index:
                    print(
                        f"{i+1:02d}. {model['name']} ({model_capability}, Param: {model_params}, Size: {model_size} GB, vRAM: {size_vram} GB, CTX_LEN: {num_ctx}, Expires: {expires_at})"
                    )
                else:
                    print(
                        f"- {model['name']} ({model_capability}, Param: {model_params}, Size: {model_size} GB, vRAM: {size_vram} GB, CTX_LEN: {num_ctx}, Expires: {expires_at})"
                    )

        else:
            print("\nNo models currently running.")

    except Exception as e:
        print(f"Error: {e}")

    return models


def load_model(client):
    """
    Load a model from the available models list.

    This function lists all available models, prompts the user to enter a model name or index,
    converts the input into the corresponding model name if it's an integer, and then loads the model.
    It handles exceptions such as invalid inputs or errors during loading.

    Returns:
        str: The name of the loaded model.
    """

    # List all available models
    all_models = list_all_models(client)
    # Enter model name to load
    model_name = input("\nEnter model name/index to load: ")
    # Convert index to model name
    try:
        model_index = int(model_name)
        model_name = all_models[model_index - 1]['name']
    except ValueError:
        pass
    except IndexError:
        pass
    # Load model
    try:
        print(f"\n[INFO] Loading model {model_name}...")
        response = client.load_model(model_name)
        if response.get("status") != "aborted":
            print(f"Model loaded successfully: {response}")
        else:
            print(f"Model is not loaded: {response}")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")


def unload_model(client):
    """
    Unloads a specified model from the system.

    This function lists all available models, prompts the user to enter the name or index of the model they want to unload,
    converts the input into the corresponding model name if necessary, and then unloads the model using the client's API.
    It handles potential errors such as invalid inputs or failures during the unloading process.

    Returns:
        None
    """

    # List all available models
    all_models = list_running_models(client)
    # Enter model name/index to unload
    model_name = input("\nEnter model name/index to unload: ")
    # Convert index to model name
    try:
        model_index = int(model_name)
        model_name = all_models[model_index - 1]['name']
    except ValueError:
        pass
    except IndexError:
        pass
    # Unload model
    try:
        print(f"\n[INFO] Unloading model {model_name}...")
        response = client.unload_model(model_name)
        print(f"[INFO] Model unloaded successfully: {response}")
    except Exception as e:
        print(f"[ERROR] Error unloading model: {e}")


def handle_common_commands(user_input: str, client, current_system: str = ""):
    """
    Helper function to parse and handle slash commands shared between chat and generate modes.
    Returns a dictionary dictating the next action for the session loop.
    """

    parts = user_input.strip().split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == '/exit':
        return {"status": "exit"}

    elif cmd == '/help':
        return {"status": "help"}

    elif cmd == '/edit':
        print(
            f"\033[90m[Opening {os.environ.get('EDITOR', 'vim')}... Save and quit to submit]\033[0m"
        )
        edited_text = get_input_from_editor(initial_text=arg)
        if not edited_text:
            print("\033[90m[Empty input, cancelled]\033[0m")
            return {"status": "continue"}
        print(
            f"\033[90m[Captured {len(edited_text)} characters from editor]\033[0m"
        )
        print(f'>>> You (via Editor):\n{edited_text}')
        return {"status": "ready", "prompt": edited_text}

    elif cmd == '/system':
        # If user provides inline text, use it. Otherwise, open the editor pre-filled with current system prompt.
        if arg:
            sys_content = arg
        else:
            print(
                f"\033[90m[Opening {os.environ.get('EDITOR', 'vim')} to edit system prompt...]\033[0m"
            )
            sys_content = get_input_from_editor(initial_text=current_system)

        if sys_content:
            print(f">>> System Prompt Updated:\n{sys_content}")
            return {"status": "update_system", "content": sys_content}
        else:
            print("[INFO] System prompt cleared.")
            return {"status": "update_system", "content": ""}

    elif cmd == '/persona':
        if arg:
            persona_name = arg.lower()
            if persona_name in PERSONAS:
                sys_content = PERSONAS[persona_name]
                print(f"\033[92m[INFO] Loaded persona: {persona_name}\033[0m")
                print(f">>> System Prompt Updated:\n{sys_content}")
                return {"status": "update_system", "content": sys_content}
            else:
                print(
                    f"\033[91m[ERROR] Persona '{persona_name}' not found.\033[0m"
                )
                print(f"Available personas: {', '.join(PERSONAS.keys())}")
                return {"status": "continue"}
        else:
            # If they just type /persona without a name, list the available options
            print("\n\033[96m=== Available Personas ===\033[0m")
            for name, desc in PERSONAS.items():
                # Print the name and a short preview of the prompt
                preview = desc[:65] + "..." if len(desc) > 65 else desc
                print(f"- \033[93m{name:<10}\033[0m : {preview}")
            return {"status": "continue"}

    elif cmd == '/ctx':
        if arg:
            client.num_ctx = int(arg) if arg.isdigit() else client.num_ctx
            print(f"[INFO] Context length updated to {client.num_ctx}")
        else:
            print(f"[INFO] Current context length: {client.num_ctx}")
        return {"status": "continue"}

    elif cmd == '/keepalive':
        if arg:
            client.keep_alive = int(arg) if arg.isdigit() else arg
            print(f"[INFO] Keep-alive updated to {client.keep_alive}")
        else:
            print(f"[INFO] Current keep-alive: {client.keep_alive}")
        return {"status": "continue"}

    elif cmd in ['/image', '/file']:
        paths = []
        item_type = "image" if cmd == '/image' else "file"

        # Improved UX: Just press Enter on an empty line to finish adding paths
        print(
            f"\n[Enter {item_type} paths. Leave empty and press Enter to finish]"
        )
        while True:
            path = input(f"{item_type.capitalize()} path: ").strip()
            if not path:
                break
            paths.append(path)

        new_prompt = input(f"\n>>> You (Prompt for {item_type}s): \n").strip()
        return {
            "status": "ready",
            "prompt": new_prompt,
            "image_paths": paths if cmd == '/image' else [],
            "file_paths": paths if cmd == '/file' else []
        }

    # If it's not a common command, pass it back so the specific mode can handle it or treat it as text
    return {
        "status": "unhandled",
        "cmd": cmd,
        "arg": arg,
        "prompt": user_input
    }


def generate_completion_with_model(client, running_only=False):
    # Get available models
    all_models = list_running_models(client, with_index=running_only)
    if not running_only or all_models == []:
        all_models = list_all_models(client)

    if len(all_models) == 1:
        model_name = all_models[0]['name']
    else:
        model_name = input("\nEnter model name to generate_completion with: ")
        try:
            model_index = int(model_name)
            model_name = all_models[model_index - 1]['name']
        except (ValueError, IndexError):
            pass
        if model_name not in [model['name'] for model in all_models]:
            print(f'[ERROR] Not a valid model name: {model_name}')
            return

    history = ""
    system_prompt = ""
    session_stats = OllamaStats(client.num_ctx)

    # Define the help menu
    generate_help_menu = f"""
        \033[96m=== Generate completion with {model_name} ===\033[0m

        \033[93mCore Commands:\033[0m
        - /help             : Show this menu
        - /exit             : Exit the session
        - /edit             : Open editor (Vim/Nano) to write a multi-line prompt
        - /system [text]    : Update system prompt (leave blank to open editor)
        - /persona [name]   : Load a preset system prompt (leave blank to list all)

        \033[93mHistory & Flow:\033[0m
        - /continue (or /c) : Continue generation (opens editor to review/append)
        - /history (or /h)  : View and edit full generation history in editor
        - /clear            : Clear current history completely
        - /save [file]      : Save history to Markdown (default: generation_history.md)

        \033[93mModel Settings:\033[0m
        - /ctx [number]     : View or change context length (e.g., /ctx 8192)
        - /keepalive [time] : View or change model keep-alive time (e.g., 30m)

        \033[93mAttachments:\033[0m
        - /image            : Load image paths for Vision Models
        - /file             : Load text file paths to include in prompt
        """
    print(generate_help_menu)  # Print once on startup

    # Load model
    start_time = time.time()
    _ = client.load_model(model_name)
    print(f'\nLoading {model_name} took {time.time() - start_time:.3f} sec')

    # Load tokenizer
    start_time = time.time()
    tok = client.get_tokenizer(model_name)
    print(
        f'Loading {tok.model_family} tokenizer took {time.time() - start_time:.3f} sec'
    )

    while True:
        image_paths, file_paths = [], []
        client.ctx_window_used_token = 0

        user_input = input("\n>>> You: \n").strip()
        if not user_input:
            continue

        # Parse Commands
        cmd_res = handle_common_commands(user_input, client)

        if cmd_res["status"] == "exit":
            break

        elif cmd_res["status"] == "help":
            print(generate_help_menu)
            continue

        elif cmd_res["status"] == "update_system":
            system_prompt = cmd_res["content"]
            continue

        elif cmd_res["status"] == "continue":
            continue

        elif cmd_res["status"] == "ready":
            user_input = cmd_res["prompt"]
            image_paths = cmd_res.get("image_paths", [])
            file_paths = cmd_res.get("file_paths", [])

        elif cmd_res["status"] == "unhandled":
            cmd, arg = cmd_res["cmd"], cmd_res["arg"]

            if cmd in ['/history', '/h']:
                print(
                    f"\033[90m[Opening {os.environ.get('EDITOR', 'vim')} to view/edit history...]\033[0m"
                )
                edited_history = get_input_from_editor(initial_text=history)

                # Update history if the user saved changes
                if edited_history != history:
                    history = edited_history
                    print(f">>> History Updated:\n{history}")
                else:
                    print("\033[90m[History unchanged]\033[0m")
                continue

            elif cmd == '/clear':
                history = ""
                print('\n[INFO] History cleared.')
                continue

            elif cmd in ['/c', '/cont', '/continue']:
                print(
                    f"\033[90m[Opening {os.environ.get('EDITOR', 'vim')} to continue generation...]\033[0m"
                )

                # Pre-fill the editor with the current history, plus a couple of newlines
                prefill = f"{history}\n\n" if history else ""
                edited_text = get_input_from_editor(initial_text=prefill)

                if not edited_text:
                    print("\033[90m[Empty input, cancelled]\033[0m")
                    continue

                print(f">>> You (Continued):\n{edited_text}")
                user_input = edited_text

            elif cmd == '/save':
                filename = arg if arg else "generation_history.md"
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(history)
                    print(
                        f"\033[92m[INFO] Generation saved successfully to {filename}\033[0m"
                    )
                except Exception as e:
                    print(
                        f"\033[91m[ERROR] Could not save history: {e}\033[0m")
                continue

            else:
                user_input = cmd_res["prompt"]

        if not user_input and not image_paths and not file_paths:
            continue

        print(f"\n<<< Model ({model_name}): ")
        stop_spinner = threading.Event()
        spinner_thread = threading.Thread(target=client._spinner_task,
                                          args=(stop_spinner,
                                                client.stdout_lock))
        spinner_thread.start()

        start_time = time.time()
        first_token_time, last_token_time = None, None
        full_response, first_token_latency = "", None
        try:
            # Send request to server
            response = client.generate(model_name,
                                       prompt=user_input,
                                       num_ctx=client.num_ctx)

            # Get first full token to measure first token time
            first_token_received = False
            if client.stream:
                # Decode response
                for chunk in response:
                    if "response" in chunk:
                        content = chunk["response"]
                        # Record time of first token
                        if not first_token_received:
                            first_token_time = time.time()
                            first_token_received = True
                            # Stop the spinner
                            stop_spinner.set()
                            spinner_thread.join(timeout=1)
                        # Record time of latest token
                        last_token_time = time.time()
                        # Display with typing effect
                        client._print_typing_effect(content,
                                                    client.stdout_lock)
                        full_response += content

                    # Capture final statistics from the last chunk
                    if chunk.get("done"):
                        metadata = chunk

                # Calculate TTFT
                if first_token_time is not None:
                    first_token_latency = first_token_time - start_time
                print()

            else:
                metadata = response
                first_token_latency = -1
                full_response = response.get("response", "No response")
                print(full_response)

                # Stop the spinner
                stop_spinner.set()
                spinner_thread.join(timeout=1)

            # Content window warning
            print_context_warning(
                prompt_tokens=metadata.get("prompt_eval_count", 0),
                ctx_window=client.num_ctx,
            )

            # Stats
            session_stats.TTFT = first_token_latency
            prompt_token = metadata.get('prompt_eval_count', 0)
            generation_token = metadata.get('eval_count', 0)
            session_stats.total_input_token = prompt_token
            session_stats.total_output_token = generation_token
            session_stats.system_prompt_token = tok.count(system_prompt)
            session_stats.user_prompt_token = tok.count(user_input)
            session_stats.think_token = 0  # tok.count(thinking_response)
            session_stats.content_token = 0  #tok.count(content_response)
            session_stats.prompt_TPS = \
                prompt_token / metadata.get('prompt_eval_duration', -1) * 1e9
            session_stats.generation_TPS = \
                generation_token / metadata.get('eval_duration', -1) * 1e9
            session_stats.display(client.num_ctx)

        except KeyboardInterrupt:
            stop_spinner.set()  # Stop the spinner thread
            spinner_thread.join(timeout=1)
            print("\n\n[INTERRUPTED] Stopping generation...")
            continue

        # Handle exceptions
        except Exception as e:
            print(f"\n[ERROR] Error during chat: {e}")

        # Add generated response to history
        else:
            history = f'{user_input} {full_response}'


def chat_with_model(client, running_only=False):

    # Get available models
    all_models = list_running_models(client, with_index=running_only)
    if not running_only or all_models == []:
        all_models = list_all_models(client)

    if len(all_models) == 1:
        model_name = all_models[0]['name']
    else:
        model_name = input("\nEnter model name to chat with: ")
        try:
            model_index = int(model_name)
            model_name = all_models[model_index - 1]['name']
        except (ValueError, IndexError):
            pass
        if model_name not in [model['name'] for model in all_models]:
            print(f'[ERROR] Not a valid model name: {model_name}')
            return

    # Define the help menu
    chat_help_menu = f"""
        \033[96m=== Chat with {model_name} ===\033[0m

        \033[93mCore Commands:\033[0m
        - /help             : Show this menu
        - /exit             : Exit the chat session
        - /edit             : Open editor (Vim/Nano) to write a multi-line prompt
        - /system [text]    : Update system prompt (leave blank to open editor)
        - /persona [name]   : Load a preset system prompt (leave blank to list all)
        - /save [file]      : Save chat to JSON (default: chat_history.json)
        - /load [file]      : Load chat from JSON (default: chat_history.json)

        \033[93mModel Settings:\033[0m
        - /think [on|off]   : Toggle reasoning/thinking mode (for supported models)
        - /ctx [number]     : View or change context length (e.g., /ctx 8192)
        - /keepalive [time] : View or change model keep-alive time (e.g., 30m)

        \033[93mAttachments:\033[0m
        - /image            : Load image paths for Vision Models
        - /file             : Load text file paths to include in prompt
        """
    print(chat_help_menu)  # Print once on startup

    # Load model
    start_time = time.time()
    _ = client.load_model(model_name)
    print(f'\nLoading {model_name} took {time.time() - start_time:.3f} sec')

    # Load tokenizer
    start_time = time.time()
    tok = client.get_tokenizer(model_name)
    print(
        f'Loading {tok.model_family} tokenizer took {time.time() - start_time:.3f} sec'
    )

    # Init stats
    session_stats = OllamaStats(client.num_ctx)

    messages = []
    while True:

        # --- 1. Parse user input ---
        image_paths, file_paths = [], []
        user_input = input("\n>>> You: \n").strip()
        if not user_input:
            continue

        # Extract current system prompt from messages array
        current_sys = messages[0]["content"] \
            if messages and messages[0].get( "role") == "system" else ""

        # Pass it to the handler
        cmd_res = handle_common_commands(user_input,
                                         client,
                                         current_system=current_sys)

        if cmd_res["status"] == "exit":
            break

        elif cmd_res["status"] == "help":
            print(chat_help_menu)
            continue

        elif cmd_res["status"] == "update_system":
            sys_content = cmd_res["content"]
            # Manage system prompt at index 0 of messages array
            if messages and messages[0].get("role") == "system":
                if sys_content: messages[0]["content"] = sys_content
                else: messages.pop(0)  # Clear if empty
            elif sys_content:
                messages.insert(0, {"role": "system", "content": sys_content})
            continue

        elif cmd_res["status"] == "continue":
            continue

        elif cmd_res["status"] == "ready":
            user_input = cmd_res["prompt"]
            image_paths = cmd_res.get("image_paths", [])
            file_paths = cmd_res.get("file_paths", [])

        elif cmd_res["status"] == "unhandled":
            cmd, arg = cmd_res["cmd"], cmd_res["arg"]
            # Handle Chat-Specific commands
            if cmd == '/think':
                if arg.lower() in ['on', 'true']: client.think = True
                elif arg.lower() in ['off', 'false']: client.think = False
                else: client.think = arg
                print(f"[INFO] Think mode updated to {client.think}")
                continue

            elif cmd == '/save':
                filename = arg if arg else "chat_history.json"
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(messages, f, indent=2, ensure_ascii=False)
                    print(
                        f"\033[92m[INFO] Chat saved successfully to {filename}\033[0m"
                    )
                except Exception as e:
                    print(f"\033[91m[ERROR] Could not save chat: {e}\033[0m")
                continue

            elif cmd == '/load':
                filename = arg if arg else "chat_history.json"
                try:
                    if os.path.exists(filename):
                        with open(filename, 'r', encoding='utf-8') as f:
                            messages = json.load(f)
                        print(
                            f"\033[92m[INFO] Chat loaded successfully from {filename} ({len(messages)} messages)\033[0m"
                        )
                        # Show the loaded message
                        for message in messages:
                            role = message['role']
                            content = message['content']
                            if role == "user":
                                print(f'\n>>> You (Loaded):\n{content}')
                            elif role == 'assistant':
                                print(
                                    f"\n<<< Model ({model_name}) (Loaded):\n{content}"
                                )
                            else:
                                print(f"\n {role} (Loaded):\n{content}")
                    else:
                        print(
                            f"\033[91m[ERROR] File {filename} does not exist.\033[0m"
                        )
                except Exception as e:
                    print(f"\033[91m[ERROR] Could not load chat: {e}\033[0m")
                continue

            else:
                user_input = cmd_res["prompt"]  # Normal text

        # Failsafe: if input is empty after command processing, skip
        if not user_input and not image_paths and not file_paths:
            continue

        # --- 2. API Call & Rendering ---
        print(f"\n<<< Model ({model_name}): ")

        # Track full response and performance data
        start_time = time.time()
        full_response, token_count = "", 0
        thinking_response, content_response = "", ""
        first_token_time, last_token_time = None, None
        first_token_latency = None
        is_generating = False

        # Tracking states for formatting
        currently_thinking = False
        content_started = False

        # Spinner for loading animation
        stop_spinner = threading.Event()
        spinner_thread = threading.Thread(target=client._spinner_task,
                                          args=(stop_spinner,
                                                client.stdout_lock))
        spinner_thread.start()

        try:
            # Send request to server
            response, messages = client.chat(model_name,
                                             prompt=user_input,
                                             messages=messages,
                                             image_paths=image_paths,
                                             file_paths=file_paths,
                                             think=client.think,
                                             num_ctx=client.num_ctx)
            is_generating = True

            # Get first full token to measure first token time
            first_token_received = False
            if client.stream:
                # Decode response
                for chunk in response:
                    if "message" in chunk and chunk["message"].get("thinking"):
                        thinking = chunk["message"]["thinking"]
                        # Record time of first token
                        if not first_token_received:
                            first_token_time = time.time()
                            first_token_received = True
                            # Stop the spinner
                            stop_spinner.set()
                            spinner_thread.join(timeout=1)

                            # Print Thinking Block Header
                            print(f"\n\033[90m[THINKING]\033[0m")
                            currently_thinking = True

                        # Record time of latest token
                        last_token_time = time.time()
                        # Display with typing effect
                        client._print_typing_effect(thinking,
                                                    client.stdout_lock)
                        full_response += thinking
                        thinking_response += thinking

                    if "message" in chunk and chunk["message"].get("content"):
                        content = chunk["message"]["content"]

                        # If we were just thinking, close the block before printing content
                        if currently_thinking:
                            print(f"\n\033[90m[END THOUGHTS]\033[0m\n")
                            currently_thinking = False

                        # Record time of first token
                        if not first_token_received:
                            first_token_time = time.time()
                            first_token_received = True
                            # Stop the spinner
                            stop_spinner.set()
                            spinner_thread.join(timeout=1)

                        # Record time of latest token
                        last_token_time = time.time()
                        # Display with typing effect
                        client._print_typing_effect(content,
                                                    client.stdout_lock)
                        full_response += content
                        content_response += content

                    # Track token info for TPS calculation
                    if "eval_count" in chunk:
                        token_count = chunk["eval_count"]

                    # Capture final statistics from the last chunk
                    if chunk.get("done"):
                        metadata = chunk

                        # Ensure we close thinking block if model finished without content
                        if currently_thinking:
                            print(f"\n\033[90m[END THOUGHTS]\033[0m\n")

                if first_token_time is not None:
                    first_token_latency = first_token_time - start_time
                print()

            else:
                metadata = response
                first_token_latency = -1
                full_response = response.get("message",
                                             {}).get("content", "No response")
                content_response = full_response
                print(full_response)

                # Stop the spinner
                stop_spinner.set()
                spinner_thread.join(timeout=1)

            # Content window warning
            print_context_warning(
                prompt_tokens=metadata.get("prompt_eval_count", 0),
                ctx_window=client.num_ctx,
            )

            # Stats
            session_stats.TTFT = first_token_latency
            prompt_token = metadata.get('prompt_eval_count', 0)
            generation_token = metadata.get('eval_count', 0)
            session_stats.total_input_token = prompt_token
            session_stats.total_output_token = generation_token
            session_stats.system_prompt_token = tok.count(current_sys)
            session_stats.user_prompt_token = tok.count(user_input)
            session_stats.think_token = tok.count(thinking_response)
            session_stats.content_token = tok.count(content_response)
            session_stats.prompt_TPS = \
                prompt_token / metadata.get('prompt_eval_duration', -1) * 1e9
            session_stats.generation_TPS = \
                generation_token / metadata.get('eval_duration', -1) * 1e9
            session_stats.display(client.num_ctx)

            # Add assistant response to messages for context
            if content_response != "":
                messages.append({
                    "role": "assistant",
                    "content": content_response  # Thinking excluded
                    #"content": full_response # Thinking included
                })

        except KeyboardInterrupt:
            # Remove user input if interrupted during output generation
            if is_generating:
                messages.pop()
            stop_spinner.set()  # Stop the spinner thread
            spinner_thread.join(timeout=1)
            print("\n\n[INTERRUPTED] Stopping generation...")
            continue

        # Handle exceptions
        except Exception as e:
            print(f"\n[ERROR] Error during chat: {e}")


def main_menu(client):

    while True:
        # We access host from the client object so it reflects changes made in config_menu
        print(f"\n=== Ollama API Client for [{client.host}:{client.port}] ===")
        print("0. Exit")
        print("1. Interactive Sessions")
        print("2. Model Management")
        print("3. Server Configuration (Host/Port)")

        choice = input("\nSelect Category (0-3): ")

        if choice == "0":
            print("Exiting program. Goodbye!")
            break
        elif choice == "1":
            interaction_menu(client)
        elif choice == "2":
            management_menu(client)
        elif choice == "3":
            config_menu(client)
        else:
            print("Invalid choice.")


def config_menu(client):

    while True:
        print("\n--- Server & Generation Configuration ---")
        print(f"1. Host:         {client.host}")
        print(f"2. Port:         {client.port}")
        print(f"3. Context Win:  {client.num_ctx}")
        print(f"4. Top_k:        {client.top_k}")
        print(f"5. Top_p:        {client.top_p}")
        print(f"6. Temperature:  {client.temperature}")
        print(f"7. Stream Mode:  {'ON' if client.stream else 'OFF'}")
        print(f"8. Keep Alive:   {client.keep_alive}")
        print("------------------------------------------")
        print("0. Back to Main Menu")

        choice = input("\nSelect setting to change (0-8): ")

        if choice == "0":
            break
        elif choice == "1":
            client.host = input(f"Enter host [{client.host}]: ") or client.host
            client.update_base_url()
        elif choice == "2":
            val = input(f"Enter port [{client.port}]: ")
            if val.isdigit():
                client.port = int(val)
                client.update_base_url()
        elif choice == "3":
            val = input(f"Enter context window [{client.num_ctx}]: ")
            if val.isdigit(): client.num_ctx = int(val)
        elif choice == "4":
            val = input(f"Enter top_k [{client.top_k}]: ")
            if val.isdigit(): client.top_k = int(val)
        elif choice == "5":
            val = input(f"Enter top_p [{client.top_p}]: ")
            if val:
                try:
                    client.top_p = float(val)
                except ValueError:
                    pass
        elif choice == "6":
            val = input(f"Enter temperature [{client.temperature}]: ")
            if val:
                try:
                    client.temperature = float(val)
                except ValueError:
                    pass
        elif choice == "7":
            client.stream = not client.stream
            print(f"Stream mode toggled to: {client.stream}")
        elif choice == "8":
            val = input(
                f"Enter keep_alive (e.g., 10m, 1h, 0, -1) [{client.keep_alive}]: "
            )
            if val:
                try:
                    client.keep_alive = int(val)
                except ValueError:
                    client.keep_alive = val


def management_menu(client):

    while True:
        print("\n--- Model Management ---")
        print("0. Back to Main Menu")
        print("1. List Currently Running Models")
        print("2. List All Available Models")
        print("3. Load Model into Memory")
        print("4. Unload Model (Free VRAM)")

        choice = input("\nChoice: ")
        if choice == "0": break
        elif choice == "1": list_running_models(client)
        elif choice == "2": list_all_models(client)
        elif choice == "3": load_model(client)
        elif choice == "4": unload_model(client)


def interaction_menu(client):

    while True:
        print("\n--- Interactive Sessions ---")
        print("0. Back to Main Menu")
        print("1. Chat (Select from RUNNING models only)")
        print("2. Chat (Select from ALL models)")
        print("3. Generate (Select from RUNNING models only)")
        print("4. Generate (Select from ALL models)")

        choice = input("\nChoice: ")
        if choice == "0": break
        elif choice == "1":
            chat_with_model(client, running_only=True)
        elif choice == "2":
            chat_with_model(client)
        elif choice == "3":
            generate_completion_with_model(client, running_only=True)
        elif choice == "4":
            generate_completion_with_model(client)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Ollama API Client")
    # Connection args
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help=
        "The hostname or IP address of the Ollama server (default: localhost)")
    parser.add_argument(
        "--port",
        type=int,
        default=11434,
        help=
        "The port number the Ollama server is listening on (default: 11434)")
    # Generation args
    parser.add_argument(
        "--ctx",
        type=int,
        default=8192,
        help=
        "The size of the context window used to generate the next token (default: 8192)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help=
        "Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers (default: 40)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help=
        "Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text (default: 0.9)"
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help=
        "The temperature of the model. Increasing the temperature will make the model answer more creatively (default: 0.7)"
    )
    parser.add_argument(
        "--keep-alive",
        type=str,
        default="10m",
        help=
        "Controls how long the model stays loaded in memory following a request (e.g., '10m', '1h', '0', '-1') (default: '10m')"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help=
        "Disable real-time response streaming and wait for the full response to be generated"
    )
    args = parser.parse_args()

    # Init ollama client
    client = OllamaClient(host=args.host, port=args.port)

    # Map arguments to client defaults
    client.num_ctx = args.ctx
    client.top_k = args.top_k
    client.top_p = args.top_p
    client.temperature = args.temp
    client.keep_alive = args.keep_alive
    client.stream = not args.no_stream

    # Menu
    try:
        main_menu(client)
    except KeyboardInterrupt:
        print("\n\n[TERMINATED] Program closed by user.")
    except Exception as e:
        print(f"\n\n[CRITICAL ERROR] {e}")
    finally:
        # This runs NO MATTER WHAT
        # Clears the line and ensures the terminal cursor is visible
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
        print("[INFO] Session ended. Terminal cleaned.")
