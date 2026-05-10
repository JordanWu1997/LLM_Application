#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Ollama server REST API Documentation
A comprehensive CLI for managing models and interacting with Ollama servers.
- https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import base64
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterator, List, Optional, Union

import ollama
import requests
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


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
        self.ctx_window_used_token = 0

    def update_base_url(self):
        self.base_url = f"http://{self.host}:{self.port}/api"

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
                                              args=(stop_spinner, ))
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
            spinner_thread.join()

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

    def chat(
        self,
        model_name: str,
        prompt: str = '',
        image_paths: List[str] = [],
        messages: List[Dict[str, str]] = [],
        num_ctx: int = -1,
        think: bool = True,
    ) -> tuple[Union[Dict, Iterator], List[Dict[str, str]]]:
        """
        Chat with a model.

        Args:
            - model_name (str): Name of the model to chat with
            - prompt (str): The prompt for the chat.
            - image_paths (list[str]): List of paths to images to include in the chat.
            - messages (list[dict]): List of message dictionaries [{"role": "user", "content": "Hello"}, ...].
            - num_ctx

        Returns:
            - dict: A dictionary containing the model's reply or a stream iterator
        """

        # Read image and encode to base64
        encoded_images = []
        for image_path in image_paths:
            with open(image_path, "rb") as img_file:
                encoded_images.append(
                    base64.b64encode(img_file.read()).decode("utf-8"))

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
                 image_paths: List[str] = [],
                 num_ctx: int = -1) -> Union[Dict, Iterator]:
        """
        Generates a response from the specified model using the given prompt and image paths.

        Args:
            model_name (str): The name of the model to use for generation.
            prompt (str, optional): The prompt text for the generation. Defaults to an empty string.
            image_paths ([str], optional): A list of file paths to images to be included in the generation. Defaults to an empty list.
            num_ctx

        Returns:
            Union[Dict, Itererator]: A dictionary containing the generated response or an iterator if streaming is enabled.
        """

        # Read image and encode to base64
        encoded_images = []
        for image_path in image_paths:
            with open(image_path, "rb") as img_file:
                encoded_images.append(
                    base64.b64encode(img_file.read()).decode("utf-8"))

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
            "images": encoded_images,
            "stream": self.stream
        }

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

    def display_session_stats(self, session_stats):
        # Stats
        print("\n[Statistics]")
        # TTFT
        if session_stats['first_token_latency'] is not None:
            print(
                f"- First token latency (TTFT): {session_stats['first_token_latency']:.2f} sec"
            )
        # TPS
        if session_stats['tps'] is not None:
            print(f"- Performance: {session_stats['tps']:.1f} tokens/sec")
        # Token stats
        total = session_stats["total_input_tokens"] + session_stats[
            "total_output_tokens"]
        print("- Token Usage")
        print(f"  - Input Tokens: {session_stats['total_input_tokens']}")
        print(f"  - Output Tokens: {session_stats['total_output_tokens']}")
        if session_stats['thinking_tokens'] is not None:
            print(
                f"    - Estimate Thinking: {session_stats['thinking_tokens']}")
        if session_stats['content_tokens'] is not None:
            print(f"    - Estimate Content: {session_stats['content_tokens']}")
        print(f"  - Total (Input + Output) Tokens: {total}")

        # Window usage (thinking excluded)
        if session_stats['content_tokens'] is not None:
            window_usage = \
                session_stats['content_tokens'] / session_stats['num_ctx']
            self.ctx_window_used_token += session_stats['content_tokens']
        else:
            window_usage = \
                session_stats['total_output_tokens'] / session_stats['num_ctx']
            self.ctx_window_used_token += session_stats['total_output_tokens']
        ctx_window_usage = \
            self.ctx_window_used_token / session_stats['num_ctx']
        print(
            f"  - Context Window: {session_stats['num_ctx']} (+{window_usage:.1%} Usage, {ctx_window_usage:.1%} Used in total)"
        )

    @staticmethod
    def _spinner_task(stop_event):
        """
        A task that displays a spinning animation using the spinner characters '\\', '|', '/'.

        Args:
        stop_event (Event): An event object used to signal the task to stop.
        """

        spinner = ['\\', '|', '/']
        i = 0
        while not stop_event.is_set():
            sys.stdout.write(f'\r{spinner[i % len(spinner)]}')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        # sys.stdout.write('\rDone')
        sys.stdout.write('\r' + ' ' * 10 + '\r')
        sys.stdout.flush()

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
    def _print_typing_effect(text):
        """
        Print text with a typing effect.

        Args:
            text: Text to print
        """
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()


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
                ctx_num = model.get('context_length', 'N/A')
                if ctx_num != 'N/A':
                    ctx_num = int(ctx_num)
                expiires_at = model.get('expires_at', 'N/A')

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
                        f"{i+1:02d}. {model['name']} ({model_capability}, Param: {model_params}, Size: {model_size} GB, vRAM: {size_vram} GB, CTX_LEN: {ctx_num}, Expires: {expiires_at})"
                    )
                else:
                    print(
                        f"- {model['name']} ({model_capability}, Param: {model_params}, Size: {model_size} GB, vRAM: {size_vram} GB, CTX_LEN: {ctx_num}, Expires: {expiires_at})"
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


def generate_completion_with_model(client, running_only=False):
    """
    Generate a completion using the specified model.

    Args:
        stream (bool): Whether to stream the response. Defaults to False.

    Returns:
        None
    """
    # Get available models
    all_models = list_running_models(client, with_index=running_only)
    if not running_only or all_models == []:
        all_models = list_all_models(client)

    # Convert index to model name
    if len(all_models) == 1:
        model_name = all_models[0]['name']
    else:
        model_name = input("\nEnter model name to generate_completion with: ")
        try:
            model_index = int(model_name)
            model_name = all_models[model_index - 1]['name']
        except ValueError:
            pass
        except IndexError:
            pass
        if model_name not in [model['name'] for model in all_models]:
            print(f'[ERROR] Not a valid model name: {model_name}')
            return

    # Initial context settings
    messages = []
    session_stats = {"total_input_tokens": 0, "total_output_tokens": 0}
    client.ctx_window_used_token = 0

    # Opening
    print(f"\nGenerate completion with {model_name}")
    print("- type /exit to exit the chat session")
    print("- type /image to enter image paths for VLM")
    print("- type /ctx [number] to change context length")
    print(
        "- type /keepalive [options] to change model keep alive time [1m/5m/1h/0/-1]"
    )
    print(
        "- type /continue to continue generation with previous results and follow-up input"
    )
    print(
        '- type /history to shwo full history (user_input and model generation)'
    )
    print('- type /clear to clear history')

    # Load model to generate completion
    start_time = time.time()
    _ = client.load_model(model_name)
    elapsed_time = time.time() - start_time
    print(f'\nLoading {model_name} took {elapsed_time:.3f} sec')

    # Main loop
    history = ''
    while True:

        # Init
        image_paths = []
        follow_up_input = None
        client.ctx_window_used_token = 0

        # User input
        user_input = input("\n>>> You: \n")
        if user_input.lower() == '/exit':
            break
        elif user_input.lower() == '/history':
            print('\n>>> History')
            print(f"\n\033[90m[START OF HiSTORY]\033[0m")
            print(history)
            print(f"\n\033[90m[END OF HISTORY]\033[0m")
            continue
            user_input = input("\n>>> You: \n")
        elif user_input.lower() == '/clear':
            print('\n[INFO] History cleared.')
            continue
        elif user_input.lower() == '/image':
            # Add image
            image_path = input('\nEnter the image path: ')
            if image_path != '':
                image_paths.append(image_path)
            # Add another image
            while True:
                image_path = input('Do you want to add another one? (y/N): ')
                if image_path != 'y':
                    break
                image_path = input('Enter the image path: ')
                image_paths.append(image_path)
            # Add prompt for image
            user_input = input("\n>>> You: \n")
        elif user_input.lower().startswith('/ctx'):
            try:
                parts = user_input.split()
                if len(parts) > 1:
                    new_num_ctx = int(parts[1])
                    client.num_ctx = new_num_ctx
                    print(
                        f"[INFO] Context length updated to {client.num_ctx} for next message."
                    )
                else:
                    print(f"[INFO] Current context length: {client.num_ctx}")
                continue  # Return to start of loop for follonw prompt
            except ValueError:
                print(
                    "[ERROR] Please provide a valid number for context length."
                )
                continue
        elif user_input.lower() in ['/c', '/cont', '/continue']:
            follow_up_input = input(
                "\n>>> Your follow-up (or just leave empty): \n")
            user_input = f'{history} {follow_up_input}'
        elif user_input.lower().startswith('/keepalive'):
            try:
                parts = user_input.split()
                if len(parts) > 1:
                    try:
                        client.keep_alive = int(parts[1])
                    except ValueError:
                        client.keep_alive = parts[1]
                    print(f"[INFO] Keep-alive updated to {client.keep_alive}")
                else:
                    print(f"[INFO] Current keep-alive: {client.keep_alive}")
                continue
            except Exception as e:
                print(f"[ERROR] {e}")
                continue

        # Track full response and performance data
        print(f"\n<<< Model ({model_name}): ")

        # Track full response and performance data
        start_time = time.time()
        full_response, token_count = "", 0

        # Spinner for loading animation
        stop_spinner = threading.Event()
        spinner_thread = threading.Thread(target=client._spinner_task,
                                          args=(stop_spinner, ))
        spinner_thread.start()

        # Init
        first_token_time, last_token_time = None, None
        first_token_latency, tps = None, None

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
                            spinner_thread.join()
                        # Record time of latest token
                        last_token_time = time.time()
                        # Display with typing effect
                        client._print_typing_effect(content)
                        full_response += content

                    # Track token info for TPS calculation
                    if "eval_count" in chunk:
                        token_count = chunk["eval_count"]

                    # Capture final statistics from the last chunk
                    if chunk.get("done"):
                        session_stats["total_input_tokens"] += chunk.get(
                            "prompt_eval_count", 0)
                        session_stats["total_output_tokens"] += chunk.get(
                            "eval_count", 0)

                if first_token_time is not None:
                    first_token_latency = first_token_time - start_time
                    elapsed_time = time.time() - start_time
                    tps = token_count / elapsed_time
                print()

            else:
                full_response = response.get("response", "No response")
                session_stats["total_input_tokens"] += response.get(
                    "prompt_eval_count", 0)
                session_stats["total_output_tokens"] += response.get(
                    "eval_count", 0)

            # Update stats
            session_stats['first_token_latency'] = first_token_latency
            session_stats['tps'] = tps
            session_stats['num_ctx'] = client.num_ctx
            session_stats["thinking_tokens"] = None
            session_stats["content_tokens"] = None

            # Show stats
            client.display_session_stats(session_stats)

            # Add generated response to history
            history = f'{user_input} {full_response}'

        except KeyboardInterrupt:
            stop_spinner.set()  # Stop the spinner thread
            spinner_thread.join()
            print("\n\n[INTERRUPTED] Stopping generation...")
            continue

        # Handle exceptions
        except Exception as e:
            print(f"\n[ERROR] Error during chat: {e}")


def chat_with_model(client, running_only=False):
    """
    Chat with a specified model using the provided API.

    Parameters:
        stream (bool): If True, enables streaming for typing effect. Defaults to False.

    Returns:
        None
    """

    # Get available models
    all_models = list_running_models(client, with_index=running_only)
    if not running_only or all_models == []:
        all_models = list_all_models(client)

    # Convert index to model name
    if len(all_models) == 1:
        model_name = all_models[0]['name']
    else:
        model_name = input("\nEnter model name to chat with: ")
        try:
            model_index = int(model_name)
            model_name = all_models[model_index - 1]['name']
        except ValueError:
            pass
        except IndexError:
            pass
        if model_name not in [model['name'] for model in all_models]:
            print(f'[ERROR] Not a valid model name: {model_name}')
            return

    # Initial context settings
    messages = []
    session_stats = {"total_input_tokens": 0, "total_output_tokens": 0}
    client.ctx_window_used_token = 0

    # Opening
    print(f"\nChat with {model_name}")
    print("- type /exit to exit the chat session")
    print("- type /image to enter image paths for VLM")
    print("- type /ctx [number] to change context length")
    print("- type /think [options] to change think mode [on/off/OTHERS]")
    print(
        "- type /keepalive [options] to change model keep alive time [1m/5m/1h/0/-1]"
    )

    # Load model to chat with
    start_time = time.time()
    _ = client.load_model(model_name)
    elapsed_time = time.time() - start_time
    print(f'\nLoading {model_name} took {elapsed_time:.3f} sec')

    # Load tokenizer
    start_time = time.time()
    tok = OllamaTokenizer(model_name)
    elapsed_time = time.time() - start_time
    print(f'Loading {tok.model_family} tokenizer took {elapsed_time:.3f} sec')

    # Main loop
    while True:

        # Init
        image_paths = []

        # User input
        user_input = input("\n>>> You: \n")
        if user_input.lower() == '/exit':
            break
        elif user_input.lower() == '/image':
            # Add image
            image_path = input('\nEnter the image path: ')
            if image_path != '':
                image_paths.append(image_path)
            # Add another image
            while True:
                image_path = input('Do you want to add another one? (y/N): ')
                if image_path != 'y':
                    break
                image_path = input('Enter the image path: ')
                image_paths.append(image_path)
            # Add prompt for image
            user_input = input("\n>>> You: \n")
        elif user_input.lower().startswith('/ctx'):
            try:
                parts = user_input.split()
                if len(parts) > 1:
                    new_num_ctx = int(parts[1])
                    client.num_ctx = new_num_ctx
                    print(
                        f"[INFO] Context length updated to {client.num_ctx} for next message."
                    )
                else:
                    print(f"[INFO] Current context length: {client.num_ctx}")
                continue  # Return to start of loop for next prompt
            except ValueError:
                print(
                    "[ERROR] Please provide a valid number for context length."
                )
                continue
        elif user_input.lower().startswith('/think'):
            try:
                parts = user_input.split()
                if len(parts) > 1:
                    option = parts[1]
                    if option.lower() == 'on':
                        client.think = True
                    elif option.lower() == 'off':
                        client.think = False
                    else:
                        client.think = option
                    print(
                        f"[INFO] Think mode updated to {client.think} for next message."
                    )
                else:
                    print(f"[INFO] Current think mode: {client.think}")
                continue  # Return to start of loop for next prompt
            except ValueError:
                print(
                    "[ERROR] Please provide a valid number for context length."
                )
                continue
        elif user_input.lower().startswith('/keepalive'):
            try:
                parts = user_input.split()
                if len(parts) > 1:
                    try:
                        client.keep_alive = int(parts[1])
                    except ValueError:
                        client.keep_alive = parts[1]
                    print(f"[INFO] Keep-alive updated to {client.keep_alive}")
                else:
                    print(f"[INFO] Current keep-alive: {client.keep_alive}")
                continue
            except Exception as e:
                print(f"[ERROR] {e}")
                continue

        print(f"\n<<< Model ({model_name}): ")

        # Track full response and performance data
        start_time = time.time()
        full_response, token_count = "", 0
        thinking_response, content_response = "", ""

        # Spinner for loading animation
        stop_spinner = threading.Event()
        spinner_thread = threading.Thread(target=client._spinner_task,
                                          args=(stop_spinner, ))
        spinner_thread.start()

        # Init
        first_token_time, last_token_time = None, None
        first_token_latency, tps = None, None

        # Tracking states for formatting
        currently_thinking = False
        content_started = False

        try:
            # Send request to server
            response, messages = client.chat(model_name,
                                             prompt=user_input,
                                             messages=messages,
                                             image_paths=image_paths,
                                             think=client.think,
                                             num_ctx=client.num_ctx)

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
                            spinner_thread.join()

                            # Print Thinking Block Header
                            print(f"\n\033[90m[THINKING]\033[0m")
                            currently_thinking = True

                        # Record time of latest token
                        last_token_time = time.time()
                        # Display with typing effect
                        client._print_typing_effect(thinking)
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
                            spinner_thread.join()

                        # Record time of latest token
                        last_token_time = time.time()
                        # Display with typing effect
                        client._print_typing_effect(content)
                        full_response += content
                        content_response += content

                    # Track token info for TPS calculation
                    if "eval_count" in chunk:
                        token_count = chunk["eval_count"]

                    # Capture final statistics from the last chunk
                    if chunk.get("done"):

                        # Ensure we close thinking block if model finished without content
                        if currently_thinking:
                            print(f"\n\033[90m[END THOUGHTS]\033[0m\n")

                        session_stats["total_input_tokens"] += chunk.get(
                            "prompt_eval_count", 0)
                        session_stats["total_output_tokens"] += chunk.get(
                            "eval_count", 0)

                if first_token_time is not None:
                    first_token_latency = first_token_time - start_time
                    elapsed_time = time.time() - start_time
                    tps = token_count / elapsed_time
                print()

            else:
                full_response = response.get("message",
                                             {}).get("content", "No response")
                session_stats["total_input_tokens"] += response.get(
                    "prompt_eval_count", 0)
                session_stats["total_output_tokens"] += response.get(
                    "eval_count", 0)

                # Stop the spinner
                stop_spinner.set()
                spinner_thread.join()
                tps = client._calculate_tokens_per_second(response)
                print(full_response)

            # Update stats
            session_stats['first_token_latency'] = first_token_latency
            session_stats['tps'] = tps
            session_stats['num_ctx'] = client.num_ctx

            # Count token and update stats
            counts = tok.get_stats(thinking_response, content_response)
            session_stats["thinking_tokens"] = counts["thinking_tokens"]
            session_stats["content_tokens"] = counts["content_tokens"]

            # Show stats
            client.display_session_stats(session_stats)

            # Add assistant response to messages for context
            #messages.append({"role": "assistant", "content": full_response})
            # Add assistant response to messages for context (thinking excluded)
            messages.append({"role": "assistant", "content": content_response})

        except KeyboardInterrupt:
            stop_spinner.set()  # Stop the spinner thread
            spinner_thread.join()
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
            if val:
                val = input(f"Enter temperature [{client.temperature}]: ")
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
    main_menu(client)
