#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Ollama server API Documentation
- https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import base64
import json
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Union

import requests


class OllamaClient:
    """Client for interacting with Ollama REST API."""

    def __init__(self, host: str = "localhost", port: int = 11434):
        """
        Initialize the Ollama client.

        Args:
            host: Hostname or IP address of the Ollama server
            port: Port number of the Ollama server
        """
        self.base_url = f"http://{host}:{port}/api"

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
                                 json={"model": model_name})
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
        response = requests.post(f"{self.base_url}/generate",
                                 json={"model": model_name})
        response.raise_for_status()
        return response.json()

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

    def chat(self,
             model_name: str,
             prompt: str = '',
             image_paths: [str] = [],
             messages: List[Dict[str, str]] = [],
             temperature: float = 0.7,
             top_p: float = 0.9,
             top_k: int = 40,
             stream: bool = False) -> Union[Dict, iter]:
        """
        Chat with a model.

        Args:
            - model_name (str): Name of the model to chat with
            - prompt (str): The prompt for the chat.
            - image_paths (list[str]): List of paths to images to include in the chat.
            - messages (list[dict]): List of message dictionaries [{"role": "user", "content": "Hello"}, ...].
            - temperature (float): Sampling temperature (higher is more creative).
            - top_p (float): Nucleus sampling parameter.
            - top_k (int): Top-k sampling parameter.
            - stream (bool): Whether to stream the response.

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
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            },
            "stream": stream,
        }
        if encoded_images != []:
            payload["images"] = encoded_images

        if stream:
            return self._stream_chat_response(payload), messages
        else:
            response = requests.post(f"{self.base_url}/chat", json=payload)
            response.raise_for_status()
            return response.json(), messages

    def generate(self,
                 model_name: str,
                 prompt: str = '',
                 image_paths: [str] = [],
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 40,
                 stream: bool = False) -> Union[Dict, iter]:
        """
        Generates a response from the specified model using the given prompt and image paths.

        Args:
            model_name (str): The name of the model to use for generation.
            prompt (str, optional): The prompt text for the generation. Defaults to an empty string.
            image_paths ([str], optional): A list of file paths to images to be included in the generation. Defaults to an empty list.
            temperature (float, optional): The temperature parameter for the generation. Defaults to 0.7.
            top_p (float, optional): The top-p parameter for the generation. Defaults to 0.9.
            top_k (int, optional): The top-k parameter for the generation. Defaults to 40.
            stream (bool, optional): Whether to generate the response in a streaming manner. Defaults to False.

        Returns:
            Union[Dict, iter]: A dictionary containing the generated response or an iterator if streaming is enabled.
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
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            },
            "images": encoded_images,
            "stream": stream
        }

        if stream:
            return self._stream_generate_response(payload)
        else:
            response = requests.post(f"{self.base_url}/generate", json=payload)
            response.raise_for_status()
            return response.json()

    def _stream_chat_response(self, payload: Dict):
        """
        Stream chat responses.

        Args:
            payload: Request payload dictionary

        Yields:
            Streamed response chunks
        """
        with requests.post(f"{self.base_url}/chat", json=payload,
                           stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield {"error": "Failed to parse streaming response"}

    def _stream_generate_response(self, payload: Dict):
        """
        Stream chat responses.

        Args:
            payload: Request payload dictionary

        Yields:
            Streamed response chunks
        """
        with requests.post(f"{self.base_url}/generate",
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


def list_all_models(with_index=True):
    """
    List all available models on the specified client.

    Args:
        with_index (bool): If True, include model index in the output. Default is True.

    Returns:
        list: A list of dictionaries containing information about each model.
    """

    try:
        models = client.list_models()
        print("\nAvailable models:")
        for i, model in enumerate(models):
            model_info = client.show_model_info(model['name'])
            model_size = model.get('size', 'N/A')
            if model_size != 'N/A':
                model_size = float(model_size) / 1024 / 1024 / 1024
                model_size = f'{model_size:.2f}'
            model_capability = model_info.get('capabilities', 'N/A')
            model_params = model_info['details'].get('parameter_size', 'N/A')
            model_quant = model_info['details'].get('quantization_level',
                                                    'N/A')
            if with_index:
                print(
                    f"{i+1:02d}. {model['name']} ({model_capability}, Param: {model_params}, Size: {model_size} GB)"
                )
            else:
                print(
                    f"- {model['name']} ({model_capability}, Param: {model_params}, Size: {model_size} GB)"
                )
    except Exception as e:
        print(f"Error: {e}")
    return models


def list_running_models(with_index=True):
    """
    List all the running models on a specified platform.

    Args:
        with_index (bool): If True, include model indices in the output. Default is True.

    Returns:
        list: A list of dictionaries containing information about each running model.
        Each dictionary includes 'name', 'capability', and 'size' fields.
    """
    try:
        models = client.list_running_models()
        if models:
            print("\nRunning models:")
            for i, model in enumerate(models):
                model_info = client.show_model_info(model['name'])
                model_size = model.get('size', 'N/A')
                if model_size != 'N/A':
                    model_size = float(model_size) / 1024 / 1024 / 1024
                    model_size = f'{model_size:.2f}'
                model_capability = model_info.get('capabilities', 'N/A')
                model_params = model_info['details'].get(
                    'parameter_size', 'N/A')
                model_quant = model_info['details'].get(
                    'quantization_level', 'N/A')
                if with_index:
                    print(
                        f"{i+1:02d}. {model['name']} ({model_capability}, Param: {model_params}, Size: {model_size} GB)"
                    )
                else:
                    print(
                        f"- {model['name']} ({model_capability}, Param: {model_params}, Size: {model_size} GB)"
                    )
        else:
            print("\nNo models currently running.")
    except Exception as e:
        print(f"Error: {e}")
    return models


def load_model():
    """
    Load a model from the available models list.

    This function lists all available models, prompts the user to enter a model name or index,
    converts the input into the corresponding model name if it's an integer, and then loads the model.
    It handles exceptions such as invalid inputs or errors during loading.

    Returns:
        str: The name of the loaded model.
    """

    # List all available models
    all_models = list_all_models()
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
        print(f"Model loaded successfully: {response}")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")


def unload_model():
    """
    Unloads a specified model from the system.

    This function lists all available models, prompts the user to enter the name or index of the model they want to unload,
    converts the input into the corresponding model name if necessary, and then unloads the model using the client's API.
    It handles potential errors such as invalid inputs or failures during the unloading process.

    Returns:
        None
    """

    # List all available models
    all_models = list_running_models()
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


def generate_completion_with_model(stream=False):
    """
    Generate a completion using the specified model.

    Args:
        stream (bool): Whether to stream the response. Defaults to False.

    Returns:
        None
    """

    # Get available models
    list_running_models(with_index=False)
    all_models = list_all_models()

    # Convert index to model name
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

    # Opening
    print(f"\nGenerate completion with {model_name}")
    print("- type /exit to exit the chat session")
    print("- type /image to enter image paths for VLM")
    print("- type /continue to continue generation with previous results")

    # Load model to generate completion
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=client._spinner_task,
                                      args=(stop_spinner, ))
    spinner_thread.start()
    start_time = time.time()
    _ = client.load_model(model_name)
    elapsed_time = time.time() - start_time
    stop_spinner.set()
    spinner_thread.join()
    print(f'\nLoading {model_name} took {elapsed_time:.3f} sec')

    # Main loop
    history = ''
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
            user_input = input("\n>>> You: \n", end="")
        elif user_input.lower() == '/continue':
            user_input = f'{history} '

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

        try:
            # Send request to server
            response = client.generate(model_name,
                                       prompt=user_input,
                                       stream=stream)

            # Get first full token to measure first token time
            first_token_received = False
            first_token_time, last_token_time = None, None

            if stream:
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
                tps = 'N/A'
                if first_token_time is not None:
                    first_token_latency = first_token_time - start_time
                    elapsed_time = time.time() - start_time
                    tps = f'{token_count / elapsed_time:.1f}'
                print()
            else:
                full_response = response.get("response", "No response")
                tps = f'{client._calculate_tokens_per_second(response):.1f}'
                print(full_response)

            # Use streaming for typing effect
            print("\n[Statistics]")
            if first_token_time is not None:
                print(f"- First token latency: {first_token_latency:.2f} sec")
            print(f"- Performance: {tps} tokens/sec")
            print("- type /exit to exit the chat session")
            print("- type /image to enter image paths for VLM")
            print(
                "- type /continue to continue generation with previous results"
            )

            # Add generated response to history
            history += full_response

        # Handle exceptions
        except Exception as e:
            print(f"\n[ERROR] Error during chat: {e}")


def chat_with_model(stream=False, running_only=False):
    """
    Chat with a specified model using the provided API.

    Parameters:
        stream (bool): If True, enables streaming for typing effect. Defaults to False.

    Returns:
        None
    """

    # Get available models
    all_models = list_running_models(with_index=False)
    if not running_only or all_models == []:
        all_models = list_all_models()

    # Convert index to model name
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

    # Opening
    print(f"\nChat with {model_name}")
    print("- type /exit to exit the chat session")
    print("- type /image to enter image paths for VLM")

    # Load model to chat with
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=client._spinner_task,
                                      args=(stop_spinner, ))
    spinner_thread.start()
    start_time = time.time()
    _ = client.load_model(model_name)
    elapsed_time = time.time() - start_time
    stop_spinner.set()
    spinner_thread.join()
    print(f'\nLoading {model_name} took {elapsed_time:.3f} sec')

    # Main loop
    messages = []
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
        print(f"\n<<< Model ({model_name}): ")

        # Track full response and performance data
        start_time = time.time()
        full_response, token_count = "", 0

        # Spinner for loading animation
        stop_spinner = threading.Event()
        spinner_thread = threading.Thread(target=client._spinner_task,
                                          args=(stop_spinner, ))
        spinner_thread.start()

        try:
            # Send request to server
            response, messages = client.chat(model_name,
                                             prompt=user_input,
                                             messages=messages,
                                             image_paths=image_paths,
                                             stream=stream)

            # Get first full token to measure first token time
            first_token_received = False
            first_token_time, last_token_time = None, None

            if stream:
                # Decode response
                for chunk in response:
                    if "message" in chunk and chunk["message"].get("content"):
                        content = chunk["message"]["content"]
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
                tps = 'N/A'
                if first_token_time is not None:
                    first_token_latency = first_token_time - start_time
                    elapsed_time = time.time() - start_time
                    tps = f'{token_count / elapsed_time:.1f}'
                print()
            else:
                full_response = response.get("message",
                                             {}).get("content", "No response")
                # Stop the spinner
                stop_spinner.set()
                spinner_thread.join()
                tps = f'{client._calculate_tokens_per_second(response):.1f}'
                print(full_response)

            # Calculate first token latency if available
            print("\n[Statistics]")
            if first_token_time is not None:
                print(f"- First token latency: {first_token_latency:.2f} sec")
            print(f"- Performance: {tps} tokens/sec")

            # Add assistant response to messages for context
            messages.append({"role": "assistant", "content": full_response})

        # Handle exceptions
        except Exception as e:
            print(f"\n[ERROR] Error during chat: {e}")


# Example usage
if __name__ == "__main__":

    # Host
    if len(sys.argv) > 1:
        host = sys.argv[1]
    else:
        host = input("\nEnter Ollama server host IP (default: localhost): ") \
            or "localhost"

    # Init client
    client = OllamaClient(host=host)

    # Main
    while True:

        # Help message
        print(f"\n=== Ollama API Client for [{host}] ===")
        print("0. Exit")
        print("1. List all models")
        print("2. List running models")
        print("3. Load a model")
        print("4. Unload a running model")
        print("5. Chat with model")
        print("6. Chat with running model")
        print("7. Generate completion with model")

        # Enter choice
        choice = input("\nEnter your choice (0,1-6): ")
        if choice == "0":
            print("Exiting program. Goodbye!")
            break
        elif choice == "1":
            list_all_models()
        elif choice == "2":
            list_running_models()
        elif choice == "3":
            load_model()
        elif choice == "4":
            unload_model()
        elif choice == "5":
            chat_with_model(stream=True)
        elif choice == "6":
            chat_with_model(stream=True, running_only=True)
        elif choice == "7":
            generate_completion_with_model(stream=True)
        else:
            print("Invalid choice. Please try again.")
