#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8
r"""
[ADD MODULE DOCUMENTATION HERE]

# ========================================================================== #
#  _  __   _   _                                          __        ___   _  #
# | |/ /  | | | |  Author: Jordan Kuan-Hsien Wu           \ \      / / | | | #
# | ' /   | |_| |  E-mail: jordankhwu@gmail.com            \ \ /\ / /| | | | #
# | . \   |  _  |  Github: https://github.com/JordanWu1997  \ V  V / | |_| | #
# |_|\_\  |_| |_|  Datetime: 2026-05-16 00:19:54             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import json
import os
import subprocess
import sys
import tempfile

import requests


def get_available_models(base_url):
    """Fetch available models from the Ollama server."""

    try:
        # Note: /api/tags (GET) or /api/ps (GET/POST) are common endpoints
        # /api/tags is the standard for listing local models
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            return [m['name'] for m in response.json().get('models', [])]
        return []
    except Exception:
        return []


def check_args_connections(args):

    base_url = f"http://{args.host}:{args.port}"
    print(f"\n[*] Checking connection to {base_url}...")
    try:
        # We use the version or tags endpoint to verify the server is alive
        hb = requests.get(f"{base_url}/api/tags", timeout=5)
        hb.raise_for_status()
    except (requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError):
        print(
            f"Error: Could not connect to Ollama at {base_url}. Is the server running?"
        )
        sys.exit(1)

    # Check if the selected model is available
    available_models = get_available_models(base_url)
    selected_model = args.model
    if selected_model not in available_models:
        print(f"[!] Model '{selected_model}' not found locally.")

        if not available_models:
            print(
                "No local models found. Please pull a model first (e.g., 'ollama pull llama3')."
            )
            sys.exit(1)
        print("\nAvailable models:")
        for i, m in enumerate(available_models):
            print(f"{i + 1}. {m}")
        try:
            choice = int(
                input(
                    f"\nSelect a model number (1-{len(available_models)}): "))
            selected_model = available_models[choice - 1]
            args.model = selected_model
            print(f"[*] Switched to model: {selected_model}")
        except (ValueError, IndexError):
            print("Invalid selection. Exiting.")
            sys.exit(1)

    print()
    print(f"--- Configuration Loaded ---")
    print(f"URL: {base_url}")
    print(f"Model: {args.model}")
    print(f"Context Window: {args.ctx}")
    print(f"---------------------------")
    print()

    return args


def stream_ollama_generate(prompt,
                           ollama_url="http://localhost:11434",
                           model="gemma3:12b",
                           ctx_window=4096,
                           verbose=False):

    # Payload
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_ctx": ctx_window,
            "temperature": 0
        }
    }

    full_response = ""
    stats = {}
    with requests.post(f"{ollama_url}/api/generate", json=payload,
                       stream=True) as response:
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if not chunk.get("done"):
                    content = chunk.get("response", "")
                    full_response += content
                    if verbose:
                        print(content, end="", flush=True)
                else:
                    stats = chunk

    # Performance Reporting
    p_tokens = stats.get('prompt_eval_count', 0)
    o_tokens = stats.get('eval_count', 0)
    duration = stats.get('eval_duration', 1) / 1e9
    if verbose:
        print(
            f"\n--- Stats: {p_tokens} token-in / {o_tokens} token-out | {o_tokens/duration:.1f} t/s ---"
        )
    # Prompt truncated alert
    print_context_warning(prompt_tokens=p_tokens, ctx_window=ctx_window)

    return full_response


def get_input_from_editor(initial_text=""):
    """Opens the user's default terminal editor to capture multi-line input."""

    # Look for the user's preferred editor, default to vim (or nano)
    editor = os.environ.get('EDITOR', 'vim')

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".md",
                                     delete=False) as tf:
        if initial_text:
            tf.write(initial_text)
            tf.flush()

        # Close the file handle temporarily so the editor can safely open it
        tf_name = tf.name

    try:
        # Launch the editor and wait for the user to close it
        subprocess.call([editor, tf_name])

        # Read the contents after the user saves and exits
        with open(tf_name, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        return content
    finally:
        # Always clean up the temporary file afterward
        if os.path.exists(tf_name):
            os.remove(tf_name)


def print_context_warning(prompt_tokens, ctx_window):
    """
    Warn if prompt tokens are approaching or exceeding context window.
    """
    if not ctx_window or ctx_window <= 0:
        return

    usage_ratio = prompt_tokens / ctx_window
    usage_percent = usage_ratio * 100

    # Hard truncation warning
    if prompt_tokens >= ctx_window:
        print(f"\n\033[91m[WARNING] Input may be TRUNCATED "
              f"({prompt_tokens}/{ctx_window} tokens, "
              f"{usage_percent:.1f}% used)\033[0m")

    # Near-limit warning
    elif usage_ratio >= 0.90:
        print(f"\n\033[93m[WARNING] Context window nearly full "
              f"({prompt_tokens}/{ctx_window} tokens, "
              f"{usage_percent:.1f}% used)\033[0m")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Argparse Template for ollama utils")
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
    parser.add_argument("--model",
                        default="gemma4:e4b-8k-gpu",
                        help="Model name to use (default: gemma4:e4b-8k-gpu)")
    parser.add_argument(
        "--ctx",
        type=int,
        default=8192,
        help=
        "The size of the context window used to generate the next token (default: 8192)"
    )
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="Enable verbose mode")
    args = parser.parse_args()

    # Check if Ollama connection/model is available
    check_args_connections(args)
