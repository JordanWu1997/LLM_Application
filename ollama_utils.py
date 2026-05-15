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
"""

"""

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
            print(f"[*] Switched to model: {selected_model}")
        except (ValueError, IndexError):
            print("Invalid selection. Exiting.")
            sys.exit(1)

    print()
    print(f"--- Configuration Loaded ---")
    print(f"URL: {base_url}")
    print(f"Model: {selected_model}")
    print(f"Context Window: {args.ctx}")
    print(f"---------------------------")
    print()


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
    args = parser.parse_args()

    # Check if Ollama connection/model is available
    check_args_connections(args)
