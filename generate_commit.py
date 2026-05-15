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
# |_|\_\  |_| |_|  Datetime: 2026-01-16 22:45:17             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import json
import os
import subprocess
import sys
import tempfile

import requests

from ollama_utils import check_args_connections, get_available_models


def get_git_diff():
    try:
        return subprocess.check_output(['git', 'diff', '--cached'],
                                       text=True).strip()
    except Exception:
        return None


def generate_streaming_commit(diff,
                              ollama_url='http://localhost:11434',
                              model="gemma4:e4b",
                              context_window=4096):

    system_prompt = "Write a concise Conventional Commit message. Output ONLY the message text."

    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\nDiff:\n{diff}",
        "stream": True,  # Enable streaming
        "options": {
            "num_ctx": context_window,
            "temperature": 0
        }
    }

    full_message = ""
    metadata = {}

    try:
        # Use stream=True in requests
        response = requests.post(f'{ollama_url}/api/generate',
                                 json=payload,
                                 stream=True)
        response.raise_for_status()

        print("\n[INFO] 🤖 Generating: ", end="", flush=True)
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)

                # Check if the chunk has response text
                if 'response' in chunk:
                    text = chunk['response']
                    print(text, end="", flush=True)
                    full_message += text

                # The last chunk contains the metadata
                if chunk.get('done'):
                    metadata = chunk
                    print("\n")

        # Token Truncation Check
        processed = metadata.get("prompt_eval_count", 0)
        if processed >= context_window:
            print(
                f"\033[93m⚠️  Warning: Input reached {context_window} tokens and was truncated.\033[0m"
            )
        else:
            print(
                f"\033[90m(Tokens used: {processed}/{context_window})\033[0m")

        return full_message.strip()

    except Exception as e:
        print(f"\n[ERROR] ❌ API Error: {e}")
        return ""


def edit_message(initial_message):
    editor = os.environ.get('EDITOR', 'vim')
    with tempfile.NamedTemporaryFile(suffix=".tmp", mode='w+',
                                     delete=False) as tf:
        tf.write(initial_message)
        temp_path = tf.name

    subprocess.call([editor, temp_path])

    with open(temp_path, 'r') as f:
        edited_message = f.read().strip()

    os.remove(temp_path)
    return edited_message


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Git Commit w/ Ollama")
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

    # Git diff
    diff = get_git_diff()
    if not diff:
        sys.exit("[ERROR] ❌ No changes staged. Run 'git add' first.")

    # Message
    ai_message = generate_streaming_commit(
        diff,
        ollama_url=f'http://{args.host}:{args.port}',
        model=args.model,
        context_window=args.ctx)
    if not ai_message:
        sys.exit()

    # Choice
    choice = input("\nCommit? [y]es / [n]o / [e]dit: ").lower()
    if choice == 'y':
        final_message = ai_message
    elif choice == 'e':
        final_message = edit_message(ai_message)
    else:
        sys.exit("[INFO] Aborted.")

    # Git commit
    if final_message:
        subprocess.run(['git', 'commit', '-m', final_message])
        sys.exit("[INFO] ✅ Committed!")
