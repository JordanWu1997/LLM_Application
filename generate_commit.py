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


def get_git_diff():
    try:
        return subprocess.check_output(['git', 'diff', '--cached'],
                                       text=True).strip()
    except Exception:
        return None


def generate_streaming_commit(diff):
    system_prompt = "Write a concise Conventional Commit message. Output ONLY the message text."

    payload = {
        "model": MODEL,
        "prompt": f"{system_prompt}\n\nDiff:\n{diff}",
        "stream": True,  # Enable streaming
        "options": {
            "num_ctx": CONTEXT_WINDOW,
            "temperature": 0
        }
    }

    full_message = ""
    metadata = {}

    try:
        # Use stream=True in requests
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
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
        if processed >= CONTEXT_WINDOW:
            print(
                f"\033[93m⚠️  Warning: Input reached {CONTEXT_WINDOW} tokens and was truncated.\033[0m"
            )
        else:
            print(
                f"\033[90m(Tokens used: {processed}/{CONTEXT_WINDOW})\033[0m")

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

    # Configuration
    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL = "qwen2.5-coder:14b"
    CONTEXT_WINDOW = 4096  # Ollama default is 2048 or 4096

    # Git diff
    diff = get_git_diff()
    if not diff:
        sys.exit("[ERROR] ❌ No changes staged. Run 'git add' first.")

    # Message
    ai_message = generate_streaming_commit(diff)
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
