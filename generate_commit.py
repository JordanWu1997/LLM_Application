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

from llamacpp_utils import stream_llamacpp_chat, stream_llamacpp_generate
from ollama_utils import (check_args_connections, print_context_warning,
                          stream_ollama_generate)


def get_git_diff():
    try:
        return subprocess.check_output(['git', 'diff', '--cached'],
                                       text=True).strip()
    except Exception:
        return None


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
    parser.add_argument("--engine",
                        choices=['ollama', 'llamacpp'],
                        default='ollama',
                        help="LLM Engine")
    parser.add_argument(
        "--ctx",
        type=int,
        default=8192,
        help=
        "The size of the context window used to generate the next token (default: 8192)"
    )
    args = parser.parse_args()

    # Check if Ollama connection/model is available
    if args.engine == 'ollama':
        args = check_args_connections(args)

    # Git diff
    diff = get_git_diff()
    if not diff:
        sys.exit("[ERROR] ❌ No changes staged. Run 'git add' first.")

    prompt = f'Diff:\n{diff}'
    system_prompt = "Write a concise Conventional Commit message. Output ONLY the message text.",

    # Message
    if args.engine == 'ollama':
        ai_message = stream_ollama_generate(
            prompt,
            system_prompt=system_prompt,
            ollama_url=f'http://{args.host}:{args.port}',
            model=args.model,
            ctx_window=args.ctx,
            verbose=True)
    else:
        ai_message = stream_llamacpp_chat(
            prompt,
            system_prompt=system_prompt,
            llamacpp_url=f'http://{args.host}:{args.port}',
            ctx_window=args.ctx,
            verbose=True)
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
