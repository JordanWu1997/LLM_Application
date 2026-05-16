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
# |_|\_\  |_| |_|  Datetime: 2026-05-16 20:53:42             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from ollama_utils import check_args_connections, stream_ollama_generate


def get_last_real_command(commands):
    """Filters out the 'fix' command itself to find the actual failed command."""
    for cmd in reversed(commands):
        cmd = cmd.strip()
        # Ignore empty lines and the trigger command itself
        if cmd and not cmd.startswith('fix') and 'fix.py' not in cmd:
            return cmd
    return None


def run_auto_mode(command_args,
                  ollama_url="http://localhost:11434",
                  model="gemma3:12b",
                  ctx_window=4096):
    """Re-runs the passed command, catches output, and analyzes if it fails."""

    # Rebuild the command from arguments
    command = " ".join(command_args).strip()

    if not command:
        print("❌ No command provided.")
        print(
            "Usage: fix <command>  (Hint: use 'fix !!' to run your last command)"
        )
        sys.exit(1)

    print(f"🔄 Running command to capture context: `{command}`...\n")

    # Execute the command
    result = subprocess.run(command,
                            shell=True,
                            capture_output=True,
                            text=True)

    # If it works, just print the output and exit gracefully
    if result.returncode == 0:
        print("✅ The command succeeded! Here is the output:\n")
        print(result.stdout)
        sys.exit(0)

    # If it fails, build the prompt and ask Ollama
    prompt = f"""You are an expert Linux/macOS terminal assistant.
The user ran the following command which resulted in an error: `{command}`

Standard Output:
{result.stdout}

Standard Error:
{result.stderr}

Analyze the error and provide the exact fix. Format your response exactly as:
**Issue:** [Brief explanation]
**Fix:** `[Corrected command]`
"""

    # Call ollama generate
    stream_ollama_generate(prompt,
                           ollama_url=ollama_url,
                           model=model,
                           ctx_window=ctx_window,
                           verbose=True)


def run_consult_mode(ollama_url="http://localhost:11434",
                     model="gemma3:12b",
                     ctx_window=4096):
    """Mode 2: Reads debug logs pasted by the user or piped from another command."""

    # Check if data is being piped (e.g., `cat log.txt | fix`)
    if not sys.stdin.isatty():
        debug_text = sys.stdin.read()
    else:
        # Interactive paste mode
        print(
            "📋 Paste your debug message below. Press Ctrl+D (or Ctrl+Z on Windows) on an empty line when finished:"
        )
        debug_text = sys.stdin.read()
        print("\n---")

    if not debug_text.strip():
        print("No input provided. Exiting.")
        sys.exit(0)

    prompt = f"""You are an expert software engineer and debugger.
The user has provided the following debug logs/error messages. Analyze them, explain what is going wrong, and provide the exact steps or code needed to fix it.

Debug Context:
{debug_text}
"""
    # Call ollama generate
    stream_ollama_generate(prompt,
                           ollama_url=ollama_url,
                           model=model,
                           ctx_window=ctx_window,
                           verbose=True)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="AI Terminal Assistant powered by Ollama",
        usage=
        "%(prog)s [-c] [command ...]\n       (Hint: use '%(prog)s !!' to auto-fix the last command)"
    )
    parser.add_argument('-c',
                        '--consult',
                        action='store_true',
                        help="Consult mode: paste debug messages directly.")
    parser.add_argument('command',
                        nargs=argparse.REMAINDER,
                        help="The command to auto-fix")
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

    # Parse known args so argparse doesn't trip over flags meant for the underlying command
    args, unknown = parser.parse_known_args()

    # Check if Ollama connection/model is available
    check_args_connections(args)

    # Combine the recognized 'command' with any unknown flags (like `fix ls -la`)
    full_command = args.command + unknown

    # If the user piped data into the script, force consult mode automatically
    if not sys.stdin.isatty():
        args.consult = True

    # Force consult mode if data is piped or -c flag is used
    if not sys.stdin.isatty() or args.consult:
        run_consult_mode()
    else:
        run_auto_mode(full_command)
