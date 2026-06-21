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
# |_|\_\  |_| |_|  Datetime: 2026-01-16 23:28:53             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import datetime
import json
import subprocess
import sys

import requests

from ollama_utils import check_args_connections, stream_ollama_generate


def get_journal_logs(since, until, priority='3'):
    """Fetches systemd logs for the specific time range."""
    try:
        # -p 4 filters for Warning, Error, Critical, and Alert
        cmd = [
            "journalctl", "--since", since, "--until", until, "--no-pager",
            "-p", priority
        ]
        output = subprocess.check_output(cmd, text=True)
        return output.strip()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ❌ Failed to fetch logs: {e}")
        return None


def analayze_log(log_data,
                 ollama_url='http://localhost:11434',
                 model="gemma4:e4b",
                 ctx_window=4096):

    system_prompt = (
        "You are a Linux System Expert. Analyze the following logs. "
        "1. Identify the most critical error. "
        "2. Explain the likely root cause. "
        "3. Provide a command-line solution to fix it.")

    gen_prompt = (f"{system_prompt}\n"
                  f"\nLOG DATA:\n{log_data}")

    print(f"\n[INFO] 🔍 Analyzing logs with {model}...\n" + "=" * 30)
    stream_ollama_generate(gen_prompt,
                           system_prompt=system_prompt,
                           ollama_url=ollama_url,
                           model=model,
                           ctx_window=ctx_window,
                           verbose=True)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Analyze journal w/ Ollama")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Hostname or IP address of the Ollama server (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11434,
        help="Port number the Ollama server is listening on (default: 11434)")
    parser.add_argument("--model",
                        default="gemma4:e4b-8k-gpu",
                        help="Model name to use (default: gemma4:e4b-8k-gpu)")
    parser.add_argument(
        "--ctx",
        type=int,
        default=8192,
        help=
        "Size of the context window for ollama model used to generate the next token (default: 8192)"
    )
    args = parser.parse_args()

    # Check if Ollama connectio/model is available
    args = check_args_connections(args)

    # Calculate Defaults
    today_start = datetime.datetime.now().strftime("%Y-%m-%d 00:00:00")
    today_end = datetime.datetime.now().strftime("%Y-%m-%d 23:59:59")

    print(f"[INFO] 📋 Local System Log Insight Tool")
    print(
        f"[INFO] Leave blank for today's logs ({today_start} to {today_end})")

    user_since = input(f"[INFO] Start time (default: {today_start}): ").strip()
    user_until = input(f"[INFO] End time   (default: {today_end}): ").strip()

    # `journalctl` options
    since = user_since if user_since else today_start
    until = user_until if user_until else today_end
    priority = '4'

    # Get log
    logs = get_journal_logs(since, until, priority=priority)
    print(f'[INFO] 📋 Total {len(logs):d} lines (-p {priority}) are found')
    if not logs:
        sys.exit("[INFO] ✅ No warning or error logs found for this period.")

    # # Analyze retrieved logs
    analayze_log(logs,
                 ollama_url=f'http://{args.host}:{args.port}',
                 model=args.model,
                 ctx_window=args.ctx)
