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


def stream_log_analysis(log_data):
    system_prompt = (
        "You are a Linux System Expert. Analyze the following logs. "
        "1. Identify the most critical error. "
        "2. Explain the likely root cause. "
        "3. Provide a command-line solution to fix it.")

    payload = {
        "model": MODEL,
        "prompt": f"{system_prompt}\n\nLOG DATA:\n{log_data}",
        "stream": True,
        "options": {
            "num_ctx": CONTEXT_WINDOW,
            "temperature": 0.1
        }
    }

    full_response = ""
    metadata = {}

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        response.raise_for_status()

        print(f"\n[INFO] 🔍 Analyzing logs with {MODEL}...\n" + "=" * 30)

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)

                if 'response' in chunk:
                    content = chunk['response']
                    print(content, end="", flush=True)
                    full_response += content

                if chunk.get('done'):
                    metadata = chunk

        print("\n" + "=" * 30)

        # Token Truncation Check
        prompt_tokens = metadata.get("prompt_eval_count", 0)
        if prompt_tokens >= CONTEXT_WINDOW:
            print(f"\033[91m⚠️  CRITICAL: Log data was TRUNCATED.\033[0m")
            print(
                f"The logs used {prompt_tokens} tokens, hitting the {CONTEXT_WINDOW} limit."
            )
            print(
                "Action: Shorten the time interval or increase CONTEXT_WINDOW.\n"
            )
        else:
            print(
                f"\033[90m[Analysis complete. Context used: {prompt_tokens}/{CONTEXT_WINDOW} tokens]\033[0m\n"
            )

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] ❌ API Connection Error: {e}")


if __name__ == "__main__":

    # Configuration
    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL = "qwen2.5-coder:7b"
    CONTEXT_WINDOW = 128 * 1024  # Logs need a larger window

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

    # Analyze retrieved logs
    stream_log_analysis(logs)
