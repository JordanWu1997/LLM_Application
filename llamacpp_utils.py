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
# |_|\_\  |_| |_|  Datetime: 2026-06-26 22:27:59             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import json

import requests

from ollama_utils import print_context_warning


def stream_llamacpp_chat(prompt,
                         system_prompt=None,
                         llamacpp_url="http://localhost:8082",
                         ctx_window=4096,
                         verbose=False):

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": f"{system_prompt}"})
    messages.append({"role": "user", "content": f"{prompt}"})

    # OpenAI-compatible payload
    payload = {
        # llama.cpp's OpenAI endpoint usually ignores this, but it's safe to include a dummy name
        "model": "local-model",
        "messages": messages,
        "stream": True,
        "temperature": 0.1
    }

    full_response = ""
    stats = {}

    with requests.post(f"{llamacpp_url}/v1/chat/completions",
                       json=payload,
                       stream=True) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')

                # OpenAI SSE format always starts with 'data: '
                if decoded_line.startswith("data: "):
                    json_str = decoded_line[6:]

                    # '[DONE]' is the strict OpenAI indicator that the stream is finished
                    if json_str.strip() == "[DONE]":
                        break

                    if not json_str.strip():
                        continue

                    chunk = json.loads(json_str)

                    # 1. Safely extract the streaming text from the choices array
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})

                        # Only grab 'content' if it exists (the final chunk won't have it)
                        if "content" in delta:
                            text = delta["content"]
                            if text:
                                print(text, end="", flush=True)
                                full_response += text

                    # 2. Catch the llama.cpp timings in the final chunk
                    if "timings" in chunk:
                        timings = chunk["timings"]
                        p_tokens = timings.get("prompt_n", 0)
                        o_tokens = timings.get("predicted_n", 0)
                        t_s = timings.get("predicted_per_second", 0.0)

                        print(
                            f"\n\n--- Stats: {p_tokens} token-in / {o_tokens} token-out | {t_s:.1f} t/s ---"
                        )

    # Performance Reporting Extraction
    # llama.cpp uses different keys for its evaluation metrics
    p_tokens = stats.get('tokens_evaluated', 0)
    o_tokens = stats.get('tokens_predicted', 0)

    # llama.cpp nests duration details inside a "timings" dictionary
    timings = stats.get('timings', {})

    if verbose:
        # We can pull the tokens-per-second metric directly since llama.cpp pre-calculates it,
        # or calculate it manually using 'predicted_ms' to match your original script's math.
        predicted_ms = timings.get(
            'predicted_ms', 1000)  # Default to 1000ms to avoid div by zero
        calculated_t_s = (o_tokens /
                          (predicted_ms / 1000.0)) if predicted_ms > 0 else 0.0

        # Prefer the native pre-calculated metric if available
        t_s = timings.get('predicted_per_second', calculated_t_s)

        print(
            f"\n--- Stats: {p_tokens} token-in / {o_tokens} token-out | {t_s:.1f} t/s ---"
        )

    # Prompt truncated alert (assuming this is defined elsewhere)
    print_context_warning(prompt_tokens=p_tokens, ctx_window=ctx_window)

    return full_response


def stream_llamacpp_generate(prompt,
                             system_prompt=None,
                             llamacpp_url="http://localhost:8082",
                             ctx_window=4096,
                             verbose=False):

    if system_prompt is not None:
        prompt = f'{system_prompt} {prompt}'

    # OpenAI-compatible payload
    payload = {
        "model": "local-model",
        "prompt": prompt,
        "stream": True,
        "temperature": 0.1
    }

    full_response = ""
    stats = {}

    with requests.post(f"{llamacpp_url}/v1/completions",
                       json=payload,
                       stream=True) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')

                # OpenAI SSE format always starts with 'data: '
                if decoded_line.startswith("data: "):
                    json_str = decoded_line[6:]

                    # '[DONE]' is the strict OpenAI indicator that the stream is finished
                    if json_str.strip() == "[DONE]":
                        break

                    if not json_str.strip():
                        continue

                    chunk = json.loads(json_str)

                    # 1. Safely extract the streaming text from the choices array
                    choices = chunk.get("choices", [])
                    if choices:
                        # Generate API uses 'text' directly on the choice object, not inside a 'delta'
                        text = choices[0].get("text", "")
                        if text:
                            print(text, end="", flush=True)
                            full_response += text

                    # 2. Catch the llama.cpp timings in the final chunk
                    if "timings" in chunk:
                        timings = chunk["timings"]
                        p_tokens = timings.get("prompt_n", 0)
                        o_tokens = timings.get("predicted_n", 0)
                        t_s = timings.get("predicted_per_second", 0.0)

                        print(
                            f"\n\n--- Stats: {p_tokens} token-in / {o_tokens} token-out | {t_s:.1f} t/s ---"
                        )

    # Performance Reporting Extraction
    # llama.cpp uses different keys for its evaluation metrics
    p_tokens = stats.get('tokens_evaluated', 0)
    o_tokens = stats.get('tokens_predicted', 0)

    # llama.cpp nests duration details inside a "timings" dictionary
    timings = stats.get('timings', {})

    if verbose:
        # We can pull the tokens-per-second metric directly since llama.cpp pre-calculates it,
        # or calculate it manually using 'predicted_ms' to match your original script's math.
        predicted_ms = timings.get(
            'predicted_ms', 1000)  # Default to 1000ms to avoid div by zero
        calculated_t_s = (o_tokens /
                          (predicted_ms / 1000.0)) if predicted_ms > 0 else 0.0

        # Prefer the native pre-calculated metric if available
        t_s = timings.get('predicted_per_second', calculated_t_s)

        print(
            f"\n--- Stats: {p_tokens} token-in / {o_tokens} token-out | {t_s:.1f} t/s ---"
        )

    # Prompt truncated alert (assuming this is defined elsewhere)
    print_context_warning(prompt_tokens=p_tokens, ctx_window=ctx_window)

    return full_response


if __name__ == '__main__':

    prompt = """
diff --git a/generate_commit.py b/generate_commit.py
index 8997189..b7e94da 100644
--- a/generate_commit.py
+++ b/generate_commit.py
@@ -22,6 +22,7 @@ import tempfile

 import requests

+from llamacpp_utils import stream_llamacpp_generate
 from ollama_utils import (check_args_connections, print_context_warning,
                           stream_ollama_generate)
"""

    system_prompt = """
Write a concise Conventional Commit message. Output ONLY the message text.
"""

    print("Generate")
    ai_message = stream_llamacpp_generate(
        prompt,
        system_prompt=system_prompt,
        llamacpp_url=f'http://192.168.1.117:8082',
        ctx_window=4096,
        verbose=True)

    print("Chat")
    ai_message = stream_llamacpp_chat(
        prompt,
        system_prompt=system_prompt,
        llamacpp_url=f'http://192.168.1.117:8082',
        ctx_window=4096,
        verbose=True)
