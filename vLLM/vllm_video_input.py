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
# |_|\_\  |_| |_|  Datetime: 2026-08-30 23:23:56             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import time

from openai import OpenAI


def test_vllm_video_input(file_url,
                          base_url="http://localhost:8000/v1",
                          verbose=False):

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    model = client.models.list().data[0].id

    # 1. Start the timer before making the request
    start_time = time.perf_counter()
    first_token_time = None
    response_text = ""
    reasoning_text = ""
    completion_tokens = 0
    usage_data = None
    transitioned_to_answer = False

    if verbose:
        print(f"Sending request to vLLM server (Model: {model})...\n")
        print("--- Thinking Process ---")

    # 2. Call the API with stream=True and include_usage=True
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role":
                "system",
                "content":
                """
You are an expert video analysis AI. Your task is to analyze the provided video and generate a precise, chronological log of all significant events, actions, and scene changes.
You must output your analysis strictly using the following format for each distinct event: MM:SS-MM:SS Description of the action

Rules:
- Answer in **Traditional Chinese**
- Output only the timestamped list. Do not include conversational filler, greetings, or summary paragraphs.
- Ensure timestamps are accurate to the second and cover the entire duration of the video sequentially.
- Keep descriptions concise, focusing on primary physical actions, visual transitions, and key audio events.
"""
            },
            {
                "role":
                "user",
                "content": [
                    {
                        "type":
                        "text",
                        "text":
                        """
Analyze the attached video and provide a complete timeline of events.
Break the video down into distinct segments and describe exactly what happens in each timeframe using the strict MM:SS-MM:SS format.
Answer in **Traditional Chinese**
"""
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"file://{file_url}",
                        },
                    },
                ],
            },
        ],
        seed=42,
        max_tokens=1024,
        stream=True,
        stream_options={"include_usage": True})

    # 3. Iterate over the stream chunks
    for chunk in stream:
        # A. The final chunk contains the usage statistics (and empty choices)
        if chunk.usage:
            usage_data = chunk.usage
            completion_tokens = chunk.usage.completion_tokens

        # Skip the rest of the loop if choices is empty (happens on the final usage chunk)
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # Safely extract reasoning and content tokens
        reasoning_token = getattr(delta, "reasoning", None) or getattr(
            delta, "reasoning_content", None)
        content_token = delta.content

        # B. Capture TTFT on the very first token (either reasoning or text)
        if first_token_time is None and (reasoning_token or content_token):
            first_token_time = time.perf_counter()

        # C. Process Reasoning Tokens
        if reasoning_token:
            reasoning_text += reasoning_token
            if verbose:
                print(reasoning_token, end="", flush=True)

        # D. Process Final Content Tokens
        if content_token:
            if not transitioned_to_answer:
                transitioned_to_answer = True
                if verbose and reasoning_text:
                    print("\n\n--- Final Answer ---")

            response_text += content_token
            if verbose:
                print(content_token, end="", flush=True)

    # 4. Stop the timer when the stream ends
    end_time = time.perf_counter()

    # 5. Calculate statistics
    # Fallback just in case the model returns an empty response
    if first_token_time is None:
        first_token_time = end_time

    # TTFT: Time from request start to first token received
    ttft = first_token_time - start_time

    # Generation Time: Time from first token to the last token
    generation_time = end_time - first_token_time

    # TPS: Total generated tokens divided by the generation time
    tps = completion_tokens / generation_time if generation_time > 0 else 0

    if verbose:
        print("\n\n" + "=" * 40)
        print("📊 PERFORMANCE STATISTICS")
        print("=" * 40)
        print(f"Time to First Token (TTFT): {ttft:.3f} seconds")
        print(f"Generation Time:            {generation_time:.3f} seconds")
        print(f"Tokens Per Second (TPS):    {tps:.2f} tokens/sec")
        print(f"Total Output Tokens:        {completion_tokens} tokens")

        # Print reasoning token count if the model reported it
        if usage_data and getattr(usage_data, "completion_tokens_details",
                                  None):
            reasoning_count = getattr(usage_data.completion_tokens_details,
                                      "reasoning_tokens", 0)
            if reasoning_count > 0:
                print(f"Reasoning Tokens:           {reasoning_count} tokens")

    return response_text


if __name__ == '__main__':

    import sys

    base_url = 'http://localhost:8000/v1'
    for input_file_path in sys.argv[1:]:
        _ = test_vllm_video_input(input_file_path,
                                  base_url=base_url,
                                  verbose=True)
