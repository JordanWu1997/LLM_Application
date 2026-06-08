#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8

import json
import sys

import requests

MODEL = "Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
PORT = 8081

url = f"http://localhost:{PORT}/v1/chat/completions"
headers = {"Content-Type": "application/json"}

payload = {
    "model": f"{MODEL}",
    "messages": [{
        "role":
        "user",
        "content":
        "What are the three laws of robotics? Think step by step."
    }],
    "temperature":
    0.7,
    "max_tokens":
    2048,  # INCREASED: Reasoning models need lots of room to think!
    "stream":
    True
}

print("Sending request... \n")

# Trackers to make the output look clean
has_started_thinking = False
has_started_answering = False

try:
    response = requests.post(url, headers=headers, json=payload, stream=True)
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8').strip()

            if decoded_line.startswith('data:'):
                data_str = decoded_line[5:].strip()

                if data_str == '[DONE]':
                    break
                if not data_str:
                    continue

                try:
                    data_json = json.loads(data_str)
                    choices = data_json.get('choices', [])

                    if choices:
                        delta = choices[0].get('delta', {})

                        # 1. Check for the model's inner thoughts
                        reasoning = delta.get('reasoning_content', '')
                        if reasoning:
                            if not has_started_thinking:
                                print("\n--- 🧠 Model is Thinking ---")
                                has_started_thinking = True

                            sys.stdout.write(reasoning)
                            sys.stdout.flush()

                        # 2. Check for the final actual answer
                        content = delta.get('content', '')
                        if content:
                            if not has_started_answering:
                                print("\n\n--- 🤖 Final Answer ---")
                                has_started_answering = True

                            sys.stdout.write(content)
                            sys.stdout.flush()

                except json.JSONDecodeError:
                    pass

    print("\n\n--- Done ---")

except Exception as e:
    print(f"\nConnection or Parsing Error: {e}")
