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
# |_|\_\  |_| |_|  Datetime: 2026-02-07 17:18:47             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import json
import os
import re
import sys

import requests
from tqdm import tqdm


def get_existing_master_tags(filepath):
    """Gets ## Headings from TAG.md"""
    if not os.path.exists(filepath) or filepath is None:
        return []
    with open(filepath, 'r') as f:
        return [
            line.strip('# ').strip() for line in f if line.startswith('##')
        ]


def get_current_file_tags(content):
    """Extracts existing tags from the first two lines of the note."""
    lines = content.splitlines()
    found_tags = []
    # Check first 2 lines for #tags or :tags:
    for line in lines[:2]:
        # Match #tag or :tag:
        tags = re.findall(r'#(\w+)|:(\w+):', line)
        for t in tags:
            found_tags.append(t[0] or t[1])
    return list(set(found_tags))


def call_ollama_streaming(prompt,
                          phase_name,
                          ollama_url="http://localhost:11434",
                          model_name="gemma3:12b",
                          ctx_window=4096,
                          verbose=False):

    # Payload
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_ctx": ctx_window,
            "temperature": 0
        }
    }

    if verbose:
        print(f"\n>>> Phase: {phase_name}")
    full_response = ""
    stats = {}

    with requests.post(f"{ollama_url}/api/generate", json=payload,
                       stream=True) as response:
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if not chunk.get("done"):
                    content = chunk.get("response", "")
                    full_response += content
                    if verbose:
                        print(content, end="", flush=True)
                else:
                    stats = chunk

    # Performance Reporting
    p_tokens = stats.get('prompt_eval_count', 0)
    o_tokens = stats.get('eval_count', 0)
    duration = stats.get('eval_duration', 1) / 1e9
    if verbose:
        print(
            f"\n--- {phase_name} Stats: {p_tokens} token-in / {o_tokens} token-out | {o_tokens/duration:.1f} t/s ---"
        )

    # Prompt truncated alert
    if verbose:
        if p_tokens >= ctx_window:
            print(
                f"⚠️  CRITICAL: Prompt truncated! ({p_tokens}/{ctx_window} tokens)"
            )

    return full_response


def update_note_in_place(input_note_path, tags, verbose=False):
    with open(input_note_path, 'r') as f:
        lines = f.readlines()

    # Define the new tag lines
    obsidian_line = "- Tags: " + " ".join([f"#{t}" for t in tags]) + "\n"
    vimwiki_line = "- Vimwiki: :" + ":".join(tags) + ":\n"
    separator = "\n---\n"

    # 1. Remove old tags and separators at the very top
    # This prevents stacking multiple --- if you run the script twice
    content_start = 0
    while content_start < len(lines):
        line = lines[content_start].strip()
        if line.startswith('- Tags: #') \
                and not line.startswith('# '):  # Likely Obsidian tags
            content_start += 1
        elif line.startswith('- Vimwiki: :') \
            and line.endswith(':'):  # Likely Vimwiki tags
            content_start += 1
        elif line == '---':  # Existing separator
            content_start += 1
        elif line == '':  # Empty space
            content_start += 1
        else:
            break

    # 2. Find the index of the first Level 1 Heading (# Title)
    h1_index = -1
    for i in range(content_start, len(lines)):
        if lines[i].startswith('# '):
            h1_index = i
            break

    # 3. Reconstruct the file
    new_header = [obsidian_line, vimwiki_line, separator, "\n"]

    # If we found an H1, insert the header before it.
    # If no H1 exists, just put them at the top.
    if h1_index != -1:
        final_lines = new_header + lines[h1_index:]
    else:
        final_lines = new_header + lines[content_start:]
    # Update in place
    with open(input_note_path, 'w') as f:
        f.writelines(final_lines)

    # Summary
    if verbose:
        print(f"\n✅ File updated. Tags: {', '.join(tags)}")


def extract_md_file_keyword(inpute_md_file_path,
                            tag_file_path=None,
                            ollama_url="http://localhost:11434",
                            model_name="gemma3:12b",
                            ctx_window=4096,
                            max_input_char=-1,
                            verbose=False,
                            update_in_place=False):

    # Load note file
    if not os.path.exists(inpute_md_file_path):
        print(f'[ERROR] {inpute_md_file_path} is not a file ...')
        return
    with open(inpute_md_file_path, 'r') as f:
        full_content = f.read()

    # Get existing tags
    master_tags = get_existing_master_tags(tag_file_path)
    file_tags = get_current_file_tags(full_content)

    # Keyword Extraction phase
    gen_prompt = (
        f"Extract 3-5 keywords from this note. "
        f"Current tags in file: {', '.join(file_tags)}. "
        f"Return ONLY a comma-separated list.\n\nTEXT:\n{full_content[:max_input_char]}"
    )
    raw_keywords = [
        t.strip() for t in call_ollama_streaming(gen_prompt,
                                                 "Extraction",
                                                 ollama_url=ollama_url,
                                                 model_name=model_name,
                                                 ctx_window=ctx_window,
                                                 verbose=verbose).split(',')
    ]

    # Reconcile & Abbreviate phase
    all_known_tags = list(set(master_tags + file_tags))
    match_prompt = f"""
    TASK: Reconcile keywords with known tags and abbreviate.
    KEYWORDS: {', '.join(raw_keywords)}
    KNOWN TAGS: {', '.join(all_known_tags)}

    RULES:
    1. If a keyword matches a Known Tag, use it.
    2. Use short abbreviations (e.g., 'Development' -> 'dev').
    3. Use abbreviations but do not overshorten (e.g., DO NOT shorten 'tool' -> 'tl')
    4. Return ONLY a comma-separated list of 3-8 tags.
    """
    final_tags_str = call_ollama_streaming(match_prompt,
                                           "Abbreviation",
                                           ollama_url=ollama_url,
                                           model_name=model_name,
                                           ctx_window=ctx_window,
                                           verbose=verbose)

    # Update markdown note in place
    if update_in_place:
        final_tags = [
            t.strip().lower().replace(" ", "_")
            for t in final_tags_str.split(',')
        ]
        update_note_in_place(inpute_md_file_path, final_tags, verbose=verbose)

    return final_tags_str.strip('\n').split(',')


if __name__ == '__main__':

    # Ollama parameter
    ollama_url = "http://localhost:11434"
    model_name = "gemma3:12b"
    ctx_window = 4096

    # Tag file in vimwiki (tag stores as level-2 heading in markdown)
    tag_file_path = "$HOME/Documents/KNOWLEDGE_BASE/TAG.md"

    # Main
    inpute_md_file_paths = sys.argv[1:]
    for inpute_md_file_path in tqdm(inpute_md_file_paths):
        print(f'[INFO] INPUT: {inpute_md_file_path}')
        tags = extract_md_file_keyword(inpute_md_file_path,
                                       tag_file_path,
                                       ollama_url=ollama_url,
                                       model_name=model_name,
                                       ctx_window=ctx_window,
                                       verbose=True,
                                       update_in_place=True)
