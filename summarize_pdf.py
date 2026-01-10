#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8
"""
Inspired by:
- https://freedium-mirror.cfd/https://blog.stackademic.com/11-python-scripts-to-automate-your-daily-tasks-4a48fc34ac8e
"""

import datetime
import os

import ollama
import pymupdf
from tqdm import tqdm


def find_pdfs(input_dir):
    return [
        f'{input_dir}/{f}' for f in os.listdir(input_dir) if f.endswith('.pdf')
    ]


def summarize_arxiv_pdf(pdf_path,
                        model="gemma3:4b",
                        text_max_str=4000,
                        stream=True,
                        verbose=False):

    # 1. Extract text from the PDF
    text = ""
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()

    # 2. Prepare the prompt (focusing on the abstract/intro)
    # We slice the text to avoid hitting context window limits if the paper is long
    input_text = text[:text_max_str]
    prompt = f"Please provide a concise summary of the following research paper abstract/introduction:\n\n{input_text}"

    # 3. Query Ollama
    stream = ollama.chat(model=model,
                         messages=[
                             {
                                 'role': 'user',
                                 'content': prompt,
                             },
                         ],
                         stream=True)

    # 4. Print
    full_summary, final_metadata = "", None
    for chunk in stream:
        content = chunk['message']['content']
        if stream:
            print(content, end='', flush=True)
        full_summary += content
        if chunk.get('done'):
            final_metadata = chunk

    # 5. Ollama verbose
    if final_metadata and verbose:
        # Convert nanoseconds to seconds
        total_sec = final_metadata.get('total_duration', 0) / 1e9
        # Avoid div by zero
        eval_sec = final_metadata.get('eval_duration', 1) / 1e9
        prompt_tokens = final_metadata.get('prompt_eval_count', 0)
        response_tokens = final_metadata.get('eval_count', 0)
        # Calculate tokens per second
        tokens_per_sec = response_tokens / eval_sec
        # Print
        print(f"\n\n{'-'*20} PERFORMANCE REPORT {'-'*20}")
        print(f"• Tokens Generated:   {response_tokens}")
        print(f"• Prompt Tokens:      {prompt_tokens}")
        print(f"• Generation Speed:   {tokens_per_sec:.2f} tokens/s")
        print(f"• Total Time:         {total_sec:.2f}s")
        print(f"{'-'*60}\n")

    return full_summary


def save_as_markdown(input_file_path, summary, output_folder="summaries"):

    # Init output
    os.makedirs(output_folder, exist_ok=True)

    # Create a clean filename
    filename = os.path.basename(input_file_path)
    safe_name = filename.replace(".pdf", ".md")

    # Get time
    datetime_dt = datetime.datetime.today()
    datetime_str = datetime_dt.strftime("%Y/%m/%d %H:%M:%S")

    # Content
    content = f"""---
source: {input_file_path}
date: {datetime_str}
---

# Summary of {filename}

{summary}
"""

    # Save content
    with open(f"{output_folder}/{safe_name}", "w") as f:
        f.write(content)


def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize Arxiv PDFs using local Ollama LLM.")
    parser.add_argument("-i",
                        "--input",
                        dest="input_pdf_dir",
                        required=True,
                        help="Path to the directory containing PDF files")
    parser.add_argument(
        "-o",
        "--output",
        dest="output_md_dir",
        default="./output",
        help=
        "Path to save the generated Markdown summaries (default: ./summaries)")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show Ollama performance statistics (tokens, speed, etc.)")
    parser.add_argument(
        "-s",
        "--stream",
        action="store_true",
        help="Stream the LLM output to the terminal in real-time")
    args = parser.parse_args()

    # Find pdf file paths
    pdf_file_paths = find_pdfs(args.input_pdf_dir)

    # Loop through pdf files
    for pdf_file_path in tqdm(pdf_file_paths):
        print(f'[INFO] Input: {pdf_file_path}\n')
        summary = summarize_arxiv_pdf(pdf_file_path,
                                      verbose=args.verbose,
                                      stream=args.stream)
        # Store summary
        save_as_markdown(pdf_file_path,
                         summary,
                         output_folder=args.output_md_dir)


if __name__ == '__main__':
    main()
