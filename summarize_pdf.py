#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8
"""
Inspired by:
- https://freedium-mirror.cfd/https://blog.stackademic.com/11-python-scripts-to-automate-your-daily-tasks-4a48fc34ac8e
"""

import datetime
import json
import os
import re
import sys

import pymupdf
import requests
from tqdm import tqdm

# Global structure mapping to ensure consistency between CLI arguments and parsing indices
STRUCT_KEYS = [
    "Introduction", "Methods", "Results", "Discussion", "Conclusion"
]


def collect_target_pdfs(input_targets):
    """
    Accepts a list of target paths (files or directories) and flattens them
    into a unique, clean list of absolute PDF file paths.
    """
    discovered_pdfs = set()

    for target in input_targets:
        target = os.path.abspath(target.strip())

        if os.path.isdir(target):
            # Recursively walk the directory tree to gather all PDFs
            for root, _, files in os.walk(target):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        discovered_pdfs.add(os.path.join(root, file))
        elif os.path.isfile(target):
            if target.lower().endswith('.pdf'):
                discovered_pdfs.add(target)
            else:
                print(
                    f"[WARN] ⚠️ Skipping non-PDF file target: {os.path.basename(target)}"
                )
        else:
            print(f"[ERROR] ❌ Input target path does not exist: {target}")

    return sorted(list(discovered_pdfs))


def interactive_section_selector(available_sections):
    """Asks the user via terminal which sections they want to prioritize for analysis."""
    print("\n" + "=" * 50)
    print(" 🛠️  ACADEMIC PAPER SECTION SELECTION")
    print("=" * 50)
    print("Title, Authors, and Abstract are selected automatically.")

    selectable_keys = [k for k in available_sections.keys() if k != "Abstract"]
    valid_options = []

    for idx, sec in enumerate(selectable_keys, start=1):
        status = "✅ Found Content" if available_sections[
            sec] else "❌ Empty/Not Found"
        print(f" [{idx}] {sec:<15} ({status})")
        if available_sections[sec]:
            valid_options.append(str(idx))

    print(" [A] All Available Sections")
    print(" [N] None (Abstract Only)")
    print("=" * 50)

    user_choice = input(
        "[INFO] Choose section numbers to include (e.g., 1,3,4 or A/N): "
    ).strip().upper()

    selected_sections = []
    if user_choice == "A":
        selected_sections = [
            k for k in selectable_keys if available_sections[k]
        ]
    elif user_choice == "N" or not user_choice:
        selected_sections = []
    else:
        choices = [c.strip() for c in user_choice.split(",")]
        for c in choices:
            if c in valid_options:
                target_key = selectable_keys[int(c) - 1]
                selected_sections.append(target_key)

    return selected_sections


def resolve_automated_sections(cli_args, available_sections):
    """
    Validates user choices provided via the non-interactive --sections option.
    Falls back to interactive selections if the input values are missing or malformed.
    """
    if cli_args.sections is None:
        return interactive_section_selector(available_sections)

    cleaned_tokens = [str(t).strip().upper() for t in cli_args.sections]

    if "A" in cleaned_tokens:
        print(
            "[INFO] 🤖 Automation Engine: Selecting ALL available document sections."
        )
        return [k for k in STRUCT_KEYS if available_sections[k]]

    if "N" in cleaned_tokens:
        print(
            "[INFO] 🤖 Automation Engine: Selecting NONE (Abstract-only summary mode)."
        )
        return []

    selected_sections = []
    for token in cleaned_tokens:
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(STRUCT_KEYS):
                target_key = STRUCT_KEYS[idx]
                if available_sections[target_key]:
                    selected_sections.append(target_key)
                else:
                    print(
                        f"[WARN] ⚠️ Automation section match '{target_key}' is empty in this paper. Skipping."
                    )
            else:
                print(
                    f"[ERROR] ❌ Index '{token}' out of bounds. Must be 1 to {len(STRUCT_KEYS)}."
                )
        else:
            print(
                f"[WARN] ⚠️ Invalid section identifier format '{token}' ignored."
            )

    print(
        f"[INFO] 🤖 Automation Engine Selected Sections: {['Abstract'] + selected_sections}"
    )
    return selected_sections


def parse_arxiv_pdf_sections(pdf_path):
    """
    Scans the PDF to extract Title, Authors, Abstract, and maps sections
    like Introduction, Methods, Results, Discussion, and Conclusion.
    """
    title = "Unknown Title"
    authors = "Unknown Authors"

    # Structural keys to search for text blocks
    sections = {
        "Abstract": "",
        "Introduction": "",
        "Methods": "",
        "Results": "",
        "Discussion": "",
        "Conclusion": ""
    }

    # Regex rules matching typical academic headers
    sec_regex = {
        "Abstract":
        r'(?:abstract|ABSTRACT)[\s\.:\-–\n]+(.*?)(?=\n\s*(?:1\.?\s+|introduction|INTRODUCTION|\Z))',
        "Introduction":
        r'(?:1\.?\s+|\b)(?:introduction|INTRODUCTION)[\s\.:\-–\n]+(.*?)(?=\n\s*(?:2\.?\s+|\b)(?:methods|methodology|related|background|METHODS|\Z))',
        "Methods":
        r'(?:\d\.?\s+|\b)(?:methods|methodology|experimental setup|METHODS)[\s\.:\-–\n]+(.*?)(?=\n\s*(?:\d\.?\s+|\b)(?:results|evaluation|RESULTS|\Z))',
        "Results":
        r'(?:\d\.?\s+|\b)(?:results|evaluation|experimental findings|RESULTS)[\s\.:\-–\n]+(.*?)(?=\n\s*(?:\d\.?\s+|\b)(?:discussion|limitations|DISCUSSION|\Z))',
        "Discussion":
        r'(?:\d\.?\s+|\b)(?:discussion|DISCUSSION)[\s\.:\-–\n]+(.*?)(?=\n\s*(?:\d\.?\s+|\b)(?:conclusion|future work|CONCLUSION|\Z))',
        "Conclusion":
        r'(?:\d\.?\s+|\b)(?:conclusion|conclusions|summary and outlook|CONCLUSION)[\s\.:\-–\n]+(.*?)(?=\n\s*(?:references|acknowledgements|REFERENCES|\Z))'
    }

    try:
        # 1. Gather all document text to run targeted structural slicing
        full_text = ""
        with pymupdf.open(pdf_path) as doc:
            for page in doc:
                full_text += page.get_text()

        # Clean text to ease regex tracking across line breaks
        cleaned_text = re.sub(r'[ \t]+', ' ', full_text)
        cleaned_text = re.sub(r'\n+', '\n', cleaned_text).strip()

        # 2. Slice standard sections out of body text
        for sec_name, pattern in sec_regex.items():
            match = re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[sec_name] = match.group(1).strip()

        # 3. Handle Title & Authors from the layout matrix of page 1
        with pymupdf.open(pdf_path) as doc:
            if len(doc) > 0:
                blocks = doc[0].get_text("blocks")
                valid_blocks = [
                    b[4].strip() for b in blocks
                    if b[4].strip() and "arXiv:" not in b[4]
                ]
                if len(valid_blocks) >= 2:
                    title = valid_blocks[0].replace('\n', ' ')
                    authors = valid_blocks[1].replace('\n', ' ')

    except Exception as e:
        print(f"[WARN] ⚠️ Structural layout parsing errored out: {e}")

    return title, authors, sections


def summarize_arxiv_pdf(pdf_path,
                        selected_additional_sections=None,
                        ollama_url="http://localhost:11434",
                        model="gemma3:12b",
                        context_window=4096,
                        stream=True,
                        verbose=False):

    # Extract targeted metadata using structural heuristics
    title, authors, sections = parse_arxiv_pdf_sections(pdf_path)

    # Prepare the structural prompt
    prompt_body = (f"You are evaluating a research paper.\n\n")

    prompt_body += (
        f"Title: {title}\n"
        f"Authors: {authors}\n\n"
        f"Abstract:\n{sections['Abstract'] if sections['Abstract'] else 'Not available.'}\n\n"
    )

    if selected_additional_sections is not None:
        prompt_body += "--- ADDITIONAL SECTIONS CHOSEN BY USER ---\n"
        for sec in selected_additional_sections:
            truncated_section = sections[sec][:12000]
            prompt_body += f"\n### {sec}:\n{truncated_section}\n"

    prompt_body += (
        f"\nInstructions:\n"
        f"Please provide a cohesive, structured, and informative summary of this paper "
        f"based strictly on the metadata and structural textual blocks provided above."
    )

    # Generate context
    payload = {
        "model": model,
        "messages": [
            {
                'role': 'user',
                'content': prompt_body,
            },
        ],
        "stream": True,
        "options": {
            "num_ctx": context_window,
        }
    }

    # Query ollama and Print
    full_summary, final_metadata = "", None
    try:
        response = requests.post(f'{ollama_url}/api/chat',
                                 json=payload,
                                 stream=True)
        response.raise_for_status()

        if verbose:
            print(f"\n[INFO] 🔍 Summarizing with {model}...\n" + "=" * 40)

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "message" in chunk and chunk["message"].get("content"):
                    content = chunk['message']['content']
                    full_summary += content
                    if stream:
                        print(content, end='', flush=True)
                if chunk.get('done'):
                    final_metadata = chunk

        # --- Token Truncation Engine ---
        prompt_tokens = final_metadata.get("prompt_eval_count", 0)
        print()
        if prompt_tokens >= context_window:
            print(
                f"\033[91m⚠️  CRITICAL: Paper metadata prompt was TRUNCATED.\033[0m"
            )
            print(
                f"The input metadata context hit your threshold limit: {prompt_tokens}/{context_window} tokens."
            )
            print(
                "Action: Consider expanding your execution --ctx values configuration.\n"
            )
        else:
            if verbose:
                print(
                    f"\033[90m[Analysis complete. Context used: {prompt_tokens}/{context_window} tokens]\033[0m\n"
                )

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] ❌ API Connection Error: {e}")

    # Ollama verbose
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

    return full_summary, title, authors, sections


def save_as_markdown(input_file_path,
                     summary,
                     title,
                     authors,
                     sections,
                     chosen_sections=[],
                     output_folder="./output"):
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.basename(input_file_path)
    safe_name = filename.replace(".pdf", ".md")
    datetime_str = datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")

    # Frontmatter builds context based on user targets
    content = f"""---
source: {filename}
extracted_title: "{title}"
authors: "{authors}"
analyzed_sections: {["Abstract"] + chosen_sections}
date: {datetime_str}
---

# {title}

**Authors:** {authors}

## Original Abstract
> {sections['Abstract'] if sections['Abstract'] else 'No abstract parsed.'}

---

## LLM Summary
{summary}
"""

    with open(os.path.join(output_folder, safe_name), "w",
              encoding="utf-8") as f:
        f.write(content)


def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize Arxiv PDFs via flexible file/directory trees")
    # nargs="+" allows passing a space-separated list of multiple files, directories, or both
    parser.add_argument(
        "input_targets",
        nargs="+",
        help=
        "Paths to PDF files, directories, or a mix of both (space-separated)")
    parser.add_argument(
        "-o",
        "--output",
        dest="output_md_dir",
        default="./output",
        help=
        "Path to save the generated Markdown summaries (default: ./summaries)")
    # Fully detailed section definitions inside the argument configuration
    parser.add_argument(
        "-s",
        "--sections",
        dest="sections",
        nargs="+",
        default=None,
        help="Automated selections. Pass option values space-separated.\n"
        "Title, Authors, and Abstract are always included by default.\n\n"
        "Available Indices Mapping:\n"
        "  1 : Introduction\n"
        "  2 : Methods (Methodology, Experimental Setup)\n"
        "  3 : Results (Evaluation, Findings)\n"
        "  4 : Discussion (Limitations)\n"
        "  5 : Conclusion (Summary and Outlook)\n\n"
        "Control Shortcuts:\n"
        "  A : All available sections combined\n"
        "  N : None (Extracts Abstract only)\n\n"
        "Example: '-s 1 3 5' runs Intro, Results, and Conclusion automatically.\n"
        "Omitting this argument completely triggers the interactive selector menu."
    )
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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show Ollama performance statistics (tokens, speed, etc.)")
    parser.add_argument(
        "-n",
        "--no-stream",
        action="store_true",
        help="Disable stream the LLM output to the terminal in real-time")
    parser.add_argument(
        "--model",
        default="gemma4:e4b-8k-gpu",
        help="Model name to deploy (default: gemma4:e4b-8k-gpu)")
    parser.add_argument(
        "--ctx",
        type=int,
        default=8192,
        help=
        "Size of the context window for ollama model used to generate the next token (default: 8192)"
    )
    args = parser.parse_args()

    # Find pdf file paths
    print(f"[INFO] 📋 Academic PDF Structural Extraction Engine")

    # Run target engine to build list from files and/or directories recursively
    pdf_file_paths = collect_target_pdfs(args.input_targets)
    print(
        f'[INFO] 📋 Total {len(pdf_file_paths):d} valid unique paper tracks isolated for analysis queue.'
    )
    if not pdf_file_paths:
        sys.exit(
            "[INFO] ✅ No valid targets matching *.pdf criteria found inside execution arguments."
        )

    # Build section selector from the first available document
    print(
        f"\n[INFO] Gathering layout matrices from initial target to open interactive prompt dashboard..."
    )
    _, _, sample_sections = parse_arxiv_pdf_sections(pdf_file_paths[0])
    selected_additional_sections = resolve_automated_sections(
        args, sample_sections)

    # Loop through pdf files
    for pdf_file_path in tqdm(pdf_file_paths):
        if args.verbose:
            print(f'[INFO] Input: {pdf_file_path}\n')

        summary, title, authors, sections = summarize_arxiv_pdf(
            pdf_file_path,
            selected_additional_sections=selected_additional_sections,
            ollama_url=f"http://{args.host}:{args.port}",
            model=args.model,
            context_window=args.ctx,
            verbose=args.verbose,
            stream=not args.no_stream)

        if summary:
            save_as_markdown(pdf_file_path,
                             summary,
                             title,
                             authors,
                             sections,
                             selected_additional_sections,
                             output_folder=args.output_md_dir)


if __name__ == '__main__':
    main()
