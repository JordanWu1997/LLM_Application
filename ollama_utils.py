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
# |_|\_\  |_| |_|  Datetime: 2026-05-16 00:19:54             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import json
import os
import subprocess
import sys
import tempfile

import requests
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

console = Console()


def get_available_models(base_url):
    """Fetch available models from the Ollama server."""

    try:
        # Note: /api/tags (GET) or /api/ps (GET/POST) are common endpoints
        # /api/tags is the standard for listing local models
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            return [m['name'] for m in response.json().get('models', [])]
        return []
    except Exception:
        return []


def check_args_connections(args):

    base_url = f"http://{args.host}:{args.port}"
    print(f"\n[*] Checking connection to {base_url}...")
    try:
        # We use the version or tags endpoint to verify the server is alive
        hb = requests.get(f"{base_url}/api/tags", timeout=5)
        hb.raise_for_status()
    except (requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError):
        print(
            f"Error: Could not connect to Ollama at {base_url}. Is the server running?"
        )
        sys.exit(1)

    # Check if the selected model is available
    available_models = get_available_models(base_url)
    selected_model = args.model
    if selected_model not in available_models:
        print(f"[!] Model '{selected_model}' not found locally.")

        if not available_models:
            print(
                "No local models found. Please pull a model first (e.g., 'ollama pull llama3')."
            )
            sys.exit(1)
        print("\nAvailable models:")
        for i, m in enumerate(available_models):
            print(f"{i + 1}. {m}")
        try:
            choice = int(
                input(
                    f"\nSelect a model number (1-{len(available_models)}): "))
            selected_model = available_models[choice - 1]
            args.model = selected_model
            print(f"[*] Switched to model: {selected_model}")
        except (ValueError, IndexError):
            print("Invalid selection. Exiting.")
            sys.exit(1)

    print()
    print(f"--- Configuration Loaded ---")
    print(f"URL: {base_url}")
    print(f"Model: {args.model}")
    print(f"Context Window: {args.ctx}")
    print(f"---------------------------")
    print()

    return args


def stream_ollama_generate(prompt,
                           ollama_url="http://localhost:11434",
                           model="gemma3:12b",
                           ctx_window=4096,
                           verbose=False):

    # Payload
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_ctx": ctx_window,
            "temperature": 0
        }
    }

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
            f"\n--- Stats: {p_tokens} token-in / {o_tokens} token-out | {o_tokens/duration:.1f} t/s ---"
        )
    # Prompt truncated alert
    print_context_warning(prompt_tokens=p_tokens, ctx_window=ctx_window)

    return full_response


def get_input_from_editor(initial_text=""):
    """Opens the user's default terminal editor to capture multi-line input."""

    # Look for the user's preferred editor, default to vim (or nano)
    editor = os.environ.get('EDITOR', 'vim')

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".md",
                                     delete=False) as tf:
        if initial_text:
            tf.write(initial_text)
            tf.flush()

        # Close the file handle temporarily so the editor can safely open it
        tf_name = tf.name

    try:
        # Launch the editor and wait for the user to close it
        subprocess.call([editor, tf_name])

        # Read the contents after the user saves and exits
        with open(tf_name, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        return content
    finally:
        # Always clean up the temporary file afterward
        if os.path.exists(tf_name):
            os.remove(tf_name)


def print_context_warning(prompt_tokens, ctx_window):
    """
    Warn if prompt tokens are approaching or exceeding context window.
    """
    if not ctx_window or ctx_window <= 0:
        return

    usage_ratio = prompt_tokens / ctx_window
    usage_percent = usage_ratio * 100

    # Hard truncation warning
    if prompt_tokens >= ctx_window:
        print(f"\n\033[91m[WARNING] Input may be TRUNCATED "
              f"({prompt_tokens}/{ctx_window} tokens, "
              f"{usage_percent:.1f}% used)\033[0m")

    # Near-limit warning
    elif usage_ratio >= 0.90:
        print(f"\n\033[93m[WARNING] Context window nearly full "
              f"({prompt_tokens}/{ctx_window} tokens, "
              f"{usage_percent:.1f}% used)\033[0m")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Argparse Template for ollama utils")
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
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="Enable verbose mode")
    args = parser.parse_args()

    # Check if Ollama connection/model is available
    check_args_connections(args)


def preview_text_file(filepath):
    """Shows a syntax-highlighted snippet of the attached text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.splitlines()
        preview_lines = lines[:10]  # Grab the first 10 lines
        preview_text = "\n".join(preview_lines)

        if len(lines) > 10:
            preview_text += f"\n\n... [dim](and {len(lines) - 10} more lines)[/dim]"

        # Try to guess the language for syntax highlighting based on extension
        extension = filepath.split(
            '.')[-1].lower() if '.' in filepath else 'txt'

        # rich.syntax handles code highlighting automatically
        syntax = Syntax(preview_text,
                        extension,
                        theme="monokai",
                        line_numbers=True,
                        word_wrap=True)

        console.print()
        console.print(
            Panel(
                syntax,
                title=f"📄 [bold green]File Attached: {filepath}[/bold green]",
                border_style="green"))

    except Exception as e:
        console.print(
            f"[bold red]❌ Error reading file {filepath}: {e}[/bold red]")


def preview_image_file(filepath):
    """Shows the actual image in the terminal using ANSI blocks, along with metadata."""

    try:
        size_bytes = os.path.getsize(filepath)
        size_kb = size_bytes / 1024

        dimensions = "Unknown"
        image_format = filepath.split('.')[-1].upper()

        # Default placeholder if Pillow isn't installed
        preview_renderable = "[dim italic]Install 'Pillow' (pip install Pillow) to see visual previews.[/dim italic]"

        try:
            from PIL import Image
            with Image.open(filepath) as img:
                dimensions = f"{img.width} x {img.height} px"
                image_format = img.format or image_format

                # --- The ANSI Image Renderer ---
                img_rgb = img.convert("RGB")
                max_width = 50  # Keep it small enough to not wrap on standard terminals

                width, height = img_rgb.size
                aspect_ratio = height / width

                # Calculate dimensions (terminal chars are roughly 2:1 height:width)
                new_width = min(max_width, width)
                new_height = max(1, int((new_width * aspect_ratio) / 2))

                # Ensure compatibility with older and newer Pillow versions
                resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
                img_resized = img_rgb.resize((new_width, new_height * 2),
                                             resample_filter)

                lines = []
                for y in range(0, img_resized.height, 2):
                    line = Text()
                    for x in range(img_resized.width):
                        # Get top and bottom pixels
                        r1, g1, b1 = img_resized.getpixel((x, y))
                        r2, g2, b2 = img_resized.getpixel((x, y + 1))

                        # Convert to hex for Rich styling
                        hex_top = f"#{r1:02x}{g1:02x}{b1:02x}"
                        hex_bot = f"#{r2:02x}{g2:02x}{b2:02x}"

                        # ▀ is the upper half-block character
                        line.append("▀", style=f"{hex_top} on {hex_bot}")
                    lines.append(line)

                preview_renderable = Text("\n").join(lines)

        except ImportError:
            pass
        except Exception as e:
            preview_renderable = f"[red]Image render failed: {e}[/red]"

        # Create a borderless metadata table to sit underneath the image
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="bold white")
        table.add_row("File Name", os.path.basename(filepath))
        table.add_row("Format", image_format)
        table.add_row("Dimensions", dimensions)
        table.add_row("Size", f"{size_kb:.1f} KB")

        # Group the ANSI image and the table together
        content_group = Group(
            preview_renderable,
            "",  # Blank line spacer
            table)

        console.print()
        console.print(
            Panel(content_group,
                  title="🖼️ [bold magenta]Image Attached[/bold magenta]",
                  border_style="magenta",
                  expand=False))

    except Exception as e:
        console.print(
            f"[bold red]❌ Error reading image {filepath}: {e}[/bold red]")
