#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import ast
import os
import re
import sys

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# Import your prebuilt engine
from ollama_utils import check_args_connections, stream_ollama_generate

console = Console()


def get_system_prompt(py_version: str, indent_spaces: int = 0) -> str:
    """Generates the strict ruleset for the LLM based on Python version."""

    prompt = f"""You are an expert Python developer and technical documentation specialist.
Your task is to take the provided Python code snippet and output it with:
1. Complete Google-style docstrings (Summary, Args:, Returns:, Raises:).
2. Precise Type Hints for all arguments and the return value.

CRITICAL RULES:
1. Target Python Version: {py_version}.
   - If >= 3.10: strictly use `|` for unions (e.g. `str | None`) and builtin generics (`list[int]`, `dict[str, Any]`).
   - If <= 3.9: strictly use `typing.Union`, `typing.Optional`, `typing.List`, etc.
   - If you use any types from the `typing` module, prefix them (e.g., `typing.Optional`, `typing.Any`) so we don't break scope.
2. PRESERVE ALL LOGIC 100%. Do not rename variables, re-order statements, or change default parameter values.
3. Base Indentation: The code provided starts at an indentation of {indent_spaces} spaces. Maintain this exact base indentation.
4. Output ONLY the raw executable Python code. No conversational filler, no markdown ```python wrappers."""

    return prompt


def clean_llm_code(raw_response: str) -> str:
    """Strips markdown code fences that LLMs love to ignore instructions and add anyway."""
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove top fence (e.g. ```python)
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove bottom fence
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return cleaned


def parse_file_ast(filepath: str):
    """Safely extracts all functions/methods and their exact line boundaries."""
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        console.print(
            f"[bold red]❌ Syntax Error in target file at line {e.lineno}: {e.msg}[/bold red]"
        )
        return None, source.splitlines()

    lines = source.splitlines()
    funcs = []

    class FunctionHarvester(ast.NodeVisitor):

        def __init__(self):
            self.class_path = []

        def visit_ClassDef(self, node):
            self.class_path.append(node.name)
            self.generic_visit(node)
            self.class_path.pop()

        def visit_FunctionDef(self, node):
            self._harvest(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            self._harvest(node)
            self.generic_visit(node)

        def _harvest(self, node):
            doc = ast.get_docstring(node)
            # Rough check if type hints exist
            has_types = (node.returns is not None
                         or any(arg.annotation for arg in node.args.args)
                         or any(arg.annotation
                                for arg in node.args.kwonlyargs))

            prefix = ".".join(self.class_path)
            full_name = f"{prefix}.{node.name}" if prefix else node.name

            # Calculate base indentation of the def line
            def_line = lines[node.lineno - 1]
            indent = len(def_line) - len(def_line.lstrip())

            funcs.append({
                "name":
                full_name,
                "start":
                node.lineno,
                "end":
                node.end_lineno,
                "has_doc":
                doc is not None,
                "has_types":
                has_types,
                "indent":
                indent,
                "code":
                "\n".join(lines[node.lineno - 1:node.end_lineno])
            })

    visitor = FunctionHarvester()
    visitor.visit(tree)
    # Sort top-to-bottom
    return sorted(funcs, key=lambda x: x["start"]), lines


def inject_code_into_file(filepath: str, start_line: int, end_line: int,
                          new_snippet: str):
    """Splices the LLM code back into the file and checks if 'import typing' is needed."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    start_idx = start_line - 1
    new_lines = new_snippet.splitlines()

    updated_lines = lines[:start_idx] + new_lines + lines[end_line:]
    full_text = "\n".join(updated_lines)

    # Safe top-level injection of 'import typing' if the LLM used it
    if "typing." in full_text and "import typing" not in full_text and "from typing" not in full_text:
        insert_pos = 0
        for i, line in enumerate(updated_lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_pos = i
                break
        updated_lines.insert(insert_pos, "import typing")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(updated_lines) + "\n")


def run_full_script_mode(args, target_file: str):
    """Mode 1: Send the whole damn file and let the LLM sort it out."""
    with open(target_file, "r", encoding="utf-8") as f:
        source = f.read()

    console.print(
        f"\n[*] Processing entire file: [bold cyan]{target_file}[/bold cyan]..."
    )

    sys_prompt = f"""You are an expert Python Documenter. Rewrite this entire file to add Google-style docstrings and Python {args.py_version} type hints to every function and class.
Return ONLY the newly written Python file. Preserve all existing code logic."""

    base_url = f"http://{args.host}:{args.port}"
    raw_out = stream_ollama_generate(prompt=source,
                                     system_prompt=sys_prompt,
                                     ollama_url=base_url,
                                     model=args.model,
                                     ctx_window=args.ctx,
                                     verbose=args.verbose)

    clean_code = clean_llm_code(raw_out)

    if args.in_place:
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(clean_code + "\n")
        console.print(
            f"[bold green]✔ File successfully overwritten![/bold green]")
    else:
        out_path = target_file.replace(".py", "_documented.py")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(clean_code + "\n")
        console.print(
            f"[bold green]✔ Saved to safe copy:[/bold green] {out_path}")


def run_selective_mode(args, target_file: str):
    """Mode 2: Interactive loop with safe AST re-parsing."""
    base_url = f"http://{args.host}:{args.port}"

    while True:
        funcs, _ = parse_file_ast(target_file)
        if funcs is None:
            break

        if not funcs:
            console.print(
                f"[yellow]No functions discovered inside {target_file}.[/yellow]"
            )
            break

        table = Table(
            title=
            f"Target: [bold cyan]{os.path.basename(target_file)}[/bold cyan] (Python {args.py_version})"
        )
        table.add_column("Idx", justify="right", style="cyan", no_wrap=True)
        table.add_column("Function / Method", style="bold green")
        table.add_column("Docstring", justify="center")
        table.add_column("Type Hints", justify="center")
        table.add_column("Lines", style="dim")

        for i, f in enumerate(funcs):
            doc_flag = "[bold green]✔[/bold green]" if f[
                "has_doc"] else "[bold red]✘[/bold red]"
            type_flag = "[bold green]✔[/bold green]" if f[
                "has_types"] else "[yellow]?[/yellow]"
            table.add_row(str(i + 1), f["name"], doc_flag, type_flag,
                          f"{f['start']}→{f['end']}")

        console.print("\n")
        console.print(table)

        choice = input("\nPick a function Idx to document (or 'q' to quit): "
                       ).strip().lower()
        if choice == 'q':
            break

        if not choice.isdigit() or not (1 <= int(choice) <= len(funcs)):
            console.print("[red]Invalid index.[/red]")
            continue

        target_fn = funcs[int(choice) - 1]

        console.print(
            f"\n[*] Generating for: [bold yellow]{target_fn['name']}[/bold yellow]..."
        )

        sys_prompt = get_system_prompt(args.py_version, target_fn["indent"])
        if args.verbose:
            print(f'\n[SYSTEM PROMPT]\n{sys_prompt}')
            print(f'\n[PROMPT]\n{target_fn["code"]}')

        raw_code = stream_ollama_generate(prompt=target_fn["code"],
                                          system_prompt=sys_prompt,
                                          ollama_url=base_url,
                                          model=args.model,
                                          ctx_window=args.ctx,
                                          verbose=args.verbose)

        new_code = clean_llm_code(raw_code)
        if args.verbose:
            print(f'\n[OUTPUT]\n{new_code}\n')

        if not new_code or "def " not in new_code:
            console.print(
                "[bold red]❌ LLM returned malformed code. Aborting this splice.[/bold red]"
            )
            continue

        inject_code_into_file(target_file, target_fn["start"],
                              target_fn["end"], new_code)
        console.print(
            f"\n[bold green]✔ Applied to {target_fn['name']}! Re-indexing file...[/bold green]\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Docstring & Type Hint Generator")
    parser.add_argument("file", help="The target Python script")
    parser.add_argument("-m",
                        "--mode",
                        choices=["select", "full"],
                        default="select",
                        help="Operating mode (default: select)")
    parser.add_argument(
        "--py-version",
        default=f"{sys.version_info.major}.{sys.version_info.minor}",
        help=
        "Target Python version for type syntax (default: your system's version)"
    )
    parser.add_argument(
        "-i",
        "--in-place",
        action="store_true",
        help=
        "Overwrite file in-place (Only applies to 'full' mode; selective is always in-place)"
    )

    # Inherit Ollama connection flags
    parser.add_argument("--host", default="localhost", help="Ollama Host")
    parser.add_argument("--port", type=int, default=11434, help="Ollama Port")
    parser.add_argument("--model", default="gemma4:26b", help="LLM to use")
    parser.add_argument("--ctx", type=int, default=8192, help="Context Window")
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="Print streaming output live")

    args = parser.parse_args()

    # 1. Fire your existing connection checker
    check_args_connections(args)

    if not os.path.exists(args.file):
        console.print(f"[red]File not found: {args.file}[/red]")
        sys.exit(1)

    if args.mode == "full":
        run_full_script_mode(args, args.file)
    else:
        run_selective_mode(args, args.file)
