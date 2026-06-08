#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8

import json
import subprocess
import sys
import urllib.request

# Configuration
API_URL = "http://localhost:8081"


def fetch_json(endpoint):
    try:
        req = urllib.request.Request(f"{API_URL}/{endpoint}")
        with urllib.request.urlopen(req, timeout=2) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception:
        return None


def get_vram():
    try:
        # Ask NVIDIA drivers directly for VRAM usage
        cmd = [
            "nvidia-smi", "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader"
        ]
        output = subprocess.check_output(cmd, text=True).strip()
        # Output looks like: "11500 MiB, 12288 MiB"
        return output
    except Exception:
        return "N/A"


def get_container_ram():
    try:
        # Ask Docker exactly how much system RAM the llama container is eating
        cmd = [
            "docker", "stats", "--no-stream", "--format", "{{.MemUsage}}",
            "llamacpp-server"
        ]
        output = subprocess.check_output(cmd, text=True).strip()
        return output
    except Exception:
        return "N/A"


def main():
    props = fetch_json("props")
    slots = fetch_json("slots")

    if not props or slots is None:
        print(
            "\n🔴 Error: Cannot connect to Llama.cpp server at http://localhost:8080"
        )
        print("   Make sure the server is running.\n")
        sys.exit(1)

    # Calculate Context and Slot Usage
    total_ctx = props.get('default_generation_settings', {}).get('n_ctx', 0)

    active_slots = 0
    used_tokens = 0
    total_slots = len(slots)

    for slot in slots:
        used_tokens += slot.get('n_tokens', 0)
        # state 0 usually means idle, 1 means processing/generating
        if slot.get('state', 0) != 0:
            active_slots += 1

    ctx_percent = (used_tokens / total_ctx * 100) if total_ctx > 0 else 0

    # Get Hardware Metrics
    vram_usage = get_vram()
    ram_usage = get_container_ram()

    # Print Dashboard
    print("+" + "-" * 52 + "+")
    print(f"| {'🦙 LLAMA.CPP STATUS DASHBOARD':^50} |")
    print("+" + "-" * 52 + "+")
    print(f"| API Status   : 🟢 ONLINE                           |")
    print(f"| GPU VRAM     : {vram_usage:<35} |")
    print(f"| System RAM   : {ram_usage:<35} |")
    print(
        f"| Context Used : {used_tokens} / {total_ctx} Tokens ({ctx_percent:.1f}%)"
        + " " * (19 - len(str(used_tokens)) - len(str(total_ctx))) + " |")
    print(
        f"| Active Slots : {active_slots} / {total_slots} Busy                            |"
    )
    print("+" + "-" * 52 + "+")


if __name__ == "__main__":
    main()
