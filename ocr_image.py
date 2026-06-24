#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import base64
import json
import os
import sys

import requests
from rich.console import Console

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("❌ Missing Pillow! Please install it via: pip install pillow")
    sys.exit(1)

import ollama_utils

console = Console()

OCR_SYSTEM_PROMPT = """You are an expert Optical Character Recognition (OCR) engine.
Extract the text from the provided image verbatim.

STRICT RULES:
1. Output ONLY the extracted text.
2. Maintain original paragraph breaks, punctuation, and language.
3. DO NOT wrap the output in markdown code blocks.
4. DO NOT add conversational filler.
5. If the image contains structured data, format it cleanly using Markdown tables."""


def get_cjk_font(user_font_path: str = None, font_size: int = 18):
    """Scavenges OS for a CJK-compatible font to prevent Chinese 'Tofu' boxes."""
    if user_font_path and os.path.exists(user_font_path):
        return ImageFont.truetype(user_font_path, font_size)

    cjk_candidates = [
        "msyh.ttc", "simhei.ttf", "PingFang.ttc", "STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
    ]
    for f_name in cjk_candidates:
        try:
            return ImageFont.truetype(f_name, font_size)
        except IOError:
            continue
    return ImageFont.load_default()


def smart_hybrid_wrap(paragraph: str, max_width: int, draw_obj,
                      font_obj) -> list[str]:
    """Wraps text respecting Western word spaces, but splitting CJK char-by-char."""
    if not paragraph.strip():
        return [""]

    def get_w(text_str):
        try:
            return draw_obj.textlength(text_str, font=font_obj)
        except AttributeError:
            return draw_obj.textbbox((0, 0), text_str, font=font_obj)[2]

    lines, curr_line = [], ""
    for chunk in paragraph.split(" "):
        separator = " " if curr_line else ""
        test_line = curr_line + separator + chunk

        if get_w(test_line) <= max_width:
            curr_line = test_line
        else:
            if curr_line:
                lines.append(curr_line)
                curr_line = ""
                test_line = chunk

            if get_w(test_line) > max_width:
                for char in chunk:
                    if get_w(curr_line + char) <= max_width: curr_line += char
                    else:
                        lines.append(curr_line)
                        curr_line = char
                curr_line += " "
            else:
                curr_line = chunk

    if curr_line.strip(): lines.append(curr_line.strip())
    return lines


def gather_image_targets(input_paths: list[str]) -> list[str]:
    """Flattens mixed file/folder inputs into a unique list of valid images."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    collected = []
    for path in input_paths:
        if os.path.isfile(path) and os.path.splitext(
                path)[1].lower() in valid_exts:
            collected.append(path)
        elif os.path.isdir(path):
            for item in os.listdir(path):
                f_path = os.path.join(path, item)
                if os.path.isfile(f_path) and os.path.splitext(
                        item)[1].lower() in valid_exts:
                    collected.append(f_path)
    return list(dict.fromkeys(collected))


def generate_vision_ocr(args, img_path: str) -> tuple[str, dict]:
    """Streams the OCR and extracts nanosecond hardware metrics."""
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    url = f"http://{args.host}:{args.port}/api/generate"
    payload = {
        "model": args.model,
        "prompt": "Transcribe the text in this image strictly.",
        "system": OCR_SYSTEM_PROMPT,
        "images": [img_b64],
        "stream": True,
        "options": {
            "num_ctx": args.ctx,
            "temperature": 0.05
        }
    }

    response = requests.post(url,
                             json=payload,
                             stream=True,
                             timeout=(5.0, 60.0))
    response.raise_for_status()

    collected_text = []
    stats = {"ttft": 0.0, "tps": 0.0, "tokens_in": 0, "tokens_out": 0}

    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            collected_text.append(chunk.get("response", ""))
            if args.verbose:
                print(chunk.get("response", ""), end="", flush=True)

            # Grab Ollama's native C++ closing telemetry
            if chunk.get("done"):
                p_dur_sec = chunk.get("prompt_eval_duration", 0) / 1e9
                eval_dur_sec = chunk.get("eval_duration", 0) / 1e9
                gen_count = chunk.get("eval_count", 0)

                stats["tokens_in"] = chunk.get("prompt_eval_count", 0)
                stats["tokens_out"] = gen_count
                stats["ttft"] = round(p_dur_sec, 2)
                if eval_dur_sec > 0:
                    stats["tps"] = round(gen_count / eval_dur_sec, 1)

    if args.verbose: print("\n")
    return "".join(collected_text).strip(), stats


def create_comparison_canvas(img_path: str, ocr_text: str, stats: dict,
                             output_img_path: str, custom_font_path: str):
    """Renders the 1000px side-by-side verification canvas with a telemetry HUD."""
    try:
        resample_algo = Image.Resampling.LANCZOS
    except AttributeError:
        resample_algo = Image.LANCZOS

    # 1. Scale original image to 1000px high
    orig_img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = orig_img.size
    target_h = 1000
    target_w = int(orig_w * (target_h / orig_h))
    scaled_orig = orig_img.resize((target_w, target_h), resample_algo)

    # 2. Setup Blank Canvas
    canvas_bg = (250, 250, 250)
    text_canvas = Image.new("RGB", (target_w, target_h), canvas_bg)
    draw = ImageDraw.Draw(text_canvas)

    font_size = 18
    font = get_cjk_font(custom_font_path, font_size)

    # 3. Draw OCR Text (leaving bottom 85px safe-zone for the HUD)
    # padding = 35
    padding = 20
    max_line_width = target_w - (padding * 2)
    wrapped_lines = []
    for p in ocr_text.splitlines():
        wrapped_lines.extend(smart_hybrid_wrap(p, max_line_width, draw, font))

    y_pos = padding
    line_step = font_size + 8
    hud_safe_horizon = target_h - 90  # Stop writing text above the HUD box

    for line in wrapped_lines:
        if y_pos + line_step > hud_safe_horizon:
            draw.text((padding, y_pos),
                      "[... Output truncated to fit canvas ...]",
                      font=font,
                      fill=(200, 40, 40))
            break
        draw.text((padding, y_pos), line, font=font, fill=(24, 24, 24))
        y_pos += line_step

    # 4. Draw the Floating Telemetry HUD at the bottom
    hud_font = get_cjk_font(custom_font_path, font_size=13)
    hud_box_x1, hud_box_y1 = 20, target_h - 65
    hud_box_x2, hud_box_y2 = target_w - 20, target_h - 20

    # Draw rounded pill container (fallback to sharp rectangle on very old Pillow versions)
    try:
        draw.rounded_rectangle(
            [hud_box_x1, hud_box_y1, hud_box_x2, hud_box_y2],
            radius=6,
            fill=(235, 240, 245),
            outline=(210, 215, 220))
    except AttributeError:
        draw.rectangle([hud_box_x1, hud_box_y1, hud_box_x2, hud_box_y2],
                       fill=(235, 240, 245),
                       outline=(210, 215, 220))

    hud_str = f"⏱️ TTFT: {stats['ttft']}s    │    ⚡ TPS: {stats['tps']} t/s    │    🪙 Tokens: {stats['tokens_in']} in / {stats['tokens_out']} out"

    # Perfectly center the HUD string inside its pill container
    try:
        txt_w = draw.textlength(hud_str, font=hud_font)
    except AttributeError:
        txt_w = draw.textbbox((0, 0), hud_str, font=hud_font)[2]

    txt_x = hud_box_x1 + ((hud_box_x2 - hud_box_x1) - txt_w) // 2
    txt_y = hud_box_y1 + 14
    draw.text((txt_x, txt_y), hud_str, font=hud_font, fill=(70, 80, 95))

    # 5. Stitch & Export
    master_img = Image.new("RGB", (target_w * 2, target_h), (200, 200, 200))
    master_img.paste(scaled_orig, (0, 0))
    master_img.paste(text_canvas, (target_w, 0))
    master_img.save(output_img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ollama Batch OCR & Visualizer with Telemetry HUD")
    parser.add_argument("inputs", nargs="+", help="Image files or directories")
    parser.add_argument("-o",
                        "--output-dir",
                        default=None,
                        help="Assigned destination folder")
    parser.add_argument("--font",
                        default=None,
                        help="Force specific .ttf/.ttc font file")

    parser.add_argument("--model",
                        default="glm-ocr:latest",
                        help="Vision model")
    parser.add_argument("--host", default="localhost", help="Ollama Host")
    parser.add_argument("--port", type=int, default=11434, help="Ollama Port")
    parser.add_argument("--ctx", type=int, default=4096, help="Context Window")
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="Print OCR stream live")

    args = parser.parse_args()
    ollama_utils.check_args_connections(args)

    targets = gather_image_targets(args.inputs)
    if not targets:
        console.print("[bold red]❌ No valid images discovered.[/bold red]")
        sys.exit(1)

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    console.print(
        f"[*] Processing [bold cyan]{len(targets)}[/bold cyan] images...\n")

    for i, img_path in enumerate(targets, 1):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        dest = args.output_dir if args.output_dir else os.path.dirname(
            img_path) or "."
        out_txt, out_img = os.path.join(dest,
                                        f"{basename}_ocr.txt"), os.path.join(
                                            dest, f"{basename}_comparison.png")

        console.print(
            f"┌─ ({i}/{len(targets)}) [bold yellow]{os.path.basename(img_path)}[/bold yellow]"
        )

        text_result, stats_dict = generate_vision_ocr(args, img_path)
        if not text_result: continue

        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(text_result)
        create_comparison_canvas(img_path, text_result, stats_dict, out_img,
                                 args.font)

        console.print(
            f"├─ Telemetry: [cyan]TTFT: {stats_dict['ttft']}s[/cyan] | [green]{stats_dict['tps']} TPS[/green] | [dim]{stats_dict['tokens_out']} tokens[/dim]"
        )
        console.print(f"└─ Saved: [dim]{out_img}[/dim]\n")

    console.print("[bold green]✔ Batch complete![/bold green]")
