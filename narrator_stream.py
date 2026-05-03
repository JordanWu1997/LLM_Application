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
# |_|\_\  |_| |_|  Datetime: 2026-01-15 22:59:21             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import base64
import json
import os
import queue
import sys
import threading
import time

import cv2
import requests
from tqdm import tqdm

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"
PROMPT = ("Describe the main event in ONE short sentence. "
          "Maximum 8 words. No adjectives. No explanations. No speculation.")
CONTEXT_WINDOW = 4096

# Global state for background updates
latest_caption = "Initializing..."
latest_prefix = ""
is_running = True

# Use a session for connection pooling (faster than repeated posts)
session = requests.Session()


def generate_output_video_writer(input_video_path,
                                 input_video_cap,
                                 output_frame_width=-1,
                                 output_video_dir='',
                                 output_suffix='',
                                 output_video='',
                                 verbose=False):

    FPS = input_video_cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(input_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_frame_width = frame_width

    if output_frame_width > 0:
        output_frame_height = int(output_frame_width *
                                  (frame_height / frame_width))
    # Assign default frame width/height for RTSP streaming
    elif output_frame_width == 0:
        output_frame_width = 960
        output_frame_height = 640

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Handle web cam
    if isinstance(input_video_path, int):
        input_video_name = str(input_video_path)
    else:
        input_video_name, _ = os.path.splitext(input_video_path)

    # Main
    output_video_path = f'{input_video_name}_{output_suffix}.mp4'
    if output_video_dir != '':
        # Init output dir
        if not os.path.isdir(output_video_dir):
            os.makedirs(output_video_dir)
        output_video_path = f'{output_video_dir}/{os.path.basename(input_video_name)}_{output_suffix}.mp4'
    output_video_writer = cv2.VideoWriter(
        output_video_path, fourcc, FPS,
        (output_frame_width, output_frame_height))
    if verbose:
        print(
            f'[INFO] INPUT: {input_video_path} ({frame_width}x{frame_height}@{FPS:.2f})'
        )
        print(f'[INFO] OUTPUT: {output_video_path}')

    return output_video_writer, (output_frame_width, output_frame_height)


def frame_to_base64(frame, format=".jpg", quality=90):
    encode_param = []
    if format == ".jpg":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(format, frame, encode_param)
    if not success:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buffer).decode("utf-8")


def frame_to_hhmmss(frame_idx, fps):
    total_seconds = int(frame_idx / fps)

    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60

    return f"{h:02d}:{m:02d}:{s:02d}"


def resize_with_padding(img, target_size=(640, 640), pad_color=(0, 0, 0)):
    h, w = img.shape[:2]
    target_w, target_h = target_size

    # Scale factor
    scale = min(target_w / w, target_h / h)

    # New size
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(resized,
                                top,
                                bottom,
                                left,
                                right,
                                borderType=cv2.BORDER_CONSTANT,
                                value=pad_color)

    return padded


def wrap_text_to_width(text, max_width, font, font_scale, thickness):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = word if not current else current + " " + word
        (w, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def caption_frame_stream(frame,
                         target_size=(320, 320),
                         caption_prefix='',
                         display_size=None,
                         max_word_num=-1,
                         live_display=True,
                         verbose=False):

    # Init
    current_text = ''

    # Resize and embed to base64
    resized_frame = resize_with_padding(frame, target_size=target_size)
    img_b64 = frame_to_base64(resized_frame)

    # Init canvas
    base_frame = frame.copy()
    if display_size is not None:
        base_frame = resize_with_padding(base_frame, target_size=display_size)

    # Construct payload
    payload = {
        "model": MODEL,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [
                    img_b64,
                ]
            },
        ],
        "options": {
            "num_ctx": CONTEXT_WINDOW
        }
    }

    # Print out video time
    if verbose:
        print(f'\n{caption_prefix} ')

    # Post to ollama server
    with requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=120,
    ) as resp:
        resp.raise_for_status()

        full_text = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue

            chunk = json.loads(line)
            delta = chunk.get("message", {}).get("content", "")
            if delta:
                current_text += delta

                # Early stop
                if too_many_words(current_text, word_limit=max_word_num):
                    print()
                    return current_text.strip(), True

                # Visualization
                if live_display:
                    canvas = base_frame.copy()
                    canvas = draw_subtitle(canvas,
                                           f'{caption_prefix} {current_text}',
                                           margin=0,
                                           position='bottom')
                    cv2.imshow(f'VLM Narrator', canvas)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[INFO] QUIT pressed")
                        return current_text.strip(), False  # Close connection

                # Print out in terminal
                if verbose:
                    print(delta, end="", flush=True)

                # Get data
            if chunk.get("done"):
                metadata = chunk
                break

        # Newline after streaming
        if verbose:
            print()

        # Token Truncation Check
        if verbose:
            processed = metadata.get("prompt_eval_count", 0)
            if processed >= CONTEXT_WINDOW:
                print(
                    f"\033[93m⚠️  Warning: Input reached {CONTEXT_WINDOW} tokens and was truncated.\033[0m"
                )
            else:
                print(
                    f"\033[90m(Tokens used: {processed}/{CONTEXT_WINDOW})\033[0m"
                )

    return current_text.strip(), True


def too_many_words(text, word_limit=10):
    if word_limit > 0:
        return len(text.strip().split()) > word_limit
    else:
        return False


def draw_subtitle(frame,
                  text,
                  position="bottom",
                  max_width_ratio=0.9,
                  font=cv2.FONT_HERSHEY_SIMPLEX,
                  font_scale=0.8,
                  thickness=2,
                  margin=20,
                  line_spacing=10):

    # Subtitle geometry
    h, w = frame.shape[:2]
    max_width = int(w * max_width_ratio)
    lines = wrap_text_to_width(text, max_width, font, font_scale, thickness)
    (line_h, _), _ = cv2.getTextSize("Ay", font, font_scale, thickness)
    total_height = len(lines) * (line_h + line_spacing)

    # Init canvas
    overlay = frame.copy()

    # Determine y-coordinate based on position
    if position == "bottom":
        y0 = h - margin - total_height
        rect_top = y0 - margin
        rect_bottom = h
    elif position == "top":
        y0 = margin
        rect_top = 0
        rect_bottom = y0 + total_height + margin
    else:
        raise ValueError("position must be 'top' or 'bottom'")

    # Background rectangle
    cv2.rectangle(overlay, (0, rect_top), (w, rect_bottom), (0, 0, 0), -1)

    # Alpha blending
    alpha = 0.6
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Draw lines
    y = y0
    for line in lines:
        cv2.putText(frame, line, (margin, y + line_h), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h + line_spacing
    return frame


def streaming_inference_worker(task_queue, target_size, max_word_num):
    """Background thread to handle Ollama API calls without blocking video."""

    global latest_caption, latest_prefix, is_running

    while is_running:
        try:
            # Block for a short time to wait for a frame
            frame, prefix = task_queue.get(timeout=1)
        except queue.Empty:
            continue

        resized_frame = cv2.resize(frame, target_size)
        img_b64 = frame_to_base64(resized_frame)

        payload = {
            "model": MODEL,
            "stream": True,
            "messages": [{
                "role": "user",
                "content": PROMPT,
                "images": [img_b64]
            }],
            "options": {
                "num_ctx": CONTEXT_WINDOW
            }
        }

        if max_word_num < 0:
            max_word_num = 1e9

        try:
            with session.post(OLLAMA_URL,
                              json=payload,
                              stream=True,
                              timeout=10) as resp:
                resp.raise_for_status()
                full_text = ""
                for line in resp.iter_lines(decode_unicode=True):
                    if not line: continue
                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        full_text += delta
                        # Update global caption dynamically for a "typing" effect
                        latest_caption = full_text.strip()
                        latest_prefix = prefix
                        if len(full_text.split()) > max_word_num:
                            break
                    if chunk.get("done"): break
        except Exception as e:
            print(f"Inference Error: {e}")

        task_queue.task_done()


def run_streaming_caption_pipeline(input_video_path,
                                   infer_every_sec=3,
                                   target_size=(320, 320),
                                   display_size=None,
                                   max_word_num=-1,
                                   output_video_dir=None,
                                   output_suffix='output',
                                   live_display=True,
                                   verbose=False):

    # Global variables
    global is_running, latest_caption, latest_prefix

    # Load video info
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(fps * infer_every_sec)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Generate output video writer
    if output_video_dir is not None:
        output_video_writer, (output_frame_width, output_frame_height) = \
            generate_output_video_writer(input_video_path, cap,
                                         output_video_dir=output_video_dir,
                                         output_suffix=output_suffix,
                                         output_frame_width=-1,
                                         verbose=True)

    # Queue size 1 ensures we only process the *freshest* frame
    task_queue = queue.Queue(maxsize=1)

    # Start the worker thread
    worker = threading.Thread(target=streaming_inference_worker,
                              args=(task_queue, target_size, max_word_num),
                              daemon=True)
    worker.start()

    # Main
    frame_num = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Hand off frame to inference thread if it's time and worker is free
            if frame_num % step == 0:
                if task_queue.empty():
                    video_time = f"{int(frame_num/fps)//3600:02d}:{(int(frame_num/fps)%3600)//60:02d}:{int(frame_num/fps)%60:02d}"
                    task_queue.put((frame.copy(), f"[{video_time}]"))

            # Always draw and show the most recent caption we have
            canvas = frame.copy()
            display_text = f"{latest_prefix} {latest_caption}"
            draw_subtitle(canvas, display_text)

            # Save result as video
            if output_video_dir is not None:
                output_video_writer.write(canvas)

            # Live display
            if live_display:
                if display_size is not None:
                    canvas = resize_with_padding(canvas, display_size)
                cv2.imshow("Smooth VLM Stream", canvas)
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_num += 1
    finally:
        is_running = False
        cap.release()
        if output_video_dir is not None:
            # Release video writer
            output_video_writer.release()
        cv2.destroyAllWindows()


def run_video_caption_pipeline(input_video_path,
                               infer_every_sec=5,
                               target_size=(320, 320),
                               display_size=None,
                               max_word_num=-1,
                               output_video_dir=None,
                               output_suffix='output',
                               live_display=True,
                               verbose=False):

    # Load video info
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(fps * infer_every_sec)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Generate output video writer
    if output_video_dir is not None:
        output_video_writer, (output_frame_width, output_frame_height) = \
            generate_output_video_writer(input_video_path, cap,
                                         output_video_dir=output_video_dir,
                                         output_suffix=output_suffix,
                                         output_frame_width=-1,
                                         verbose=True)

    # Main
    frame_num = 0
    caption_prefix, caption = '', ''
    progress_bar = tqdm(total=total_frames)
    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Caption
        if frame_num % step == 0 and frame_num >= step:
            caption_start = time.time()
            video_time = frame_to_hhmmss(frame_num, fps)
            caption_prefix = f'[{video_time}]'
            caption, status = caption_frame_stream(
                frame,
                caption_prefix=caption_prefix,
                target_size=target_size,
                display_size=display_size,
                max_word_num=max_word_num,
                live_display=live_display,
                verbose=verbose)
            if verbose:
                print(f'[INFO] FPS: {1 / (time.time() - caption_start):.1f}')
            if not status:
                break

        # Visualization for saving
        if output_video_dir is not None:
            canvas = frame.copy()
            canvas = draw_subtitle(canvas,
                                   f'{caption_prefix} {caption}',
                                   margin=0,
                                   position='bottom')
            output_video_writer.write(canvas)

        # Update frame
        frame_num += 1
        progress_bar.update(1)

    # Release video writer
    if output_video_dir is not None:
        output_video_writer.release()

    # Close all opencv windows
    cv2.destroyAllWindows()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_video_paths', nargs='+', type=str)
    parser.add_argument('-i', '--infer_interval', default=3)
    parser.add_argument('-s', '--streaming', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    for input_video_path in args.input_video_paths:
        input_video_path = int(input_video_path)
        if args.streaming:
            run_streaming_caption_pipeline(input_video_path,
                                           infer_every_sec=args.infer_interval,
                                           target_size=(320, 320),
                                           display_size=(640, 360),
                                           max_word_num=10,
                                           output_video_dir='.',
                                           live_display=not False,
                                           verbose=args.verbose)
        else:
            run_video_caption_pipeline(input_video_path,
                                       infer_every_sec=args.infer_interval,
                                       target_size=(320, 320),
                                       display_size=(640, 360),
                                       max_word_num=10,
                                       output_video_dir='.',
                                       live_display=not False,
                                       verbose=args.verbose)
