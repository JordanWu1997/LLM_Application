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
import sys
import textwrap
import time

import cv2
import requests
from tqdm import tqdm


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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    input_video_name, _ = os.path.splitext(input_video_path)
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
        ]
    }

    # Print out video time
    if verbose:
        print(f'{caption_prefix} ')

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
                break

        if verbose:
            print()  # newline after streaming

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
        if frame_num % step == 0 and frame_num > step:
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

    OLLAMA_URL = "http://localhost:11434/api/chat"
    MODEL = "gemma3:4b"
    # MODEL = "moondream:latest"
    # MODEL = "qwen2.5vl:3b"
    # MODEL = "qwen3-vl:2b"
    PROMPT = ("Describe the main event in ONE short sentence."
              "Maximum 8 words."
              "No adjectives."
              "No explanations."
              "No speculation.")

    input_video_paths = sys.argv[1:]
    for input_video_path in input_video_paths:
        run_video_caption_pipeline(input_video_path,
                                   infer_every_sec=3,
                                   target_size=(320, 320),
                                   display_size=(640, 360),
                                   max_word_num=10,
                                   output_video_dir='.',
                                   live_display=False,
                                   verbose=False)
