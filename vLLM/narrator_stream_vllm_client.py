#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8

import argparse
import base64
import io
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from narrator_stream_vllm import (GridSceneWrapper, ROIWrapper,
                                  YOLOTrackerWrapper)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
vlm_input_queue = queue.Queue(maxsize=1)
vlm_output_queue = queue.Queue()
vlm_ready_event = threading.Event()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Real-time Video Analysis Pipeline (vLLM Client)")

    parser.add_argument("input_video_path",
                        type=str,
                        help="Path to the input video file.")
    parser.add_argument("-o",
                        "--output_video_path",
                        type=str,
                        default=None,
                        help="Path to save output video.")
    # Connection Parameters for existing server
    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://localhost:8000/v1",
        help="The base URL endpoint of your running vLLM OpenAI API Server.")
    parser.add_argument(
        "--vllm_model_name",
        type=str,
        default="cyankiwi/Qwen3.5-4B-AWQ-4bit",
        help=
        "The exact model ID string matching the backend engine container configuration."
    )
    parser.add_argument("--mode",
                        type=str,
                        choices=[
                            "person", "scene", "grid_1x2", "grid_1x3",
                            "grid_2x1", "grid_2x2", "grid_2x3", "grid_3x1",
                            "grid_3x2", "grid_3x3"
                        ],
                        default="person",
                        help="The analysis mode to run.")
    parser.add_argument("-i",
                        "--instruction",
                        type=str,
                        default=None,
                        help="Text instruction override.")
    parser.add_argument(
        "-I",
        "--interval",
        type=int,
        default=30,
        help="Frames checkpoint gap between evaluation intervals.")
    parser.add_argument("-r",
                        "--display_ratio",
                        type=float,
                        default=1.0,
                        help="Scaling display ratio.")
    parser.add_argument("--img_size",
                        type=int,
                        default=256,
                        help="Thumbnail dimension (square) for VLM input.")

    return parser.parse_args()


# =========================================================
# DECOUPLED CLIENT WORKER
# =========================================================


def encode_pil_to_base64_jpeg(pil_img):
    """Converts a PIL image in-memory to a Base64 string compliant with the Vision API."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def remote_vlm_client_loop(api_url, model_name, task_instruction, ready_event):
    """Asynchronous worker that communicates with vLLM concurrently."""
    print(f"[Client Worker] Connecting to vLLM Server at: {api_url}")
    client = OpenAI(base_url=api_url, api_key="token")
    ready_event.set()

    def send_single_crop(img, track_id):
        """Helper to send a single image over the network."""
        try:
            base64_image = encode_pil_to_base64_jpeg(img)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role":
                    "system",
                    "content":
                    "You are a concise captioning assistant. Provide a brief description immediately."
                }, {
                    "role":
                    "user",
                    "content": [{
                        "type": "text",
                        "text": f"{task_instruction} Output directly."
                    }, {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }]
                }],
                temperature=0.0,
                max_tokens=24,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                })
            caption = response.choices[0].message.content.strip()
            return track_id, caption
        except Exception as e:
            print(f"[Crop Error] Track {track_id} failed -> {e}")
            return track_id, "[Inference Error]"

    while True:
        try:
            task = vlm_input_queue.get()
            if task is None:
                break

            crops, metadata = task["crops"], task["metadata"]

            # vLLM handles continuous batching on its end automatically.
            results = []
            with ThreadPoolExecutor(max_workers=len(crops)) as executor:
                # Map send_single_crop across all images and track IDs simultaneously
                futures = [
                    executor.submit(send_single_crop, img, tid)
                    for img, tid in zip(crops, metadata)
                ]
                for future in futures:
                    results.append(future.result())

            # Return all concurrent results back to the main processing thread queue
            vlm_output_queue.put(results)
            vlm_input_queue.task_done()

        except Exception as e:
            print(f"[Client Worker Error]: -> {e}")
            vlm_input_queue.task_done()


# =========================================================
# CORE VISUALIZATION PIPELINE
# =========================================================


def draw_bottom_wrapped_text(
        img,
        text,
        bbox,
        font_size=30,
        chinese_font_path="/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        reverse=False):

    x1, y1, x2, y2 = map(int, bbox)
    box_width = x2 - x1
    max_width = box_width - 10 if box_width > 20 else box_width

    try:
        font = ImageFont.truetype(chinese_font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    lines = []
    current_line = ''
    for char in text:
        test_line = current_line + char
        width = font.getlength(test_line) if hasattr(
            font, 'getlength') else font.getsize(test_line)[0]
        if width > max_width and current_line != '':
            lines.append(current_line)
            current_line = char
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)

    if hasattr(font, 'getbbox'):
        bbox_font = font.getbbox("测Test")
        line_height = bbox_font[3] - bbox_font[1]
    else:
        line_height = font.getsize("测Test")[1]

    line_spacing = 4
    total_text_height = len(lines) * line_height \
            + (len(lines) - 1) * line_spacing
    vertical_padding = 6
    bg_height = total_text_height + (vertical_padding * 2)

    if reverse:

        bg_y1 = y1
        bg_y2 = min(y2, y1 + bg_height)

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        current_y = bg_y1 + vertical_padding
        for line in lines:
            if current_y > y2:
                break
            draw.text((x1 + 6, current_y + 1), line, font=font, fill=(0, 0, 0))
            draw.text((x1 + 5, current_y),
                      line,
                      font=font,
                      fill=(255, 255, 255))
            current_y += line_height + line_spacing

    else:
        bg_y1 = max(y1, y2 - bg_height)

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, bg_y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        current_y = bg_y1 + vertical_padding
        for line in lines:
            if current_y > y2:
                break
            draw.text((x1 + 6, current_y + 1), line, font=font, fill=(0, 0, 0))
            draw.text((x1 + 5, current_y),
                      line,
                      font=font,
                      fill=(255, 255, 255))
            current_y += line_height + line_spacing

    np.copyto(img, cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))


def process_video(video_path,
                  detector,
                  vlm_infer_frame_interval=30,
                  output_video_path=None,
                  crop_resize_size=None,
                  font_size=18,
                  text_loc='bottom',
                  display_ratio=1.0):
    try:
        cap = cv2.VideoCapture(int(video_path))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except ValueError:
        cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    display_width = int(display_ratio * frame_width)
    display_height = int(display_ratio * frame_height)
    FPS = cap.get(cv2.CAP_PROP_FPS)

    output_video_writer = None
    if output_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_writer = cv2.VideoWriter(output_video_path, fourcc, FPS,
                                              (display_width, display_height))

    persistent_captions = {}
    active_animations = {}
    progress_bar, frame_count = tqdm(total=total_frames), 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        progress_bar.update(1)
        current_time = time.time()

        results = detector.track(frame)
        canvas = results[0].plot(labels=False, conf=False)
        boxes = results[0].boxes if results[0].boxes is not None else []

        try:
            while not vlm_output_queue.empty():
                vlm_results = vlm_output_queue.get_nowait()
                for track_id, caption in vlm_results:
                    persistent_captions[track_id] = caption
                    if track_id not in active_animations or active_animations[
                            track_id]["full_text"] != caption:
                        active_animations[track_id] = {
                            "full_text": caption,
                            "current_text": "",
                            "last_update": current_time
                        }
        except queue.Empty:
            pass

        if frame_count % vlm_infer_frame_interval == 0 \
                and vlm_input_queue.empty():

            crops = []
            metadata = []

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(
                    box.id[0].item()) if box.id is not None else f"temp_{idx}"
                image_crop = frame[y1:y2, x1:x2]

                if image_crop.size > 0:
                    image_crop_rgb = cv2.cvtColor(image_crop,
                                                  cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_crop_rgb)

                    if crop_resize_size is not None:
                        pil_image.thumbnail(crop_resize_size)
                    crops.append(pil_image)
                    metadata.append(track_id)

            if crops:
                vlm_input_queue.put({"crops": crops, "metadata": metadata})

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0].item()) \
                    if box.id is not None else f"temp_{idx}"

            if track_id in active_animations:
                anim = active_animations[track_id]
                full_text = anim["full_text"]
                curr_text = anim["current_text"]

                chars_per_tick = max(2, len(full_text) // 10)
                if len(curr_text) < len(full_text) and (
                        current_time - anim["last_update"]) > 0.005:
                    next_len = min(len(full_text),
                                   len(curr_text) + chars_per_tick)
                    anim["current_text"] = full_text[:next_len]
                    anim["last_update"] = current_time

                draw_bottom_wrapped_text(
                    canvas,
                    anim["current_text"], [x1, y1, x2, y2],
                    font_size=font_size,
                    reverse=True if text_loc == 'top' else False)
            elif track_id in persistent_captions:
                draw_bottom_wrapped_text(
                    canvas,
                    persistent_captions[track_id], [x1, y1, x2, y2],
                    font_size=font_size,
                    reverse=True if text_loc == 'top' else False)

        canvas = cv2.resize(canvas, (display_width, display_height))
        if output_video_writer is not None:
            output_video_writer.write(canvas)

        cv2.imshow("Video Analysis Pipeline", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    vlm_input_queue.put(None)

    if output_video_writer is not None:
        output_video_writer.release()
        print(f"\nVideo saved to {output_video_path}")


def main():
    args = parse_arguments()

    print("Loading Local Object Detector Module...")
    if args.mode == "scene":
        detector = ROIWrapper([[0, 0, -1, -1]])
        # instruction = "Describe this scene, the environment, and the overall atmosphere concisely in 4 to 7 words."
        instruction = "以中文精準地描述畫面中發生的事"
        text_loc, font_size = 'top', 30
    elif args.mode.startswith("grid"):
        rows, cols = map(int, args.mode.split('_')[-1].split('x'))
        detector = GridSceneWrapper(rows=rows, cols=cols)
        instruction = "以中文精準地描述畫面中發生的事"
        text_loc, font_size = 'top', 18
    else:
        detector = YOLOTrackerWrapper(model_id='yolov8n-pose.pt', classes=[0])
        instruction = "以中文簡單概要畫面中人的行為"
        text_loc, font_size = 'top', 18

    if args.instruction is not None:
        instruction = args.instruction

    # Establish localized target constraints to prevent frame scaling payload bloat over network link
    crop_resize_size = (args.img_size, args.img_size)

    # Initialize client background worker loop
    worker_thread = threading.Thread(target=remote_vlm_client_loop,
                                     args=(args.vllm_url, args.vllm_model_name,
                                           instruction, vlm_ready_event),
                                     daemon=True)
    worker_thread.start()

    print("Connecting to live backend network service...")
    vlm_ready_event.wait()
    print("Success! Live connection complete. Starting stream iteration.")

    process_video(args.input_video_path,
                  detector,
                  vlm_infer_frame_interval=args.interval,
                  output_video_path=args.output_video_path,
                  display_ratio=args.display_ratio,
                  crop_resize_size=crop_resize_size,
                  font_size=font_size,
                  text_loc=text_loc)


if __name__ == "__main__":
    main()
