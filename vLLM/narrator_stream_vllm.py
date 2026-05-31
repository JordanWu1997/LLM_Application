#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8

import argparse
import queue
import threading
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoProcessor
from ultralytics import YOLO
from ultralytics.engine.results import Results
from vllm import LLM, SamplingParams

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
vlm_input_queue = queue.Queue(maxsize=1)
vlm_output_queue = queue.Queue()
vlm_ready_event = threading.Event()


class ROIWrapper:
    """
    Fakes a YOLO tracking model using Custom ROIs.
    Supports `-1` to snap to the frame's max width/height.
    """

    def __init__(self, rois):
        self.rois = rois
        self.names = {0: "roi"}

    def track(self, frame):
        h, w = frame.shape[:2]

        box_list = []

        for idx, roi in enumerate(self.rois):
            x1 = float(w if roi[0] == -1 else roi[0])
            y1 = float(h if roi[1] == -1 else roi[1])
            x2 = float(w if roi[2] == -1 else roi[2])
            y2 = float(h if roi[3] == -1 else roi[3])

            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Pack the 7-column tensor: [x1, y1, x2, y2, track_id, conf, class_id]
            track_id = float(idx + 1)
            box_list.append([x1, y1, x2, y2, track_id, 1.0, 0.0])

        # If list is empty, create an empty (0, 7) tensor to avoid crashes
        if not box_list:
            box_data = torch.empty((0, 7), dtype=torch.float32)
        else:
            box_data = torch.tensor(box_list, dtype=torch.float32)

        # Pass the 7-column tensor directly to Results
        result = Results(orig_img=frame,
                         path="memory",
                         names=self.names,
                         boxes=box_data)

        return [result]


class GridSceneWrapper:
    """
    Fakes a YOLO tracking model by dividing the frame into a uniform grid.
    """

    def __init__(self, rows=2, cols=2):
        self.rows = rows
        self.cols = cols
        self.names = {0: "region"}

    def track(self, frame):
        h, w = frame.shape[:2]
        cell_w = w / self.cols
        cell_h = h / self.rows

        box_list = []
        track_id_counter = 1

        for r in range(self.rows):
            for c in range(self.cols):
                x1 = c * cell_w
                y1 = r * cell_h
                x2 = (c + 1) * cell_w
                y2 = (r + 1) * cell_h

                track_id = float(track_id_counter)
                # Pack the 7-column tensor: [x1, y1, x2, y2, track_id, conf, class_id]
                box_list.append([x1, y1, x2, y2, track_id, 1.0, 0.0])
                track_id_counter += 1

        if not box_list:
            box_data = torch.empty((0, 7), dtype=torch.float32)
        else:
            box_data = torch.tensor(box_list, dtype=torch.float32)

        result = Results(orig_img=frame,
                         path="memory",
                         names=self.names,
                         boxes=box_data)

        return [result]


class YOLOTrackerWrapper:
    """Standard YOLO tracking wrapped to share the same initialization interface."""

    def __init__(self, model_id="yolov8n.pt", classes=[0]):
        self.model = YOLO(model_id)
        self.classes = classes

    def track(self, frame):
        # Native YOLO naturally returns a list of Results objects
        return self.model.track(frame,
                                persist=True,
                                classes=self.classes,
                                verbose=False)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Real-time Human Behavior Analysis Pipeline")

    parser.add_argument("input_video_path",
                        type=str,
                        help="Path to the input video file.")

    parser.add_argument(
        "-o",
        "--output_video_path",
        type=str,
        default=None,
        help=
        "Path to save the output video. (default: None, video will not be saved)"
    )

    parser.add_argument(
        "--vlm_family",
        type=str,
        choices=["llava", "gemma", "qwen"],
        default="qwen",
        help="Vision-Language Model family. Options: 'llava', 'gemma', 'qwen'."
    )

    parser.add_argument("--mode",
                        type=str,
                        choices=["person", "scene", "grid"],
                        default="person",
                        help="The analysis mode to run.")

    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help=
        "The specific text prompt/instruction sent to the VLM. (Leave empty to use default template)"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help=
        "How many frames to wait between sending crops to the VLM. (default: 30)"
    )

    parser.add_argument("-r",
                        "--display_ratio",
                        type=float,
                        default=1.0,
                        help="Scaling ratio to display visualization results")

    return parser.parse_args()


# =========================================================
# DECOUPLED VLM WORKERS
# =========================================================


def llava_worker_loop(vlm_model_id, task_instruction, sampling_params,
                      ready_event):
    """Worker specifically designed for LLaVA family models using raw strings."""
    print(f"[LLaVA Worker] Loading 4-bit Quantized VLM ({vlm_model_id})...")
    vlm = LLM(model=vlm_model_id,
              quantization="awq",
              max_model_len=2048,
              gpu_memory_utilization=0.70,
              enforce_eager=True,
              limit_mm_per_prompt={"image": 1},
              disable_log_stats=True)

    print("[LLaVA Worker] Initialization complete. Engine is hot.")
    ready_event.set()

    while True:
        try:
            task = vlm_input_queue.get()
            if task is None: break

            crops, metadata = task["crops"], task["metadata"]

            inputs = []
            for img in crops:
                # LLaVA raw string templating
                prompt = f"USER: <image>\n{task_instruction}\nASSISTANT:"
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": img
                    }
                })

            outputs = vlm.generate(inputs,
                                   sampling_params=sampling_params,
                                   use_tqdm=False)

            results = [(metadata[i], out.outputs[0].text.strip())
                       for i, out in enumerate(outputs)]
            vlm_output_queue.put(results)
            vlm_input_queue.task_done()
        except Exception as e:
            print(f"[LLaVA Worker Error]: {e}")


def gemma_worker_loop(
    vlm_model_id,
    task_instruction,
    sampling_params,
    ready_event,
):
    """Worker specifically designed for Gemma family models using AutoProcessor."""
    print(f"[Gemma Worker] Loading AutoProcessor for {vlm_model_id}...")
    processor = AutoProcessor.from_pretrained(vlm_model_id,
                                              trust_remote_code=True)

    print(f"[Gemma Worker] Loading 4-bit Quantized VLM ({vlm_model_id})...")
    vlm = LLM(model=vlm_model_id,
              trust_remote_code=True,
              max_model_len=2048,
              gpu_memory_utilization=0.60,
              max_num_seqs=2,
              enforce_eager=True,
              limit_mm_per_prompt={"image": 1},
              disable_log_stats=True)

    print("[Gemma Worker] Initialization complete. Engine is hot.")
    ready_event.set()

    while True:
        try:
            task = vlm_input_queue.get()
            if task is None: break

            crops, metadata = task["crops"], task["metadata"]

            inputs = []
            for img in crops:
                # Gemma dynamic templating via AutoProcessor
                chat = [{
                    "role":
                    "user",
                    "content": [{
                        "type": "image"
                    }, {
                        "type": "text",
                        "text": task_instruction
                    }]
                }]
                prompt = processor.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True)
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": img
                    }
                })

            outputs = vlm.generate(inputs,
                                   sampling_params=sampling_params,
                                   use_tqdm=False)

            results = [(metadata[i], out.outputs[0].text.strip())
                       for i, out in enumerate(outputs)]
            vlm_output_queue.put(results)
            vlm_input_queue.task_done()
        except Exception as e:
            print(f"[Gemma Worker Error]: {e}")


def qwen_worker_loop(vlm_model_id, task_instruction, sampling_params,
                     ready_event):
    """Worker specifically designed for Qwen-VL family models."""
    print(f"[Qwen Worker] Loading AutoProcessor for {vlm_model_id}...")
    processor = AutoProcessor.from_pretrained(vlm_model_id,
                                              trust_remote_code=True)

    print(f"[Qwen Worker] Loading 4-bit Quantized VLM ({vlm_model_id})...")
    vlm = LLM(
        model=vlm_model_id,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.75,
        max_num_seqs=
        2,  # Critical to prevent Qwen's memory profiler from crashing
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        disable_log_stats=True)

    print("[Qwen Worker] Initialization complete. Engine is hot.")
    ready_event.set()

    while True:
        try:
            task = vlm_input_queue.get()
            if task is None: break

            crops, metadata = task["crops"], task["metadata"]

            inputs = []
            for img in crops:
                # Qwen-VL specific chat template structure
                chat = [{
                    "role": "system",
                    "content": "You are a highly analytical assistant."
                }, {
                    "role":
                    "user",
                    "content": [{
                        "type": "image"
                    }, {
                        "type": "text",
                        "text": task_instruction
                    }]
                }]

                prompt = processor.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True)
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": img
                    }
                })

            outputs = vlm.generate(inputs,
                                   sampling_params=sampling_params,
                                   use_tqdm=False)

            results = [(metadata[i], out.outputs[0].text.strip())
                       for i, out in enumerate(outputs)]
            vlm_output_queue.put(results)
            vlm_input_queue.task_done()
        except Exception as e:
            print(f"[Qwen Worker Error]: {e}")


# =========================================================
# CORE PIPELINE
# =========================================================


def draw_bottom_wrapped_text(
        img,
        text,
        bbox,
        font_size=18,
        chinese_font_path="/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
    """Wraps CJK/English text and renders it using PIL for Unicode support."""
    x1, y1, x2, y2 = map(int, bbox)
    box_width = x2 - x1
    max_width = box_width - 10 if box_width > 20 else box_width

    # Load the Chinese font
    try:
        font = ImageFont.truetype(chinese_font_path, font_size)
    except IOError:
        print(
            f"Warning: Could not load font at {chinese_font_path}. Falling back to default."
        )
        font = ImageFont.load_default()

    # --- 1. CJK-Friendly Text Wrapping ---
    # Evaluates character-by-character since Chinese doesn't use spaces
    lines = []
    current_line = ''

    for char in text:
        test_line = current_line + char
        # Get pixel width of the string
        width = font.getlength(test_line) if hasattr(
            font, 'getlength') else font.getsize(test_line)[0]

        if width > max_width and current_line != '':
            lines.append(current_line)
            current_line = char
        else:
            current_line = test_line

    if current_line:
        lines.append(current_line)

    # --- 2. Geometry & Layout Calculation ---
    # Get the height of a standard character
    if hasattr(font, 'getbbox'):
        bbox_font = font.getbbox("测Test")
        line_height = bbox_font[3] - bbox_font[1]
    else:
        line_height = font.getsize("测Test")[1]

    line_spacing = 4
    total_text_height = len(lines) * line_height + (len(lines) -
                                                    1) * line_spacing
    vertical_padding = 6
    bg_height = total_text_height + (vertical_padding * 2)

    bg_y1 = max(y1, y2 - bg_height)

    # --- 3. Draw Semi-Transparent Background (OpenCV is faster for shapes) ---
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, bg_y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # --- 4. Draw Chinese Text (PIL is required for Unicode) ---
    # Convert OpenCV image (BGR) to PIL Image (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    current_y = bg_y1 + vertical_padding
    for line in lines:
        if current_y > y2:
            break
        # Draw a slight black shadow/outline for better contrast
        draw.text((x1 + 6, current_y + 1), line, font=font, fill=(0, 0, 0))
        # Draw the actual white text
        draw.text((x1 + 5, current_y), line, font=font, fill=(255, 255, 255))
        current_y += line_height + line_spacing

    # Convert PIL Image back to OpenCV array and apply it in-place to the original frame
    np.copyto(img, cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))


def process_video(video_path,
                  detector,
                  vlm_infer_frame_interval=30,
                  output_video_path=None,
                  crop_resize_size=None,
                  display_ratio=1.0):
    """Core Video Pipeline Loop"""

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

                    # Optional: resize crops for Gemma to save context tokens
                    if crop_resize_size is not None:
                        pil_image.thumbnail(crop_resize_size)
                    crops.append(pil_image)
                    metadata.append(track_id)

            if crops:
                # Main thread no longer builds prompts, only sends raw data
                vlm_input_queue.put({"crops": crops, "metadata": metadata})

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(
                box.id[0].item()) if box.id is not None else f"temp_{idx}"

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

                draw_bottom_wrapped_text(canvas, anim["current_text"],
                                         [x1, y1, x2, y2])
            elif track_id in persistent_captions:
                draw_bottom_wrapped_text(canvas, persistent_captions[track_id],
                                         [x1, y1, x2, y2])

        canvas = cv2.resize(canvas, (display_width, display_height))
        if output_video_writer is not None:
            output_video_writer.write(canvas)

        cv2.imshow("Human Behavior Analysis", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    vlm_input_queue.put(None)

    if output_video_writer is not None:
        output_video_writer.release()
        print(f"\nVideo is saved as {output_video_path}")

    print("\nProcessing stopped.")


# ---------------------------------------------------------
# Entrypoint Guard Setup
# ---------------------------------------------------------
def main():

    # Load argument
    args = parse_arguments()

    # Load detector, task instruction
    print("Loading Detector ...")
    if args.mode == "scene":
        detector = ROIWrapper([[0, 0, -1, -1]])
        instruction = "Describe this scene, the environment, and the overall atmosphere concisely in 4 to 7 words."
        # instruction = "以中文精準地描述畫面中發生的事"
    elif args.mode == "grid":
        # Divide the screen into 4 distinct quadrants
        detector = GridSceneWrapper(rows=1, cols=3)
        # instruction = "Describe the main object or activity happening specifically in this cropped region in 4 to 7 words."
        instruction = "以中文精準地描述畫面中發生的事"
    else:  # YOLO "person"
        detector = YOLOTrackerWrapper(model_id='yolov8n-pose.pt', classes=[0])
        instruction = "Describe the actions or behavior of this person concisely in 4 to 7 words."

    # User-defined instruction
    if args.instruction is not None:
        instruction = args.instruction

    # Load VLM
    if args.vlm_family == "gemma":
        vlm_model_id = "RedHatAI/gemma-3-4b-it-quantized.w4a16"
        target_worker = gemma_worker_loop
        crop_resize_size = (336, 336)
    elif args.vlm_family == "qwen":
        vlm_model_id = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
        target_worker = qwen_worker_loop
        crop_resize_size = (256, 256)
    else:
        vlm_model_id = "ybelkada/llava-1.5-7b-hf-awq"
        target_worker = llava_worker_loop
        crop_resize_size = None

    sampling_params = SamplingParams(temperature=0.1, max_tokens=32)

    # Dynamic thread assignment based on chosen family
    worker_thread = threading.Thread(target=target_worker,
                                     args=(vlm_model_id, instruction,
                                           sampling_params, vlm_ready_event),
                                     daemon=True)
    worker_thread.start()

    print("--------------------------------------------------")
    print("Waiting for vLLM Server to allocate VRAM...")
    print("Video playback will remain paused until engine is ready.")
    print("--------------------------------------------------")
    vlm_ready_event.wait()
    print("Starting real-time video inference loop!")

    process_video(args.input_video_path,
                  detector,
                  vlm_infer_frame_interval=args.interval,
                  output_video_path=args.output_video_path,
                  display_ratio=args.display_ratio,
                  crop_resize_size=crop_resize_size)


if __name__ == "__main__":
    main()
