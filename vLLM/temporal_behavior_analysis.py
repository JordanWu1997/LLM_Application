#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8

import argparse
import collections
import math
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

from narrator_stream_vllm import YOLOTrackerWrapper, draw_bottom_wrapped_text

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
vlm_input_queue = queue.Queue(maxsize=1)
vlm_output_queue = queue.Queue()
vlm_ready_event = threading.Event()


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

    parser.add_argument(
        "-i",
        "--instruction",
        type=str,
        default=None,
        help=
        "The specific text prompt/instruction sent to the VLM. (Leave empty to use default template)"
    )

    parser.add_argument("--temporal_window",
                        type=int,
                        default=5,
                        help="Frame window for VLM to analysize. (default: 5)")

    parser.add_argument(
        "-I",
        "--interval",
        type=int,
        default=5,
        help=
        "How many frames to wait between sending crops to the VLM. (default: 5)"
    )

    parser.add_argument("-r",
                        "--display_ratio",
                        type=float,
                        default=1.0,
                        help="Scaling ratio to display visualization results")

    return parser.parse_args()


def qwen_temporal_worker_loop(vlm_model_id,
                              task_instruction,
                              sampling_params,
                              ready_event,
                              temporal_window=5):
    print(
        f"[Qwen Temporal Worker] Loading AutoProcessor for {vlm_model_id}...")
    processor = AutoProcessor.from_pretrained(vlm_model_id,
                                              trust_remote_code=True)

    print(
        f"[Qwen Temporal Worker] Loading 4-bit Quantized VLM ({vlm_model_id})..."
    )
    vlm = LLM(
        model=vlm_model_id,
        trust_remote_code=True,
        max_model_len=4096,  # Increased context window for multiple images
        gpu_memory_utilization=0.65,
        max_num_seqs=2,
        enforce_eager=True,
        # --- CRITICAL FIX: Allow multiple images per prompt ---
        limit_mm_per_prompt={"image": temporal_window},
        disable_log_stats=True)

    print("[Qwen Temporal Worker] Engine is hot.")
    ready_event.set()

    # Refined instruction for temporal sequences
    temporal_instruction = f"These are {temporal_window} sequential frames of a person. " + task_instruction

    while True:
        try:
            task = vlm_input_queue.get()
            if task is None: break

            # Now we receive a list of sequences, not a list of single images
            sequences, metadata = task["sequences"], task["metadata"]

            inputs = []
            for seq in sequences:
                # Dynamically build the content array based on sequence length
                content_block = [{"type": "image"} for _ in range(len(seq))]
                content_block.append({
                    "type": "text",
                    "text": temporal_instruction
                })

                chat = [{
                    "role":
                    "system",
                    "content":
                    "You are a highly analytical assistant evaluating continuous video frames."
                }, {
                    "role": "user",
                    "content": content_block
                }]

                prompt = processor.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True)

                # multi_modal_data accepts a list of PIL images
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": seq
                    }
                })

            # Run inference
            outputs = vlm.generate(inputs,
                                   sampling_params=sampling_params,
                                   use_tqdm=False)

            results = [(metadata[i], out.outputs[0].text.strip())
                       for i, out in enumerate(outputs)]
            vlm_output_queue.put(results)
            vlm_input_queue.task_done()
        except Exception as e:
            print(f"[Qwen Temporal Worker Error]: {e}")


def process_video(video_path,
                  detector,
                  vlm_infer_frame_interval=30,
                  output_video_path=None,
                  crop_resize_size=None,
                  temporal_window=5,
                  display_ratio=1.0):
    """Core Video Pipeline Loop"""

    # Maps track_id -> sliding window of PIL images
    temporal_buffers = collections.defaultdict(
        lambda: collections.deque(maxlen=temporal_window))
    last_seen_frames = {}  # Tracks when we last saw an ID to clear stale data

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
        # canvas = results[0].plot(labels=False, conf=False)
        canvas = results[0].plot()
        boxes = results[0].boxes if results[0].boxes is not None else []

        try:
            while not vlm_output_queue.empty():
                vlm_results = vlm_output_queue.get_nowait()
                for track_id, caption in vlm_results:
                    persistent_captions[track_id] = caption

                    if track_id not in active_animations:
                        # Initialize with a queue of pending texts
                        active_animations[track_id] = {
                            "text_queue": [caption],
                            "current_full": caption,
                            "current_typed": "",
                            "last_update": current_time,
                            "hold_time_start": 0
                        }
                    else:
                        # Only add to queue if it's a NEW caption
                        if caption != active_animations[track_id][
                                "text_queue"][-1]:
                            active_animations[track_id]["text_queue"].append(
                                caption)
        except queue.Empty:
            pass

        # 2. Extract crops EVERY frame and add to the temporal buffer
        current_frame_ids = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(
                box.id[0].item()) if box.id is not None else f"temp_{idx}"
            current_frame_ids.append(track_id)

            # Update last seen tracker
            last_seen_frames[track_id] = frame_count

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size > 0:
                person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(person_crop_rgb)

                # CRITICAL MEMORY FIX: Downscale before storing in RAM!
                # 5 full-res crops per person will destroy your RAM.
                pil_image.thumbnail((256, 256))

                # Append to the sliding window (automatically drops oldest if full)
                temporal_buffers[track_id].append(pil_image)

        # 3. Dispatch to VLM if interval is reached and queue is empty
        if frame_count % vlm_infer_frame_interval == 0 \
                and vlm_input_queue.empty():
            sequences = []
            metadata = []

            for track_id in current_frame_ids:
                buffer = temporal_buffers[track_id]

                # Only send if the person has been on screen long enough to fill the window
                if len(buffer) == temporal_window:
                    # Convert deque to standard list for the VLM queue
                    sequences.append(list(buffer))
                    metadata.append(track_id)

            if sequences:
                vlm_input_queue.put({
                    "sequences": sequences,
                    "metadata": metadata
                })

        # 4. Garbage Collection (Clear memory for people who left the screen)
        # If an ID hasn't been seen in 60 frames, delete their buffer and text cache
        stale_ids = [
            tid for tid, last_frame in last_seen_frames.items()
            if (frame_count - last_frame) > 60
        ]
        for tid in stale_ids:
            temporal_buffers.pop(tid, None)
            last_seen_frames.pop(tid, None)
            persistent_captions.pop(tid, None)
            active_animations.pop(tid, None)

        # 5. Draw text HUD
        if track_id in active_animations:
            anim = active_animations[track_id]

            full_text = anim["current_full"]
            curr_text = anim["current_typed"]

            # Standard typing speed (comfortable for reading)
            chars_per_tick = max(1, len(full_text) // 15)

            # Check if we are done typing the current sentence
            if len(curr_text) >= len(full_text):
                # Start the hold timer if we haven't already
                if anim["hold_time_start"] == 0:
                    anim["hold_time_start"] = current_time

                # Wait 1.5 seconds for the user to read it.
                # If there are more texts waiting in the queue, load the next one!
                if (current_time - anim["hold_time_start"]) > 1.5 and len(
                        anim["text_queue"]) > 1:
                    anim["text_queue"].pop(0)  # Remove the old text
                    anim["current_full"] = anim["text_queue"][
                        0]  # Load the new text
                    anim["current_typed"] = ""  # Reset typing
                    anim["hold_time_start"] = 0  # Reset timer
            else:
                # Still typing...
                if (current_time - anim["last_update"]) > 0.01:
                    next_len = min(len(full_text),
                                   len(curr_text) + chars_per_tick)
                    anim["current_typed"] = full_text[:next_len]
                    anim["last_update"] = current_time

            draw_bottom_wrapped_text(canvas, anim["current_typed"],
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
    detector = YOLOTrackerWrapper(model_id='yolov8n-pose.pt', classes=[0])
    # instruction = "Describe the actions or behavior of this person concisely in 4 to 7 words."
    instruction = "以中文描述畫面中人的行為"

    # User-defined instruction
    if args.instruction is not None:
        instruction = args.instruction

    # VLM model
    vlm_model_id = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
    target_worker = qwen_temporal_worker_loop
    crop_resize_size = (256, 256)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=32)

    # temporal_window = 5
    # args.interval = 5

    # Dynamic thread assignment based on chosen family
    worker_thread = threading.Thread(
        target=target_worker,
        args=(vlm_model_id, instruction, sampling_params, vlm_ready_event),
        kwargs={'temporal_window': args.temporal_window},
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
                  crop_resize_size=crop_resize_size,
                  temporal_window=args.temporal_window)


if __name__ == "__main__":
    main()
