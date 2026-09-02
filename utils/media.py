import subprocess
import base64
import io
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import decord
import numpy as np
from PIL import Image
from tqdm import tqdm


def is_valid_image(file_bytes: bytes) -> bool:
    IMAGE_SIGNATURES = {
        b"\xff\xd8\xff": "jpeg",
        b"\x89PNG\r\n\x1a\n": "png",
        b"GIF87a": "gif",
        b"GIF89a": "gif",
        b"RIFF": "webp",
        b"BM": "bmp",
    }
    for sig in IMAGE_SIGNATURES:
        if file_bytes.startswith(sig):
            return True
    return False

def _load_single_image_as_base64(path, **kwargs):
    if not path:
        return None

    resize_images = kwargs.get("resize_images", False)
    resize_factor = kwargs.get("resize_factor", 0.5)
    max_pixels = kwargs.get("max_pixels", None)

    try:
        path = Path(path)
        ext = path.suffix[1:].lower() or "jpeg"

        if resize_images or max_pixels is not None:
            img = Image.open(path).convert("RGB")
            width, height = img.size

            if max_pixels is not None:
                if width * height > max_pixels:
                    scale = (max_pixels / (width * height)) ** 0.5
                    width, height = int(width * scale), int(height * scale)
            elif resize_images:
                height, width = int(height * resize_factor), int(width * resize_factor)

            img = img.resize((width, height), Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"

        else:
            with open(path, "rb") as f:
                file_bytes = f.read()
            if not is_valid_image(file_bytes):
                raise ValueError(f"Not a valid image: {path}")
            Image.open(io.BytesIO(file_bytes)).load()
            encoded = base64.b64encode(file_bytes).decode("utf-8")
            return f"data:image/{ext};base64,{encoded}"

    except Exception as e:
        tqdm.write(f"Error loading image {path}: {e}")
        return None

def _load_single_video_as_base64(path, **kwargs):
    if not path:
        return None

    size = kwargs.get("size", None)

    try:
        path = Path(path)

        if size is not None:
            result = subprocess.run(
                [
                    "ffmpeg", "-i", path,
                    "-vf", (
                        f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
                        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2"
                    ),
                    "-c:v", "libx264",
                    "-an",
                    "-movflags", "frag_keyframe+empty_moov",
                    "-f", "mp4",
                    "-y",
                    "pipe:1"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True
            )
            file_bytes = result.stdout
        else:
            with open(path, "rb") as f:
                file_bytes = f.read()

        ext = "mp4" if size is not None else (path.suffix[1:].lower() or "mp4")
        encoded = base64.b64encode(file_bytes).decode("utf-8")
        return f"data:video/{ext};base64,{encoded}"

    except subprocess.CalledProcessError as e:
        tqdm.write(f"ffmpeg error for {path}: {e}")
    except Exception as e:
        tqdm.write(f"Error loading video {path}: {e}")

    return None

def _load_single_video_frames_base64(path, **kwargs):
    if not path:
        return []

    num_frames = kwargs.get("num_frames", 16)
    frame_size = kwargs.get("frame_size", None)

    data_urls = []

    try:
        decord.bridge.set_bridge("native")

        vr = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=4)
        total_frames = len(vr)

        if total_frames <= 0:
            tqdm.write(f"Error: Could not read frame count for {path}")
            return []

        num_frames = min(num_frames, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

        frames = vr.get_batch(frame_indices).asnumpy()

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        prefix = "data:image/jpeg;base64,"

        for frame_rgb in frames:
            frame_bgr = frame_rgb[:, :, ::-1]

            if frame_size is not None:
                frame_bgr = cv2.resize(frame_bgr, (frame_size, frame_size), interpolation=cv2.INTER_AREA)

            success, buffer = cv2.imencode(".jpg", frame_bgr, encode_params)
            if success:
                data_urls.append(prefix + base64.b64encode(buffer).decode("utf-8"))

    except Exception as e:
        tqdm.write(f"Error loading video frames from {path}: {e}")

    return data_urls

def _load_single_media(kind, path, **kwargs):
    try:
        if kind == "image":
            return _load_single_image_as_base64(path, **kwargs)
        elif kind == "video":
            return _load_single_video_as_base64(path, **kwargs)
        elif kind == "video_frames":
            return _load_single_video_frames_base64(path, **kwargs)
    except Exception as e:
        tqdm.write(f"Error loading media {kind} from {path}: {e}")
        return None


def load_batch_media(model_id, batch_image_paths, batch_video_paths,
                     resize_images, resize_factor, video_as_frames,
                     num_frames, example_k_images_for_batch, max_pixels=None):
    model_is_intern = "intern" in model_id.lower()
    model_is_llava = "llava" in model_id.lower()

    # An explicit max_pixels (e.g. example-driven capping qwen's huge prompts)
    # overrides the per-model default for every image, query and demonstration.
    default_max_pixels = (
        448 * 448 * 12 if model_is_intern
        else 384 * 384 if model_is_llava
        else None
    )
    eff_max_pixels = max_pixels if max_pixels is not None else default_max_pixels

    tasks = []
    for idx, (image_paths, video_path) in enumerate(zip(batch_image_paths, batch_video_paths)):
        if image_paths:
            for j, p in enumerate(image_paths):
                tasks.append({
                    "kind": "image", "batch_idx": idx, "slot": "query_images", "pos": j, "path": p,
                    "kwargs": {
                        "resize_images": resize_images,
                        "resize_factor": resize_factor,
                        "max_pixels": eff_max_pixels,
                    }
                })
        if video_path:
            if not video_as_frames:
                tasks.append({
                    "kind": "video", "batch_idx": idx, "slot": "query_video", "pos": 0, "path": video_path,
                    "kwargs": {"size": 384 if model_is_llava else None}
                })
            else:
                tasks.append({
                    "kind": "video_frames", "batch_idx": idx, "slot": "query_video", "pos": 0, "path": video_path,
                    "kwargs": {"num_frames": num_frames, "frame_size": 448 if model_is_intern else None}
                })
        if example_k_images_for_batch is not None:
            for j, p in enumerate(example_k_images_for_batch[idx]):
                for img_index, image in enumerate(p):
                    tasks.append({
                        "kind": "image", "batch_idx": idx, "slot": "example_images",
                        "fs_example": j, "img_index": img_index, "path": image,
                        "kwargs": {"resize_images": resize_images, "resize_factor": resize_factor,
                                   "max_pixels": eff_max_pixels}
                    })

    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_load_single_media, task["kind"], task["path"], **task["kwargs"]): i
            for i, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    batch_media = defaultdict(lambda: {"query_images": [], "example_images": [], "query_video": None})
    for task, result in zip(tasks, results):
        idx, slot = task["batch_idx"], task["slot"]
        if slot == "query_video":
            batch_media[idx][slot] = result
        elif slot == "example_images":
            batch_media[idx][slot].append((task["fs_example"], task["img_index"], result))
        else:
            batch_media[idx][slot].append((task["pos"], result))

    for idx in batch_media:
        batch_media[idx]["query_images"] = [
            x[1] for x in sorted(batch_media[idx]["query_images"], key=lambda x: x[0])
        ]
        grouped = defaultdict(dict)
        for fs_idx, img_index, img in batch_media[idx]["example_images"]:
            grouped[fs_idx][img_index] = img
        n_example_driven = len(example_k_images_for_batch[idx]) if example_k_images_for_batch is not None else 0
        batch_media[idx]["example_images"] = [
            [grouped[fs_idx][k] for k in sorted(grouped[fs_idx])] if fs_idx in grouped else []
            for fs_idx in range(n_example_driven)
        ]
    return batch_media
