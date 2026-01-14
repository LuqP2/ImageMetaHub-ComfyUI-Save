"""
Video Metadata Utils for MetaHub Save Node
Handles video metadata injection using ffmpeg.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Video extensions supported
VIDEO_EXTENSIONS = {'.mp4', '.webm', '.mkv', '.mov', '.avi'}


def find_ffmpeg_binary() -> Optional[Path]:
    """
    Locates ffmpeg executable.

    Search order:
    1. FFMPEG_PATH environment variable
    2. System PATH (shutil.which)
    3. Common installation locations

    Returns:
        Path to ffmpeg executable or None if not found
    """
    # Check environment variable first
    env_path = os.environ.get('FFMPEG_PATH')
    if env_path:
        ffmpeg_path = Path(env_path)
        if ffmpeg_path.exists():
            return ffmpeg_path

    # Check system PATH
    which_result = shutil.which('ffmpeg')
    if which_result:
        return Path(which_result)

    # Common locations on Windows
    common_paths = [
        Path('C:/ffmpeg/bin/ffmpeg.exe'),
        Path('C:/Program Files/ffmpeg/bin/ffmpeg.exe'),
        Path('C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe'),
    ]

    for path in common_paths:
        if path.exists():
            return path

    return None


def find_ffprobe_binary() -> Optional[Path]:
    """
    Locates ffprobe executable (for verification).

    Returns:
        Path to ffprobe executable or None if not found
    """
    # Check environment variable
    env_path = os.environ.get('FFPROBE_PATH')
    if env_path:
        ffprobe_path = Path(env_path)
        if ffprobe_path.exists():
            return ffprobe_path

    # Check system PATH
    which_result = shutil.which('ffprobe')
    if which_result:
        return Path(which_result)

    # Try to find next to ffmpeg
    ffmpeg = find_ffmpeg_binary()
    if ffmpeg:
        ffprobe = ffmpeg.parent / ('ffprobe' + ffmpeg.suffix)
        if ffprobe.exists():
            return ffprobe

    return None


def detect_video_format(file_path: Path) -> Optional[str]:
    """
    Detects video format from file extension.

    Args:
        file_path: Path to video file

    Returns:
        Format string ('mp4', 'webm', 'mkv') or None if not a video
    """
    ext = file_path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return ext[1:]  # Remove the dot
    return None


def is_video_file(file_path: Path) -> bool:
    """
    Checks if file is a supported video format.

    Args:
        file_path: Path to check

    Returns:
        True if file is a supported video format
    """
    return file_path.suffix.lower() in VIDEO_EXTENSIONS


def build_video_a1111_metadata(params: dict) -> str:
    """
    Builds A1111/Civitai compatible metadata string for video.

    Similar to image metadata but includes video-specific fields.

    Format:
    {positive_prompt}
    Negative prompt: {negative_prompt}
    Steps: {steps}, Sampler: {sampler}, CFG scale: {cfg}, Seed: {seed},
    Size: {width}x{height}, Model: {model_name}, Model hash: {hash},
    Frames: {frame_count}, FPS: {frame_rate}

    Args:
        params: Dict containing all generation parameters

    Returns:
        Formatted A1111 metadata string
    """
    # Build LoRA hashes string
    lora_hashes_str = ", ".join(
        f"{name}: {hash_val}"
        for name, hash_val in params.get('lora_hashes', {}).items()
    )

    # Build config parts
    config_parts = [
        f"Steps: {params.get('steps', 20)}",
        f"Sampler: {params.get('sampler', 'euler')}",
        f"CFG scale: {params.get('cfg', 7.0)}",
        f"Seed: {params.get('seed', 0)}",
        f"Size: {params.get('width', 512)}x{params.get('height', 512)}",
        f"Model: {params.get('model_name', '')}",
        f"Model hash: {params.get('model_hash', '0000000000')}",
    ]

    # Add video-specific fields
    if params.get('frame_count'):
        config_parts.append(f"Frames: {params['frame_count']}")
    if params.get('frame_rate'):
        config_parts.append(f"FPS: {params['frame_rate']}")

    # Add motion model if present
    motion_model = params.get('motion_model_name')
    if motion_model:
        config_parts.append(f"Motion Model: {motion_model}")

    # Add LoRA hashes if present
    if lora_hashes_str:
        config_parts.append(f'Lora hashes: "{lora_hashes_str}"')

    config_line = ", ".join(config_parts)

    # Build final metadata string
    positive = params.get('positive', '')
    negative = params.get('negative', '')

    metadata = f"""{positive}
Negative prompt: {negative}
{config_line}"""

    return metadata


def build_video_metahub_metadata(params: dict, workflow_json: dict) -> dict:
    """
    Builds Video MetaHub metadata as JSON object.

    Similar to image metadata but with video-specific fields.

    Args:
        params: Dict containing generation parameters
        workflow_json: Complete ComfyUI workflow from extra_pnginfo

    Returns:
        JSON-serializable dict for video metadata
    """
    import numpy as np

    def _sanitize(value: Any) -> Any:
        if isinstance(value, float) and np.isnan(value):
            return None
        if isinstance(value, dict):
            return {k: _sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sanitize(v) for v in value]
        return value

    safe_workflow = _sanitize(workflow_json.get('workflow', {}))
    safe_prompt = _sanitize(workflow_json.get('prompt', {}))

    # Calculate duration if we have frame count and rate
    duration_seconds = None
    frame_count = params.get('frame_count')
    frame_rate = params.get('frame_rate')
    if frame_count and frame_rate and frame_rate > 0:
        duration_seconds = round(frame_count / frame_rate, 2)

    return {
        # CRITICAL: Required field for detection
        "generator": "ComfyUI",
        "media_type": "video",

        # Main generation fields
        "prompt": params.get('positive', ''),
        "negativePrompt": params.get('negative', ''),
        "seed": params.get('seed', 0),
        "steps": params.get('steps', 20),
        "cfg": params.get('cfg', 7.0),
        "sampler_name": params.get('sampler', 'euler'),
        "scheduler": params.get('scheduler', 'normal'),
        "model": params.get('model_name', ''),
        "model_hash": params.get('model_hash', ''),
        "vae": params.get('vae_name', ''),
        "denoise": params.get('denoise', 1.0),
        "width": params.get('width', 512),
        "height": params.get('height', 512),

        # Video-specific fields
        "video": {
            "frame_rate": frame_rate,
            "frame_count": frame_count,
            "duration_seconds": duration_seconds,
            "width": params.get('width', 512),
            "height": params.get('height', 512),
            "format": params.get('video_format', 'mp4'),
            "codec": params.get('video_codec', 'h264'),
        },

        # Motion model (AnimateDiff, etc.)
        "motion_model": {
            "name": params.get('motion_model_name'),
            "hash": params.get('motion_model_hash'),
        } if params.get('motion_model_name') else None,

        # LoRAs in array format
        "loras": [
            {"name": lora['name'], "weight": lora['weight']}
            for lora in params.get('lora_list', [])
        ],

        # IMH Pro fields
        "imh_pro": {
            "user_tags": params.get('user_tags', ''),
            "notes": params.get('notes', ''),
            "project_name": params.get('project_name', ''),
        },

        # Analytics/Performance metrics
        "analytics": {
            "vram_peak_mb": params.get('vram_peak_mb'),
            "gpu_device": params.get('gpu_device'),
            "generation_time_ms": params.get('generation_time_ms'),
            "steps_per_second": params.get('steps_per_second'),
            "comfyui_version": params.get('comfyui_version'),
            "torch_version": params.get('torch_version'),
            "python_version": params.get('python_version'),
        },

        # Complete workflow data
        "workflow": safe_workflow,
        "prompt_api": safe_prompt,
    }


def inject_video_metadata(
    video_path: Path,
    a1111_metadata: str,
    metahub_metadata: dict,
    output_path: Optional[Path] = None,
    keep_original: bool = False,
) -> Path:
    """
    Injects metadata into video container without re-encoding.

    Uses ffmpeg with -c copy to add metadata to video file.
    Metadata is stored in:
    - description: A1111/Civitai format
    - comment: JSON videometahub_data

    Args:
        video_path: Path to input video file
        a1111_metadata: A1111 formatted metadata string
        metahub_metadata: MetaHub metadata dict (will be JSON serialized)
        output_path: Optional output path (default: replace original)
        keep_original: If True, keep original file (only works if output_path differs)

    Returns:
        Path to video with injected metadata

    Raises:
        RuntimeError: If ffmpeg is not found or injection fails
    """
    ffmpeg = find_ffmpeg_binary()
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg and ensure it's in PATH, "
            "or set FFMPEG_PATH environment variable."
        )

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Determine output path
    if output_path is None:
        # Replace original - use temp file then rename
        replace_original = True
        temp_fd, temp_path = tempfile.mkstemp(suffix=video_path.suffix)
        os.close(temp_fd)
        final_output = Path(temp_path)
    else:
        replace_original = False
        final_output = Path(output_path)

    # Serialize metadata JSON
    metahub_json = json.dumps(metahub_metadata, ensure_ascii=False, separators=(',', ':'))

    # Detect format for format-specific flags
    video_format = detect_video_format(video_path)

    # Build ffmpeg command
    cmd = [
        str(ffmpeg),
        '-y',  # Overwrite output
        '-i', str(video_path),
        '-c', 'copy',  # No re-encoding
        '-map_metadata', '0',  # Preserve existing metadata
    ]

    # Add format-specific flags
    if video_format == 'mp4':
        cmd.extend(['-movflags', 'use_metadata_tags'])

    # Add metadata fields
    # Use description for A1111 format (human-readable)
    cmd.extend(['-metadata', f'description={a1111_metadata}'])
    # Use comment for full JSON data
    cmd.extend(['-metadata', f'comment={metahub_json}'])
    # Also add a title with basic info
    title = f"Generated with ComfyUI - Seed: {metahub_metadata.get('seed', 'unknown')}"
    cmd.extend(['-metadata', f'title={title}'])

    # Output file
    cmd.append(str(final_output))

    # Execute ffmpeg
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg timed out while injecting metadata")
    except Exception as e:
        # Clean up temp file on error
        if replace_original and final_output.exists():
            final_output.unlink()
        raise RuntimeError(f"Failed to inject metadata: {e}")

    # If replacing original, move temp file to original location
    if replace_original:
        try:
            # Remove original
            video_path.unlink()
            # Move temp to original location
            shutil.move(str(final_output), str(video_path))
            return video_path
        except Exception as e:
            # Try to clean up
            if final_output.exists():
                final_output.unlink()
            raise RuntimeError(f"Failed to replace original file: {e}")

    return final_output


def verify_video_metadata(video_path: Path) -> Optional[dict]:
    """
    Verifies that metadata was successfully injected into video.

    Uses ffprobe to read metadata from video file.

    Args:
        video_path: Path to video file

    Returns:
        Dict with metadata fields or None if verification failed
    """
    ffprobe = find_ffprobe_binary()
    if not ffprobe:
        print("[MetaHub Video] Warning: ffprobe not found, cannot verify metadata")
        return None

    try:
        cmd = [
            str(ffprobe),
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            str(video_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        tags = data.get('format', {}).get('tags', {})

        return {
            'title': tags.get('title'),
            'description': tags.get('description'),
            'comment': tags.get('comment'),
            'has_metahub_data': 'videometahub_data' in tags.get('comment', '') or
                               '"generator":"ComfyUI"' in tags.get('comment', ''),
        }

    except Exception as e:
        print(f"[MetaHub Video] Warning: Could not verify metadata: {e}")
        return None


def get_video_info(video_path: Path) -> Optional[dict]:
    """
    Gets video information using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Dict with video info (frame_count, frame_rate, width, height, duration) or None
    """
    ffprobe = find_ffprobe_binary()
    if not ffprobe:
        return None

    try:
        cmd = [
            str(ffprobe),
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-show_format',
            str(video_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)

        # Find video stream
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break

        if not video_stream:
            return None

        # Parse frame rate (can be "30/1" or "29.97")
        frame_rate = None
        r_frame_rate = video_stream.get('r_frame_rate', '')
        if '/' in r_frame_rate:
            num, den = r_frame_rate.split('/')
            if int(den) != 0:
                frame_rate = round(int(num) / int(den), 2)
        else:
            try:
                frame_rate = float(r_frame_rate)
            except ValueError:
                pass

        # Get frame count
        frame_count = video_stream.get('nb_frames')
        if frame_count:
            frame_count = int(frame_count)

        # Get duration
        duration = data.get('format', {}).get('duration')
        if duration:
            duration = float(duration)

        return {
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'frame_rate': frame_rate,
            'frame_count': frame_count,
            'duration': duration,
            'codec': video_stream.get('codec_name'),
        }

    except Exception as e:
        print(f"[MetaHub Video] Warning: Could not get video info: {e}")
        return None


def extract_video_files(filenames_tuple: tuple) -> List[Path]:
    """
    Extracts video file paths from VHS_FILENAMES tuple.

    VHS_FILENAMES format: (save_output: bool, [file_paths: List[str]])

    Args:
        filenames_tuple: VHS_FILENAMES tuple from Video Combine node

    Returns:
        List of Path objects for video files
    """
    video_paths = []

    try:
        if not isinstance(filenames_tuple, (tuple, list)):
            return video_paths

        # VHS_FILENAMES is (bool, [paths])
        if len(filenames_tuple) < 2:
            return video_paths

        file_list = filenames_tuple[1]
        if not isinstance(file_list, (list, tuple)):
            return video_paths

        for path_str in file_list:
            path = Path(path_str)
            if path.exists() and is_video_file(path):
                video_paths.append(path)

    except Exception as e:
        print(f"[MetaHub Video] Warning: Could not parse VHS_FILENAMES: {e}")

    return video_paths
