"""
MetaHub Input Node for ComfyUI
Reads prepared Image MetaHub bridge assets from local disk.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


BRIDGE_IMAGE_NAME = "image.png"
BRIDGE_MASK_NAME = "mask.png"
BRIDGE_METADATA_NAME = "metadata.json"


def resolve_default_bridge_dir() -> Path:
    return Path.home() / "ImageMetaHub" / "comfyui_bridge"


def resolve_bridge_dir(bridge_dir: str = "") -> Path:
    raw_dir = (bridge_dir or "").strip()
    if not raw_dir:
        return resolve_default_bridge_dir()
    return Path(raw_dir).expanduser().resolve()


def sanitize_session_id(session_id: str = "latest") -> str:
    normalized = (session_id or "latest").strip() or "latest"
    if "/" in normalized or "\\" in normalized or normalized in {".", ".."}:
        raise ValueError("MetaHub Input session_id cannot contain path separators.")
    return normalized


def resolve_payload_dir(bridge_dir: str = "", session_id: str = "latest") -> Path:
    root = resolve_bridge_dir(bridge_dir)
    normalized_session = sanitize_session_id(session_id)
    if normalized_session == "latest":
        return root / "latest"
    return root / "sessions" / normalized_session


def read_metadata(payload_dir: Path) -> tuple[dict, str]:
    metadata_path = payload_dir / BRIDGE_METADATA_NAME
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"MetaHub Input could not find {metadata_path}. "
            "Use Send to ComfyUI Bridge in Image MetaHub first, or set bridge_dir."
        )
    metadata_json = metadata_path.read_text(encoding="utf-8")
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as error:
        raise ValueError(f"MetaHub Input metadata.json is invalid JSON: {error}") from error
    return metadata, metadata_json


def image_to_tensor(image_path: Path) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"MetaHub Input could not find prepared image: {image_path}")
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def mask_to_tensor(mask_path: Path | None, width: int, height: int) -> torch.Tensor:
    if mask_path is None or not mask_path.exists():
        return torch.zeros((1, height, width), dtype=torch.float32)
    mask = ImageOps.exif_transpose(Image.open(mask_path)).convert("L")
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    array = np.asarray(mask).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def file_digest(path: Path) -> str:
    if not path.exists():
        return f"missing:{path}"
    digest = hashlib.sha256()
    digest.update(str(path).encode("utf-8", errors="replace"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_digest(bridge_dir: str = "", session_id: str = "latest") -> str:
    try:
        payload_dir = resolve_payload_dir(bridge_dir, session_id)
        metadata_path = payload_dir / BRIDGE_METADATA_NAME
        image_path = payload_dir / BRIDGE_IMAGE_NAME
        digest_parts = [
            file_digest(metadata_path),
            file_digest(image_path),
        ]
        try:
            metadata, _metadata_json = read_metadata(payload_dir)
            mask_available = bool(metadata.get("files", {}).get("mask", {}).get("available"))
        except Exception:
            mask_available = False
        if mask_available:
            digest_parts.append(file_digest(payload_dir / BRIDGE_MASK_NAME))
        return "|".join(digest_parts)
    except Exception as error:
        return f"error:{type(error).__name__}:{error}"


class MetaHubInputNode:
    """
    Loads the latest prepared Image MetaHub bridge image, optional mask, and metadata.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bridge_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Image MetaHub bridge directory. Blank uses ~/ImageMetaHub/comfyui_bridge.",
                }),
                "session_id": ("STRING", {
                    "default": "latest",
                    "tooltip": "Use latest, or a session id from bridge/sessions.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "FLOAT", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "metadata_json", "denoise", "intent", "source_path", "session_id", "width", "height")
    FUNCTION = "load_bridge"
    CATEGORY = "image/load"
    DESCRIPTION = "Load prepared Image MetaHub image, mask, and metadata from the local ComfyUI Bridge."
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, bridge_dir="", session_id="latest", **kwargs):
        return payload_digest(bridge_dir, session_id)

    def load_bridge(self, bridge_dir="", session_id="latest"):
        payload_dir = resolve_payload_dir(bridge_dir, session_id)
        metadata, metadata_json = read_metadata(payload_dir)
        image_path = payload_dir / BRIDGE_IMAGE_NAME
        image = image_to_tensor(image_path)
        height = int(image.shape[1])
        width = int(image.shape[2])

        mask_available = bool(metadata.get("files", {}).get("mask", {}).get("available"))
        mask_path = payload_dir / BRIDGE_MASK_NAME if mask_available else None
        mask = mask_to_tensor(mask_path, width, height)

        denoise = metadata.get("denoise", 0.65)
        try:
            denoise = float(denoise)
        except (TypeError, ValueError):
            denoise = 0.65
        denoise = min(1.0, max(0.0, denoise))

        intent = metadata.get("intent") or "img2img"
        source_path = metadata.get("source", {}).get("path") or ""
        resolved_session_id = metadata.get("session_id") or sanitize_session_id(session_id)

        return (
            image,
            mask,
            metadata_json,
            denoise,
            str(intent),
            str(source_path),
            str(resolved_session_id),
            width,
            height,
        )


NODE_CLASS_MAPPINGS = {
    "MetaHubInputNode": MetaHubInputNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaHubInputNode": "MetaHub Input",
}
