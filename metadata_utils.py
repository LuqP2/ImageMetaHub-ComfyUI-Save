"""
Metadata Utils for MetaHub Save Node
Handles hash calculation, metadata formatting, and metadata injection.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape as xml_escape
import numpy as np
from PIL import Image, PngImagePlugin

try:
    import piexif
    from piexif import helper as piexif_helper
except Exception:
    piexif = None
    piexif_helper = None


# ============================================================================
# HASH CALCULATION SYSTEM
# ============================================================================

# Global cache for model hashes (performance optimization)
_HASH_CACHE = {}


def get_model_search_paths(model_type: str) -> List[str]:
    """
    Returns search paths for a model type.

    Priority:
    1. Environment variable override (COMFYUI_CHECKPOINT_PATH, etc.)
    2. ComfyUI folder_paths API
    3. Default relative paths

    Args:
        model_type: "checkpoint", "lora", or "vae"

    Returns:
        List of absolute paths to search
    """
    paths = []

    # Check environment variable
    env_var = f"COMFYUI_{model_type.upper()}_PATH"
    if env_var in os.environ:
        paths.append(os.environ[env_var])

    # Try ComfyUI folder_paths API
    try:
        import folder_paths
        type_map = {
            "checkpoint": "checkpoints",
            "lora": "loras",
            "vae": "vae"
        }
        if model_type in type_map:
            comfy_paths = folder_paths.get_folder_paths(type_map[model_type])
            paths.extend(comfy_paths)
    except (ImportError, Exception):
        pass

    # Add default relative paths if no paths found
    if not paths:
        default_paths = {
            "checkpoint": ["models/checkpoints"],
            "lora": ["models/loras"],
            "vae": ["models/vae"],
        }
        paths.extend(default_paths.get(model_type, []))

    return paths


def find_model_file(model_name: str, model_type: str) -> Optional[Path]:
    """
    Finds a model file in search paths.

    Args:
        model_name: Name of model (with or without .safetensors)
        model_type: Type of model ("checkpoint", "lora", "vae")

    Returns:
        Path to model file or None if not found
    """
    # Ensure .safetensors extension
    if not model_name.endswith('.safetensors'):
        model_name += '.safetensors'

    # Get search paths
    search_paths = get_model_search_paths(model_type)

    # Search for file
    for base_path in search_paths:
        potential_path = Path(base_path) / model_name
        if potential_path.exists():
            return potential_path

    return None


def calculate_model_hash(model_name: str, model_type: str = "checkpoint") -> str:
    """
    Calculates SHA256 hash (first 10 chars) for a model file - AutoV2/Civitai format.

    Uses cache for performance. Fallback to "0000000000" if file not found.

    Args:
        model_name: Name of model file (with or without .safetensors)
        model_type: Type of model ("checkpoint", "lora", "vae")

    Returns:
        10-character SHA256 hash prefix or "0000000000" on failure
    """
    # Check cache first
    cache_key = f"{model_type}:{model_name}"
    if cache_key in _HASH_CACHE:
        return _HASH_CACHE[cache_key]

    try:
        # Find model file
        model_path = find_model_file(model_name, model_type)

        if not model_path:
            raise FileNotFoundError(f"Model '{model_name}' not found in search paths")

        # Calculate SHA256 hash (chunk-based for memory efficiency)
        sha256 = hashlib.sha256()
        with open(model_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)

        # Get first 10 characters (AutoV2 format)
        hash_value = sha256.hexdigest()[:10]

        # Cache the result
        _HASH_CACHE[cache_key] = hash_value

        return hash_value

    except Exception as e:
        # Silent fallback - NEVER interrupt generation
        print(f"[MetaHub] Warning: Could not calculate hash for {model_name}: {e}")
        return "0000000000"


def calculate_lora_hashes(lora_list: List[dict]) -> Dict[str, str]:
    """
    Calculates hashes for multiple LoRA models.

    Args:
        lora_list: List of dicts with 'name' and 'weight' keys

    Returns:
        Dict mapping lora_name to hash (without .safetensors extension)
    """
    hashes = {}

    for lora in lora_list:
        try:
            lora_name = lora['name']
            # Remove .safetensors for display name in A1111 format
            display_name = lora_name.replace('.safetensors', '')
            hash_value = calculate_model_hash(lora_name, model_type="lora")
            hashes[display_name] = hash_value
        except Exception as e:
            # Log warning but continue with other LoRAs
            print(f"[MetaHub] Warning: Could not calculate hash for LoRA '{lora.get('name')}': {e}")

    return hashes


# ============================================================================
# PERFORMANCE METRICS COLLECTION
# ============================================================================

def collect_gpu_metrics() -> Dict[str, Any]:
    """
    Auto-detects GPU metrics (VRAM peak, device name).

    Handles CUDA, MPS (Mac Metal), and CPU-only setups.
    Silent failures - never interrupts generation.

    Returns:
        dict with keys:
            - vram_peak_mb: Peak VRAM usage in MB (None if unavailable)
            - gpu_device: GPU device name string
            - gpu_available: Boolean indicating if GPU is available
    """
    metrics = {
        "vram_peak_mb": None,
        "gpu_device": None,
        "gpu_available": False,
    }

    try:
        import torch

        # Check CUDA availability
        if torch.cuda.is_available():
            metrics["gpu_available"] = True

            # Get GPU device name
            try:
                device_name = torch.cuda.get_device_name(0)
                metrics["gpu_device"] = device_name
            except Exception as e:
                print(f"[MetaHub] Warning: Could not get GPU name: {e}")

            # Get peak VRAM usage (convert bytes to MB)
            try:
                vram_bytes = torch.cuda.max_memory_allocated(0)
                vram_mb = vram_bytes / (1024 * 1024)
                metrics["vram_peak_mb"] = round(vram_mb, 2)

                # Reset peak memory counter for next generation
                torch.cuda.reset_peak_memory_stats(0)
            except Exception as e:
                print(f"[MetaHub] Warning: Could not get VRAM usage: {e}")

        # Check MPS (Mac Metal)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            metrics["gpu_available"] = True
            metrics["gpu_device"] = "Apple Metal Performance Shaders (MPS)"
            # MPS doesn't support memory tracking yet
            metrics["vram_peak_mb"] = None

        else:
            # CPU-only mode
            metrics["gpu_available"] = False
            metrics["gpu_device"] = "CPU (CUDA not available)"

    except ImportError:
        # torch not installed (should never happen in ComfyUI)
        print("[MetaHub] Warning: PyTorch not found, GPU metrics unavailable")
    except Exception as e:
        print(f"[MetaHub] Warning: GPU metrics collection failed: {e}")

    return metrics


def collect_version_info() -> Dict[str, Optional[str]]:
    """
    Auto-detects software versions (Python, PyTorch, ComfyUI).

    Silent failures - returns None for unavailable versions.

    Returns:
        dict with keys:
            - python_version: Python version string (e.g., "3.10.12")
            - torch_version: PyTorch version string (e.g., "2.0.1+cu118")
            - comfyui_version: ComfyUI version string (e.g., "0.1.0")
    """
    versions = {
        "comfyui_version": None,
        "torch_version": None,
        "python_version": None,
    }

    # Python version (always available)
    try:
        import sys
        versions["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    except Exception:
        pass

    # PyTorch version
    try:
        import torch
        versions["torch_version"] = torch.__version__
    except Exception:
        pass

    # ComfyUI version (multiple detection methods)
    try:
        # Method 1: Try importing comfy module
        try:
            import comfy
            if hasattr(comfy, '__version__'):
                versions["comfyui_version"] = comfy.__version__
        except (ImportError, AttributeError):
            pass

        # Method 2: Check for version file
        if not versions["comfyui_version"]:
            version_file = Path(__file__).parent.parent.parent / "comfy_version.txt"
            if version_file.exists():
                versions["comfyui_version"] = version_file.read_text().strip()
    except Exception:
        pass

    return versions


# ============================================================================
# METADATA FORMATTING
# ============================================================================

def build_a1111_metadata(params: dict) -> str:
    """
    Builds A1111/Civitai compatible metadata string for tEXt chunk.

    Format:
    {positive_prompt}
    Negative prompt: {negative_prompt}
    Steps: {steps}, Sampler: {sampler}, CFG scale: {cfg}, Seed: {seed},
    Size: {width}x{height}, Model: {model_name}, Model hash: {hash},
    Lora hashes: "lora1: hash1, lora2: hash2"

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

    # Build config line
    config_parts = [
        f"Steps: {params['steps']}",
        f"Sampler: {params['sampler']}",
        f"CFG scale: {params['cfg']}",
        f"Seed: {params['seed']}",
        f"Size: {params['width']}x{params['height']}",
        f"Model: {params['model_name']}",
        f"Model hash: {params['model_hash']}",
    ]

    # Add LoRA hashes if present
    if lora_hashes_str:
        config_parts.append(f'Lora hashes: "{lora_hashes_str}"')

    config_line = ", ".join(config_parts)

    # Build final metadata string
    metadata = f"""{params['positive']}
Negative prompt: {params['negative']}
{config_line}"""

    return metadata


def build_imh_metadata(params: dict, workflow_json: dict) -> dict:
    """
    Builds Image MetaHub iTXt metadata as JSON object.
    100% compatible with comfyUIParser.ts

    CRITICAL Fields:
    - "generator": "ComfyUI" (required for IMH detection)
    - "workflow": workflow JSON from extra_pnginfo
    - "prompt_api": prompt JSON from extra_pnginfo
    - Main fields: prompt, negativePrompt, seed, steps, cfg, model, sampler_name, etc.
    - LoRAs in array format: [{"name": "...", "weight": 0.8}]

    Args:
        params: Dict containing generation parameters and IMH fields
        workflow_json: Complete ComfyUI workflow from extra_pnginfo

    Returns:
        JSON-serializable dict for IMH metadata
    """
    return {
        # CRITICAL: Required field for IMH detection
        "generator": "ComfyUI",

        # Main fields (compatible with comfyUIParser.ts)
        "prompt": params['positive'],
        "negativePrompt": params['negative'],
        "seed": params['seed'],
        "steps": params['steps'],
        "cfg": params['cfg'],
        "sampler_name": params['sampler'],
        "scheduler": params['scheduler'],
        "model": params['model_name'],
        "model_hash": params['model_hash'],
        "vae": params.get('vae_name', ''),
        "denoise": params.get('denoise', 1.0),
        "width": params['width'],
        "height": params['height'],

        # LoRAs in the format expected by parser: array of {name, weight}
        "loras": [
            {
                "name": lora['name'],
                "weight": lora['weight']
            }
            for lora in params.get('lora_list', [])
        ],

        # IMH Pro fields (custom extension)
        "imh_pro": {
            "user_tags": params.get('user_tags', ''),
            "notes": params.get('notes', ''),
            "project_name": params.get('project_name', ''),
        },

        # Analytics (custom extension) - Performance/Benchmark metrics
        "analytics": {
            # Tier 1: CRITICAL metrics
            "vram_peak_mb": params.get('vram_peak_mb'),
            "gpu_device": params.get('gpu_device'),
            "generation_time_ms": params.get('generation_time_ms'),

            # Tier 2: VERY USEFUL metrics
            "steps_per_second": params.get('steps_per_second'),
            "comfyui_version": params.get('comfyui_version'),

            # Tier 3: NICE-TO-HAVE metrics
            "torch_version": params.get('torch_version'),
            "python_version": params.get('python_version'),

            # Legacy field (kept for backward compatibility)
            "generation_time": params.get('generation_time', 0.0),
        },

        # Complete workflow (used by parser for re-parsing if needed)
        "workflow": workflow_json.get('workflow', {}),
        "prompt_api": workflow_json.get('prompt', {}),
    }


def _serialize_metadata_json(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return None


def _get_workflow_prompt_texts(imh_metadata: dict) -> Tuple[Optional[str], Optional[str]]:
    if not imh_metadata:
        return None, None
    workflow_text = _serialize_metadata_json(imh_metadata.get("workflow"))
    prompt_text = _serialize_metadata_json(
        imh_metadata.get("prompt_api") or imh_metadata.get("prompt")
    )
    return workflow_text, prompt_text


def build_comfyui_xmp_packet(imh_metadata: dict) -> Optional[bytes]:
    workflow_text, prompt_text = _get_workflow_prompt_texts(imh_metadata)
    if not workflow_text and not prompt_text:
        return None

    fields = []
    if workflow_text:
        fields.append(f"<comfyui:workflow>{xml_escape(workflow_text)}</comfyui:workflow>")
    if prompt_text:
        fields.append(f"<comfyui:prompt>{xml_escape(prompt_text)}</comfyui:prompt>")

    xmp_payload = (
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:comfyui="https://comfyui.org/ns/1.0/">'
        f"{''.join(fields)}"
        '</rdf:Description>'
        '</rdf:RDF>'
        '</x:xmpmeta>'
        '<?xpacket end="w"?>'
    )
    return xmp_payload.encode("utf-8")


# ============================================================================
# WORKFLOW PARSING
# ============================================================================

def extract_loras_from_workflow(workflow_json: dict) -> List[dict]:
    """
    Auto-detects and extracts LoRA loaders from ComfyUI workflow JSON.

    Searches for nodes with class_type containing "lora".
    Supports both UI format (nodes array) and API format (prompt dict).

    Args:
        workflow_json: Workflow dict from extra_pnginfo

    Returns:
        List of dicts with 'name' and 'weight' keys:
        [{"name": "detail_tweaker.safetensors", "weight": 0.8}, ...]
    """
    loras = []

    if not workflow_json:
        return loras

    try:
        # Try UI format (nodes array with widgets_values)
        workflow = workflow_json.get('workflow', {})
        for node in workflow.get('nodes', []):
            class_type = node.get('class_type', '').lower()
            if 'lora' in class_type:
                values = node.get('widgets_values', [])
                if len(values) >= 2:
                    loras.append({
                        "name": values[0],
                        "weight": float(values[1])
                    })

        # Try API format (prompt dict with inputs)
        prompt = workflow_json.get('prompt', {})
        for node_data in prompt.values():
            class_type = node_data.get('class_type', '').lower()
            if 'lora' in class_type:
                inputs = node_data.get('inputs', {})
                lora_name = inputs.get('lora_name', '')
                if lora_name:
                    weight = inputs.get('strength_model')
                    if weight is None:
                        weight = inputs.get('strength', inputs.get('strength_clip', 1.0))
                    loras.append({
                        "name": lora_name,
                        "weight": float(weight)
                    })

    except Exception as e:
        print(f"[MetaHub] Warning: Could not extract LoRAs from workflow: {e}")

    return loras


def get_workflow_json(extra_pnginfo: Optional[dict]) -> dict:
    """
    Extracts workflow JSON from ComfyUI's extra_pnginfo parameter.

    Args:
        extra_pnginfo: Hidden parameter passed by ComfyUI

    Returns:
        Complete workflow JSON dict or empty dict if not available
    """
    if not extra_pnginfo:
        return {}

    # extra_pnginfo can be a dict or list
    if isinstance(extra_pnginfo, list) and len(extra_pnginfo) > 0:
        return extra_pnginfo[0]
    elif isinstance(extra_pnginfo, dict):
        return extra_pnginfo

    return {}


def ensure_prompt_in_workflow(workflow_json: dict, prompt_data: Optional[dict]) -> dict:
    """
    Ensures workflow JSON includes prompt data when extra_pnginfo is missing it.
    """
    if not isinstance(workflow_json, dict):
        workflow_json = {}

    if isinstance(prompt_data, dict) and prompt_data:
        existing_prompt = workflow_json.get("prompt")
        if not isinstance(existing_prompt, dict) or not existing_prompt:
            workflow_json["prompt"] = prompt_data

    return workflow_json


def ensure_metahub_save_node(workflow_json: dict, save_node_id: Optional[str]) -> None:
    """
    Ensures saved workflow keeps MetaHubSaveNode instead of SaveImage.
    """
    if not isinstance(workflow_json, dict):
        return

    target_id = str(save_node_id) if save_node_id is not None else None

    prompt = workflow_json.get("prompt")
    if isinstance(prompt, dict):
        if target_id and target_id in prompt:
            node = prompt.get(target_id)
            if isinstance(node, dict):
                node["class_type"] = "MetaHubSaveNode"
        else:
            save_nodes = [
                node_id
                for node_id, node in prompt.items()
                if isinstance(node, dict) and node.get("class_type") == "SaveImage"
            ]
            if len(save_nodes) == 1:
                prompt[save_nodes[0]]["class_type"] = "MetaHubSaveNode"

    workflow = workflow_json.get("workflow")
    if not isinstance(workflow, dict):
        return
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return

    def update_workflow_node(node: dict) -> None:
        if "type" in node:
            node["type"] = "MetaHubSaveNode"
        if "class_type" in node:
            node["class_type"] = "MetaHubSaveNode"
        if node.get("title") in ("Save Image", "SaveImage"):
            node["title"] = "MetaHub Save Image"
        props = node.get("properties")
        if isinstance(props, dict) and props.get("node_name") in ("SaveImage", "Save Image"):
            props["node_name"] = "MetaHub Save Image"

    if target_id:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            if node_id is not None and str(node_id) == target_id:
                update_workflow_node(node)
                return
    else:
        save_nodes = [
            node
            for node in nodes
            if isinstance(node, dict)
            and (node.get("type") == "SaveImage" or node.get("class_type") == "SaveImage")
        ]
        if len(save_nodes) == 1:
            update_workflow_node(save_nodes[0])


# ============================================================================
# PNG CHUNK INJECTION
# ============================================================================

def inject_metadata_chunks(image_path: str, a1111_metadata: str, imh_metadata: dict):
    """
    Injects metadata chunks into PNG file without re-encoding.

    Injects two chunks:
    - tEXt chunk "parameters" (A1111/Civitai compatible)
    - iTXt chunk "imagemetahub_data" (IMH format, UTF-8 support)

    Uses Pillow's PngInfo API for efficient header-only modification.
    Compression level 4 matches ComfyUI default.

    Args:
        image_path: Path to PNG file
        a1111_metadata: A1111 formatted metadata string
        imh_metadata: IMH metadata dict (will be JSON serialized)
    """
    try:
        img = Image.open(image_path)
        save_png_with_metadata(img, image_path, a1111_metadata, imh_metadata)

    except Exception as e:
        # Log error but don't raise - image is already saved
        print(f"[MetaHub] Warning: Could not inject metadata into {image_path}: {e}")


def save_png_with_metadata(image: Image.Image, image_path: str, a1111_metadata: str, imh_metadata: dict) -> None:
    """
    Saves PNG image with A1111 tEXt and IMH iTXt metadata.
    """
    try:
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("parameters", a1111_metadata or "")
        imh_json = json.dumps(imh_metadata or {}, ensure_ascii=False)
        png_info.add_itxt("imagemetahub_data", imh_json)
        workflow_text, prompt_text = _get_workflow_prompt_texts(imh_metadata)
        if workflow_text:
            png_info.add_itxt("workflow", workflow_text)
        if prompt_text:
            png_info.add_itxt("prompt", prompt_text)
        image.save(image_path, "PNG", pnginfo=png_info, compress_level=4)
    except Exception as e:
        print(f"[MetaHub] Warning: PNG metadata save failed for {image_path}: {e}")
        image.save(image_path, "PNG", compress_level=4)


def _build_exif_bytes(a1111_metadata: str, imh_metadata: dict) -> Optional[bytes]:
    if piexif is None or piexif_helper is None:
        return None
    try:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}
        imh_json = json.dumps(imh_metadata or {}, ensure_ascii=False)
        if imh_json:
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = imh_json.encode("utf-8", errors="replace")
        if a1111_metadata:
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif_helper.UserComment.dump(
                a1111_metadata, encoding="unicode"
            )
        return piexif.dump(exif_dict)
    except Exception:
        return None


def save_jpeg_with_metadata(
    image: Image.Image,
    image_path: str,
    quality: int,
    a1111_metadata: str,
    imh_metadata: dict,
) -> None:
    """
    Saves JPEG image with EXIF metadata (UserComment + ImageDescription).
    """
    try:
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        save_kwargs: Dict[str, Any] = {"format": "JPEG", "quality": quality}
        xmp_bytes = build_comfyui_xmp_packet(imh_metadata)
        exif_bytes = _build_exif_bytes(a1111_metadata, imh_metadata)
        if exif_bytes:
            save_kwargs["exif"] = exif_bytes
        elif a1111_metadata:
            save_kwargs["comment"] = a1111_metadata.encode("utf-8", errors="replace")
        if xmp_bytes:
            save_kwargs["xmp"] = xmp_bytes
        try:
            image.save(image_path, **save_kwargs)
        except TypeError:
            if "xmp" in save_kwargs:
                save_kwargs.pop("xmp", None)
                image.save(image_path, **save_kwargs)
            else:
                raise
    except Exception as e:
        print(f"[MetaHub] Warning: JPEG metadata save failed for {image_path}: {e}")
        image.save(image_path, "JPEG", quality=quality)


def save_webp_with_metadata(
    image: Image.Image,
    image_path: str,
    quality: int,
    a1111_metadata: str,
    imh_metadata: dict,
) -> None:
    """
    Saves WebP image with EXIF metadata.
    """
    try:
        save_kwargs: Dict[str, Any] = {"format": "WEBP", "quality": quality}
        xmp_bytes = build_comfyui_xmp_packet(imh_metadata)
        exif_bytes = _build_exif_bytes(a1111_metadata, imh_metadata)
        if exif_bytes:
            save_kwargs["exif"] = exif_bytes
        elif a1111_metadata:
            save_kwargs["comment"] = a1111_metadata
        if xmp_bytes:
            save_kwargs["xmp"] = xmp_bytes
        try:
            image.save(image_path, **save_kwargs)
        except TypeError:
            if "xmp" in save_kwargs:
                save_kwargs.pop("xmp", None)
                image.save(image_path, **save_kwargs)
            else:
                raise
    except Exception as e:
        print(f"[MetaHub] Warning: WebP metadata save failed for {image_path}: {e}")
        image.save(image_path, "WEBP", quality=quality)


# ============================================================================
# FILE MANAGEMENT
# ============================================================================

_INVALID_FILENAME_CHARS = '<>:"/\\|?*'
_KNOWN_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def _normalize_file_format(file_format: str) -> str:
    fmt = (file_format or "PNG").strip().upper()
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt not in ("PNG", "JPEG", "WEBP"):
        fmt = "PNG"
    return fmt


def _get_extension(file_format: str) -> str:
    fmt = _normalize_file_format(file_format)
    if fmt == "JPEG":
        return ".jpg"
    if fmt == "WEBP":
        return ".webp"
    return ".png"


def _strip_known_extension(name: str) -> str:
    lower = name.lower()
    for ext in _KNOWN_EXTENSIONS:
        if lower.endswith(ext):
            return name[: -len(ext)]
    return name


def sanitize_filename(name: str) -> str:
    if not name:
        return "unknown"
    cleaned = str(name)
    for ch in _INVALID_FILENAME_CHARS:
        cleaned = cleaned.replace(ch, "_")
    cleaned = cleaned.strip().strip(".")
    return cleaned or "unknown"


def _format_placeholder_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, str) and not value.strip():
        return "unknown"
    return str(value)


def resolve_placeholders(pattern: str, params: dict, counter: int) -> str:
    now = datetime.now()
    model_name = params.get("model_name") or ""
    model_base = Path(model_name).stem if model_name else ""

    replacements = {
        "%counter%": f"{counter:05d}",
        "%seed%": _format_placeholder_value(params.get("seed")),
        "%date%": now.strftime("%Y-%m-%d"),
        "%time%": now.strftime("%H-%M-%S"),
        "%datetime%": now.strftime("%Y-%m-%d_%H-%M-%S"),
        "%model%": _format_placeholder_value(model_base),
        "%sampler%": _format_placeholder_value(params.get("sampler")),
        "%steps%": _format_placeholder_value(params.get("steps")),
        "%cfg%": _format_placeholder_value(params.get("cfg")),
        "%width%": _format_placeholder_value(params.get("width")),
        "%height%": _format_placeholder_value(params.get("height")),
    }

    resolved = pattern or "ComfyUI_%counter%"
    for placeholder, value in replacements.items():
        resolved = resolved.replace(placeholder, value)
    return resolved


def get_output_directory(output_path: str) -> Path:
    """
    Determines output directory with automatic creation (mkdir -p).
    Fallback to ComfyUI default if output_path is empty.

    Args:
        output_path: Custom output path from user input

    Returns:
        Path object for output directory (created if doesn't exist)
    """
    if output_path and output_path.strip():
        output_dir = Path(output_path)
    else:
        # Use ComfyUI's default output directory
        try:
            import folder_paths
            output_dir = Path(folder_paths.get_output_directory())
        except ImportError:
            # Fallback if not in ComfyUI environment
            output_dir = Path("output")

    # Create directory if it doesn't exist (mkdir -p)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        # This is a critical error - cannot save without directory
        raise RuntimeError(f"Cannot create output directory '{output_dir}': {e}")

    return output_dir


def get_next_filename(output_dir: Path, filename_pattern: str, params: dict, file_format: str) -> Path:
    """
    Generates next available filename with auto-increment counter.

    Format: resolved pattern + extension
    Example: ComfyUI_00001.png, ComfyUI_00002.png

    Args:
        output_dir: Directory to save file
        filename_pattern: Filename pattern with placeholders
        params: Dict with placeholder values
        file_format: PNG, JPEG, or WEBP

    Returns:
        Full path to next available filename
    """
    counter = 1
    uses_counter = "%counter%" in (filename_pattern or "")
    extension = _get_extension(file_format)
    while True:
        resolved = resolve_placeholders(filename_pattern, params, counter)
        if not uses_counter and counter > 1:
            resolved = f"{resolved}_{counter:05d}"
        filename = sanitize_filename(_strip_known_extension(resolved)) + extension
        full_path = output_dir / filename
        if not full_path.exists():
            return full_path
        counter += 1


def tensor_to_pil(image_tensor: np.ndarray) -> Image.Image:
    """
    Converts ComfyUI image tensor to PIL Image.

    ComfyUI format: [H, W, C] with values in [0, 1] range

    Args:
        image_tensor: NumPy array from ComfyUI IMAGE type

    Returns:
        PIL Image in RGB mode
    """
    # Accept torch tensors or numpy arrays without hard dependency on torch.
    if hasattr(image_tensor, "detach"):
        image_tensor = image_tensor.detach()
        if hasattr(image_tensor, "cpu"):
            image_tensor = image_tensor.cpu()
        image_tensor = image_tensor.numpy()
    elif not isinstance(image_tensor, np.ndarray):
        image_tensor = np.array(image_tensor)

    # Convert from [0, 1] to [0, 255]
    img_array = np.clip(image_tensor * 255, 0, 255).astype(np.uint8)

    # Create PIL Image
    if img_array.ndim == 2:
        mode = "L"
    elif img_array.shape[-1] == 1:
        img_array = img_array[:, :, 0]
        mode = "L"
    elif img_array.shape[-1] == 4:
        mode = "RGBA"
    else:
        mode = "RGB"
    pil_image = Image.fromarray(img_array, mode=mode)

    return pil_image


def save_image_with_metadata(
    image: Image.Image,
    file_path: Path,
    file_format: str,
    quality: int,
    a1111_metadata: str,
    imh_metadata: dict,
) -> None:
    fmt = _normalize_file_format(file_format)
    if fmt == "PNG":
        save_png_with_metadata(image, str(file_path), a1111_metadata, imh_metadata)
    elif fmt == "JPEG":
        save_jpeg_with_metadata(image, str(file_path), quality, a1111_metadata, imh_metadata)
    elif fmt == "WEBP":
        save_webp_with_metadata(image, str(file_path), quality, a1111_metadata, imh_metadata)
    else:
        save_png_with_metadata(image, str(file_path), a1111_metadata, imh_metadata)


def save_image_batch(
    images: np.ndarray,
    output_dir: Path,
    filename_pattern: str,
    params: dict,
    file_format: str = "PNG",
    quality: int = 95,
    a1111_metadata: str = "",
    imh_metadata: Optional[dict] = None,
) -> List[Path]:
    """
    Saves batch of images with auto-increment filenames and metadata.

    Args:
        images: Batch of images from ComfyUI (shape: [B, H, W, C])
        output_dir: Output directory
        filename_pattern: Filename pattern with placeholders
        params: Dict with placeholder values
        file_format: PNG, JPEG, or WEBP
        quality: JPEG/WebP quality (1-100)
        a1111_metadata: A1111 metadata string
        imh_metadata: IMH metadata dict

    Returns:
        List of saved file paths
    """
    saved_paths = []

    for image_tensor in images:
        # Convert to PIL
        pil_image = tensor_to_pil(image_tensor)

        # Get next filename
        file_path = get_next_filename(output_dir, filename_pattern, params, file_format)

        # Save image with metadata
        try:
            save_image_with_metadata(pil_image, file_path, file_format, quality, a1111_metadata, imh_metadata or {})
            saved_paths.append(file_path)
        except Exception as e:
            print(f"[MetaHub] Warning: Could not save image {file_path}: {e}")
            try:
                fmt = _normalize_file_format(file_format)
                if fmt == "JPEG":
                    if pil_image.mode in ("RGBA", "LA", "P"):
                        pil_image = pil_image.convert("RGB")
                    pil_image.save(file_path, "JPEG", quality=quality)
                elif fmt == "WEBP":
                    pil_image.save(file_path, "WEBP", quality=quality)
                else:
                    pil_image.save(file_path, "PNG", compress_level=4)
                saved_paths.append(file_path)
            except Exception as save_error:
                print(f"[MetaHub] Warning: Fallback save failed for {file_path}: {save_error}")

    return saved_paths


def build_ui_preview(saved_paths: List[Path], output_base: Path) -> dict:
    """
    Builds UI preview structure for ComfyUI frontend.

    ComfyUI frontend expects relative paths from output directory with format:
    {
        "ui": {
            "images": [
                {"filename": "file.png", "subfolder": "subdir", "type": "output"}
            ]
        }
    }

    Args:
        saved_paths: List of saved file paths
        output_base: Base output directory (ComfyUI default output/)

    Returns:
        Dict with {"ui": {"images": [...]}} structure for preview
    """
    ui_images = []

    for file_path in saved_paths:
        try:
            # Calculate relative path from output directory
            relative_path = file_path.relative_to(output_base)

            # Split into subfolder + filename
            parts = relative_path.parts
            if len(parts) > 1:
                # Has subfolder(s): output/project/img.png -> subfolder="project"
                subfolder = str(Path(*parts[:-1]))
                filename = parts[-1]
            else:
                # No subfolder: output/img.png -> subfolder=""
                subfolder = ""
                filename = str(relative_path.name)

            ui_images.append({
                "filename": filename,
                "subfolder": subfolder,
                "type": "output"
            })

        except ValueError:
            # Path is outside output_base (e.g., absolute path like D:/MyImages)
            # Preview won't work - frontend can only serve from output/ or temp/
            print(f"[MetaHub] Warning: Preview unavailable for {file_path.name} (outside output directory)")

    return {
        "ui": {
            "images": ui_images
        }
    }
