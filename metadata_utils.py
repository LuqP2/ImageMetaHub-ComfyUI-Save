"""
Metadata Utils for MetaHub Save Node
Handles hash calculation, metadata formatting, and PNG chunk injection.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from PIL import Image, PngImagePlugin


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

        # Analytics (custom extension)
        "analytics": {
            "generation_time": params.get('generation_time', 0.0),
        },

        # Complete workflow (used by parser for re-parsing if needed)
        "workflow": workflow_json.get('workflow', {}),
        "prompt_api": workflow_json.get('prompt', {}),
    }


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
                    weight = inputs.get('strength_model', inputs.get('strength', 1.0))
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
        # Load existing image
        img = Image.open(image_path)

        # Create PngInfo object
        png_info = PngImagePlugin.PngInfo()

        # Add tEXt chunk for A1111 compatibility
        png_info.add_text("parameters", a1111_metadata)

        # Add iTXt chunk for IMH data (supports UTF-8)
        imh_json = json.dumps(imh_metadata, ensure_ascii=False)
        png_info.add_itxt("imagemetahub_data", imh_json)

        # Save with metadata (Pillow optimizes to avoid full re-encode)
        img.save(image_path, "PNG", pnginfo=png_info, compress_level=4)

    except Exception as e:
        # Log error but don't raise - image is already saved
        print(f"[MetaHub] Warning: Could not inject metadata into {image_path}: {e}")


# ============================================================================
# FILE MANAGEMENT
# ============================================================================

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


def get_next_filename(output_dir: Path, prefix: str) -> Path:
    """
    Generates next available filename with auto-increment counter.

    Format: {prefix}_{counter:05d}.png
    Example: ComfyUI_00001.png, ComfyUI_00002.png

    Args:
        output_dir: Directory to save file
        prefix: Filename prefix

    Returns:
        Full path to next available filename
    """
    counter = 1
    while True:
        filename = f"{prefix}_{counter:05d}.png"
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


def save_image_batch(images: np.ndarray, output_dir: Path, prefix: str) -> List[Path]:
    """
    Saves batch of images with auto-increment filenames.

    Args:
        images: Batch of images from ComfyUI (shape: [B, H, W, C])
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        List of saved file paths
    """
    saved_paths = []

    for image_tensor in images:
        # Convert to PIL
        pil_image = tensor_to_pil(image_tensor)

        # Get next filename
        file_path = get_next_filename(output_dir, prefix)

        # Save image (metadata will be added later)
        pil_image.save(file_path, "PNG", compress_level=4)

        saved_paths.append(file_path)

    return saved_paths
