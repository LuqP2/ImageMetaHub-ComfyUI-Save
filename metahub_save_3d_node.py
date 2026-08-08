"""Image MetaHub 3D save node for ComfyUI.

The node deliberately uses the legacy node registration surface so the package
continues to load on older ComfyUI releases.  The wildcard input accepts both
the current ``MESH`` value and the newer File3D wrappers.
"""

from __future__ import annotations

import json
import struct
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import folder_paths
except ImportError:  # pragma: no cover - exercised only outside ComfyUI
    folder_paths = None

try:
    from comfy.cli_args import args
except ImportError:  # pragma: no cover - test/legacy fallback
    class _Args:
        disable_metadata = False

    args = _Args()

try:
    from . import metadata_utils as utils
    from .workflow_extractor import WorkflowExtractor
except ImportError:
    import metadata_utils as utils
    from workflow_extractor import WorkflowExtractor


class AnyType(str):
    """ComfyUI V1 wildcard socket compatible with all upstream 3D types."""

    def __ne__(self, _value: object) -> bool:
        return False


ANY_3D = AnyType("*")
SUPPORTED_FORMATS = {"glb", "gltf", "obj", "fbx", "stl"}


def _tensor_numpy(value: Any, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _mesh_batch_item(mesh: Any, index: int) -> Tuple[Any, Any, Any, Any, Any]:
    vertex_count = None
    face_count = None
    if getattr(mesh, "vertex_counts", None) is not None:
        vertex_count = int(mesh.vertex_counts[index].item())
        face_count = int(mesh.face_counts[index].item())

    vertices = mesh.vertices[index]
    faces = mesh.faces[index]
    colors = getattr(mesh, "vertex_colors", None)
    uvs = getattr(mesh, "uvs", None)
    texture = getattr(mesh, "texture", None)

    if vertex_count is not None:
        vertices = vertices[:vertex_count]
        faces = faces[:face_count]
        if colors is not None:
            colors = colors[index, :vertex_count]
        if uvs is not None:
            uvs = uvs[index, :vertex_count]
    else:
        colors = colors[index] if colors is not None else None
        uvs = uvs[index] if uvs is not None else None

    texture = texture[index] if texture is not None else None
    return vertices, faces, colors, uvs, texture


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def _write_glb(
    output_path: Path,
    vertices: Any,
    faces: Any,
    metadata: Optional[Dict[str, Any]],
    *,
    vertex_colors: Any = None,
    uvs: Any = None,
    texture: Any = None,
    unlit: bool = False,
) -> Dict[str, Any]:
    positions = _tensor_numpy(vertices, np.float32).reshape(-1, 3)
    indices_signed = _tensor_numpy(faces, np.int64).reshape(-1, 3)
    if positions.size == 0 or indices_signed.size == 0:
        raise ValueError("Cannot save an empty 3D mesh")
    if indices_signed.min() < 0 or indices_signed.max() >= len(positions):
        raise ValueError("Mesh contains an out-of-range face index")

    indices = indices_signed.astype(np.uint32, copy=False).reshape(-1)
    uv_array = _tensor_numpy(uvs, np.float32).reshape(-1, 2) if uvs is not None else None
    color_array = _tensor_numpy(vertex_colors, np.float32) if vertex_colors is not None else None
    if color_array is not None:
        color_array = np.clip(color_array.reshape(len(positions), -1), 0.0, 1.0)
        if color_array.shape[1] not in (3, 4):
            color_array = None
    if uv_array is not None and len(uv_array) != len(positions):
        uv_array = None

    texture_bytes = b""
    if texture is not None:
        texture_array = _tensor_numpy(texture, np.float32)
        texture_array = np.clip(texture_array * 255.0, 0, 255).astype(np.uint8)
        if texture_array.ndim == 3 and texture_array.shape[-1] in (3, 4):
            buffer = BytesIO()
            Image.fromarray(texture_array, mode="RGBA" if texture_array.shape[-1] == 4 else "RGB").save(buffer, "PNG")
            texture_bytes = buffer.getvalue()

    chunks: list[bytes] = []
    buffer_views: list[dict] = []
    accessors: list[dict] = []

    def append_buffer(data: bytes, target: Optional[int] = None) -> int:
        offset = sum(len(chunk) for chunk in chunks)
        padded = _pad4(data)
        chunks.append(padded)
        view: dict = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return len(buffer_views) - 1

    position_view = append_buffer(positions.tobytes(), 34962)
    accessors.append({
        "bufferView": position_view,
        "componentType": 5126,
        "count": len(positions),
        "type": "VEC3",
        "min": positions.min(axis=0).tolist(),
        "max": positions.max(axis=0).tolist(),
    })
    index_view = append_buffer(indices.tobytes(), 34963)
    accessors.append({
        "bufferView": index_view,
        "componentType": 5125,
        "count": len(indices),
        "type": "SCALAR",
    })
    attributes: dict = {"POSITION": 0}

    if uv_array is not None:
        uv_view = append_buffer(uv_array.tobytes(), 34962)
        accessors.append({"bufferView": uv_view, "componentType": 5126, "count": len(uv_array), "type": "VEC2"})
        attributes["TEXCOORD_0"] = len(accessors) - 1
    if color_array is not None:
        color_view = append_buffer(color_array.astype(np.float32).tobytes(), 34962)
        accessors.append({
            "bufferView": color_view,
            "componentType": 5126,
            "count": len(color_array),
            "type": "VEC4" if color_array.shape[1] == 4 else "VEC3",
        })
        attributes["COLOR_0"] = len(accessors) - 1

    primitive: dict = {"attributes": attributes, "indices": 1, "mode": 4}
    gltf: dict = {
        "asset": {"version": "2.0", "generator": "Image MetaHub ComfyUI Save"},
        "buffers": [],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "meshes": [{"primitives": [primitive]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    if metadata:
        gltf["asset"]["extras"] = {"imagemetahub_data": metadata}

    materials: list[dict] = []
    has_embedded_texture = bool(texture_bytes and uv_array is not None)
    if has_embedded_texture:
        texture_view = append_buffer(texture_bytes)
        gltf["images"] = [{"bufferView": texture_view, "mimeType": "image/png"}]
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}]
        gltf["textures"] = [{"source": 0, "sampler": 0}]
        materials.append({
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0, "texCoord": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "doubleSided": True,
        })
        primitive["material"] = 0
    elif unlit:
        materials.append({
            "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 1.0},
            "extensions": {"KHR_materials_unlit": {}},
            "doubleSided": True,
        })
        gltf["extensionsUsed"] = ["KHR_materials_unlit"]
        primitive["material"] = 0
    if materials:
        gltf["materials"] = materials

    binary = b"".join(chunks)
    gltf["buffers"] = [{"byteLength": len(binary)}]
    json_bytes = _pad4(json.dumps(gltf, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), b" ")
    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total_length))
        handle.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        handle.write(json_bytes)
        handle.write(struct.pack("<II", len(binary), 0x004E4942))
        handle.write(binary)

    return {
        "format": "glb",
        "vertexCount": int(len(positions)),
        "faceCount": int(len(indices) // 3),
        "materialCount": len(materials),
        "hasTextures": has_embedded_texture,
        "bounds": {"min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
    }


def _inject_glb_metadata(file_path: Path, metadata: Dict[str, Any]) -> bool:
    try:
        data = file_path.read_bytes()
        if len(data) < 20 or data[:4] != b"glTF":
            return False
        _magic, version, _total = struct.unpack_from("<4sII", data, 0)
        if version != 2:
            return False
        json_length, json_type = struct.unpack_from("<II", data, 12)
        if json_type != 0x4E4F534A or 20 + json_length > len(data):
            return False
        document = json.loads(data[20:20 + json_length].rstrip(b" \x00").decode("utf-8"))
        asset = document.setdefault("asset", {"version": "2.0"})
        extras = asset.setdefault("extras", {})
        if not isinstance(extras, dict):
            extras = {}
            asset["extras"] = extras
        extras["imagemetahub_data"] = metadata
        new_json = _pad4(json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), b" ")
        remaining = data[20 + json_length:]
        rebuilt = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(new_json) + len(remaining))
        rebuilt += struct.pack("<II", len(new_json), 0x4E4F534A) + new_json + remaining
        file_path.write_bytes(rebuilt)
        return True
    except Exception as error:
        print(f"[ImageMetaHub-Save] Warning: could not embed GLB metadata: {error}")
        return False


def _next_path(filename_prefix: str, extension: str) -> Tuple[Path, str]:
    if folder_paths is None:
        raise RuntimeError("ComfyUI folder_paths is unavailable")
    output_root = folder_paths.get_output_directory()
    full_folder, filename, counter, subfolder, _prefix = folder_paths.get_save_image_path(filename_prefix, output_root)
    candidate = Path(full_folder) / f"{filename}_{counter:05}_.{extension}"
    while candidate.exists():
        counter += 1
        candidate = Path(full_folder) / f"{filename}_{counter:05}_.{extension}"
    return candidate, subfolder


def _build_metadata(
    prompt: Any,
    extra_pnginfo: Any,
    unique_id: Any,
    tags: str,
    notes: str,
    project_name: str,
    generation_time_override: Optional[float],
) -> Dict[str, Any]:
    workflow_json = utils.get_workflow_json(extra_pnginfo)
    prompt_data = prompt if isinstance(prompt, dict) else workflow_json.get("prompt", {})
    if not isinstance(prompt_data, dict):
        prompt_data = {}
    workflow_json = utils.ensure_prompt_in_workflow(workflow_json, prompt_data)
    save_node_id = str(unique_id) if unique_id is not None else None
    utils.ensure_metahub_save_node(
        workflow_json,
        save_node_id,
        class_type="MetaHubSave3DModel",
        display_name="MetaHub Save 3D Model",
    )
    extracted, _missing = WorkflowExtractor(prompt_data).extract(save_node_id=save_node_id)
    loras = extracted.get("lora_list") or utils.extract_loras_from_workflow(workflow_json)
    model_name = extracted.get("model_name") or ""
    steps = int(extracted.get("steps") or 0)
    elapsed = time.time() - generation_time_override if generation_time_override and generation_time_override > 0 else 0.0
    generation_time_ms = int(elapsed * 1000) if elapsed > 0 else None
    gpu = utils.collect_gpu_metrics()
    versions = utils.collect_version_info()
    fields = ["seed", "steps", "cfg", "sampler_name", "scheduler", "model_name", "positive", "negative", "denoise", "vae_name"]
    sources = utils.build_metadata_sources({}, extracted, fields)
    params = {
        "positive": extracted.get("positive") or "",
        "negative": extracted.get("negative") or "",
        "seed": int(extracted.get("seed") or 0),
        "steps": steps,
        "cfg": float(extracted.get("cfg") or 0.0),
        "sampler": extracted.get("sampler_name") or "",
        "scheduler": extracted.get("scheduler") or "",
        "model_name": model_name,
        "model_hash": utils.calculate_model_hash(model_name, model_type="checkpoint") if model_name else "",
        "vae_name": extracted.get("vae_name") or "",
        "denoise": float(extracted.get("denoise") or 1.0),
        "width": 0,
        "height": 0,
        "lora_list": loras,
        "user_tags": tags,
        "notes": notes,
        "project_name": project_name,
        "generation_time": elapsed,
        "generation_time_ms": generation_time_ms,
        "steps_per_second": round(steps / elapsed, 2) if steps > 0 and elapsed > 0 else None,
        "vram_peak_mb": gpu.get("vram_peak_mb"),
        "gpu_device": gpu.get("gpu_device"),
        "comfyui_version": versions.get("comfyui_version"),
        "torch_version": versions.get("torch_version"),
        "python_version": versions.get("python_version"),
        "metadata_status": utils.build_metadata_status(sources),
        "metadata_sources": sources,
        "imh_attribution": utils.extract_workflow_attribution(workflow_json, save_node_id),
        "source_image": extracted.get("source_image"),
        "generation_type": extracted.get("generation_type"),
    }
    result = utils.build_imh_metadata(params, workflow_json)
    result["schema_version"] = 1
    result["media_type"] = "model3d"
    return result


def _write_sidecar(model_path: Path, metadata: Dict[str, Any]) -> Path:
    sidecar_path = Path(str(model_path) + ".imagemetahub.json")
    sidecar_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return sidecar_path


def _source_node_class(prompt: Any, unique_id: Any) -> Optional[str]:
    if not isinstance(prompt, dict) or unique_id is None:
        return None
    save_node = prompt.get(str(unique_id)) or prompt.get(unique_id)
    if not isinstance(save_node, dict):
        return None
    model_input = (save_node.get("inputs") or {}).get("model_3d")
    if not isinstance(model_input, (list, tuple)) or not model_input:
        return None
    upstream = prompt.get(str(model_input[0])) or prompt.get(model_input[0])
    if not isinstance(upstream, dict):
        return None
    class_type = upstream.get("class_type")
    return str(class_type) if class_type else None


class MetaHubSave3DModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_3d": (ANY_3D, {"tooltip": "ComfyUI MESH or File3D model"}),
                "filename_prefix": ("STRING", {"default": "3d/ComfyUI"}),
            },
            "optional": {
                "tags": ("STRING", {"default": ""}),
                "notes": ("STRING", {"default": "", "multiline": True}),
                "project_name": ("STRING", {"default": ""}),
                "generation_time_override": ("FLOAT", {"default": None, "forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_model"
    OUTPUT_NODE = True
    CATEGORY = "3d/save"
    DESCRIPTION = "Save a 3D model with Image MetaHub generation metadata"

    def save_model(
        self,
        model_3d: Any,
        filename_prefix: str = "3d/ComfyUI",
        tags: str = "",
        notes: str = "",
        project_name: str = "",
        generation_time_override: Optional[float] = None,
        prompt: Any = None,
        extra_pnginfo: Any = None,
        unique_id: Any = None,
    ) -> Dict[str, Any]:
        metadata = None if getattr(args, "disable_metadata", False) else _build_metadata(
            prompt,
            extra_pnginfo,
            unique_id,
            tags,
            notes,
            project_name,
            generation_time_override,
        )
        results = []
        saved_paths: list[str] = []
        source_node_class = _source_node_class(prompt, unique_id)

        if hasattr(model_3d, "save_to"):
            extension = str(getattr(model_3d, "format", "glb") or "glb").lower().lstrip(".")
            if extension not in SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported 3D format: {extension}")
            output_path, subfolder = _next_path(filename_prefix, extension)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            model_3d.save_to(str(output_path))
            model_info = {"format": extension, "sourceNodeClass": source_node_class or type(model_3d).__name__}
            payload = dict(metadata or {})
            payload["model_3d"] = model_info
            if metadata:
                if extension == "glb":
                    _inject_glb_metadata(output_path, payload)
                _write_sidecar(output_path, payload)
            results.append({"filename": output_path.name, "subfolder": subfolder, "type": "output"})
            saved_paths.append(str(output_path))
        elif hasattr(model_3d, "vertices") and hasattr(model_3d, "faces"):
            batch_size = int(model_3d.vertices.shape[0])
            for index in range(batch_size):
                vertices, faces, colors, uvs, texture = _mesh_batch_item(model_3d, index)
                output_path, subfolder = _next_path(filename_prefix, "glb")
                payload = dict(metadata or {})
                model_info = _write_glb(
                    output_path,
                    vertices,
                    faces,
                    None,
                    vertex_colors=colors,
                    uvs=uvs,
                    texture=texture,
                    unlit=bool(getattr(model_3d, "unlit", False)),
                )
                payload["model_3d"] = model_info
                model_info["sourceNodeClass"] = source_node_class or type(model_3d).__name__
                if metadata:
                    _inject_glb_metadata(output_path, payload)
                    _write_sidecar(output_path, payload)
                results.append({"filename": output_path.name, "subfolder": subfolder, "type": "output"})
                saved_paths.append(str(output_path))
        else:
            raise TypeError("MetaHub Save 3D Model expected a MESH or File3D value")

        return {"ui": {"3d": results, "imagemetahub_files": saved_paths}}


NODE_CLASS_MAPPINGS = {"MetaHubSave3DModel": MetaHubSave3DModel}
NODE_DISPLAY_NAME_MAPPINGS = {"MetaHubSave3DModel": "MetaHub Save 3D Model"}
