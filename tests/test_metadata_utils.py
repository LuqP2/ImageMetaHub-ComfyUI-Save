import json
import math
import sys
from pathlib import Path
import copy

from metadata_utils import (
    build_a1111_metadata,
    build_imh_metadata,
    build_metadata_sources,
    build_metadata_status,
    ensure_metahub_save_node,
    ensure_prompt_in_workflow,
    extract_workflow_attribution,
    find_model_file,
    get_next_filename,
    get_output_directory,
    get_workflow_json,
    make_civitai_safe_text,
    resolve_filename_pattern_path,
    save_png_with_metadata,
)
from PIL import Image


def _read_png_text_chunks(path: Path) -> dict:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("Not a PNG file")
    offset = 8
    chunks = {}
    while offset + 8 <= len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        chunk_type = data[offset + 4:offset + 8]
        chunk_data = data[offset + 8:offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"tEXt":
            if b"\x00" in chunk_data:
                keyword, text = chunk_data.split(b"\x00", 1)
                chunks[keyword.decode("latin-1")] = text.decode("latin-1")
    return chunks


def _save_simple_png(path: Path) -> None:
    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    image.save(path, "PNG")


def test_output_path_relative_resolves_under_comfy_output(tmp_path, monkeypatch):
    class FakeFolderPaths:
        @staticmethod
        def get_output_directory():
            return str(tmp_path / "output")

    monkeypatch.setitem(sys.modules, "folder_paths", FakeFolderPaths)

    output = get_output_directory("project/%date:yyyyMMdd%")

    assert output.parent.name == "project"
    assert output.parent.parent == tmp_path / "output"
    assert output.exists()


def test_output_path_allows_trailing_separator(tmp_path, monkeypatch):
    class FakeFolderPaths:
        @staticmethod
        def get_output_directory():
            return str(tmp_path / "output")

    monkeypatch.setitem(sys.modules, "folder_paths", FakeFolderPaths)

    output = get_output_directory("project/")

    assert output == tmp_path / "output" / "project"
    assert output.exists()


def test_filename_pattern_supports_subfolders_and_tokens():
    params = {"seed": 123, "model_name": "model.safetensors"}

    relative = resolve_filename_pattern_path("%date:yyyyMMdd%/image_%seed%", params, 1, "PNG")

    assert len(relative.parts) == 2
    assert relative.parts[1] == "image_123.png"


def test_filename_pattern_rejects_unsafe_segments():
    try:
        resolve_filename_pattern_path("../escape", {}, 1, "PNG")
    except ValueError as error:
        assert "Unsafe path segment" in str(error)
    else:
        raise AssertionError("Expected unsafe pattern to fail")


def test_get_next_filename_creates_nested_collision_path(tmp_path):
    existing = tmp_path / "set" / "image.png"
    existing.parent.mkdir()
    existing.write_bytes(b"exists")

    path = get_next_filename(tmp_path, "set/image", {}, "PNG")

    assert path == tmp_path / "set" / "image_00002.png"


def test_metadata_status_marks_defaulted_fields_as_partial():
    sources = build_metadata_sources(
        {"positive": None, "model_name": None, "seed": None},
        {"positive": "prompt"},
        ["positive", "model_name", "seed"],
    )

    assert sources["positive"] == "detected"
    assert sources["model_name"] == "default"
    assert build_metadata_status(sources, ["positive", "model_name", "seed"]) == "partial"


def test_blank_string_override_is_unknown_not_detected():
    sources = build_metadata_sources(
        {"positive": ""},
        {"positive": "extracted prompt"},
        ["positive"],
    )

    assert sources["positive"] == "unknown"
    assert build_metadata_status(sources, ["positive"]) == "fallback"


def test_find_model_file_tries_safetensors_for_dotted_extensionless_names(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "dreamshaper.v8.safetensors"
    model_path.write_bytes(b"model")
    monkeypatch.setenv("COMFYUI_CHECKPOINT_PATH", str(model_dir))

    assert find_model_file("dreamshaper.v8", "checkpoint") == model_path


def test_ensure_prompt_in_workflow_adds_prompt():
    workflow = {"workflow": {"nodes": []}}
    prompt = {"1": {"class_type": "KSampler", "inputs": {}}}

    result = ensure_prompt_in_workflow(workflow, prompt)

    assert result["prompt"] == prompt


def test_ensure_prompt_in_workflow_preserves_existing_prompt():
    existing = {"9": {"class_type": "SaveImage", "inputs": {}}}
    workflow = {"prompt": existing}
    prompt = {"1": {"class_type": "KSampler", "inputs": {}}}

    result = ensure_prompt_in_workflow(workflow, prompt)

    assert result["prompt"] is existing


def test_ensure_metahub_save_node_updates_prompt_by_id():
    workflow = {"prompt": {"7": {"class_type": "SaveImage", "inputs": {}}}}

    ensure_metahub_save_node(workflow, "7")

    assert workflow["prompt"]["7"]["class_type"] == "MetaHubSaveNode"


def test_ensure_metahub_save_node_updates_single_prompt_saveimage():
    workflow = {"prompt": {"1": {"class_type": "SaveImage", "inputs": {}}}}

    ensure_metahub_save_node(workflow, None)

    assert workflow["prompt"]["1"]["class_type"] == "MetaHubSaveNode"


def test_ensure_metahub_save_node_skips_multiple_prompt_saveimage():
    workflow = {
        "prompt": {
            "1": {"class_type": "SaveImage", "inputs": {}},
            "2": {"class_type": "SaveImage", "inputs": {}},
        }
    }

    ensure_metahub_save_node(workflow, None)

    assert workflow["prompt"]["1"]["class_type"] == "SaveImage"
    assert workflow["prompt"]["2"]["class_type"] == "SaveImage"


def test_ensure_metahub_save_node_updates_workflow_node_by_id():
    workflow = {
        "workflow": {
            "nodes": [
                {
                    "id": 5,
                    "type": "SaveImage",
                    "title": "Save Image",
                    "properties": {"node_name": "Save Image"},
                }
            ]
        }
    }

    ensure_metahub_save_node(workflow, "5")

    node = workflow["workflow"]["nodes"][0]
    assert node["type"] == "MetaHubSaveNode"
    assert node["title"] == "MetaHub Save Image Advanced"
    assert node["properties"]["node_name"] == "MetaHub Save Image Advanced"


def test_ensure_metahub_save_node_can_preserve_simple_node_type():
    workflow = {
        "prompt": {"5": {"class_type": "SaveImage", "inputs": {}}},
        "workflow": {
            "nodes": [
                {
                    "id": 5,
                    "type": "SaveImage",
                    "title": "Save Image",
                    "properties": {"node_name": "Save Image"},
                }
            ]
        },
    }

    ensure_metahub_save_node(
        workflow,
        "5",
        class_type="MetaHubSaveImage",
        display_name="MetaHub Save Image",
    )

    node = workflow["workflow"]["nodes"][0]
    assert workflow["prompt"]["5"]["class_type"] == "MetaHubSaveImage"
    assert node["type"] == "MetaHubSaveImage"
    assert node["title"] == "MetaHub Save Image"
    assert node["properties"]["node_name"] == "MetaHub Save Image"


def test_get_workflow_json_wraps_raw_workflow():
    extra_pnginfo = {"nodes": [], "links": []}

    result = get_workflow_json(extra_pnginfo)

    assert result == {"workflow": extra_pnginfo}


def test_get_workflow_json_accepts_extra_pnginfo_envelope():
    extra_pnginfo = {"extra_pnginfo": {"workflow": {"nodes": []}}}

    result = get_workflow_json(extra_pnginfo)

    assert result == {"workflow": {"nodes": []}}


def test_get_workflow_json_accepts_prompt_dict():
    prompt = {"1": {"class_type": "KSampler", "inputs": {}}}

    result = get_workflow_json(prompt)

    assert result == {"prompt": prompt}


def test_build_imh_metadata_includes_workflow_and_prompt():
    params = {
        "positive": "pos",
        "negative": "neg",
        "seed": 123,
        "steps": 20,
        "cfg": 7.5,
        "sampler": "euler",
        "scheduler": "normal",
        "model_name": "model.safetensors",
        "model_hash": "abcdef1234",
        "width": 512,
        "height": 768,
        "lora_list": [],
        "generation_type": "img2img",
        "parent_image": {"fileName": "selected.png", "relativePath": "library/selected.png"},
        "source_image": {"fileName": "base.png", "relativePath": "inputs/base.png"},
        "metadata_status": "partial",
        "metadata_sources": {"model_name": "default"},
    }
    workflow_json = {"workflow": {"nodes": []}, "prompt": {"1": {"class_type": "KSampler"}}}

    imh = build_imh_metadata(params, workflow_json)

    assert imh["generator"] == "ComfyUI"
    assert imh["workflow"] == workflow_json["workflow"]
    assert imh["prompt_api"] == workflow_json["prompt"]
    assert imh["generation_type"] == "img2img"
    assert imh["parent_image"]["fileName"] == "selected.png"
    assert imh["source_image"]["fileName"] == "base.png"
    assert imh["metadata_status"] == "partial"
    assert imh["metadata_sources"] == {"model_name": "default"}


def test_extract_workflow_attribution_from_node_properties():
    workflow_json = {
        "workflow": {
            "nodes": [
                {
                    "id": 7,
                    "type": "MetaHubSaveNode",
                    "properties": {
                        "imh_attribution": {
                            "schema_version": 1,
                            "token": " imhcrt_br_creator_workflow_v1_random ",
                            "source": "metahub_save_node",
                        }
                    },
                }
            ]
        }
    }

    attribution = extract_workflow_attribution(workflow_json, "7")

    assert attribution["token"] == "imhcrt_br_creator_workflow_v1_random"
    assert attribution["source"] == "metahub_save_node"
    assert attribution["node_version"] == "1.1.3"


def test_build_imh_metadata_includes_attribution_without_a1111_parameters():
    params = {
        "positive": "pos",
        "negative": "neg",
        "steps": 20,
        "sampler": "euler",
        "scheduler": "normal",
        "cfg": 7.5,
        "seed": 123,
        "width": 512,
        "height": 768,
        "model_name": "model.safetensors",
        "model_hash": "abcdef1234",
        "lora_list": [],
        "imh_attribution": {
            "schema_version": 1,
            "token": "imhcrt_br_creator_workflow_v1_random",
            "source": "metahub_save_node",
            "node_version": "1.0.9",
        },
    }

    imh = build_imh_metadata(params, {"workflow": {"nodes": []}, "prompt": {}})
    a1111 = build_a1111_metadata(params)

    assert imh["imh_attribution"]["token"] == "imhcrt_br_creator_workflow_v1_random"
    assert "imhcrt_br_creator_workflow_v1_random" not in a1111


def test_build_imh_metadata_sanitizes_nan():
    params = {
        "positive": "pos",
        "negative": "neg",
        "seed": 123,
        "steps": 20,
        "cfg": 7.5,
        "sampler": "euler",
        "scheduler": "normal",
        "model_name": "model.safetensors",
        "model_hash": "abcdef1234",
        "width": 512,
        "height": 768,
        "lora_list": [],
    }
    workflow_json = {
        "workflow": {"nodes": [{"id": 1, "widgets_values": [float("nan")]}]},
        "prompt": {"1": {"class_type": "KSampler", "inputs": {"seed": float("nan")}}},
    }

    imh = build_imh_metadata(params, workflow_json)

    assert imh["workflow"]["nodes"][0]["widgets_values"][0] is None
    assert imh["prompt_api"]["1"]["inputs"]["seed"] is None


def test_build_a1111_metadata_formats_text():
    params = {
        "positive": "pos",
        "negative": "neg",
        "steps": 20,
        "sampler": "euler",
        "cfg": 7.5,
        "seed": 123,
        "width": 512,
        "height": 768,
        "model_name": "model.safetensors",
        "model_hash": "abcdef1234",
        "lora_hashes": {},
    }

    text = build_a1111_metadata(params)

    assert "Negative prompt: neg" in text
    assert "Steps: 20" in text
    assert "Sampler: euler" in text
    assert "CFG scale: 7.5" in text
    assert "Seed: 123" in text
    assert "Size: 512x768" in text
    assert "Model: model.safetensors" in text
    assert "Model hash: abcdef1234" in text


def test_make_civitai_safe_text_normalizes_smart_punctuation():
    text = "McDonald\u2019s \u201cquote\u201d \u2013 dash \u2014 dash\u2026space\u00a0here"

    assert make_civitai_safe_text(text) == "McDonald's \"quote\" - dash - dash...space here"


def test_save_png_with_metadata_sanitizes_only_parameters_chunk(tmp_path):
    imh_metadata = {
        "prompt": "McDonald\u2019s \u201cquote\u201d",
        "workflow": {"nodes": [{"title": "McDonald\u2019s \u201cquote\u201d"}]},
        "prompt_api": {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "McDonald\u2019s \u201cquote\u201d"},
            }
        },
    }
    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    file_path = tmp_path / "civitai-safe.png"

    save_png_with_metadata(
        image,
        str(file_path),
        "McDonald\u2019s \u201cquote\u201d \u2013 dash\u2026",
        imh_metadata,
    )

    loaded = Image.open(file_path)
    text = getattr(loaded, "text", {})

    assert text["parameters"] == "McDonald's \"quote\" - dash..."
    assert json.loads(text["imagemetahub_data"])["prompt"] == "McDonald\u2019s \u201cquote\u201d"
    assert json.loads(text["workflow"])["nodes"][0]["title"] == "McDonald\u2019s \u201cquote\u201d"
    assert json.loads(text["prompt"])["1"]["inputs"]["text"] == "McDonald\u2019s \u201cquote\u201d"


def test_save_png_with_metadata_writes_workflow_prompt(tmp_path):
    imh_metadata = {
        "workflow": {"nodes": []},
        "prompt_api": {"1": {"class_type": "KSampler"}},
    }
    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    file_path = tmp_path / "test.png"

    save_png_with_metadata(image, str(file_path), "params", imh_metadata)

    loaded = Image.open(file_path)
    text = getattr(loaded, "text", {})

    workflow_text = text.get("workflow")
    prompt_text = text.get("prompt")

    assert workflow_text
    assert prompt_text
    assert json.loads(workflow_text) == imh_metadata["workflow"]
    assert json.loads(prompt_text) == imh_metadata["prompt_api"]

    chunk_text = _read_png_text_chunks(file_path)
    assert "workflow" in chunk_text
    assert "prompt" in chunk_text


def test_save_png_with_metadata_roundtrip_for_comfyui(tmp_path):
    params = {
        "positive": "pos",
        "negative": "neg",
        "seed": 123,
        "steps": 20,
        "cfg": 7.5,
        "sampler": "euler",
        "scheduler": "normal",
        "model_name": "model.safetensors",
        "model_hash": "abcdef1234",
        "width": 512,
        "height": 768,
        "lora_list": [],
    }
    workflow_json = {
        "workflow": {"nodes": [{"id": 7, "type": "SaveImage", "title": "Save Image"}], "links": []},
        "prompt": {"7": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}},
    }

    ensure_metahub_save_node(workflow_json, "7")
    imh_metadata = build_imh_metadata(params, workflow_json)

    file_path = tmp_path / "roundtrip.png"
    _save_simple_png(file_path)
    image = Image.open(file_path)
    save_png_with_metadata(image, str(file_path), "params", imh_metadata)

    text_chunks = _read_png_text_chunks(file_path)
    assert "workflow" in text_chunks
    assert "prompt" in text_chunks

    workflow_loaded = json.loads(text_chunks["workflow"])
    prompt_loaded = json.loads(text_chunks["prompt"])

    assert workflow_loaded["nodes"][0]["type"] == "MetaHubSaveNode"
    assert prompt_loaded["7"]["class_type"] == "MetaHubSaveNode"
