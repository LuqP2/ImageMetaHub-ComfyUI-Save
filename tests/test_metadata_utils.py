import json
import copy

from metadata_utils import (
    build_a1111_metadata,
    build_imh_metadata,
    ensure_metahub_save_node,
    ensure_prompt_in_workflow,
    get_workflow_json,
    save_png_with_metadata,
)
from PIL import Image


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
    assert node["title"] == "MetaHub Save Image"
    assert node["properties"]["node_name"] == "MetaHub Save Image"


def test_get_workflow_json_wraps_raw_workflow():
    extra_pnginfo = {"nodes": [], "links": []}

    result = get_workflow_json(extra_pnginfo)

    assert result == {"workflow": extra_pnginfo}


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
    }
    workflow_json = {"workflow": {"nodes": []}, "prompt": {"1": {"class_type": "KSampler"}}}

    imh = build_imh_metadata(params, workflow_json)

    assert imh["generator"] == "ComfyUI"
    assert imh["workflow"] == workflow_json["workflow"]
    assert imh["prompt_api"] == workflow_json["prompt"]


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
