import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from metahub_input_node import MetaHubInputNode, payload_digest, resolve_payload_dir


def write_payload(root: Path, session_id="prep_test", mask=True, color=(10, 20, 30)):
    payload_dir = root / "sessions" / session_id
    latest_dir = root / "latest"
    payload_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": 1,
        "session_id": session_id,
        "prepared_at": "2026-06-24T18:30:12.000Z",
        "intent": "outpaint" if mask else "img2img",
        "denoise": 0.72,
        "files": {
            "image": {"name": "image.png", "width": 3, "height": 2},
            "mask": {"name": "mask.png", "available": mask, "width": 3, "height": 2},
        },
        "source": {"path": "D:/images/source.png", "name": "source.png"},
    }

    image = Image.new("RGB", (3, 2), color=color)
    mask_image = Image.new("L", (3, 2), color=255)
    for directory in (payload_dir, latest_dir):
        image.save(directory / "image.png")
        if mask:
            mask_image.save(directory / "mask.png")
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def test_loads_latest_payload_with_mask(tmp_path):
    metadata = write_payload(tmp_path, mask=True)

    result = MetaHubInputNode().load_bridge(str(tmp_path), "latest")

    image, mask, metadata_json, denoise, intent, source_path, session_id, width, height = result
    assert tuple(image.shape) == (1, 2, 3, 3)
    assert tuple(mask.shape) == (1, 2, 3)
    assert torch.all(mask == 1)
    assert json.loads(metadata_json)["session_id"] == metadata["session_id"]
    assert denoise == pytest.approx(0.72)
    assert intent == "outpaint"
    assert source_path == "D:/images/source.png"
    assert session_id == "prep_test"
    assert (width, height) == (3, 2)


def test_missing_mask_returns_black_mask(tmp_path):
    write_payload(tmp_path, session_id="prep_nomask", mask=False)

    image, mask, _metadata_json, _denoise, intent, _source_path, session_id, width, height = (
        MetaHubInputNode().load_bridge(str(tmp_path), "prep_nomask")
    )

    assert tuple(image.shape) == (1, 2, 3, 3)
    assert tuple(mask.shape) == (1, 2, 3)
    assert torch.all(mask == 0)
    assert intent == "img2img"
    assert session_id == "prep_nomask"
    assert (width, height) == (3, 2)


def test_rejects_path_traversal_session_id(tmp_path):
    with pytest.raises(ValueError):
        resolve_payload_dir(str(tmp_path), "../bad")


def test_is_changed_digest_updates_when_payload_changes(tmp_path):
    write_payload(tmp_path, mask=False, color=(10, 20, 30))
    first_digest = payload_digest(str(tmp_path), "latest")

    write_payload(tmp_path, session_id="prep_test_2", mask=False, color=(40, 50, 60))
    second_digest = payload_digest(str(tmp_path), "latest")

    assert first_digest != second_digest


def test_node_registration_includes_input_node():
    import __init__ as package

    assert package.NODE_CLASS_MAPPINGS["MetaHubInputNode"] is MetaHubInputNode
    assert package.NODE_DISPLAY_NAME_MAPPINGS["MetaHubInputNode"] == "MetaHub Input"
    assert "MetaHubSaveImage" in package.NODE_CLASS_MAPPINGS
    assert "MetaHubTimerNode" in package.NODE_CLASS_MAPPINGS
