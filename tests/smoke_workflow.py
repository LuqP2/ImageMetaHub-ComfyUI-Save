import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from metadata_utils import build_imh_metadata, ensure_metahub_save_node, save_png_with_metadata


def _read_png_text_chunks(path: Path) -> dict:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("Not a PNG file")
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


def main() -> int:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("smoke_workflow.png")

    params = {
        "positive": "smoke",
        "negative": "",
        "seed": 123,
        "steps": 20,
        "cfg": 7.5,
        "sampler": "euler",
        "scheduler": "normal",
        "model_name": "model.safetensors",
        "model_hash": "abcdef1234",
        "width": 256,
        "height": 256,
        "lora_list": [],
    }
    workflow_json = {
        "workflow": {"nodes": [{"id": 7, "type": "SaveImage", "title": "Save Image"}], "links": []},
        "prompt": {"7": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}},
    }

    ensure_metahub_save_node(workflow_json, "7")
    imh_metadata = build_imh_metadata(params, workflow_json)

    image = Image.new("RGB", (2, 2), color=(0, 0, 0))
    save_png_with_metadata(image, str(output), "params", imh_metadata)

    chunks = _read_png_text_chunks(output)
    if "workflow" not in chunks or "prompt" not in chunks:
        print("[FAIL] Missing tEXt workflow/prompt chunks")
        return 1

    try:
        workflow = json.loads(chunks["workflow"])
        prompt = json.loads(chunks["prompt"])
    except json.JSONDecodeError as exc:
        print(f"[FAIL] Invalid JSON in chunks: {exc}")
        return 1

    if workflow.get("nodes", [{}])[0].get("type") != "MetaHubSaveNode":
        print("[FAIL] workflow save node is not MetaHubSaveNode")
        return 1
    if prompt.get("7", {}).get("class_type") != "MetaHubSaveNode":
        print("[FAIL] prompt save node is not MetaHubSaveNode")
        return 1

    print("[OK] tEXt workflow/prompt present and MetaHubSaveNode preserved")
    print(f"[OK] Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
