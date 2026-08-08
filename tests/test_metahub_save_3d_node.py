import json
import struct
from types import SimpleNamespace

import numpy as np

import metahub_save_3d_node as node3d


class FakeFolderPaths:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def get_output_directory(self):
        return str(self.output_dir)

    def get_save_image_path(self, _prefix, _output_root):
        folder = self.output_dir / "3d"
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder), "ComfyUI", 1, "3d", "3d/ComfyUI"


def _mesh(with_optional_data=True):
    vertices = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
    faces = np.array([[[0, 1, 2]]], dtype=np.int64)
    return SimpleNamespace(
        vertices=vertices,
        faces=faces,
        vertex_colors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float32) if with_optional_data else None,
        uvs=np.array([[[0, 0], [1, 0], [0, 1]]], dtype=np.float32) if with_optional_data else None,
        texture=np.ones((1, 2, 2, 3), dtype=np.float32) if with_optional_data else None,
        unlit=False,
    )


def _read_glb_json(path):
    data = path.read_bytes()
    assert data[:4] == b"glTF"
    json_length, json_type = struct.unpack_from("<II", data, 12)
    assert json_type == 0x4E4F534A
    return json.loads(data[20:20 + json_length].rstrip(b" \x00"))


def test_mesh_writes_glb_sidecar_and_embedded_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(node3d, "folder_paths", FakeFolderPaths(tmp_path))
    monkeypatch.setattr(node3d.args, "disable_metadata", False)
    monkeypatch.setattr(node3d, "_build_metadata", lambda *args, **kwargs: {
        "schema_version": 1,
        "media_type": "model3d",
        "prompt": "synthetic prompt",
        "workflow": {"nodes": []},
        "prompt_api": {},
    })

    result = node3d.MetaHubSave3DModel().save_model(_mesh())

    model_path = tmp_path / "3d" / result["ui"]["3d"][0]["filename"]
    sidecar_path = model_path.with_name(model_path.name + ".imagemetahub.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    embedded = _read_glb_json(model_path)["asset"]["extras"]["imagemetahub_data"]
    assert sidecar == embedded
    assert sidecar["model_3d"]["vertexCount"] == 3
    assert sidecar["model_3d"]["faceCount"] == 1
    assert sidecar["model_3d"]["hasTextures"] is True
    assert result["ui"]["imagemetahub_files"] == [str(model_path)]


def test_disable_metadata_writes_model_without_sidecar_or_extras(tmp_path, monkeypatch):
    monkeypatch.setattr(node3d, "folder_paths", FakeFolderPaths(tmp_path))
    monkeypatch.setattr(node3d.args, "disable_metadata", True)

    result = node3d.MetaHubSave3DModel().save_model(_mesh(with_optional_data=False))

    model_path = tmp_path / "3d" / result["ui"]["3d"][0]["filename"]
    assert not model_path.with_name(model_path.name + ".imagemetahub.json").exists()
    assert "extras" not in _read_glb_json(model_path)["asset"]


def test_texture_without_usable_uvs_is_not_reported_as_embedded(tmp_path):
    mesh = _mesh()

    geometry = node3d._write_glb(
        tmp_path / "no-uvs.glb",
        mesh.vertices[0],
        mesh.faces[0],
        {},
        texture=mesh.texture[0],
    )
    gltf = _read_glb_json(tmp_path / "no-uvs.glb")

    assert geometry["hasTextures"] is False
    assert "textures" not in gltf


def test_file3d_preserves_received_format_and_writes_sidecar(tmp_path, monkeypatch):
    class FakeFile3D:
        format = "obj"

        def save_to(self, path):
            with open(path, "wb") as handle:
                handle.write(b"o synthetic\n")

    monkeypatch.setattr(node3d, "folder_paths", FakeFolderPaths(tmp_path))
    monkeypatch.setattr(node3d.args, "disable_metadata", False)
    monkeypatch.setattr(node3d, "_build_metadata", lambda *args, **kwargs: {
        "schema_version": 1,
        "media_type": "model3d",
        "workflow": {},
        "prompt_api": {},
    })

    result = node3d.MetaHubSave3DModel().save_model(FakeFile3D())

    model_path = tmp_path / "3d" / result["ui"]["3d"][0]["filename"]
    assert model_path.suffix == ".obj"
    assert model_path.read_bytes() == b"o synthetic\n"
    assert model_path.with_name(model_path.name + ".imagemetahub.json").exists()


def test_source_node_class_is_read_from_prompt_link():
    prompt = {
        "7": {"class_type": "SyntheticMeshGenerator", "inputs": {}},
        "8": {"class_type": "MetaHubSave3DModel", "inputs": {"model_3d": ["7", 0]}},
    }

    assert node3d._source_node_class(prompt, "8") == "SyntheticMeshGenerator"


def test_mesh_batch_increments_filenames(tmp_path, monkeypatch):
    mesh = _mesh(with_optional_data=False)
    mesh.vertices = np.repeat(mesh.vertices, 2, axis=0)
    mesh.faces = np.repeat(mesh.faces, 2, axis=0)
    monkeypatch.setattr(node3d, "folder_paths", FakeFolderPaths(tmp_path))
    monkeypatch.setattr(node3d.args, "disable_metadata", True)

    result = node3d.MetaHubSave3DModel().save_model(mesh)

    filenames = [item["filename"] for item in result["ui"]["3d"]]
    assert filenames == ["ComfyUI_00001_.glb", "ComfyUI_00002_.glb"]
