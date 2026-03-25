from workflow_extractor import WorkflowExtractor


def test_workflow_extractor_basic_prompt():
    prompt = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "model.safetensors"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "positive", "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "negative", "clip": ["1", 1]},
        },
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 123,
                "steps": 20,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
            },
        },
        "5": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["4", 0], "vae": ["1", 2]},
        },
        "6": {
            "class_type": "MetaHubSaveNode",
            "inputs": {"images": ["5", 0]},
        },
    }

    extractor = WorkflowExtractor(prompt)
    data, missing = extractor.extract(save_node_id="6")

    assert not missing
    assert data["seed"] == 123
    assert data["steps"] == 20
    assert data["cfg"] == 7.5
    assert data["sampler_name"] == "euler"
    assert data["scheduler"] == "normal"
    assert data["denoise"] == 1.0
    assert data["positive"] == "positive"
    assert data["negative"] == "negative"
    assert data["model_name"] == "model.safetensors"
    assert data["vae_name"] == "model.safetensors"


def test_workflow_extractor_lora_detection():
    prompt = {
        "1": {
            "class_type": "LoraLoader",
            "inputs": {"lora_name": "detail.safetensors", "strength_model": 0.8},
        }
    }

    extractor = WorkflowExtractor(prompt)
    data, missing = extractor.extract()

    assert "loras" not in missing
    assert data["lora_list"] == [{"name": "detail.safetensors", "weight": 0.8}]



def test_workflow_extractor_detects_img2img_lineage():
    prompt = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": "inputs/base.png"},
        },
        "2": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["1", 0], "vae": ["5", 2]},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 123,
                "steps": 20,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 0.45,
                "model": ["5", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["2", 0],
            },
        },
        "4": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["5", 2]},
        },
        "5": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "model.safetensors"},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "positive", "clip": ["5", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "negative", "clip": ["5", 1]},
        },
        "8": {
            "class_type": "MetaHubSaveNode",
            "inputs": {"images": ["4", 0]},
        },
    }

    extractor = WorkflowExtractor(prompt)
    data, _missing = extractor.extract(save_node_id="8")

    assert data["generation_type"] == "img2img"
    assert data["source_image"]["fileName"] == "base.png"
    assert data["source_image"]["relativePath"] == "inputs/base.png"
