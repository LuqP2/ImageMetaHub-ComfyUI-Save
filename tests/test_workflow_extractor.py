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
