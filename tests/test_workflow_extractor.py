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


def test_workflow_extractor_detects_flux_unet_model_and_vae():
    prompt = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "flux-model.safetensors"},
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "flux-vae.safetensors"},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "positive", "clip": ["9", 0]},
        },
        "4": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["1", 0], "steps": 8, "scheduler": "normal"},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 123,
                "steps": 8,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "normal",
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["3", 0],
                "latent_image": ["8", 0],
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["2", 0]},
        },
        "7": {
            "class_type": "MetaHubSaveNode",
            "inputs": {"images": ["6", 0]},
        },
        "8": {"class_type": "EmptyLatentImage", "inputs": {}},
        "9": {"class_type": "CLIPLoader", "inputs": {}},
    }

    extractor = WorkflowExtractor(prompt)
    data, _missing = extractor.extract(save_node_id="7")

    assert data["model_name"] == "flux-model.safetensors"
    assert data["vae_name"] == "flux-vae.safetensors"


def test_workflow_extractor_detects_lora_alternate_keys():
    prompt = {
        "1": {
            "class_type": "PowerLoraLoader",
            "inputs": {"lora_name_1": "style.safetensors", "strength": 0.65},
        }
    }

    extractor = WorkflowExtractor(prompt)
    data, _missing = extractor.extract()

    assert data["lora_list"] == [{"name": "style.safetensors", "weight": 0.65}]


def test_workflow_extractor_detects_rgthree_power_lora_objects():
    prompt = {
        "67": {
            "class_type": "Power Lora Loader (rgthree)",
            "inputs": {
                "lora_1": {
                    "on": True,
                    "lora": "Z-Detail-Slider.safetensors",
                    "strength": 0.5325,
                },
                "lora_2": {
                    "on": True,
                    "lora": "zy_CinematicShot_zit.safetensors",
                    "strength": 0.6875,
                },
                "lora_3": {
                    "on": False,
                    "lora": "disabled.safetensors",
                    "strength": 1.0,
                },
            },
        }
    }

    extractor = WorkflowExtractor(prompt)
    data, _missing = extractor.extract()

    assert data["lora_list"] == [
        {"name": "Z-Detail-Slider.safetensors", "weight": 0.5325},
        {"name": "zy_CinematicShot_zit.safetensors", "weight": 0.6875},
    ]


def test_workflow_extractor_reads_style_prompt_encoder_through_basic_guider():
    prompt = {
        "61": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["63", 0],
                "guider": ["101", 0],
                "sampler": ["108", 0],
                "sigmas": ["62", 0],
                "latent_image": ["113", 0],
            },
        },
        "101": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["100", 0],
                "conditioning": ["114", 0],
            },
        },
        "114": {
            "class_type": "StylePromptEncoder2 //ZImagePowerNodes",
            "inputs": {
                "style": "\"Production Photo\"",
                "text": "A white cat on the roof of a brick house",
                "clip": ["67", 1],
            },
        },
    }

    extractor = WorkflowExtractor(prompt)
    data, missing = extractor.extract()

    assert data["positive"] == "A white cat on the roof of a brick house"
    assert "positive" not in missing


def test_workflow_extractor_resolves_connected_seed_and_prompt_strings():
    prompt = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "zImageTurboNSFW_82BF16FP8.safetensors"},
        },
        "2": {
            "class_type": "Lora Loader (LoraManager)",
            "inputs": {
                "text": "<lora:Z-Detail-Slider:0.95>",
                "loras": {
                    "__value__": [
                        {
                            "name": "Z-Detail-Slider",
                            "strength": 0.75,
                            "active": True,
                            "clipStrength": 0.75,
                        }
                    ]
                },
                "model": ["1", 0],
            },
        },
        "3": {
            "class_type": "easy positive",
            "inputs": {"positive": "rabbit"},
        },
        "4": {
            "class_type": "JoinStrings",
            "inputs": {"delimiter": " ", "string1": ["3", 0]},
        },
        "5": {
            "class_type": "easy stylesSelector",
            "inputs": {"styles": "fooocus_styles", "select_styles": [], "positive": ["4", 0]},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": ["5", 0], "clip": ["2", 1]},
        },
        "7": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["6", 0]},
        },
        "8": {
            "class_type": "SeedGenerator",
            "inputs": {"seed": 862877771869053},
        },
        "9": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"shift": 3.0, "model": ["2", 0]},
        },
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "seed": ["8", 0],
                "steps": 9,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["9", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
            },
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["10", 0], "vae": ["12", 0]},
        },
        "12": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"},
        },
        "13": {
            "class_type": "MetaHubSaveNode",
            "inputs": {"images": ["11", 0]},
        },
    }

    extractor = WorkflowExtractor(prompt)
    data, missing = extractor.extract(save_node_id="13")

    assert data["seed"] == 862877771869053
    assert data["positive"] == "rabbit"
    assert data["negative"] == ""
    assert data["model_name"] == "zImageTurboNSFW_82BF16FP8.safetensors"
    assert data["vae_name"] == "ae.safetensors"
    assert data["lora_list"] == [{"name": "Z-Detail-Slider", "weight": 0.75}]
    assert "seed" not in missing
    assert "positive" not in missing



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


def test_workflow_extractor_resolves_custom_advanced_sampler_chain():
    prompt = {
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "positive prompt", "clip": ["97", 0]},
        },
        "39": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "diffusion_pytorch_model.safetensors"},
        },
        "61": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["63", 0],
                "guider": ["101", 0],
                "sampler": ["108", 0],
                "sigmas": ["62", 0],
                "latent_image": ["104", 0],
            },
        },
        "62": {
            "class_type": "BasicScheduler",
            "inputs": {"scheduler": "beta", "steps": 9, "denoise": 1.0, "model": ["100", 0]},
        },
        "63": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": 604786586961193},
        },
        "71": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["61", 0], "vae": ["39", 0]},
        },
        "91": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "jibMixZIT_v10.safetensors"},
        },
        "100": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"shift": 5.0, "model": ["91", 0]},
        },
        "101": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["100", 0], "conditioning": ["6", 0]},
        },
        "104": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "108": {
            "class_type": "ClownSampler_Beta",
            "inputs": {"eta": 0.23, "sampler_name": "linear/ralston_2s", "seed": 141088624843721},
        },
        "112": {
            "class_type": "MetaHubSaveImage",
            "inputs": {"images": ["71", 0]},
        },
    }

    extractor = WorkflowExtractor(prompt)
    data, missing = extractor.extract(save_node_id="112")

    assert data["seed"] == 604786586961193
    assert data["steps"] == 9
    assert data["sampler_name"] == "linear/ralston_2s"
    assert data["scheduler"] == "beta"
    assert data["denoise"] == 1.0
    assert data["positive"] == "positive prompt"
    assert data["model_name"] == "jibMixZIT_v10.safetensors"
    assert data["vae_name"] == "diffusion_pytorch_model.safetensors"
    assert {"seed", "steps", "sampler_name", "scheduler", "denoise", "positive", "model_name", "vae_name"}.isdisjoint(missing)
