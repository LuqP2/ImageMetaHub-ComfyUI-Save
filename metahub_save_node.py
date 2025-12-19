"""
MetaHub Save Image Node for ComfyUI
Advanced image saving with dual metadata support (A1111/Civitai + Image MetaHub)
"""

from typing import Dict, Any
from . import metadata_utils as utils


class MetaHubSaveNode:
    """
    ComfyUI custom node for saving images with comprehensive metadata.
    
    Features:
    - A1111/Civitai compatible metadata (tEXt chunk "parameters")
    - Image MetaHub metadata (iTXt chunk "imagemetahub_data")
    - Auto-detection of LoRAs from workflow
    - SHA256 model hashes (AutoV2/Civitai format)
    - Graceful degradation (never interrupts generation)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "positive": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Positive prompt"
                }),
                "negative": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Negative prompt"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Generation seed"
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Sampling steps"
                }),
                "cfg": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "CFG scale"
                }),
                "sampler_name": ("STRING", {
                    "default": "euler",
                    "tooltip": "Sampler name"
                }),
                "scheduler": ("STRING", {
                    "default": "normal",
                    "tooltip": "Scheduler"
                }),
                "model_name": ("STRING", {
                    "default": "",
                    "tooltip": "Model filename (e.g., model.safetensors)"
                }),
            },
            "optional": {
                "vae_name": ("STRING", {
                    "default": "",
                    "tooltip": "VAE filename"
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Denoise strength"
                }),
                "upscale_model": ("STRING", {
                    "default": "",
                    "tooltip": "Upscale model name"
                }),
                "generation_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "tooltip": "Generation time in seconds"
                }),
                "user_tags": ("STRING", {
                    "default": "",
                    "tooltip": "IMH Pro: User tags (comma-separated)"
                }),
                "notes": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "IMH Pro: Notes"
                }),
                "project_name": ("STRING", {
                    "default": "",
                    "tooltip": "IMH Pro: Project name"
                }),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "Filename prefix"
                }),
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Custom output directory (empty = ComfyUI default)"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image/save"
    DESCRIPTION = "Save images with A1111/Civitai and Image MetaHub metadata"

    def save_images(self, images, positive, negative, seed, steps, cfg, sampler_name, scheduler, model_name, vae_name="", denoise=1.0, upscale_model="", generation_time=0.0, user_tags="", notes="", project_name="", filename_prefix="ComfyUI", output_path="", prompt=None, extra_pnginfo=None):
        print(f"[MetaHub] Starting save_images for batch of {len(images)} image(s)")
        workflow_json = utils.get_workflow_json(extra_pnginfo)
        lora_list = utils.extract_loras_from_workflow(workflow_json)
        print(f"[MetaHub] Detected {len(lora_list)} LoRA(s) from workflow")
        print(f"[MetaHub] Calculating hash for model: {model_name}")
        model_hash = utils.calculate_model_hash(model_name, model_type="checkpoint")
        print(f"[MetaHub] Calculating hashes for LoRAs...")
        lora_hashes = utils.calculate_lora_hashes(lora_list)
        height, width = images[0].shape[0], images[0].shape[1]
        print(f"[MetaHub] Image dimensions: {width}x{height}")
        output_dir = utils.get_output_directory(output_path)
        print(f"[MetaHub] Saving to: {output_dir}")
        saved_paths = utils.save_image_batch(images, output_dir, filename_prefix)
        print(f"[MetaHub] Saved {len(saved_paths)} image(s)")
        for idx, file_path in enumerate(saved_paths):
            print(f"[MetaHub] Injecting metadata into: {file_path.name}")
            params = {'positive': positive, 'negative': negative, 'steps': steps, 'sampler': sampler_name, 'scheduler': scheduler, 'cfg': cfg, 'seed': seed, 'width': width, 'height': height, 'model_name': model_name, 'model_hash': model_hash, 'vae_name': vae_name, 'denoise': denoise, 'upscale_model': upscale_model, 'generation_time': generation_time, 'user_tags': user_tags, 'notes': notes, 'project_name': project_name, 'lora_list': lora_list, 'lora_hashes': lora_hashes}
            a1111_metadata = utils.build_a1111_metadata(params)
            imh_metadata = utils.build_imh_metadata(params, workflow_json)
            utils.inject_metadata_chunks(str(file_path), a1111_metadata, imh_metadata)
        print(f"[MetaHub] ✓ All images saved successfully with metadata")
        return {}


NODE_CLASS_MAPPINGS = {"MetaHubSaveNode": MetaHubSaveNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MetaHubSaveNode": "MetaHub Save Image"}
