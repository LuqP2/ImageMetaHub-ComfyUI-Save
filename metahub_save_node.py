"""
MetaHub Save Image Node for ComfyUI
Advanced image saving with dual metadata support (A1111/Civitai + Image MetaHub)
"""

import time
from pathlib import Path
from . import metadata_utils as utils
from .workflow_extractor import WorkflowExtractor

_HINT_SHOWN = False


class MetaHubSaveNode:
    """
    ComfyUI custom node for saving images with comprehensive metadata.
    
    Features:
    - A1111/Civitai compatible metadata (tEXt chunk "parameters")
    - Image MetaHub metadata (iTXt chunk "imagemetahub_data")
    - Auto-detection of workflow parameters (sampler, prompts, model, VAE, LoRAs)
    - SHA256 model hashes (AutoV2/Civitai format)
    - Graceful degradation (never interrupts generation)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "filename_pattern": ("STRING", {
                    "default": "ComfyUI_%counter%",
                    "tooltip": "Filename pattern with placeholders"
                }),
                "file_format": (["PNG", "JPEG", "WebP"], {
                    "default": "PNG",
                    "tooltip": "File format"
                }),
                "quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "tooltip": "JPEG/WebP quality"
                }),
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Custom output directory (empty = ComfyUI default)"
                }),
                "seed": ("INT", {
                    "default": None,
                    "forceInput": True,
                    "tooltip": "Override seed"
                }),
                "steps": ("INT", {
                    "default": None,
                    "forceInput": True,
                    "tooltip": "Override steps"
                }),
                "cfg": ("FLOAT", {
                    "default": None,
                    "forceInput": True,
                    "tooltip": "Override CFG scale"
                }),
                "sampler_name": ("STRING", {
                    "default": None,
                    "forceInput": True,
                    "tooltip": "Override sampler name"
                }),
                "scheduler": ("STRING", {
                    "default": None,
                    "forceInput": True,
                    "tooltip": "Override scheduler"
                }),
                "model_name": ("STRING", {
                    "default": None,
                    "forceInput": True,
                    "tooltip": "Override model filename"
                }),
                "positive": ("STRING", {
                    "multiline": True,
                    "default": None,
                    "forceInput": True,
                    "tooltip": "Override positive prompt"
                }),
                "negative": ("STRING", {
                    "multiline": True,
                    "default": None,
                    "forceInput": True,
                    "tooltip": "Override negative prompt"
                }),
                "denoise": ("FLOAT", {
                    "default": None,
                    "forceInput": True,
                    "tooltip": "Override denoise strength"
                }),
                "vae_name": ("STRING", {
                    "default": None,
                    "forceInput": True,
                    "tooltip": "Override VAE filename"
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
                "generation_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "tooltip": "Generation time in seconds"
                }),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "Deprecated: use filename_pattern"
                }),
                "upscale_model": ("STRING", {
                    "default": "",
                    "tooltip": "Upscale model name"
                }),
                # Performance metrics overrides (advanced users)
                "vram_peak_mb": ("FLOAT", {
                    "default": None,
                    "forceInput": True,
                    "tooltip": "Override VRAM peak (MB)"
                }),
                "gpu_device_override": ("STRING", {
                    "default": None,
                    "forceInput": True,
                    "tooltip": "Override GPU device name"
                }),
                "generation_time_override": ("FLOAT", {
                    "default": None,
                    "forceInput": True,
                    "tooltip": "Timestamp from MetaHub Timer Node (elapsed time calculated automatically)"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image/save"
    DESCRIPTION = "Save images with A1111/Civitai and Image MetaHub metadata"

    def save_images(
        self,
        images,
        filename_pattern="ComfyUI_%counter%",
        file_format="PNG",
        quality=95,
        output_path="",
        seed=None,
        steps=None,
        cfg=None,
        sampler_name=None,
        scheduler=None,
        model_name=None,
        positive=None,
        negative=None,
        denoise=None,
        vae_name=None,
        user_tags="",
        notes="",
        project_name="",
        generation_time=0.0,
        filename_prefix="ComfyUI",
        upscale_model="",
        vram_peak_mb=None,
        gpu_device_override=None,
        generation_time_override=None,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
    ):
        global _HINT_SHOWN

        try:
            workflow_json = utils.get_workflow_json(extra_pnginfo)
            prompt_data = prompt if isinstance(prompt, dict) else workflow_json.get("prompt", {})
            if not isinstance(prompt_data, dict):
                prompt_data = {}

            workflow_json = utils.ensure_prompt_in_workflow(workflow_json, prompt_data)
            save_node_id = str(unique_id) if unique_id is not None else None
            utils.ensure_metahub_save_node(workflow_json, save_node_id)

            extractor = WorkflowExtractor(prompt_data)
            extracted, missing_fields = extractor.extract(
                save_node_id=save_node_id
            )
            lora_list = extracted.get("lora_list") or utils.extract_loras_from_workflow(workflow_json)

            def resolve_value(manual_value, extracted_value, default_value):
                if manual_value is not None:
                    return manual_value
                if extracted_value is not None:
                    return extracted_value
                return default_value

            def normalize_int(value, default_value):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return default_value

            def normalize_float(value, default_value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default_value

            seed_value = normalize_int(resolve_value(seed, extracted.get("seed"), 0), 0)
            steps_value = normalize_int(resolve_value(steps, extracted.get("steps"), 20), 20)
            cfg_value = normalize_float(resolve_value(cfg, extracted.get("cfg"), 7.0), 7.0)
            sampler_value = resolve_value(sampler_name, extracted.get("sampler_name"), "euler")
            scheduler_value = resolve_value(scheduler, extracted.get("scheduler"), "normal")
            model_name_value = resolve_value(model_name, extracted.get("model_name"), "")
            positive_value = resolve_value(positive, extracted.get("positive"), "")
            negative_value = resolve_value(negative, extracted.get("negative"), "")
            denoise_value = normalize_float(resolve_value(denoise, extracted.get("denoise"), 1.0), 1.0)
            vae_name_value = resolve_value(vae_name, extracted.get("vae_name"), "")

            quality_value = normalize_int(quality if quality is not None else 95, 95)
            quality_value = max(1, min(100, quality_value))

            default_pattern = "ComfyUI_%counter%"
            pattern_value = (filename_pattern or "").strip()
            if pattern_value and pattern_value != default_pattern:
                final_pattern = pattern_value
            elif filename_prefix and str(filename_prefix).strip():
                final_pattern = f"{filename_prefix}_%counter%"
            else:
                final_pattern = default_pattern

            if model_name_value:
                model_hash = utils.calculate_model_hash(model_name_value, model_type="checkpoint")
            else:
                model_hash = "0000000000"
            lora_hashes = utils.calculate_lora_hashes(lora_list) if lora_list else {}

            height, width = images[0].shape[0], images[0].shape[1]

            # Determine final generation time
            # generation_time_override is a TIMESTAMP from Timer Node, calculate elapsed time
            print(f"[MetaHub Save] generation_time_override={generation_time_override}, generation_time={generation_time}")
            if generation_time_override is not None and generation_time_override > 0:
                # Calculate elapsed time from Timer timestamp
                final_time = time.time() - generation_time_override
                print(f"[MetaHub Save] Calculated elapsed time from Timer: {final_time:.2f}s")
            elif generation_time > 0:
                final_time = generation_time
                print(f"[MetaHub Save] Using legacy generation_time: {final_time}s")
            else:
                final_time = 0.0
                print(f"[MetaHub Save] No generation time provided")

            # Collect GPU metrics (auto-detect)
            gpu_metrics = utils.collect_gpu_metrics()

            # Collect version info
            version_info = utils.collect_version_info()

            # Calculate derived metrics
            generation_time_ms = None
            if final_time > 0:
                generation_time_ms = int(final_time * 1000)

            steps_per_second = None
            if generation_time_ms and generation_time_ms > 0 and steps_value > 0:
                steps_per_second = round((steps_value / (generation_time_ms / 1000)), 2)

            params = {
                "positive": positive_value,
                "negative": negative_value,
                "steps": steps_value,
                "sampler": sampler_value,
                "scheduler": scheduler_value,
                "cfg": cfg_value,
                "seed": seed_value,
                "width": width,
                "height": height,
                "model_name": model_name_value,
                "model_hash": model_hash,
                "vae_name": vae_name_value,
                "denoise": denoise_value,
                "upscale_model": upscale_model,
                "generation_time": generation_time,
                "user_tags": user_tags,
                "notes": notes,
                "project_name": project_name,
                "lora_list": lora_list,
                "lora_hashes": lora_hashes,
                # Performance metrics (Tier 1, 2, 3)
                "vram_peak_mb": vram_peak_mb if vram_peak_mb is not None else gpu_metrics.get("vram_peak_mb"),
                "gpu_device": gpu_device_override if gpu_device_override else gpu_metrics.get("gpu_device"),
                "generation_time_ms": generation_time_ms,
                "steps_per_second": steps_per_second,
                "comfyui_version": version_info.get("comfyui_version"),
                "torch_version": version_info.get("torch_version"),
                "python_version": version_info.get("python_version"),
            }

            a1111_metadata = utils.build_a1111_metadata(params)
            imh_metadata = utils.build_imh_metadata(params, workflow_json)

            output_dir = utils.get_output_directory(output_path)
            saved_paths = utils.save_image_batch(
                images,
                output_dir,
                final_pattern,
                params,
                file_format,
                quality_value,
                a1111_metadata,
                imh_metadata,
            )

            warn_fields = {
                "seed",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
                "model_name",
                "positive",
                "negative",
                "loras",
            }
            label_map = {
                "sampler_name": "sampler",
                "model_name": "model",
            }
            manual_inputs = {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "model_name": model_name,
                "positive": positive,
                "negative": negative,
            }
            missing_warn = []
            for field in missing_fields:
                if field not in warn_fields:
                    continue
                if field == "loras":
                    if lora_list:
                        continue
                    missing_warn.append("loras")
                    continue
                if manual_inputs.get(field) is not None:
                    continue
                missing_warn.append(label_map.get(field, field))
            if missing_warn:
                missing_warn = sorted(set(missing_warn))
                print(
                    "[ImageMetaHub-Save] ⚠ Some params not detected "
                    f"({', '.join(missing_warn)}), image saved with available metadata"
                )

            auto_summary_parts = []
            if seed is None and extracted.get("seed") is not None:
                auto_summary_parts.append(f"seed={seed_value}")
            if steps is None and extracted.get("steps") is not None:
                auto_summary_parts.append(f"steps={steps_value}")
            if model_name is None and extracted.get("model_name"):
                model_display = Path(str(model_name_value)).stem or str(model_name_value)
                auto_summary_parts.append(f"model={model_display}")
            auto_summary = ", ".join(auto_summary_parts)

            for file_path in saved_paths:
                suffix = ""
                if not _HINT_SHOWN and auto_summary:
                    suffix = f" (auto-detected: {auto_summary})"
                print(f"[ImageMetaHub-Save] ✓ Saved: {file_path.name}{suffix}")
                if not _HINT_SHOWN:
                    print("💡 Organize your AI images → github.com/LuqP2/ImageMetaHub")
                    _HINT_SHOWN = True

            # Build preview structure for ComfyUI UI
            output_base = utils.get_output_directory("")  # Get default ComfyUI output
            return utils.build_ui_preview(saved_paths, output_base)

        except Exception as e:
            print(f"[ImageMetaHub-Save] Warning: Save failed: {e}")
            # Return empty UI structure on error
            return {"ui": {"images": []}}


NODE_CLASS_MAPPINGS = {"MetaHubSaveNode": MetaHubSaveNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MetaHubSaveNode": "MetaHub Save Image"}
