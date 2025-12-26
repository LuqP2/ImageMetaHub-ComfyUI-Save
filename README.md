# MetaHub Save Image - ComfyUI Custom Node
[![Official Companion](https://img.shields.io/badge/Official%20companion-Image%20MetaHub-2b6cb0)](https://github.com/LuqP2/Image-MetaHub)




Official companion node for [Image MetaHub](https://github.com/LuqP2/Image-MetaHub).

Advanced image saving node for ComfyUI with dual metadata support.

## Features

- **Auto-Extraction** - Detects sampler params, prompts, model/VAE, and LoRAs directly from your workflow
- **Performance Metrics** - Auto-tracks GPU usage, VRAM peak, generation time, and software versions
- **Multi-Format** - PNG, JPEG, and WebP with metadata injection
- **Filename Patterns** - Placeholder-based filenames with sanitization
- **A1111/Civitai Compatible** - Saves metadata in tEXt chunk ("parameters") recognized by Automatic1111, Civitai, and most SD tools
- **Image MetaHub Compatible** - Saves extended metadata in iTXt chunk ("imagemetahub_data") with full workflow JSON
- **Model Hashes** - Calculates SHA256 hashes (AutoV2 format) for models and LoRAs
- **IMH Pro Fields** - Support for user tags, notes, and project names
- **Performance** - Hash caching and graceful degradation ensure fast generation
- **Never Fails** - Silent fallback on errors - your generation never stops

## Installation

### Method 1: Clone Repository (Recommended)

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/LuqP2/ImageMetaHub-ComfyUI-Save.git
   ```

3. Install dependencies:
   ```bash
   cd ImageMetaHub-ComfyUI-Save
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

### Method 2: Manual Installation

1. Download this repository as ZIP
2. Extract to `ComfyUI/custom_nodes/ImageMetaHub-ComfyUI-Save`
3. Install dependencies: `pip install Pillow>=10.0.0 numpy>=1.24.0 piexif>=1.1.3`
4. Restart ComfyUI

## Usage

### Quick Start

1. Add "MetaHub Save Image" node to your workflow
2. Connect the `images` output from your decoder/sampler
3. Generate! The node auto-detects and saves metadata.

Auto-detected fields:
- seed, steps, cfg
- sampler and scheduler
- model name + hash
- positive/negative prompts
- LoRAs + weights
- VAE name

### Override (Optional)

**Generation Parameters:**
Connect any override input to replace auto-detected values:
- `seed`, `steps`, `cfg`
- `sampler_name`, `scheduler`
- `model_name`, `vae_name`
- `positive`, `negative`
- `denoise`

**Performance Metrics:**
- **Automatic**: VRAM peak and GPU device are auto-detected
- **Generation Time**: Use the **MetaHub Timer** node for accurate workflow timing

**MetaHub Timer Node (Recommended for accurate timing):**
1. Add "MetaHub Timer" node to your workflow
2. Connect one of these inputs (Timer will record a timestamp when it executes):
   - `clip` - From CLIP loader or CLIPTextEncode (for total workflow time)
   - `latent` - From KSampler (for sampling + post-processing time)
   - `image` - From VAEDecode or image loader (for post-processing only)
   - `conditioning` - From CLIPTextEncode
3. Connect the corresponding output to continue your workflow (e.g., `clip` → CLIPTextEncode)
4. Connect `elapsed_time` output to Save Node's `generation_time_override`
   - The Timer records a timestamp when it executes
   - The Save Node automatically calculates elapsed time from this timestamp

**Multiple Timers (Advanced profiling):**
You can add multiple Timer nodes to measure different workflow stages:
- Timer at CLIP → total workflow time (from CLIP load to save)
- Timer after KSampler → sampling + decode time
- Timer after VAEDecode → save/post-processing time only

**Performance Overrides (Advanced):**
For custom benchmarking, connect these hidden inputs:
- `vram_peak_mb` - Override VRAM peak measurement
- `gpu_device_override` - Override GPU device name
- `generation_time_override` - Manual generation time (or from Timer node)

All overrides use `forceInput: True` (hidden from UI, connection-only).

### Filename Pattern
Use `filename_pattern` to customize names (default `ComfyUI_%counter%`). Invalid filename characters are sanitized and missing values become `unknown`. `filename_prefix` is deprecated but still supported.

| Placeholder | Description | Example |
| --- | --- | --- |
| `%counter%` | Auto-increment (00001, 00002...) | 00042 |
| `%seed%` | Seed used | 1234567890 |
| `%date%` | Date YYYY-MM-DD | 2025-01-15 |
| `%time%` | Time HH-MM-SS | 14-30-22 |
| `%datetime%` | Date + time | 2025-01-15_14-30-22 |
| `%model%` | Model name (no extension) | dreamshaper_v8 |
| `%sampler%` | Sampler name | euler_ancestral |
| `%steps%` | Steps | 20 |
| `%cfg%` | CFG scale | 7.5 |
| `%width%` | Image width | 1024 |
| `%height%` | Image height | 768 |

### File Formats
- `file_format`: PNG, JPEG, or WebP
- `quality`: JPEG/WebP quality (1-100)
- `output_path`: custom output directory (empty = ComfyUI default)
- PNG uses tEXt "parameters" and iTXt "imagemetahub_data"
- JPEG/WebP use EXIF UserComment (A1111) and ImageDescription (IMH JSON)

If structured metadata cannot be written, the image is still saved.

### Multi-Sampler Heuristic
When multiple samplers exist, the node prefers the sampler that feeds the latent used by the VAEDecode connected to the saved images; it falls back to the first sampler found.

### IMH Pro Fields (Optional)
Organize your generations with Image MetaHub Pro features:
- **user_tags**: Tag your images (e.g., "portrait, fantasy, character")
- **notes**: Add generation notes or experiments
- **project_name**: Group images by project

### Performance Metrics (Auto-Tracked)

The node automatically collects GPU and performance metrics for every generation:

**Tier 1 - Critical Metrics:**
- **VRAM Peak** - Peak VRAM usage in MB (CUDA GPUs only)
- **GPU Device** - GPU name (e.g., "NVIDIA GeForce RTX 3090")
- **Generation Time** - Total generation time in milliseconds

**Tier 2 - Very Useful:**
- **Steps per Second** - Performance benchmark (steps / generation_time)
- **ComfyUI Version** - Your ComfyUI version

**Tier 3 - Nice-to-Have:**
- **PyTorch Version** - PyTorch version with CUDA info
- **Python Version** - Python runtime version

**Auto-Detection:**
- ✅ **CUDA** (NVIDIA) - Full metrics including VRAM tracking
- ✅ **MPS** (Apple Metal) - GPU device name (VRAM not available)
- ✅ **CPU** - Marks as "CPU (CUDA not available)"

**Advanced Overrides (Optional):**
For custom benchmarking workflows, connect these inputs:
- `vram_peak_mb` - Override VRAM peak (FLOAT)
- `gpu_device_override` - Override GPU device name (STRING)
- `generation_time_override` - Override generation time in seconds (FLOAT)

All metrics are stored in the `analytics` field of the Image MetaHub metadata chunk and displayed in the **Performance** section of Image MetaHub App.

## Configuration (Optional)

If your models are in non-standard locations, set environment variables:

```bash
# Windows (PowerShell)
$env:COMFYUI_CHECKPOINT_PATH="D:\MyModels\Checkpoints"
$env:COMFYUI_LORA_PATH="D:\MyModels\Loras"
$env:COMFYUI_VAE_PATH="D:\MyModels\VAE"

# Linux/macOS
export COMFYUI_CHECKPOINT_PATH="/path/to/checkpoints"
export COMFYUI_LORA_PATH="/path/to/loras"
export COMFYUI_VAE_PATH="/path/to/vae"
```

## Metadata Formats

### A1111/Civitai Format (PNG: tEXt "parameters", JPEG/WebP: EXIF UserComment)

```
masterpiece, best quality, 1girl, portrait
Negative prompt: ugly, blurry
Steps: 20, Sampler: euler, CFG scale: 7.0, Seed: 12345, Size: 512x768, Model: mymodel, Model hash: abc1234567, Lora hashes: "detail: def9876543"
```

### Image MetaHub Format (PNG: iTXt "imagemetahub_data", JPEG/WebP: EXIF ImageDescription)

```json
{
  "generator": "ComfyUI",
  "prompt": "masterpiece, best quality, 1girl, portrait",
  "negativePrompt": "ugly, blurry",
  "seed": 12345,
  "steps": 20,
  "cfg": 7.0,
  "sampler_name": "euler",
  "model": "mymodel.safetensors",
  "model_hash": "abc1234567",
  "loras": [{"name": "detail.safetensors", "weight": 0.8}],
  "imh_pro": {
    "user_tags": "portrait, fantasy",
    "notes": "Experimental composition",
    "project_name": "Character Design"
  },
  "analytics": {
    "vram_peak_mb": 8234.5,
    "gpu_device": "NVIDIA GeForce RTX 3090",
    "generation_time_ms": 1523,
    "steps_per_second": 13.2,
    "comfyui_version": "0.1.0",
    "torch_version": "2.0.1+cu118",
    "python_version": "3.10.12"
  },
  "workflow": { }
}
```

## Integration with Image MetaHub

This node is the official companion for [Image MetaHub](https://github.com/LuqP2/Image-MetaHub).

Images saved with this node are fully compatible with Image MetaHub's:
- **Instant parsing** - MetaHub chunk extraction (no graph traversal needed)
- **Performance section** - Displays GPU metrics, VRAM, generation time, and versions
- **Search and filtering** - Full metadata indexing
- **LoRA detection** - Automatic LoRA weight extraction
- **Analytics dashboard** - Performance benchmarking
- **IMH Pro features** - Tags, notes, and project organization

The Performance section in Image MetaHub App displays:
- Generation time with smart formatting (ms/s/m+s)
- VRAM usage with GPU percentage (e.g., "8.0 GB / 24 GB (33%)")
- GPU device name
- Steps per second benchmark
- Software versions (ComfyUI, PyTorch, Python)

**Generating Variations**: Image MetaHub App includes a "Generate with ComfyUI" feature that creates simple txt2img workflows from saved metadata and sends them to your ComfyUI instance via API. See the [Image MetaHub documentation](https://github.com/LuqP2/Image-MetaHub) for details on workflow generation and ComfyUI integration features.

## Troubleshooting

### "Model hash is 0000000000"
- Your model file wasn't found in standard paths
- Set `COMFYUI_CHECKPOINT_PATH` environment variable
- Or ensure models are in `ComfyUI/models/checkpoints/`

### "LoRAs not detected"
- Make sure you're using standard LoRA Loader nodes
- Check that LoRAs are connected in the workflow
- Custom LoRA nodes might not be detected

### "Cannot create output directory"
- Check folder permissions
- Try using default output path (leave `output_path` empty)

### "Performance metrics not showing in Image MetaHub"
- Ensure you're using the latest version of both the node and Image MetaHub App
- Performance section only appears for images saved with MetaHub Save Node
- Legacy images saved with default ComfyUI Save Image won't have performance data

### "VRAM peak is None/null"
- VRAM tracking requires CUDA (NVIDIA GPUs)
- MPS (Apple Metal) and CPU modes don't support VRAM tracking
- This is expected behavior and not an error

## Performance

- **First image**: ~200ms overhead (hash calculation)
- **Subsequent images**: ~60ms overhead (hashes cached)
- **Batch processing**: Minimal overhead per image
- **GPU metrics**: <1ms overhead (PyTorch API calls)
- **Version detection**: <5ms overhead (cached after first call)

Hash calculation is cached in memory, so repeated generations with the same models are very fast. Performance metrics collection has negligible impact on generation speed.

## License

MIT License

## Contributing

Issues and pull requests welcome at https://github.com/LuqP2/ImageMetaHub-ComfyUI-Save

## Credits

Developed for the [Image MetaHub](https://github.com/LuqP2/Image-MetaHub) ecosystem.

