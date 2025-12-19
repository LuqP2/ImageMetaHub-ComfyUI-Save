# MetaHub Save Image - ComfyUI Custom Node

Advanced image saving node for ComfyUI with dual metadata support.

## Features

- **A1111/Civitai Compatible** - Saves metadata in tEXt chunk ("parameters") recognized by Automatic1111, Civitai, and most SD tools
- **Image MetaHub Compatible** - Saves extended metadata in iTXt chunk ("imagemetahub_data") with full workflow JSON  
- **Auto-Detection** - Automatically extracts LoRAs and their weights from your workflow
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
3. Install dependencies: `pip install Pillow>=10.0.0 numpy>=1.24.0`
4. Restart ComfyUI

## Usage

### Basic Usage

1. Add "MetaHub Save Image" node to your workflow
2. Connect the `images` output from your sampler
3. Fill in the generation parameters:
   - **positive**: Your positive prompt
   - **negative**: Your negative prompt
   - **seed**, **steps**, **cfg**: Generation settings
   - **sampler_name**, **scheduler**: Sampler configuration
   - **model_name**: Your checkpoint filename (e.g., `model.safetensors`)

4. Generate! Images are saved with full metadata.

### Advanced Features

#### Auto-Detection of LoRAs
The node automatically scans your workflow for LoRA Loader nodes and extracts:
- LoRA filename
- Weight/strength value

No manual input needed!

#### IMH Pro Fields (Optional)
Organize your generations with Image MetaHub Pro features:
- **user_tags**: Tag your images (e.g., "portrait, fantasy, character")
- **notes**: Add generation notes or experiments
- **project_name**: Group images by project

#### Custom Output
- **filename_prefix**: Change output filename prefix (default: "ComfyUI")
- **output_path**: Save to custom directory (empty = ComfyUI default)

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

### A1111/Civitai Format (tEXt chunk: "parameters")

```
masterpiece, best quality, 1girl, portrait
Negative prompt: ugly, blurry
Steps: 20, Sampler: euler, CFG scale: 7.0, Seed: 12345, Size: 512x768, Model: mymodel, Model hash: abc1234567, Lora hashes: "detail: def9876543"
```

### Image MetaHub Format (iTXt chunk: "imagemetahub_data")

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
  "workflow": { }
}
```

## Integration with Image MetaHub

This node is the official companion for [Image MetaHub](https://github.com/LuqP2/Image-MetaHub).

Images saved with this node are fully compatible with Image MetaHub's:
- Search and filtering
- LoRA detection
- Analytics dashboard
- IMH Pro features

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

## Performance

- **First image**: ~200ms overhead (hash calculation)
- **Subsequent images**: ~60ms overhead (hashes cached)
- **Batch processing**: Minimal overhead per image

Hash calculation is cached in memory, so repeated generations with the same models are very fast.

## License

MIT License

## Contributing

Issues and pull requests welcome at https://github.com/LuqP2/ImageMetaHub-ComfyUI-Save

## Credits

Developed for the [Image MetaHub](https://github.com/LuqP2/Image-MetaHub) ecosystem.
