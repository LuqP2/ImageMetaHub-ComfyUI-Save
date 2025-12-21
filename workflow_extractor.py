"""
Workflow Extractor for MetaHub Save Node
Auto-extracts sampler params, prompts, model, VAE, and LoRAs from ComfyUI prompt data.
"""

from typing import Any, Dict, List, Optional, Set, Tuple


class WorkflowExtractor:
    SAMPLER_NODES = [
        "KSampler",
        "KSamplerAdvanced",
        "SamplerCustom",
        "SamplerCustomAdvanced",
        "KSamplerSelect",
        "KSampler (Efficient)",
        "ImpactKSamplerBasicPipe",
        "ImpactKSamplerAdvancedPipe",
        "ImpactKSampler",
    ]

    CHECKPOINT_NODES = [
        "CheckpointLoaderSimple",
        "CheckpointLoader",
        "UNETLoader",
        "UnetLoaderGGUF",
    ]

    VAE_NODES = [
        "VAELoader",
        "CheckpointLoaderSimple",
        "CheckpointLoader",
    ]

    LORA_NODES = [
        "LoraLoader",
        "LoraLoaderModelOnly",
        "LoraLoaderModelOnlyAdvanced",
        "LoraLoaderAdvanced",
        "LoraLoaderAny",
    ]

    CLIP_NODES = [
        "CLIPTextEncode",
        "CLIPTextEncodeSDXL",
        "CLIPTextEncodeSDXLPlus",
        "CLIPTextEncodeSDXLRefiner",
        "CLIPTextEncodeSD3",
    ]

    VAE_DECODE_NODES = [
        "VAEDecode",
        "VAEDecodeTiled",
        "VAEDecode (Tiled)",
    ]

    def __init__(self, prompt: Optional[Dict[str, Any]]):
        self.prompt: Dict[str, Any] = {}
        if isinstance(prompt, dict):
            for node_id, node_data in prompt.items():
                self.prompt[str(node_id)] = node_data

    def extract(self, save_node_id: Optional[str] = None) -> Tuple[Dict[str, Any], Set[str]]:
        data: Dict[str, Any] = {"lora_list": []}
        missing: Set[str] = set()

        sampler_node_id = self.find_sampler_for_save_node(save_node_id)
        if sampler_node_id:
            sampler_params = self.extract_sampler_params(sampler_node_id)
            for key in ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"):
                value = sampler_params.get(key)
                if value is None:
                    missing.add(key)
                else:
                    data[key] = value

            positive, negative = self.extract_prompts(sampler_node_id)
            if positive is None:
                missing.add("positive")
            else:
                data["positive"] = positive
            if negative is None:
                missing.add("negative")
            else:
                data["negative"] = negative

            model_name = self.extract_model_name(sampler_node_id)
            if model_name is None:
                missing.add("model_name")
            else:
                data["model_name"] = model_name
        else:
            missing.update(
                {
                    "seed",
                    "steps",
                    "cfg",
                    "sampler_name",
                    "scheduler",
                    "denoise",
                    "positive",
                    "negative",
                    "model_name",
                }
            )

        vae_name = self.extract_vae_name(save_node_id)
        if vae_name is None:
            missing.add("vae_name")
        else:
            data["vae_name"] = vae_name

        lora_list, has_lora_nodes = self.extract_loras()
        if lora_list:
            data["lora_list"] = lora_list
        elif has_lora_nodes:
            missing.add("loras")

        return data, missing

    def find_sampler_for_save_node(self, save_node_id: Optional[str]) -> Optional[str]:
        sampler_nodes = self._find_nodes_by_type(self.SAMPLER_NODES)
        if not sampler_nodes:
            return None

        if save_node_id:
            save_node = self._get_node(save_node_id)
            if save_node:
                images_conn = save_node.get("inputs", {}).get("images")
                start_node_id = self._get_connection_node_id(images_conn)
                if start_node_id:
                    sampler_id = self._find_sampler_from_images_source(start_node_id)
                    if sampler_id:
                        return sampler_id

        return sampler_nodes[0]

    def extract_sampler_params(self, sampler_node_id: str) -> Dict[str, Any]:
        node = self._get_node(sampler_node_id)
        if not node:
            return {}
        inputs = node.get("inputs", {})

        seed = self._coerce_int(self._get_literal_input(inputs, "seed"))
        steps = self._coerce_int(self._get_literal_input(inputs, "steps"))
        cfg = self._coerce_float(self._get_first_literal(inputs, ("cfg", "cfg_scale")))
        sampler_name = self._get_first_literal(inputs, ("sampler_name", "sampler"))
        scheduler = self._get_first_literal(inputs, ("scheduler", "scheduler_name"))
        denoise = self._coerce_float(self._get_first_literal(inputs, ("denoise", "denoise_strength")))

        return {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
        }

    def extract_model_name(self, sampler_node_id: str) -> Optional[str]:
        node = self._get_node(sampler_node_id)
        if not node:
            return None
        model_conn = node.get("inputs", {}).get("model")
        start_node_id = self._get_connection_node_id(model_conn)
        if not start_node_id:
            return None
        checkpoint_node_id = self._bfs_upstream(
            [start_node_id],
            lambda n: self._class_type(n) in self.CHECKPOINT_NODES,
        )
        if not checkpoint_node_id:
            return None
        checkpoint_node = self._get_node(checkpoint_node_id)
        if not checkpoint_node:
            return None
        return self._get_checkpoint_name(checkpoint_node)

    def extract_vae_name(self, save_node_id: Optional[str]) -> Optional[str]:
        vae_decode_id = self._find_vae_decode_node(save_node_id)
        if not vae_decode_id:
            return None
        decode_node = self._get_node(vae_decode_id)
        if not decode_node:
            return None
        vae_conn = decode_node.get("inputs", {}).get("vae")
        start_node_id = self._get_connection_node_id(vae_conn)
        if not start_node_id:
            return None
        vae_node_id = self._bfs_upstream(
            [start_node_id],
            lambda n: self._class_type(n) in self.VAE_NODES,
        )
        if not vae_node_id:
            return None
        vae_node = self._get_node(vae_node_id)
        if not vae_node:
            return None
        return self._get_vae_name(vae_node)

    def extract_prompts(self, sampler_node_id: str) -> Tuple[Optional[str], Optional[str]]:
        node = self._get_node(sampler_node_id)
        if not node:
            return None, None
        inputs = node.get("inputs", {})
        positive_conn = inputs.get("positive") or inputs.get("positive_cond")
        negative_conn = inputs.get("negative") or inputs.get("negative_cond")
        positive = self._extract_text_from_connection(positive_conn)
        negative = self._extract_text_from_connection(negative_conn)
        return positive, negative

    def extract_loras(self) -> Tuple[List[Dict[str, Any]], bool]:
        loras: List[Dict[str, Any]] = []
        has_lora_nodes = False

        for _, node in self.prompt.items():
            class_type = self._class_type(node)
            if not self._is_lora_node(class_type):
                continue
            has_lora_nodes = True
            inputs = node.get("inputs", {})
            lora_name = self._get_first_literal(inputs, ("lora_name", "lora"))
            if not lora_name:
                continue
            strength_model = self._get_first_literal(
                inputs,
                ("strength_model", "strength", "strength_unet"),
            )
            weight = self._coerce_float(strength_model)
            if weight is None:
                strength_clip = self._get_first_literal(inputs, ("strength_clip",))
                weight = self._coerce_float(strength_clip)
            if weight is None:
                weight = 1.0
            loras.append({"name": lora_name, "weight": float(weight)})

        return loras, has_lora_nodes

    def _find_sampler_from_images_source(self, start_node_id: str) -> Optional[str]:
        start_node = self._get_node(start_node_id)
        if not start_node:
            return None
        if self._class_type(start_node) in self.SAMPLER_NODES:
            return start_node_id

        if self._class_type(start_node) in self.VAE_DECODE_NODES:
            samples_conn = start_node.get("inputs", {}).get("samples")
            samples_node_id = self._get_connection_node_id(samples_conn)
            if samples_node_id:
                sampler_id = self._bfs_upstream(
                    [samples_node_id],
                    lambda n: self._class_type(n) in self.SAMPLER_NODES,
                )
                if sampler_id:
                    return sampler_id

        return self._bfs_upstream(
            [start_node_id],
            lambda n: self._class_type(n) in self.SAMPLER_NODES,
        )

    def _find_vae_decode_node(self, save_node_id: Optional[str]) -> Optional[str]:
        if not save_node_id:
            return None
        save_node = self._get_node(save_node_id)
        if not save_node:
            return None
        images_conn = save_node.get("inputs", {}).get("images")
        start_node_id = self._get_connection_node_id(images_conn)
        if not start_node_id:
            return None
        return self._bfs_upstream(
            [start_node_id],
            lambda n: self._class_type(n) in self.VAE_DECODE_NODES,
        )

    def _find_nodes_by_type(self, types: List[str]) -> List[str]:
        matched = []
        for node_id, node in self.prompt.items():
            if self._class_type(node) in types:
                matched.append(node_id)
        return matched

    def _extract_text_from_connection(self, conn: Any) -> Optional[str]:
        node_id = self._get_connection_node_id(conn)
        if not node_id:
            return None
        return self._extract_text_from_node(node_id)

    def _extract_text_from_node(self, start_node_id: str) -> Optional[str]:
        texts: List[str] = []
        queue = [start_node_id]
        visited: Set[str] = set()

        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            node = self._get_node(node_id)
            if not node:
                continue
            class_type = self._class_type(node)
            if class_type in self.CLIP_NODES:
                text = self._get_clip_text(node)
                if text:
                    texts.append(text)
                    continue
            for input_val in node.get("inputs", {}).values():
                conn_id = self._get_connection_node_id(input_val)
                if conn_id and conn_id not in visited:
                    queue.append(conn_id)

        if texts:
            return "\n".join(texts)
        return None

    def _get_clip_text(self, node: Dict[str, Any]) -> Optional[str]:
        inputs = node.get("inputs", {})
        text = self._get_literal_input(inputs, "text")
        if text:
            return str(text)
        parts: List[str] = []
        for key in ("text_g", "text_l"):
            value = self._get_literal_input(inputs, key)
            if value:
                parts.append(str(value))
        if parts:
            return "\n".join(parts)
        return None

    def _get_checkpoint_name(self, node: Dict[str, Any]) -> Optional[str]:
        inputs = node.get("inputs", {})
        for key in ("ckpt_name", "checkpoint", "model_name", "unet_name"):
            value = self._get_literal_input(inputs, key)
            if value:
                return str(value)
        return None

    def _get_vae_name(self, node: Dict[str, Any]) -> Optional[str]:
        inputs = node.get("inputs", {})
        for key in ("vae_name", "ckpt_name", "model_name"):
            value = self._get_literal_input(inputs, key)
            if value:
                return str(value)
        return None

    def _get_node(self, node_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if node_id is None:
            return None
        return self.prompt.get(str(node_id))

    @staticmethod
    def _class_type(node: Dict[str, Any]) -> str:
        return str(node.get("class_type", ""))

    @staticmethod
    def _is_connection(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) >= 2

    def _get_connection_node_id(self, value: Any) -> Optional[str]:
        if not self._is_connection(value):
            return None
        return str(value[0])

    def _get_literal_input(self, inputs: Dict[str, Any], key: str) -> Any:
        value = inputs.get(key)
        if self._is_connection(value):
            return None
        return value

    def _get_first_literal(self, inputs: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
        for key in keys:
            value = self._get_literal_input(inputs, key)
            if value is not None:
                return value
        return None

    def _bfs_upstream(self, start_ids: List[str], match_fn) -> Optional[str]:
        queue = list(start_ids)
        visited: Set[str] = set()
        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            node = self._get_node(node_id)
            if not node:
                continue
            if match_fn(node):
                return node_id
            for input_val in node.get("inputs", {}).values():
                conn_id = self._get_connection_node_id(input_val)
                if conn_id and conn_id not in visited:
                    queue.append(conn_id)
        return None

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _is_lora_node(cls, class_type: str) -> bool:
        if class_type in cls.LORA_NODES:
            return True
        return "lora" in class_type.lower()
