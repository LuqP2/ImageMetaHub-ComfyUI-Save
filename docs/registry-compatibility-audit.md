# Auditoria de compatibilidade — `pyproject.toml` / ComfyUI Registry

Investigação do **código** (não do README, não do template comentado) para decidir
o que é seguro declarar em `classifiers`, `requires-python`, `Icon`/`Banner` e
`requires-comfyui`, conforme
[docs.comfy.org/registry/specifications](https://docs.comfy.org/registry/specifications).

Este documento é só o relatório. O `pyproject.toml` **não foi alterado** —
a proposta está no bloco de código no final, para aplicação manual.

---

## 1. Menor versão de Python que o código realmente aceita

**Resposta: Python ≥ 3.10.**

Evidência: [`metahub_input_node.py:69`](../metahub_input_node.py#L69)

```python
def mask_to_tensor(mask_path: Path | None, width: int, height: int) -> torch.Tensor:
```

O arquivo não tem `from __future__ import annotations` (confirmado por busca em
todo o repo), então essa anotação é avaliada **em tempo de definição da função**,
não como string. `Path | None` usando o operador `|` em tipos reais é
[PEP 604](https://peps.python.org/pep-0604/), introduzido no Python 3.10 — em
3.9 ou anterior isso levanta `TypeError: unsupported operand type(s) for |`
na hora de importar o módulo, ou seja, o node inteiro falha ao carregar.

Constraint mais fraca, mesma direção: [`metahub_input_node.py:46`](../metahub_input_node.py#L46)
usa `-> tuple[dict, str]` (PEP 585, genéricos nativos em minúsculo), que exige
apenas 3.9+ — já coberto pelo piso de 3.10 acima.

Não há nenhum outro uso de sintaxe mais nova (sem `match/case`, sem `:=`,
sem `except*`, sem `str.removeprefix/removesuffix`, sem `tomllib`) em nenhum
outro arquivo do pacote.

---

## 2. Windows / Linux / macOS — funciona igual? Existe caminho platform-dependent?

**Resposta: sim, funciona igual nos três. Há só um ponto assimétrico, não-crítico.**

- Todo I/O de arquivo passa por `pathlib.Path` (não há concatenação manual de
  string com `\` ou `/`, não há `os.path.join` platform-fráigl misturado com
  literais). Ex.: [`metahub_input_node.py:20-43`](../metahub_input_node.py#L20-L43),
  [`metadata_utils_impl.py`](../metadata_utils_impl.py) inteiro.
- Busca em todo o repo por `sys.platform`, `os.name`, `platform.system`,
  `ctypes`, `winreg`, `msvcrt`, `os.startfile`, chamadas POSIX-only
  (`os.geteuid`, `fcntl`, `pwd`/`grp`) e `subprocess(..., shell=True)`:
  **zero ocorrências** em qualquer arquivo `.py` do pacote.
- Único ponto assimétrico: [`video_metadata_utils.py:42-51`](../video_metadata_utils.py#L42-L51),
  a lista de fallback de localização do `ffmpeg` (`find_ffmpeg_binary`) só tem
  caminhos hard-coded do Windows (`C:/ffmpeg/bin/ffmpeg.exe` etc.) depois de já
  ter tentado `FFMPEG_PATH` e `shutil.which('ffmpeg')` — que são
  cross-platform e cobrem o caso comum em Linux/macOS (ffmpeg instalado via
  apt/brew fica no PATH). Se não encontrar em nenhum dos dois, o código levanta
  um `RuntimeError` com instrução clara ([`video_metadata_utils.py:317-322`](../video_metadata_utils.py#L317-L322))
  — não trava, não corrompe nada, só falha de forma explícita e igual nas
  três plataformas quando o ffmpeg realmente não está instalado.

Não há evidência de comportamento *diferente* ou incorreto em nenhuma
plataforma — só uma heurística de descoberta de binário um pouco mais fraca
fora do Windows. Isso sustenta o classifier `OS Independent`.

---

## 3. Toca em GPU/CUDA/torch, ou é I/O + hashing puro?

**Resposta: usa `torch`, mas nunca como dependência de GPU/CUDA. Nenhuma chamada
de device placement em lugar nenhum do repo (busca por `.cuda(`, `.to(device`,
`.half()`, `torch.device` — zero ocorrências).**

Dois usos, ambos read-only / opcionais:

1. **Telemetria de GPU, defensiva** — [`metadata_utils_impl.py:196-256`](../metadata_utils_impl.py#L196-L256)
   `collect_gpu_metrics()`: faz `import torch` dentro de um `try/except
   ImportError`, depois só *lê* `torch.cuda.is_available()`,
   `torch.cuda.get_device_name(0)`, `torch.cuda.max_memory_allocated(0)`, e
   equivalente para `torch.backends.mps`. Se CUDA/MPS não existir, cai para
   `"CPU (CUDA not available)"` sem erro. Nunca aloca nada na GPU, nunca move
   um tensor para lá — só introspecciona o que já está rodando.
2. **Criação de tensor CPU para o contrato de tipos do ComfyUI** —
   [`metahub_input_node.py:61-76`](../metahub_input_node.py#L61-L76)
   (`image_to_tensor`, `mask_to_tensor`): usa `torch.from_numpy(array)[None,]`
   para produzir os tipos `IMAGE`/`MASK` que o ComfyUI espera. Isso roda em
   CPU; não há `.to()`/`.cuda()` em seguida.

Fora isso, o node é I/O (leitura/escrita de PNG/JPEG/WebP/GLB, subprocess para
ffmpeg) + hashing (SHA256 de modelos/LoRAs) + string/JSON building. Ele
funciona sem problema em instalações ComfyUI CPU-only.

**Conclusão prática:** declarar `Environment :: GPU :: NVIDIA CUDA` (ou
ROCm/Metal) seria uma claim falsa — o node não exige GPU nenhuma para
funcionar, e isso só polui o filtro de busca de quem procura nodes que
realmente *precisam* de CUDA. Nenhum classifier de acelerador deve ser
adicionado.

---

## 4. Existe API do ComfyUI para estabelecer uma versão mínima verificável?

**Resposta: não, o código não usa nenhuma.**

Os dois únicos pontos de contato com o runtime do ComfyUI além dos hidden
inputs padrão (`PROMPT`, `EXTRA_PNGINFO`, `UNIQUE_ID` — parte estável do
contrato de custom nodes desde as primeiras versões) são:

- `import folder_paths` — [`metadata_utils_impl.py:59`](../metadata_utils_impl.py#L59)
  e [`metadata_utils_impl.py:1091`](../metadata_utils_impl.py#L1091), ambos
  dentro de `try/except`, usados só para localizar pastas de modelos / diretório
  de output. Se falhar, cai para caminhos relativos default.
- `import comfy` / `comfy.__version__` — [`metadata_utils_impl.py:297-299`](../metadata_utils_impl.py#L297-L299),
  também dentro de `try/except`, usado só para *reportar* a versão do ComfyUI
  nos metadados salvos — nunca comparado contra um piso mínimo.

Nenhum desses é usado como *gate* (não há `if comfy_version < X: raise` nem
equivalente), e nenhuma API nova/versionada é chamada — nada que dependa de
uma versão específica do ComfyUI para existir.

**Conclusão prática:** não há piso de versão do ComfyUI provável a partir do
código. Preencher `requires-comfyui` seria um chute. Se o mantenedor já
validou manualmente uma versão mínima em teste real, isso é um fato externo
ao código-fonte (não deriva de auditoria) e pode ser adicionado depois, com
essa evidência explícita.

---

## Princípio aplicado

Cada campo abaixo só foi preenchido onde havia evidência de código
(arquivo:linha) sustentando um limite real. Onde não havia prova, o campo foi
**omitido**, não preenchido "por via das dúvidas".

## `pyproject.toml` proposto (não aplicado — para revisão)

```toml
[project]
name = "imagemetahub-comfyui-save"
description = "Official companion node for Image MetaHub. Saves A1111/Civitai-compatible parameters plus extended Image MetaHub metadata (workflow JSON, SHA256 model hashes, VRAM peak/total, GPU device, generation time, and steps/sec) for reproducibility and benchmarking."
version = "1.1.9"
license = {file = "LICENSE"}
requires-python = ">=3.10"
classifiers = [
    "Operating System :: OS Independent",
]

dependencies = []

[project.urls]
Repository = "https://github.com/LuqP2/ImageMetaHub-ComfyUI-Save"
#  Used by Comfy Registry https://registry.comfy.org
Documentation = "https://github.com/LuqP2/ImageMetaHub-ComfyUI-Save/wiki"
"Bug Tracker" = "https://github.com/LuqP2/ImageMetaHub-ComfyUI-Save/issues"

[tool.comfy]
PublisherId = "image-metahub"
DisplayName = "ImageMetaHub-ComfyUI-Save"
includes = []
# Icon and Banner omitted: no square icon (<=400x400) or 21:9 banner asset
# exists yet. Add them (with real URLs) when they do — an empty string is
# not the same as omitting the field.
# "requires-comfyui" omitted: no code path checks or requires a specific
# ComfyUI version (see docs/registry-compatibility-audit.md, Q4). Add it only
# once a floor version has actually been tested, not as a guess.
```

**Diffs em relação ao atual, e por quê:**

| Campo | Atual | Proposto | Motivo |
|---|---|---|---|
| `requires-python` | ausente | `">=3.10"` | `metahub_input_node.py:69` usa `Path \| None` sem `from __future__ import annotations` |
| `classifiers` | bloco inteiro comentado | só `OS Independent` | nenhum caminho platform-gated encontrado; nenhum classifier de GPU porque o node não exige GPU |
| `Icon` | `""` | omitido | string vazia ≠ campo omitido; não existe asset ainda |
| `requires-comfyui` | comentado | continua omitido | nenhuma API de versão mínima é usada no código |
