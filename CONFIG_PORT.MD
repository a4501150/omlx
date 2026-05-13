# CONFIG_PORT

Portable record of per-machine setup for the local Qwen3.6-35B-A3B abliterated VLM served through oMLX. Copy this file plus the referenced config snippets to a new Mac to reproduce the setup.

Scope:
- Model alias: `Qwen3.6-35B-A3B-Abliterated-Heretic-oQ8-fp16`
- Source: `Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic-BF16` on Hugging Face
- Quant: oQ+ at oQ8 (vision encoder kept at fp16, language stack at ~8.6 bpw)
- Context: YaRN 2× → 524288 tokens
- Sampling: Qwen's "thinking, precise coding/WebDev" recipe
- Thinking: native adaptive, no hard cap; `preserve_thinking=true`
- API: Anthropic cache-hit accounting patched into oMLX (see §5)

---

## 1. Prerequisites per machine

- Apple Silicon Mac with ≥96 GB unified memory (128 GB recommended for oQ+ at 35B-A3B).
- ≥150 GB free disk (BF16 source ~70 GB + oQ8 output ~40 GB + scratch).
- oMLX checked out from this private fork and installed editable:
  ```bash
  pip install -e ".[dev]"
  ```

## 2. Download the BF16 source

```bash
hf download Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic-BF16 \
  --local-dir ~/models/Qwen3.6-35B-A3B-Abliterated-Heretic-BF16
```

Verify after download: `config.json` has `vision_config`, `processor_config.json` is present, and 21 safetensors shards totaling ~70 GB.

## 3. Quantize with oQ+ at oQ8

Via admin dashboard: open oMLX → Quantization page → pick the BF16 folder → select **oQ+ level 8** → run. Output appears as `Qwen3.6-35B-A3B-Abliterated-Heretic-oQ8-fp16` in `--model-dir`.

oQ+ at oQ8 mandatory protections (from `docs/oQ_Quantization.md`):
- `lm_head`: 8-bit
- MoE router, `shared_expert_gate`: 8-bit
- Vision encoder: fp16 (preserved for VLM use)
- SSM state params: fp32

Expected output size: ~38–40 GB across 8 shards.

With the fix in `omlx/oq.py` (see §7(c)), fresh oQ runs on Qwen3.6 HF BF16 write tensor keys with the correct `vision_tower.*` naming from the start. No rename step is needed, and the checkpoint loads cleanly as a VLM via `mlx-vlm`. If you're porting an oQ checkpoint that was **built before the oq.py fix**, see §7(d) for the one-time in-place tensor rename.

## 4. Apply YaRN 2× to `config.json`

**File**: `<model-dir>/Qwen3.6-35B-A3B-Abliterated-Heretic-oQ8-fp16/config.json`

Patch `text_config.rope_parameters` and `text_config.max_position_embeddings`. Reproducible script:

```python
# apply_yarn.py
import json
from pathlib import Path

MODEL_DIR = Path.home() / "models" / "Qwen3.6-35B-A3B-Abliterated-Heretic-oQ8-fp16"
p = MODEL_DIR / "config.json"
c = json.loads(p.read_text())
c["text_config"]["max_position_embeddings"] = 524288
c["text_config"]["rope_parameters"] = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "rope_type": "yarn",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "factor": 2.0,
    "original_max_position_embeddings": 262144,
}
p.write_text(json.dumps(c, indent=2, ensure_ascii=False))
print("YaRN 2× applied")
```

Factor choice: 2.0 is the safe default for interleaved-MRoPE. Bump to 3.0 → ~786K or 4.0 → ~1M only if the workload demands it and after testing long-context quality. Static YaRN degrades short-input quality, so enable only when >262K context is actually needed.

## 5. oMLX per-model settings — `~/.omlx/model_settings.json`

**File**: `~/.omlx/model_settings.json`

Merge (or create) this entry under `models`:

```json
{
  "version": 1,
  "models": {
    "Qwen3.6-35B-A3B-Abliterated-Heretic-oQ8-fp16": {
      "force_sampling": false,
      "thinking_budget_enabled": false,
      "turboquant_kv_enabled": false,
      "turboquant_kv_bits": 4.0,
      "turboquant_skip_last": true,
      "specprefill_enabled": false,
      "dflash_enabled": false,
      "is_pinned": false,
      "is_default": false,
      "temperature": 0.6,
      "top_p": 0.95,
      "top_k": 20,
      "min_p": 0.0,
      "presence_penalty": 0.0,
      "repetition_penalty": 1.0,
      "max_context_window": 524288,
      "max_tokens": 81920,
      "chat_template_kwargs": {
        "preserve_thinking": true,
        "enable_thinking": true
      }
    }
  }
}
```

Rationale for each field:
- Sampling set (temperature 0.6 etc.) — Qwen's official **thinking + precise coding/WebDev** recipe. For general thinking, swap to `temperature=1.0, presence_penalty=1.5`. For non-thinking, swap to `temperature=0.7, top_p=0.8, presence_penalty=1.5` and set `enable_thinking=false`.
- `max_context_window=524288` — matches the YaRN 2× scaled context.
- `max_tokens=81920` — Qwen's own reference for heavy reasoning / agentic coding. Drop to 32768 for lighter chat use.
- `chat_template_kwargs.preserve_thinking=true` — new in Qwen3.6, keeps prior turns' `<think>` blocks so multi-turn agent loops don't re-reason from scratch.
- `chat_template_kwargs.enable_thinking=true` — explicit; Qwen3.6 defaults to thinking but setting it explicit survives future template changes.

Sampling is loaded per request (no restart needed). YaRN requires a model unload + reload to take effect.

## 6. oMLX code patch — Anthropic cache-hit accounting

**Files**: `omlx/api/anthropic_utils.py`, `omlx/server.py`.

**Why**: oMLX's `BlockAwarePrefixCache` already caches prefixes automatically, but the Anthropic response builder was not wiring `output.cached_tokens` into `AnthropicUsage`. Claude Code always saw `cache_read_input_tokens: 0`, even when real prefix hits occurred.

**Changes**:

1. `convert_internal_to_anthropic_response` gains `cached_tokens: int = 0`. Populates `AnthropicUsage.cache_read_input_tokens` and subtracts from `input_tokens` (Anthropic semantics: `input_tokens` = non-cached input).
2. `create_message_start_event` gains `cache_read_input_tokens` / `cache_creation_input_tokens` params (default 0) so the streaming start event matches Anthropic's API shape.
3. `create_message_delta_event` gains `cached_tokens: int = 0`. Emits `cache_read_input_tokens` and adjusts `input_tokens` accordingly.
4. Two call sites in `server.py` (streaming `_build_anthropic_stream` near the `create_message_delta_event` call, and non-streaming `_build_anthropic_message` at the `convert_internal_to_anthropic_response` call) pass `cached_tokens=scale_anthropic_tokens(output.cached_tokens or 0, request.model)`.

Semantic mapping used (creation vs read):
- oMLX does not track whether a prefix was first-time written during this request, so `cache_creation_input_tokens` is always 0.
- All cached tokens are reported as `cache_read_input_tokens`.
- Claude Code's arithmetic still reconciles: `prompt_tokens = input_tokens + cache_read_input_tokens + cache_creation_input_tokens`.

To port to a fresh machine: pull the branch of this fork that contains the patch, or cherry-pick the commit. The changes are ~20 lines across two files.

## 7. oMLX code patch — Qwen3.6 VLM loading fixes

**Why**: Qwen3.6's HF checkpoint nests the ViT weights inside the language-model submodule as `model.language_model.visual.*`, instead of the flat `model.visual.*` layout other Qwen VLMs use. mlx-vlm's `qwen3_5_moe/qwen3_5_moe.py::Model.sanitize` has an `elif` ordering bug that mis-routes this case — upstream bug, not a fork-specific issue. The nested keys end up at `language_model.model.visual.*` where the instantiated `Qwen3_5MoeForConditionalGeneration` doesn't have them, so 333 visual tensors get silently dropped on load. Symptom in the admin UI is a popup: *"Received 333 parameters not in model: language_model.model.visual.blocks.0.attn.proj.bias, …"*. Model still works as text-only; image input produces garbage.

Verified this exists on both the pinned mlx-vlm commit (`3472132…` in `pyproject.toml`) and upstream `main` branch. The correct upstream fix is a 2-line `elif → if` reorder. We carry a local patch until that lands.

### What's in the fork

**(a) Copy `processor_config.json` from the BF16 source** — per machine

oQ doesn't carry `processor_config.json` into the quantized output. mlx-vlm needs it for image preprocessing. One-time per machine after quantization:

```bash
cp ~/models/Qwen3.6-35B-A3B-Abliterated-Heretic-BF16/processor_config.json \
   ~/models/Qwen3.6-35B-A3B-Abliterated-Heretic-oQ8-fp16/
```

Also copy `preprocessor_config.json` and `video_preprocessor_config.json` if they exist in the BF16 dir.

**(b) Sanitize patch applied at VLM load time** — `omlx/patches/qwen3_6_nested_visual.py`

Wraps mlx-vlm's `qwen3_5_moe Model.sanitize` to remap `language_model.model.visual.*` → `vision_tower.*` after the upstream sanitize runs. Idempotent; guarded via `inspect.getsource()` to auto-skip if upstream ever adds the rule.

Hook site: `omlx/engine/vlm.py::VLMBatchedEngine.start()::_load_vlm_sync()`. The patch is applied right before `vlm_load(...)`, inside the MLX executor thread. Applies once (class-level); no-op for non-qwen3_5_moe VLMs. Needed to correctly **load** Qwen3.6 HF BF16 checkpoints.

**(c) Sanitize patch applied at oQ quantization time** — `omlx/oq.py`

Same patch, different call site. `_build_model_sanitizer` in oq.py now calls `apply_qwen3_6_nested_visual_patch()` before invoking mlx-vlm's `Model.sanitize` on VLM inputs. Without this, oQ would read Qwen3.6 HF tensors via mlx-vlm's buggy sanitize, quantize them with the wrong `language_model.model.visual.*` keys baked in, and write those wrong keys to the output safetensors. On reload mlx-vlm skips sanitize for mlx-format files (`if not is_mlx_format`), so the bad keys never get remapped and the resulting checkpoint fails to load vision.

With (c) in place, **fresh oQ runs on Qwen3.6 HF BF16 produce safetensors with correct `vision_tower.*` names from the start** — no post-rename needed.

**(d) ONE-TIME in-place tensor rename** — only for oQ checkpoints built before (c) landed

If you have an oQ output that was produced **before** the oq.py fix was in place (symptom: `Received 333 parameters not in model: language_model.model.visual.blocks.0.*` on load), run this one-time rename over the existing shards instead of re-quantizing from BF16:

```python
# rename_nested_visual.py
import json
from pathlib import Path
import mlx.core as mx

MODEL_DIR = Path.home() / "models" / "Qwen3.6-35B-A3B-Abliterated-Heretic-oQ8-fp16"
NESTED = "language_model.model.visual."
TARGET = "vision_tower."

# 1. Rewrite model.safetensors.index.json
idx_path = MODEL_DIR / "model.safetensors.index.json"
idx = json.loads(idx_path.read_text())
idx["weight_map"] = {
    (TARGET + k[len(NESTED):]) if k.startswith(NESTED) else k: v
    for k, v in idx["weight_map"].items()
}
idx_path.write_text(json.dumps(idx, indent=2))

# 2. Rewrite each shard in place (atomic via temp + replace)
for shard in sorted(MODEL_DIR.glob("model-*-of-*.safetensors")):
    tensors = mx.load(str(shard))
    if not any(k.startswith(NESTED) for k in tensors):
        continue
    renamed = {
        (TARGET + k[len(NESTED):]) if k.startswith(NESTED) else k: v
        for k, v in tensors.items()
    }
    tmp = shard.with_name(shard.stem + ".tmp.safetensors")
    mx.save_safetensors(str(tmp), renamed, metadata={"format": "mlx"})
    tmp.replace(shard)
    print(f"renamed {shard.name}")
```

Run once per affected oQ checkpoint. On a 35B-A3B MoE all 333 visual tensors live in shard 1, so only one shard is rewritten (~5 GB) — takes a few seconds.

Once (c) is in the oQ source, new quant runs don't need this step.

### Verification

After the restart / model reload, look for this in the server log:

```
qwen3_6_nested_visual: patched mlx_vlm.qwen3_5_moe Model.sanitize
qwen3_6_nested_visual: remapped 333 tensor keys 'language_model.model.visual.*' -> 'vision_tower.*'
```

And the "Received 333 parameters not in model" popup should not appear.

### Upstream fix (pending)

Two-line reorder in mlx-vlm's `Model.sanitize`:

```python
# correct order — nested-visual case must come first
if "model.language_model.visual" in key:
    key = key.replace("model.language_model.visual", "vision_tower")
elif "model.language_model" in key:
    key = key.replace("model.language_model", "language_model.model")
elif "model.visual" in key:
    key = key.replace("model.visual", "vision_tower")
```

When that lands upstream and this fork's mlx-vlm pin is bumped, `qwen3_6_nested_visual.py` auto-deactivates.

## 8. Recommended launch flags

```bash
omlx serve \
  --model-dir ~/models \
  --paged-ssd-cache-dir ~/.omlx/cache
```

The SSD cold tier matters a lot for Claude Code workloads — the large system-prompt + tool-schema prefix persists across server restarts, so cold starts don't re-prefill from scratch.

## 9. Verification checklist

1. **Model loads**: appears in admin Models page, classified as VLM, vision encoder listed at fp16.
2. **Sanitize patch fires** (per §7): server log shows `qwen3_6_nested_visual: remapped 333 tensor keys …`, and the "Received 333 parameters not in model" popup is gone.
3. **CTX window shows 524288**: admin → Model Settings → Load Defaults.
4. **Sampling overrides show**: admin → Model Settings → fields populated with 0.6 / 0.95 / 20 / 0 / 0 / 1.0.
5. **Vision works end-to-end**: send a multimodal request with a small image; response references image content.
6. **YaRN scaling is live**: send a prompt longer than 262144 tokens; server accepts without truncation up to 524288.
7. **Prefix cache hits register in Anthropic usage**: send two identical long prompts in a row; the second response's `usage.cache_read_input_tokens` is > 0.
8. **`preserve_thinking` carries forward**: in a multi-turn conversation, inspect the rendered prompt and confirm prior `<think>` blocks are included.

## 10. Known tradeoffs

- **Static YaRN tax**: every request (including short ones) goes through 2× scaled RoPE. If a mixed workload is painful, run a second model-settings profile without YaRN and route short requests there.
- **`preserve_thinking` grows context**: `<think>` blocks accumulate across turns. Watch RAM in long agent loops; YaRN 2× + long thinking can fill the KV budget.
- **`cache_creation_input_tokens` is always 0 in responses**: oMLX doesn't distinguish creation from read (see §6). Claude Code's cost arithmetic still reconciles, but don't interpret the 0 as "no cache write" — it just means we don't track that axis.
- **Upstream mlx-vlm patch is pending** (see §7): until the `elif → if` fix lands upstream and this fork bumps its pin, the local sanitize wrapper is mandatory for Qwen3.6 VLM loading. Our patch is guarded and self-deactivates once upstream ships it.
- **Heretic abliteration KLD = 0.0107** on the BF16 source, measured against the original Qwen3.6 base on harmless prompts. Very gentle — near-indistinguishable on general tasks. Does not cover long-context, multilingual, or multimodal drift.

## 11. Useful references

- Qwen3.6-35B-A3B model card: <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>
- Heretic abliteration BF16 source: <https://huggingface.co/Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic-BF16>
- oQ quantization documentation: `docs/oQ_Quantization.md` in this repo
- vLLM YaRN recipe for Qwen3.5/3.6 (reference only): <https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html>
