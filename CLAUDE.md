# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

oMLX is an OpenAI/Anthropic-compatible LLM inference server for Apple Silicon built on top of Apple's MLX stack. It supports LLMs, VLMs, embeddings, rerankers, and audio (STT/TTS/STS) in a single process, with continuous batching, tiered (RAM + SSD) KV caching, and a FastAPI admin dashboard. There is also a native PyObjC menubar app packaged via `venvstacks` in `packaging/`.

## Commands

Install (editable, Apple Silicon / Python 3.10+):

```bash
pip install -e ".[dev]"              # core + dev tools (no audio extras)

# With the audio extra, pip needs the constraints file to override
# mlx-audio's transitive mlx-lm==0.31.1 pin (see pyproject.toml `[tool.uv]`
# comment and the top of `constraints.txt`):
pip install -c constraints.txt -e ".[dev,mcp,audio,grammar,modelscope]"

# Or use uv — it honors `[tool.uv] override-dependencies` directly, no
# constraints file needed:
uv sync --all-extras
```

Run the server:

```bash
omlx serve --model-dir ~/models                 # basic
omlx serve --paged-ssd-cache-dir ~/.omlx/cache  # enable SSD cold tier
omlx serve --mcp-config mcp.json                # MCP tools
```

Testing (pytest defaults to `-m "not slow and not integration"` via `pytest.ini`):

```bash
pytest                                          # fast unit tests only
pytest tests/test_scheduler.py -v               # single file
pytest tests/test_scheduler.py::TestClass::test_name  # single test
pytest -m slow                                  # tests that load real models
pytest -m integration                           # requires a running server
```

Linting / formatting (configured in `pyproject.toml`):

```bash
black omlx tests
ruff check omlx tests
mypy omlx
```

Admin Tailwind CSS rebuild (no Node.js required — downloads the Tailwind standalone CLI):

```bash
python omlx/admin/build_css.py              # one-shot minified build
python omlx/admin/build_css.py --watch      # watch mode
```

i18n normalization after editing `omlx/admin/i18n/en.json`:

```bash
python scripts/normalize_i18n.py
```

macOS app bundle (requires Python 3.11+ and `pip install venvstacks`):

```bash
cd packaging
python build.py                 # full build: venvstacks + app bundle + DMG
python build.py --skip-venv     # reuse existing venvstacks build
python build.py --dmg-only      # repackage DMG only
```

## High-level architecture

Entry point is `omlx.cli:main` (`omlx/cli.py`). `omlx serve` calls `init_settings()` → `init_server()` → `uvicorn.run(app)`. The FastAPI `app` lives in `omlx/server.py`; `/admin` routes in `omlx/admin/routes.py` are mounted on the same app.

### Request flow

```
FastAPI route (server.py / admin/routes.py / api/*)
  └─ EnginePool.get_or_load(model_id)          # omlx/engine_pool.py
      └─ engine (BatchedEngine | VLMBatchedEngine | Embedding | Reranker | STT/TTS/STS | DFlash)
          └─ Scheduler (omlx/scheduler.py) ──▶ mlx-lm BatchGenerator
              └─ Cache stack (omlx/cache/*)
```

`EnginePool` owns all loaded models. Loading is gated by `max_model_memory` (LRU-evict before loading a new one) and the global `ProcessMemoryEnforcer` (`omlx/process_memory_enforcer.py`) which also honors per-model TTL and pinning. Settings that apply per-model (alias, sampling defaults, chat template kwargs, TTL, type override) live in `omlx/model_settings.py` and persist to `~/.omlx/model_settings.json`.

### The single-threaded MLX executor (critical)

`omlx/engine_core.py::get_mlx_executor()` returns a process-wide `ThreadPoolExecutor(max_workers=1)`. **All** MLX GPU ops — across all engines, all models — must run inside this executor. Reason: mlx-lm's `BatchGenerator` tags arrays with a module-level `generation_stream` (a `mx.Stream(gpu)`), and Metal streams are bound to the thread that created them. Running GPU ops on another thread triggers "There is no Stream(gpu, 0) in current thread" or Metal command-buffer races/kernel panics.

The executor's `initializer=_init_mlx_thread` creates a new stream inside the executor and rebinds `mlx_lm.generate.generation_stream` and `omlx.scheduler.generation_stream` to it. Keep this invariant whenever adding new engines or GPU code paths.

Also note `omlx/scheduler.py::_sync_and_clear_cache()`: any call to `mx.clear_cache()` must be preceded by `mx.synchronize(generation_stream)` + `mx.synchronize()` (default stream). Without the sync, freed Metal buffers that are still referenced by in-flight command buffers cause M3/M4 kernel panics — see issues #300, #520, #888 referenced in the code.

### Cache stack (`omlx/cache/`)

- `paged_cache.py` — GPU-side block-based KV cache with copy-on-write and prefix sharing, inspired by vLLM.
- `prefix_cache.py::BlockAwarePrefixCache` — prefix-hash → blocks index used by both GPU and SSD tiers.
- `paged_ssd_cache.py` — cold tier: blocks persisted to SSD as safetensors, restored on matching prefix even after server restart.
- `hybrid_cache.py` + `type_handlers.py` + `type_registry.py` — registry of cache type handlers (`KVCache`, `RotatingKVCache`, `ArraysCache`, `CacheList`) so the scheduler can work with hybrid attention models (e.g. Qwen3.5 GatedDeltaNet, models with sliding-window + global attention mixes).
- `vision_feature_cache.py` — per-image vision-feature cache for VLMs (reusable across turns).
- `boundary_snapshot_store.py` — serializes non-KV internal state (e.g. linear-attention running state) at block boundaries so prefix hits are bit-exact across tiers.
- `tiered_manager.py`, `recovery.py`, `factory.py` — coordination, startup recovery, and construction of the cache objects the scheduler consumes.

### API layer (`omlx/api/`)

- `openai_models.py`, `anthropic_models.py`, `embedding_models.py`, `rerank_models.py`, `responses_models.py`, `audio_models.py` — Pydantic request/response schemas.
- `anthropic_utils.py`, `responses_utils.py`, `utils.py` — converters between wire formats and the internal `Request` / streaming events.
- `tool_calling.py` — per-family tool-call parsers (Qwen JSON / Qwen3.5 XML / Gemma `<start_function_call>` / GLM XML / MiniMax / Mistral `[TOOL_CALLS]` / Kimi / Longcat). Streaming suppresses tool-call markup in visible text and emits structured tool calls after the turn completes.
- `grammar.py`, `thinking.py` — JSON-schema / grammar-constrained decoding (requires `[grammar]` extra, pulls in torch) and reasoning-token extraction.
- `adapters/`, plus top-level `omlx/adapter/` (`harmony.py`, `gemma4.py`, `output_parser.py`) — protocol-specific output parsing (e.g. gpt-oss Harmony, Gemma 4).
- `mcp_routes.py` + `omlx/mcp/` — MCP client, executor, and manager when the optional `mcp` dependency is installed.

### Admin dashboard

`omlx/admin/routes.py` is a single large FastAPI router with session-cookie auth (`auth.py`), vendored JS/CSS (`static/`, `vendor_deps.py`), i18n JSON bundles for en/ja/ko/zh/zh-TW (`i18n/`), and Jinja templates under `templates/` + `templates/dashboard/`. Heavy subsystems it exposes:

- `hf_downloader.py` / `ms_downloader.py` — HuggingFace / ModelScope downloads with progress streaming.
- `benchmark.py`, `accuracy_benchmark.py` — PP/TG benchmarks and lm-eval-style accuracy runs (datasets in `omlx/eval/`).
- `oq_manager.py` — UI over `omlx/oq.py` (mixed-precision quantization producing standard mlx-lm safetensors; calibration bundled as `omlx/oq_calibration_data.json`).

### Model discovery

`omlx/model_discovery.py` walks `--model-dir` (supports two-level layouts like `mlx-community/name/`), reads each `config.json`, and classifies as `llm | vlm | embedding | reranker | audio_stt | audio_tts | audio_sts`. VLM detection uses both `model_type` (`VLM_MODEL_TYPES`) and `architectures` (`VLM_ARCHITECTURES`); admins can override per-model from the dashboard (persisted to `model_settings.json`).

### Patches (`omlx/patches/`)

Runtime monkey-patches applied at server startup:

- `index_cache.py` — augments mlx-lm caches with prefix-block indexing.
- `specprefill.py` — speculative prefill.
- `turboquant_attention.py`, `gated_delta_advance.py` — model-specific kernels / fixes (e.g. Qwen3.5 GatedDeltaNet).

When touching any of these, remember they run inside the MLX executor thread and must not be imported or invoked from other threads.

### Integrations (`omlx/integrations/`)

Small adapters that write config files for external coding tools (`codex`, `opencode`, `openclaw`, `pi`) pointing them at the running oMLX server. Exposed via `omlx launch <tool>` and the admin dashboard.

## Dependency gotchas

- `mlx-lm`, `mlx-vlm`, `mlx-embeddings`, `mlx-audio`, and `dflash-mlx` are pinned to specific git commits in `pyproject.toml` and `packaging/venvstacks.toml`. Keep both files in sync.
- `pyproject.toml` has a `[tool.uv] override-dependencies` entry that forces the same `mlx-lm` commit under `uv sync`, overriding transitive pins (e.g. `mlx-audio` wants `mlx-lm==0.31.1`).
- `transformers` is pinned `<5.4.0` because 5.4.0 made `Qwen2VLImageProcessor` require torch (issue #431). Do not bump without testing VLMs.
- `grammar` extra pulls in torch (~2GB). Keep it optional; tests guarded by `HAS_GRAMMAR` / import-try blocks.
- `mcp`, `modelscope`, `audio` are optional extras — guard code that imports them behind availability checks (see how `omlx/cache/paged_ssd_cache.py` is imported in `scheduler.py`).

## Settings precedence

CLI flags > environment (`OMLX_*`, `HF_ENDPOINT`, proxies) > `~/.omlx/settings.json` > built-in defaults. `omlx serve` with any non-None CLI override persists the overrides back to `settings.json`. Per-model overrides live separately in `~/.omlx/model_settings.json` and are editable only via the admin panel.

## Testing conventions

- Fast tests run without MLX GPU work (see `tests/conftest.py` for `MockTokenizer`, `MockModel`, `MockModelConfig`). Prefer these.
- Anything that loads a real `.safetensors` model → mark `@pytest.mark.slow`.
- Anything that needs `uvicorn`-backed HTTP → `tests/integration/*` and `@pytest.mark.integration`.
- `asyncio_mode = auto` — async tests don't need `@pytest.mark.asyncio`.
- Test file naming mirrors source: `omlx/<module>.py` → `tests/test_<module>.py`.

## License header

All new Python source files should start with:

```python
# SPDX-License-Identifier: Apache-2.0
```

## Local setup notes

Private-fork-specific setup for the Qwen3.6-35B-A3B abliterated VLM — model sourcing, oQ+ quantization, YaRN context extension, per-model `model_settings.json` overrides, and the interim mlx-vlm sanitize patch for Qwen3.6's nested visual layout — is documented in [`CONFIG_PORT.MD`](./CONFIG_PORT.MD). Copy that file + the referenced config snippets to reproduce the setup on a new machine.
