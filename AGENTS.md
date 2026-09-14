## Learned User Preferences
- Communicate in Russian unless the user writes in English.
- Use this repo to evaluate local GGUF models on DS-1000 via llama.cpp, then compare models rather than running a one-off smoke test.
- Persist generation speed, token counts, and DS-1000 library/overall stats so they can be charted; rebuild comparison artifacts when new models are added.
- Reuse the same llama.cpp settings family for dense 27B models; for gpt-oss follow Unsloth’s recommended server flags.
- When asked to run a subset, generate and evaluate only the named DS-1000 libraries (e.g. Scipy, Sklearn, Numpy, Pytorch).

## Learned Workspace Facts
- Official DS-1000 tree with local harness: `run_openai.py` talks to llama-server’s OpenAI-compatible `/v1`, `test_ds1000.py` executes generated code; Python 3.10 lives in `.venv`.
- Preferred llama-server for comparison runs is `C:\Users\user\.unsloth\llama.cpp\build\bin\Release\llama-server.exe` on port 8080 with context 65536, `--fit on`, quantized KV (`q4_0`), `--jinja`, and thinking disabled.
- Per-model metrics land under `results/` (`*-problems.csv`, `*-generation.csv`, `*-eval.json`, `*-run-config.json`); cross-model tables are `results/comparison-summary.csv` and `results/comparison-by-library.csv`.
- Windows eval must not nest `ProcessPoolExecutor` with `multiprocessing.Process`; use threads plus a top-level pickleable execution target, and do not rely on SIGALRM timeouts.
- GPU VRAM is 16 GB, so GGUFs larger than that need `--fit` (partial offload) and sequential single-model runs.
- Comparison set includes Qwen3.8-27B GSQ-RCO and Ridge, Qwen3-Coder-30B-A3B, Qwen3.6-27B MTP, Qwen3.6-28B-REAP20, Qwen3.6-35B-A3B, Qwen3.8-27B UD-Q4_K_M, plus gpt-oss-20b on a library subset.
