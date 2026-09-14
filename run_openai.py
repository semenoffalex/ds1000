import argparse
import gzip
import json
import os
import statistics
import threading
import time
import concurrent.futures as cfuts
from pathlib import Path
from urllib.parse import urlparse

import httpx
from openai import OpenAI
from tqdm import tqdm

DEFAULT_SYSTEM_PROMPT = (
    "Write a short code following the given format and indentation. "
    "Place the executable code between <code> and </code> tags, without any other non-executable things."
)
GPT4O_SYSTEM_PROMPT = "Only provide the code completion needed. Don't repeat the context code."
STOP = ["</code>", "# SOLUTION END"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate DS-1000 answers via an OpenAI-compatible API. "
            "Works with OpenAI and with llama.cpp llama-server."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLAMA_MODEL", "gpt-4o-mini-2024-07-18"),
        help="API model id; also used in the answers filename",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "sk-no-key"),
        help="API key; llama.cpp accepts any non-empty value",
    )
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent requests. Default: 1 with --base-url, 32 for OpenAI",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N problems")
    parser.add_argument(
        "--libraries",
        nargs="+",
        default=None,
        help="Only generate these DS-1000 libraries, e.g. Scipy Sklearn Numpy Pytorch",
    )
    parser.add_argument("--resume", action="store_true", help="Skip ids already present in the answers file")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--top-p", type=float, default=None, dest="top_p")
    parser.add_argument("--top-k", type=int, default=None, dest="top_k")
    parser.add_argument("--min-p", type=float, default=None, dest="min_p")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["low", "medium", "high"],
        help="gpt-oss Harmony reasoning effort (chat_template_kwargs)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Keep Qwen3-style thinking; off by default for local servers",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Answers jsonl path; default data/<model>-answers.jsonl",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Per-problem generation metrics jsonl; default results/<model>-generation.jsonl",
    )
    return parser.parse_args()


def safe_name(model: str) -> str:
    return model.replace("/", "-").replace("\\", "-")


def load_dataset():
    return [json.loads(l) for l in gzip.open("data/ds1000.jsonl.gz", "rt").readlines()]


def load_done_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done.add(int(json.loads(line)["id"]))
    return done


def server_root(base_url: str) -> str:
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def fetch_server_props(base_url: str) -> dict:
    if not base_url:
        return {}
    try:
        r = httpx.get(f"{server_root(base_url)}/props", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def extract_metrics(data: dict, wall_s: float) -> dict:
    usage = data.get("usage") or {}
    timings = data.get("timings") or {}
    details = usage.get("prompt_tokens_details") or {}
    choices = data.get("choices") or [{}]
    finish_reason = choices[0].get("finish_reason") if choices else None
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = timings.get("prompt_n")
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = timings.get("predicted_n")
    cached_tokens = details.get("cached_tokens")
    if cached_tokens is None:
        cached_tokens = timings.get("cache_n")
    return {
        "wall_s": wall_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": cached_tokens,
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_ms": timings.get("predicted_ms"),
        "prompt_tok_s": timings.get("prompt_per_second"),
        "gen_tok_s": timings.get("predicted_per_second"),
        "prompt_ms_per_tok": timings.get("prompt_per_token_ms"),
        "gen_ms_per_tok": timings.get("predicted_per_token_ms"),
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": timings.get("draft_n_accepted"),
        "finish_reason": finish_reason,
    }


def message_content(data: dict) -> str:
    message = ((data.get("choices") or [{}])[0].get("message") or {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if not content and reasoning:
        content = str(reasoning)
    return content


def call_chat_local(base_url, system_prompt, prompt, temperature, model, max_tokens, extra_body, timeout) -> tuple[str, dict]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "n": 1,
        "max_tokens": max_tokens,
    }
    # gpt-oss Harmony analysis often quotes "</code>" from the DS-1000 prompt;
    # llama.cpp would stop there and never emit the final channel.
    if "gpt-oss" not in (model or "").lower():
        payload["stop"] = STOP
    if extra_body:
        payload.update({k: v for k, v in extra_body.items() if k != "chat_template_kwargs"})
        if extra_body.get("chat_template_kwargs"):
            payload["chat_template_kwargs"] = extra_body["chat_template_kwargs"]
    url = base_url.rstrip("/") + "/chat/completions"
    while True:
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload)
            wall_s = time.perf_counter() - t0
            response.raise_for_status()
            data = response.json()
            return message_content(data), extract_metrics(data, wall_s)
        except Exception:
            time.sleep(10)


def call_chat_openai(client, system_prompt, prompt, temperature, model, max_tokens, extra_body) -> tuple[str, dict]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    kwargs = {}
    if extra_body:
        kwargs["extra_body"] = extra_body
    while True:
        t0 = time.perf_counter()
        try:
            create_kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                n=1,
                max_tokens=max_tokens,
                **kwargs,
            )
            if "gpt-oss" not in (model or "").lower():
                create_kwargs["stop"] = STOP
            result = client.chat.completions.create(**create_kwargs)
            wall_s = time.perf_counter() - t0
            data = result.model_dump() if hasattr(result, "model_dump") else {}
            extra = getattr(result, "model_extra", None)
            if extra:
                data.update(extra)
            message = result.choices[0].message
            content = message.content or ""
            reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
            if not content and reasoning:
                content = str(reasoning)
            return content, extract_metrics(data, wall_s)
        except Exception:
            time.sleep(10)


def _mean(values):
    values = [v for v in values if v is not None]
    return (sum(values) / len(values)) if values else None


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _quantile(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def summarize_generation(rows: list[dict], config: dict) -> dict:
    prompt_tokens = [r.get("prompt_tokens") for r in rows]
    completion_tokens = [r.get("completion_tokens") for r in rows]
    gen_tok_s = [r.get("gen_tok_s") for r in rows]
    prompt_tok_s = [r.get("prompt_tok_s") for r in rows]
    wall_s = [r.get("wall_s") for r in rows]
    prompt_ms = [r.get("prompt_ms") for r in rows]
    predicted_ms = [r.get("predicted_ms") for r in rows]
    cached_tokens = [r.get("cached_tokens") for r in rows]
    draft_n = [r.get("draft_n") for r in rows]
    draft_acc = [r.get("draft_n_accepted") for r in rows]
    summary = {
        "model": config.get("model"),
        "n": len(rows),
        "prompt_tokens_sum": sum(v or 0 for v in prompt_tokens),
        "prompt_tokens_mean": _mean(prompt_tokens),
        "completion_tokens_sum": sum(v or 0 for v in completion_tokens),
        "completion_tokens_mean": _mean(completion_tokens),
        "cached_tokens_sum": sum(v or 0 for v in cached_tokens),
        "wall_s_sum": sum(v or 0 for v in wall_s),
        "wall_s_mean": _mean(wall_s),
        "prompt_ms_sum": sum(v or 0 for v in prompt_ms),
        "predicted_ms_sum": sum(v or 0 for v in predicted_ms),
        "prompt_tok_s_mean": _mean(prompt_tok_s),
        "prompt_tok_s_median": _median(prompt_tok_s),
        "gen_tok_s_mean": _mean(gen_tok_s),
        "gen_tok_s_median": _median(gen_tok_s),
        "gen_tok_s_p05": _quantile(gen_tok_s, 0.05),
        "gen_tok_s_p95": _quantile(gen_tok_s, 0.95),
        "draft_n_sum": sum(v or 0 for v in draft_n),
        "draft_n_accepted_sum": sum(v or 0 for v in draft_acc),
        "config": config,
    }
    if summary["draft_n_sum"]:
        summary["draft_accept_rate"] = summary["draft_n_accepted_sum"] / summary["draft_n_sum"]
    return summary


def write_generation_csv(rows: list[dict], path: Path):
    if not rows:
        return
    keys = [
        "id",
        "library",
        "perturbation_type",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "wall_s",
        "prompt_ms",
        "predicted_ms",
        "prompt_tok_s",
        "gen_tok_s",
        "draft_n",
        "draft_n_accepted",
        "finish_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(keys) + "\n")
        for row in sorted(rows, key=lambda r: int(r["id"])):
            values = []
            for key in keys:
                value = row.get(key)
                if value is None:
                    values.append("")
                elif isinstance(value, str):
                    values.append('"' + value.replace('"', '""') + '"')
                else:
                    values.append(str(value))
            f.write(",".join(values) + "\n")


def run_openai(args, ds1000):
    workers = args.workers
    if workers is None:
        workers = 1 if args.base_url else 32

    extra_body = {}
    is_gpt_oss = "gpt-oss" in args.model.lower()
    if args.base_url and not args.thinking and not is_gpt_oss:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if args.reasoning_effort:
        extra_body.setdefault("chat_template_kwargs", {})["reasoning_effort"] = args.reasoning_effort
    if args.top_p is not None:
        extra_body["top_p"] = args.top_p
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.min_p is not None:
        extra_body["min_p"] = args.min_p
    extra_body = extra_body or None

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if "gpt-4o" in args.model:
        system_prompt = GPT4O_SYSTEM_PROMPT

    tag = safe_name(args.model)
    out_path = Path(args.output) if args.output else Path("data") / f"{tag}-answers.jsonl"
    metrics_path = Path(args.metrics) if args.metrics else Path("results") / f"{tag}-generation.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "workers": workers,
        "thinking": bool(args.thinking),
        "timeout": args.timeout,
        "limit": args.limit,
        "libraries": args.libraries,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "reasoning_effort": args.reasoning_effort,
        "stop": None if is_gpt_oss else STOP,
        "server_props": fetch_server_props(args.base_url) if args.base_url else {},
    }
    with (Path("results") / f"{tag}-run-config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    openai_client = None
    if not args.base_url:
        openai_client = OpenAI(api_key=args.api_key or "sk-no-key", timeout=args.timeout)

    problems = ds1000
    if args.libraries:
        libs = set(args.libraries)
        problems = [p for p in problems if p["metadata"]["library"] in libs]
    if args.limit:
        problems = problems[: args.limit]
    if args.resume:
        done = load_done_ids(out_path)
    else:
        done = set()
        if out_path.exists():
            out_path.unlink()
        if metrics_path.exists():
            metrics_path.unlink()

    todo = []
    for p in problems:
        pid = int(p["metadata"]["problem_id"])
        if pid not in done:
            todo.append((pid, p["prompt"]))

    write_lock = threading.Lock()

    def run_func(prompt, pid):
        if args.base_url:
            code, metrics = call_chat_local(
                args.base_url,
                system_prompt,
                prompt,
                args.temperature,
                args.model,
                args.max_tokens,
                extra_body,
                args.timeout,
            )
        else:
            code, metrics = call_chat_openai(
                openai_client,
                system_prompt,
                prompt,
                args.temperature,
                args.model,
                args.max_tokens,
                extra_body,
            )
        meta = ds1000[pid]["metadata"]
        row = {"id": pid, "code": code, "metadata": meta}
        metric_row = {
            "id": pid,
            "library": meta.get("library"),
            "perturbation_type": meta.get("perturbation_type"),
            **metrics,
        }
        with write_lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metric_row, ensure_ascii=False) + "\n")
        return pid

    if todo:
        with cfuts.ThreadPoolExecutor(max_workers=workers) as executor:
            futs = [executor.submit(run_func, prompt, pid) for pid, prompt in todo]
            for fut in tqdm(cfuts.as_completed(futs), total=len(futs)):
                fut.result()
    else:
        print(f"nothing to generate, answers already complete: {out_path}")

    rows = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    summary = summarize_generation(rows, config)
    summary_path = Path("results") / f"{tag}-generation-summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_generation_csv(rows, Path("results") / f"{tag}-generation.csv")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    args = parse_args()
    ds1000 = load_dataset()
    run_openai(args, ds1000)
