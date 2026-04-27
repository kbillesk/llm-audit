#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    tmp_path = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _backoff_s(attempt_idx: int) -> float:
    # attempt_idx: 0,1,2,... ; yields 0.5,1,2,4... (capped)
    return min(10.0, 0.5 * (2**attempt_idx))


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    endpoint: str
    timeout_s: float
    retries: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    options: dict[str, Any]
    system: str | None


@dataclass(frozen=True)
class RunConfig:
    num_of_q_repeats: int
    stream: bool
    parallelism: int
    output_json_path: str
    include_registry_fields: list[str]


@dataclass(frozen=True)
class AppConfig:
    ollama: OllamaConfig
    model: ModelConfig
    run: RunConfig

    raw: dict[str, Any]


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required key '{key}' in {ctx}")
    return d[key]


def _as_int(x: Any, ctx: str) -> int:
    if isinstance(x, bool):
        raise ValueError(f"Expected int in {ctx}, got bool")
    if not isinstance(x, int):
        raise ValueError(f"Expected int in {ctx}, got {type(x).__name__}")
    return x


def _as_str(x: Any, ctx: str) -> str:
    if not isinstance(x, str) or not x.strip():
        raise ValueError(f"Expected non-empty string in {ctx}")
    return x


def _as_bool(x: Any, ctx: str) -> bool:
    if not isinstance(x, bool):
        raise ValueError(f"Expected bool in {ctx}, got {type(x).__name__}")
    return x


def _as_list_of_str(x: Any, ctx: str) -> list[str]:
    if x is None:
        return []
    if not isinstance(x, list) or any(not isinstance(i, str) for i in x):
        raise ValueError(f"Expected list[str] in {ctx}")
    return x


def load_config(path: Path) -> AppConfig:
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping at top-level")

    ollama_raw = _require(raw, "ollama", "config")
    if not isinstance(ollama_raw, dict):
        raise ValueError("config.ollama must be a mapping")
    model_raw = _require(raw, "model", "config")
    if not isinstance(model_raw, dict):
        raise ValueError("config.model must be a mapping")
    run_raw = _require(raw, "run", "config")
    if not isinstance(run_raw, dict):
        raise ValueError("config.run must be a mapping")

    ollama = OllamaConfig(
        base_url=_as_str(_require(ollama_raw, "base_url", "config.ollama"), "config.ollama.base_url"),
        endpoint=_as_str(_require(ollama_raw, "endpoint", "config.ollama"), "config.ollama.endpoint"),
        timeout_s=float(_require(ollama_raw, "timeout_s", "config.ollama")),
        retries=_as_int(_require(ollama_raw, "retries", "config.ollama"), "config.ollama.retries"),
    )

    options = model_raw.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError("config.model.options must be a mapping")

    # Convenience alias: allow setting a max output length without knowing Ollama's option name.
    # Ollama uses `options.num_predict` to cap the number of tokens generated.
    max_out = model_raw.get("max_output_tokens", None)
    if max_out is not None:
        max_out_int = _as_int(max_out, "config.model.max_output_tokens")
        options = dict(options)
        options.setdefault("num_predict", max_out_int)

    model = ModelConfig(
        name=_as_str(_require(model_raw, "name", "config.model"), "config.model.name"),
        options=options,
        system=_as_str(model_raw["system"], "config.model.system") if "system" in model_raw else None,
    )

    run = RunConfig(
        num_of_q_repeats=_as_int(_require(run_raw, "num_of_q_repeats", "config.run"), "config.run.num_of_q_repeats"),
        stream=_as_bool(_require(run_raw, "stream", "config.run"), "config.run.stream"),
        parallelism=_as_int(run_raw.get("parallelism", 1), "config.run.parallelism"),
        output_json_path=_as_str(_require(run_raw, "output_json_path", "config.run"), "config.run.output_json_path"),
        include_registry_fields=_as_list_of_str(run_raw.get("include_registry_fields"), "config.run.include_registry_fields"),
    )

    if run.num_of_q_repeats < 1:
        raise ValueError("config.run.num_of_q_repeats must be >= 1")
    if run.parallelism < 1:
        raise ValueError("config.run.parallelism must be >= 1")

    return AppConfig(ollama=ollama, model=model, run=run, raw=raw)


def load_prompt_registry(path: Path) -> list[dict[str, Any]]:
    data = _load_yaml(path)
    if not isinstance(data, list):
        raise ValueError("prompt_registry.yaml must be a YAML list")
    prompts: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"prompt_registry entry #{idx} must be a mapping")
        prompt_id = item.get("prompt_id")
        text = item.get("text")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"prompt_registry entry #{idx} missing/invalid prompt_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"prompt_registry entry #{idx} missing/invalid text for prompt_id={prompt_id!r}")
        prompts.append(item)
    return prompts


def _ollama_generate(
    *,
    session: requests.Session,
    url: str,
    model: str,
    system: str | None,
    prompt: str,
    stream: bool,
    options: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": stream,
    }
    if payload.get("system") is None:
        payload.pop("system", None)
    if options:
        payload["options"] = options
    r = session.post(url, json=payload, timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:500]}")
    try:
        return r.json()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse Ollama JSON response: {e}") from e


def _extract_response_text(resp: dict[str, Any]) -> str:
    # /api/generate returns 'response' for non-streaming
    text = resp.get("response")
    if isinstance(text, str):
        return text
    return ""


def _strip_to_last_full_sentence(text: str) -> str:
    """
    Strip whitespace and drop any trailing unfinished sentence so the
    returned text always ends with '.' (or becomes empty).
    """
    s = text.strip()
    if not s:
        return ""
    last_dot = s.rfind(".")
    if last_dot < 0:
        return ""
    return s[: last_dot + 1].rstrip()


def _filename_safe_token(s: str) -> str:
    # Keep filenames portable across common filesystems.
    # (Ollama model names often include characters like ':' and '/'.)
    out = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in s.strip())
    out = out.strip("._-")
    return out or "unknown"


def render_output_path(template: str, *, run_id: str, started_at: str, model_name: str) -> Path:
    # Ensure filename timestamps are whole-second precision and portable.
    # Example: 2026-04-27T09:58:50+00:00 -> 20260427T095850Z
    t = dt.datetime.fromisoformat(started_at).replace(microsecond=0)
    ts = t.strftime("%Y%m%dT%H%M%S")
    if t.tzinfo is not None and t.utcoffset() == dt.timedelta(0):
        ts += "Z"
    elif t.tzinfo is not None:
        ts += t.strftime("%z")
    model_token = _filename_safe_token(model_name)
    return Path(
        template.replace("<timestamp>", ts).replace("<run_id>", run_id).replace("<model_name>", model_token)
    )


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Run prompt_registry.yaml prompts against an Ollama server.")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML (default: config.yaml)")
    p.add_argument("--registry", default="prompt_registry.yaml", help="Path to prompt registry YAML")
    p.add_argument("--dry-run", action="store_true", help="Only show what would be run")
    p.add_argument("--limit", type=int, default=None, help="Only run first N prompts (for smoke testing)")
    args = p.parse_args(argv)

    cfg_path = Path(args.config)
    reg_path = Path(args.registry)

    cfg = load_config(cfg_path)
    prompts = load_prompt_registry(reg_path)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        prompts = prompts[: args.limit]

    if args.dry_run:
        print(f"Prompts to run: {len(prompts)}")
        for item in prompts[: min(10, len(prompts))]:
            print(f"- {item['prompt_id']}")
        if len(prompts) > 10:
            print("... (truncated)")
        return 0

    run_id = str(uuid.uuid4())
    started_at = _utc_now_iso()
    output_path = render_output_path(
        cfg.run.output_json_path,
        run_id=run_id,
        started_at=started_at,
        model_name=cfg.model.name,
    )

    ollama_url = cfg.ollama.base_url.rstrip("/") + "/" + cfg.ollama.endpoint.lstrip("/")

    results: list[dict[str, Any] | None] = [None] * len(prompts)
    results_lock = threading.Lock()

    payload: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": None,
        "host": cfg.ollama.base_url,
        "client": {
            "hostname": socket.gethostname(),
            "cwd": str(Path.cwd()),
            "pid": os.getpid(),
        },
        "config": cfg.raw,
        "results": [],
    }
    _atomic_write_json(output_path, payload)
    print(f"Wrote (init): {output_path}", flush=True)
    if isinstance(cfg.model.options, dict) and "seed" in cfg.model.options and cfg.run.num_of_q_repeats > 1:
        print(
            "Note: config.model.options.seed is set; varying seed per repeat (seed + repeat_index) to avoid identical repeats.",
            flush=True,
        )

    def _run_one_prompt(prompt_item: dict[str, Any]) -> dict[str, Any]:
        prompt_id: str = prompt_item["prompt_id"]
        prompt_text: str = prompt_item["text"]

        result_item: dict[str, Any] = {
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "answers": [],
        }
        for field in cfg.run.include_registry_fields:
            if field in prompt_item:
                result_item[field] = prompt_item[field]

        for repeat_idx in range(cfg.run.num_of_q_repeats):
            attempt = 0
            last_err: str | None = None
            while attempt <= cfg.ollama.retries:
                t0 = time.time()
                try:
                    session = requests.Session()
                    options = cfg.model.options
                    if isinstance(options, dict) and "seed" in options:
                        options = dict(options)
                        try:
                            options["seed"] = int(options["seed"]) + int(repeat_idx)
                        except Exception:  # noqa: BLE001
                            # If seed isn't int-coercible, leave it as-is.
                            pass
                    resp = _ollama_generate(
                        session=session,
                        url=ollama_url,
                        model=cfg.model.name,
                        system=cfg.model.system,
                        prompt=prompt_text,
                        stream=cfg.run.stream,
                        options=options,
                        timeout_s=cfg.ollama.timeout_s,
                    )
                    answer_obj = {
                        "repeat_index": repeat_idx,
                        "response_text": _strip_to_last_full_sentence(_extract_response_text(resp)),
                    }
                    result_item["answers"].append(answer_obj)
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = str(e)
                    if attempt >= cfg.ollama.retries:
                        result_item["answers"].append(
                            {
                                "repeat_index": repeat_idx,
                                "response_text": "",
                            }
                        )
                        break
                    time.sleep(_backoff_s(attempt))
                    attempt += 1

        return result_item

    completed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.run.parallelism) as ex:
        futures: dict[concurrent.futures.Future[dict[str, Any]], tuple[int, str]] = {}
        for idx, prompt_item in enumerate(prompts):
            futures[ex.submit(_run_one_prompt, prompt_item)] = (idx, prompt_item["prompt_id"])

        for fut in concurrent.futures.as_completed(futures):
            idx, prompt_id = futures[fut]
            result_item = fut.result()

            with results_lock:
                results[idx] = result_item
                completed_count += 1
                payload["results"] = [r for r in results if r is not None]
                _atomic_write_json(output_path, payload)

            print(f"Completed {completed_count}/{len(prompts)}: {prompt_id}", flush=True)

    payload["finished_at"] = _utc_now_iso()
    _atomic_write_json(output_path, payload)
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

