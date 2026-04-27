# llm-audit runner

Runs all prompts in `prompt_registry.yaml` against an Ollama server and saves a JSON artifact containing both the **run config** and the **answers**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Edit `config.yaml`:

- `run.num_of_q_repeats`: how many times each question is sent
- `model.name` and `model.options`: Ollama model + options (temperature, etc.)
- `model.max_output_tokens`: max generated tokens (maps to Ollama `options.num_predict`)
- `ollama.base_url`: should point to `llm03.client.dm`

## Run

```bash
python run_prompts.py --config config.yaml --registry prompt_registry.yaml
```

Smoke test (only first N prompts):

```bash
python run_prompts.py --limit 3
```

Dry run (no requests):

```bash
python run_prompts.py --dry-run
```

## Output format (high level)

The output file is grouped by prompt:

- Top-level: `run_id`, `started_at`, `finished_at`, `host`, `config`, `results`
- Each `results[]` item: `prompt_id`, `prompt_text`, optional fields copied from registry, and `answers[]`
- Each `answers[]` item: `repeat_index`, `response_text`, `ollama` (full response payload), `timing_ms`, `error`

