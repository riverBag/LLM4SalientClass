import argparse
import asyncio
import json
import os
import re
import threading
from dotenv import load_dotenv
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from tenacity import RetryError, retry, stop_after_attempt, wait_random_exponential

# Load environment variables from .env file
load_dotenv()

# Global OpenAI-compatible clients, initialized lazily per provider
clients = {}

# Hajimi: multiple keys from OPENAI_hajimi_API_KEY and OPENAI_hajimi_API_KEY1..N, round-robin per request.
_hajimi_clients = None
_hajimi_rr_lock = threading.Lock()
_hajimi_rr_idx = 0


def _collect_hajimi_api_keys():
    """Returns ordered unique API keys for the Hajimi / gpt-5.4* provider."""
    keys = []
    for env_name in (
        "OPENAI_hajimi_API_KEY",
        "OPENAI_hajimi_API_KEY1",
        "OPENAI_hajimi_API_KEY2",
        "OPENAI_hajimi_API_KEY3",
    ):
        v = os.getenv(env_name)
        if v and str(v).strip():
            keys.append(str(v).strip())
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _ensure_hajimi_clients():
    """Builds one AsyncOpenAI client per Hajimi API key (same base URL, distinct keys)."""
    global _hajimi_clients
    if _hajimi_clients is not None:
        return _hajimi_clients
    keys = _collect_hajimi_api_keys()
    base_url = (os.getenv("OPENAI_hajimi_BASE_URL") or "").strip().rstrip("/")
    if not keys or not base_url:
        raise ValueError(
            "Hajimi API key(s) (OPENAI_hajimi_API_KEY or OPENAI_hajimi_API_KEY1..3) "
            "and OPENAI_hajimi_BASE_URL must be set."
        )
    try:
        import httpx
        from openai import AsyncOpenAI
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: openai/httpx. Install before LLM calls."
        ) from e
    _ua = {"User-Agent": "httpx/0.27.2"}
    _hajimi_clients = []
    for key in keys:
        http_client = httpx.AsyncClient(timeout=120.0, headers=_ua)
        _hajimi_clients.append(
            AsyncOpenAI(
                api_key=key,
                base_url=base_url,
                http_client=http_client,
                default_headers=_ua,
            )
        )
    return _hajimi_clients


def _pick_hajimi_client():
    """Round-robin across Hajimi clients to spread load across API keys / quotas."""
    global _hajimi_rr_idx
    clist = _ensure_hajimi_clients()
    with _hajimi_rr_lock:
        client = clist[_hajimi_rr_idx % len(clist)]
        _hajimi_rr_idx += 1
        return client

# Retries when the HTTP call succeeds but assistant message has no text content
MISSING_ASSISTANT_CONTENT_ATTEMPTS = 5
MISSING_CONTENT_ERROR = "missing content on assistant message"

# After inner tenacity retries give up on 429, run_task retries the whole LLM call
# this many more times (waits between attempts so other workers can proceed).
RATE_LIMIT_OUTER_RETRIES = 5

# COT-style outputs must include this block; otherwise treat as incomplete / re-run
_FINAL_ANSWER_CORE_RE = re.compile(
    r"Final Answer:\s*\n\s*<core class>",
    re.IGNORECASE,
)


def _error_looks_like_rate_limit(error: str | None) -> bool:
    """True for HTTP 429 / OpenAI RateLimitError-style messages stored in result JSON."""
    if not error or not isinstance(error, str):
        return False
    low = error.lower()
    if "429" in error:
        return True
    if "rate limit" in low or "ratelimit" in low.replace(" ", ""):
        return True
    if "too many requests" in low:
        return True
    return False


def get_provider_config(model_name: str):
    """Selects the OpenAI-compatible provider config for the requested model."""
    model_lower = model_name.lower()
    if model_lower.startswith("deepseek"):
        return (
            "deepseek",
            os.getenv("OPENAI_DEEPSEEK_API_KEY"),
            os.getenv("OPENAI_DEEPSEEK_BASE_URL"),
        )
    if model_lower.startswith("gpt-5.4"):
        return ("hajimi", None, os.getenv("OPENAI_hajimi_BASE_URL"))
    return (
        "qwen",
        os.getenv("OPENAI_QWEN_GPT_API_KEY"),
        os.getenv("OPENAI_QWEN_GPT_BASE_URL"),
    )

def get_client(model_name: str):
    """Initializes and returns the provider client, reusing it if already created."""
    provider, api_key, base_url = get_provider_config(model_name)
    if provider == "hajimi":
        return _pick_hajimi_client()
    if provider not in clients:
        try:
            from openai import AsyncOpenAI
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency: openai. Install it before running LLM calls."
            ) from e

        if not api_key or not base_url:
            raise ValueError(
                f"{provider} API key and/or base URL environment variables are not set."
            )
        base_url = base_url.rstrip("/")
        clients[provider] = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return clients[provider]

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
async def _chat_completion_create(messages: list, model_name: str):
    """Single HTTP chat completion; retries on transport/server errors only."""
    try:
        aclient = get_client(model_name)
        return await aclient.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=8192,
            temperature=0,
        )
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise


async def call_llm(messages: list, model_name: str):
    """Calls LLM; re-requests if assistant role exists but content is empty."""
    last_response = None
    for attempt in range(MISSING_ASSISTANT_CONTENT_ATTEMPTS):
        response = await _chat_completion_create(messages, model_name)
        last_response = response
        content, err = extract_completion_text(response)
        if content is not None:
            return response
        if err != MISSING_CONTENT_ERROR:
            return response
        if attempt < MISSING_ASSISTANT_CONTENT_ATTEMPTS - 1:
            print(
                f"Retrying LLM ({MISSING_CONTENT_ERROR}), "
                f"{attempt + 2}/{MISSING_ASSISTANT_CONTENT_ATTEMPTS}"
            )
    return last_response

def make_progress(note: str = "processing"):
    """Creates a rich progress bar."""
    return Progress(
        TextColumn(f"{note} • [progress.percentage]{{task.percentage:>3.0f}}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )

def get_task_messages(task: dict):
    """Returns chat messages from the current task format or the legacy prompt format."""
    messages = task.get("messages")
    if messages:
        return messages
    if "message" in task:
        return [{"role": "user", "content": task["message"]}]
    return None

def build_result(
    task: dict,
    model_name: str,
    response_content=None,
    error=None,
    raw_completion=None,
):
    """Builds one prediction record."""
    result = {
        "task_id": task.get("task_id"),
        "model": model_name,
        "response": response_content,
    }
    if error:
        result["error"] = error
    if raw_completion is not None:
        result["raw_completion"] = raw_completion
    return result


def response_has_final_answer_core_block(text: str) -> bool:
    """True if assistant text ends with the expected Final Answer / core class block."""
    if not isinstance(text, str) or not text.strip():
        return False
    return _FINAL_ANSWER_CORE_RE.search(text) is not None


# Few-shot / zero-shot: single-line <core class>...</core class>
_CORE_CLASS_TAG_RE = re.compile(
    r"<core class>\s*[^<]+\s*</core class>",
    re.IGNORECASE | re.DOTALL,
)


def _output_is_cot_predictions_path(output_file: str) -> bool:
    return "predictions_cot_" in os.path.basename(output_file).lower()


def response_meets_completed_schema(text: str, output_file: str) -> bool:
    """COT files require Final Answer + core block; other preds only need core class tags."""
    if _output_is_cot_predictions_path(output_file):
        return response_has_final_answer_core_block(text)
    if not isinstance(text, str) or not text.strip():
        return False
    return _CORE_CLASS_TAG_RE.search(text) is not None


def row_is_success(result: dict, output_file: str) -> bool:
    """True when this JSONL row is a completed prediction (resume / completed set)."""
    if result.get("error"):
        return False
    r = result.get("response")
    if r is None:
        return False
    if isinstance(r, str) and not r.strip():
        return False
    if not response_meets_completed_schema(r, output_file):
        return False
    return True


def row_is_debug_preservable(result: dict) -> bool:
    """Failed row that carries raw API payload for inspection (not in response)."""
    if not result.get("error"):
        return False
    r = result.get("raw_completion")
    return isinstance(r, str) and bool(r.strip())


def completion_to_debug_payload(response) -> str:
    """Serializes API object for inspection when text content is missing."""
    try:
        if hasattr(response, "model_dump"):
            payload = response.model_dump()
        elif hasattr(response, "dict"):
            payload = response.dict()
        else:
            payload = {"repr": repr(response)}
        return json.dumps({"completion_debug": payload}, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps(
            {"completion_debug_error": str(e), "repr": repr(response)},
            ensure_ascii=False,
        )


def extract_completion_text(response) -> tuple:
    """Returns (content, error_message). error_message is None when content is usable."""
    choices = getattr(response, "choices", None)
    if choices is None or len(choices) == 0:
        return None, "empty or missing choices in API response"
    first = choices[0]
    if first is None:
        return None, "first choice is None in API response"
    message = getattr(first, "message", None)
    if message is None:
        return None, "missing message on first choice"
    content = getattr(message, "content", None)
    if content is None:
        return None, MISSING_CONTENT_ERROR
    return content, None

def load_completed_task_ids(output_file: str):
    """Reads existing successful predictions so interrupted runs can resume."""
    completed_task_ids = set()
    if not os.path.exists(output_file):
        return completed_task_ids

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue

            task_id = result.get("task_id")
            if task_id and row_is_success(result, output_file):
                completed_task_ids.add(task_id)
    return completed_task_ids

def prune_incomplete_predictions(output_file: str):
    """Rewrites file: keep one success per task_id if any, else latest debug row."""
    if not os.path.exists(output_file):
        return

    entries = []
    total_nonempty = 0
    bad_json = 0
    no_tid = 0

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_nonempty += 1
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            task_id = result.get("task_id")
            if not task_id:
                no_tid += 1
                continue

            entries.append((total_nonempty - 1, result))

    if not entries and not bad_json and not no_tid:
        return

    by_tid = {}
    for order, result in entries:
        tid = result["task_id"]
        by_tid.setdefault(tid, []).append((order, result))

    kept = []
    for tid in sorted(by_tid.keys(), key=lambda t: min(o for o, _ in by_tid[t])):
        items = by_tid[tid]
        successes = [(o, r) for o, r in items if row_is_success(r, output_file)]
        if successes:
            kept.append(min(successes, key=lambda x: x[0]))
            continue
        debugs = [(o, r) for o, r in items if row_is_debug_preservable(r)]
        if debugs:
            kept.append(max(debugs, key=lambda x: x[0]))

    kept.sort(key=lambda x: x[0])
    kept_results = [r for _, r in kept]

    removed = total_nonempty - len(kept_results)
    n_ok = sum(1 for r in kept_results if row_is_success(r, output_file))
    n_dbg = len(kept_results) - n_ok
    if removed:
        with open(output_file, "w", encoding="utf-8") as f:
            for result in kept_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(
            f"Pruned {removed} row(s) from {output_file}; kept {len(kept_results)} "
            f"({n_ok} completed, {n_dbg} debug). "
            f"Tasks without a completed row will run again on this launch."
        )

async def run_task(
    task: dict,
    p: Progress,
    task_id,
    output_file: str,
    semaphore: asyncio.Semaphore,
    model_name: str,
):
    """Runs a single task, including the LLM call and writing the result."""
    messages = get_task_messages(task)
    if not messages:
        p.update(task_id, advance=1)
        result = build_result(task, model_name, error="missing messages")
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    error = None
    response_obj = None
    outer_rl_attempt = 0
    while True:
        async with semaphore:
            try:
                response_obj = await call_llm(messages, model_name)
            except RetryError as e:
                cause = None
                try:
                    cause = e.last_attempt.exception()
                except Exception:
                    pass
                if cause is not None:
                    error = f"{type(cause).__name__}: {cause}"
                else:
                    error = f"RetryError: {e}"
                response_obj = None
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                response_obj = None
            else:
                error = None

        if response_obj is not None:
            break
        if _error_looks_like_rate_limit(error) and outer_rl_attempt < RATE_LIMIT_OUTER_RETRIES:
            outer_rl_attempt += 1
            delay = min(120.0, 15.0 * (2 ** (outer_rl_attempt - 1)))
            print(
                f"Rate limit for task {task.get('task_id')!r}, "
                f"outer retry {outer_rl_attempt}/{RATE_LIMIT_OUTER_RETRIES}, sleeping {delay:.0f}s"
            )
            await asyncio.sleep(delay)
            continue
        break

    p.update(task_id, advance=1)

    if response_obj is None:
        result = build_result(task, model_name, response_content=None, error=error)
    else:
        response_content, parse_error = extract_completion_text(response_obj)
        if parse_error:
            err_msg = parse_error
            if parse_error == MISSING_CONTENT_ERROR:
                err_msg = (
                    f"{parse_error} (after {MISSING_ASSISTANT_CONTENT_ATTEMPTS} attempts)"
                )
            dbg = completion_to_debug_payload(response_obj)
            result = build_result(
                task,
                model_name,
                response_content=None,
                error=err_msg,
                raw_completion=dbg,
            )
        else:
            result = build_result(
                task, model_name, response_content=response_content
            )

    # Append the result to the output file
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result

async def run_tasks_file(
    input_file: str,
    output_file: str,
    concurrency: int,
    model_name: str,
    overwrite: bool = False,
):
    """Runs tasks from a single file."""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if overwrite:
        with open(output_file, "w", encoding="utf-8") as f:
            pass
    else:
        prune_incomplete_predictions(output_file)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            tasks = []
            for line in f:
                if line.strip():
                    try:
                        tasks.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        print(f"Loaded {len(tasks)} tasks from {input_file}.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    
    if not tasks:
        print(f"No tasks found in {input_file}")
        return

    if model_name.lower().startswith("gpt-5.4"):
        hk = _collect_hajimi_api_keys()
        print(f"Hajimi: {len(hk)} API key(s) will round-robin for this run.")

    completed_task_ids = load_completed_task_ids(output_file)
    if completed_task_ids:
        tasks = [task for task in tasks if task.get("task_id") not in completed_task_ids]
        print(f"Found {len(completed_task_ids)} completed tasks in {output_file}.")
        print(f"Remaining tasks to run: {len(tasks)}.")

    if not tasks:
        print(f"All tasks already completed for {input_file}")
        return

    semaphore = asyncio.Semaphore(concurrency)
    with make_progress(f"Generating {os.path.basename(output_file)}") as p:
        progress_task_id = p.add_task("Querying LLM", total=len(tasks))
        coros = [
            run_task(t, p, progress_task_id, output_file, semaphore, model_name)
            for t in tasks
        ]
        await asyncio.gather(*coros)

async def main(input_path: str, output_path: str, concurrency: int, model_name: str, overwrite: bool):
    """Main function to load tasks and orchestrate the run."""
    
    if os.path.isdir(input_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        files = [f for f in os.listdir(input_path) if f.endswith(".jsonl")]
        print(f"Found {len(files)} JSONL files in {input_path} to process.")
        
        for filename in files:
            input_file = os.path.join(input_path, filename)
            
            # Construct output filename
            # E.g. tasks_BM25... -> predictions_BM25...
            if filename.startswith("tasks_"):
                out_filename = filename.replace("tasks_", "predictions_", 1)
            else:
                out_filename = f"predictions_{filename}"
                
            output_file = os.path.join(output_path, out_filename)
            
            print(f"Processing {input_file} -> {output_file}")
            await run_tasks_file(input_file, output_file, concurrency, model_name, overwrite)
            
    else:
        # Single file mode
        await run_tasks_file(input_path, output_path, concurrency, model_name, overwrite)
        
    print(f"\nAll processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process LLM tasks from a JSONL file or directory."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSONL file or directory containing tasks.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSONL file or directory to save responses.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of concurrent tasks (default: 20).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-v3.2-20251201-128k",
        help="The name of the model to use (e.g., deepseek-v3.2-20251201-128k).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear existing prediction files and rerun all tasks.",
    )
    args = parser.parse_args()

    asyncio.run(main(args.input, args.output, args.workers, args.model, args.overwrite))
