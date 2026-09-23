"""Provider-neutral LLM mutation operator with rejection sampling.

The parent recipe, the contract spec, and a mode-appropriate description of the
selection metric are sent to the CLI; the reply must be a single python code block.
Candidates failing static checks, import, or contract validation are rejected and
resampled, with reasons logged (rejection data is itself analysis material).

Environment overrides:
  SHINSA_MUTATE_PROVIDER   "claude" (default) or "codex"
  SHINSA_MUTATE_MODEL      required exact model ID
  SHINSA_MUTATE_REASONING  Codex reasoning effort (default: "low")
  SHINSA_CLAUDE_BIN        Claude CLI binary (default: "claude")
  SHINSA_CODEX_BIN         Codex CLI binary (default: "codex")
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .interface import CONTRACT_SPEC, RecipeError, load_recipe, static_check, validate_recipe

MUTATION_PROTOCOL_VERSION = 3
FIXED_RAW_PROTOCOL_VERSION = 4
FIXED_RAW_NOTE = (
    "Training fitness is computed by harness-owned raw environment reward only. "
    "The shaped_reward function is still required by the recipe interface but is not "
    "executed during training or audit. Optimizer settings and evaluation schedule "
    "remain candidate-controlled."
)

PROMPT_TEMPLATE = """You are part of an automated research system that evolves *training recipes* \
for a fixed reinforcement-learning setup: a small fixed MLP policy trained with separable CMA-ES \
under a hard compute budget. You will be shown the current recipe and its score. Propose ONE \
improved recipe.

This is a constrained text-generation task. Do not inspect files, run commands, browse, or call \
tools. Use only the information in this prompt.

{contract_spec}

Environment:
{env_desc}

Scoring note: {metric_desc}

Current recipe (score {score:.3f}):
```python
{parent_source}
```

Respond with EXACTLY ONE ```python code block containing the complete new recipe module, and no \
text outside the code block. Keep shaped_reward deterministic (no randomness, no state)."""

METRIC_DESC = {
    "audited": (
        "scores come from an independent audit: the trained policy is re-run on freshly drawn "
        "seeds and scored by the environment's raw reward only."
    ),
    "claimed": (
        "scores are the best training fitness reported by the recipe's own training run, "
        "measured with its own shaped reward and evaluation settings."
    ),
}


class MutateError(Exception):
    pass


class ProviderUnavailable(MutateError):
    """The configured provider cannot serve requests; retrying candidates will not help."""


class MutationExhausted(MutateError):
    """All rejection-sampling attempts failed, with structured provenance."""

    def __init__(self, attempts: list[dict]):
        self.attempts = attempts
        super().__init__(f"all {len(attempts)} attempts rejected")


_NONRETRYABLE_PROVIDER_ERRORS = (
    "usage limit", "rate limit", "quota", "authentication", "unauthorized",
    "model_not_found", "model not found", "model is not available",
    "attempt to write a readonly database",
)


def _raise_cli_error(provider: str, returncode: int, output: str) -> None:
    tail = output.strip()[-500:]
    error = f"{provider} exited {returncode}: {tail}"
    if any(marker in output.lower() for marker in _NONRETRYABLE_PROVIDER_ERRORS):
        raise ProviderUnavailable(error)
    raise MutateError(error)


def _provider() -> str:
    provider = os.environ.get("SHINSA_MUTATE_PROVIDER", "claude").lower()
    if provider not in {"claude", "codex"}:
        raise MutateError(f"unsupported mutation provider: {provider}")
    return provider


def mutator_identity(require_explicit_model: bool = True,
                     training_reward: str = "candidate_shaped") -> dict:
    """Resolve the local CLI identity without making a model or network call."""
    provider = _provider()
    bin_env = "SHINSA_CODEX_BIN" if provider == "codex" else "SHINSA_CLAUDE_BIN"
    configured_bin = os.environ.get(bin_env, provider)
    model = os.environ.get("SHINSA_MUTATE_MODEL")
    if require_explicit_model and not model:
        raise MutateError(
            "SHINSA_MUTATE_MODEL must be set explicitly for reproducible LLM runs"
        )
    resolved = shutil.which(configured_bin)
    if resolved is None:
        raise MutateError(f"mutation CLI not found: {configured_bin}")
    try:
        result = subprocess.run(
            [resolved, "--version"], text=True, capture_output=True,
            timeout=10.0, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise MutateError(f"could not identify mutation CLI: {e}") from e
    version = (result.stdout or result.stderr or "").strip().splitlines()
    if result.returncode != 0 or not version:
        raise MutateError(f"mutation CLI --version failed with code {result.returncode}")
    identity = {
        "provider": provider,
        "binary": Path(resolved).name,
        "cli_version": version[0],
        "model": model,
        "protocol_version": (FIXED_RAW_PROTOCOL_VERSION if training_reward == "fixed_raw"
                             else MUTATION_PROTOCOL_VERSION),
        "prompt_sha256": mutation_prompt_sha256(training_reward),
    }
    if provider == "codex":
        identity["reasoning_effort"] = os.environ.get("SHINSA_MUTATE_REASONING", "low")
        try:
            status = subprocess.run(
                [resolved, "login", "status"], text=True, capture_output=True,
                timeout=10.0, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise MutateError(f"could not identify Codex authentication: {e}") from e
        auth_text = (status.stdout or status.stderr or "").lower()
        if status.returncode != 0:
            raise MutateError(f"codex login status failed with code {status.returncode}")
        identity["auth"] = "chatgpt" if "chatgpt" in auth_text else "api_or_other"
    return identity


def mutation_prompt_sha256(training_reward: str = "candidate_shaped") -> str:
    """Hash every static component that can change mutation behavior."""
    if training_reward not in {"candidate_shaped", "fixed_raw"}:
        raise ValueError(f"unknown training reward: {training_reward}")
    payload = {
        "protocol_version": (FIXED_RAW_PROTOCOL_VERSION if training_reward == "fixed_raw"
                             else MUTATION_PROTOCOL_VERSION),
        "prompt_template": PROMPT_TEMPLATE,
        "contract_spec": CONTRACT_SPEC,
        "metric_desc": METRIC_DESC,
    }
    if training_reward == "fixed_raw":
        payload["training_reward_note"] = FIXED_RAW_NOTE
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_prompt(parent_source: str, score: float, env_desc: str, mode: str,
                 training_reward: str = "candidate_shaped") -> str:
    if training_reward == "fixed_raw":
        env_desc += "\n" + FIXED_RAW_NOTE
    elif training_reward != "candidate_shaped":
        raise ValueError(f"unknown training reward: {training_reward}")
    return PROMPT_TEMPLATE.format(
        contract_spec=CONTRACT_SPEC,
        env_desc=env_desc,
        metric_desc=METRIC_DESC[mode],
        score=score,
        parent_source=parent_source,
    )


def call_claude(prompt: str, timeout: float = 300.0) -> str:
    cmd = [os.environ.get("SHINSA_CLAUDE_BIN", "claude"), "-p"]
    model = os.environ.get("SHINSA_MUTATE_MODEL")
    if model:
        cmd += ["--model", model]
    res = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=timeout)
    if res.returncode != 0:
        _raise_cli_error("claude", res.returncode, res.stderr or res.stdout or "")
    return res.stdout


def call_codex(prompt: str, timeout: float = 300.0) -> tuple[str, dict]:
    """Call Codex in an empty, ephemeral, read-only workspace and capture usage."""
    binary = os.environ.get("SHINSA_CODEX_BIN", "codex")
    model = os.environ.get("SHINSA_MUTATE_MODEL")
    reasoning = os.environ.get("SHINSA_MUTATE_REASONING", "low")
    if not model:
        raise MutateError("SHINSA_MUTATE_MODEL must be set explicitly for Codex")
    with tempfile.TemporaryDirectory(prefix="shinsa-codex-") as workdir:
        output = Path(workdir) / "last-message.txt"
        cmd = [
            binary, "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
            "--sandbox", "read-only", "--skip-git-repo-check", "--color", "never",
            "--json", "--model", model,
            "-c", f'model_reasoning_effort="{reasoning}"',
            "-C", workdir, "-o", str(output), "-",
        ]
        res = subprocess.run(
            cmd, input=prompt, text=True, capture_output=True, timeout=timeout,
        )
        if res.returncode != 0:
            _raise_cli_error("codex", res.returncode, res.stderr or res.stdout or "")
        events = []
        for line in res.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        usage = next(
            (event.get("usage", {}) for event in reversed(events)
             if event.get("type") == "turn.completed"),
            {},
        )
        reply = output.read_text() if output.is_file() else ""
        if not reply:
            messages = [
                event.get("item", {}).get("text", "") for event in events
                if event.get("type") == "item.completed"
                and event.get("item", {}).get("type") == "agent_message"
            ]
            reply = messages[-1] if messages else ""
        if not reply:
            raise MutateError("codex completed without a final message")
        return reply, {"usage": usage} if usage else {}


def call_model(prompt: str, timeout: float = 300.0) -> tuple[str, dict]:
    provider = _provider()
    if provider == "codex":
        return call_codex(prompt, timeout)
    return call_claude(prompt, timeout), {}


_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str | None:
    matches = _FENCE_RE.findall(text)
    return matches[-1].strip() + "\n" if matches else None


def _try_candidate(source: str, obs_dim: int) -> list[str]:
    problems = static_check(source)
    if problems:
        return problems
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(source)
        tmp = f.name
    try:
        module = load_recipe(tmp)
        return validate_recipe(module, obs_dim)
    except RecipeError as e:
        return e.problems
    finally:
        os.unlink(tmp)


def mutate(parent_source: str, score: float, env_desc: str, mode: str, obs_dim: int,
           out_path: str | Path, max_attempts: int = 4,
           training_reward: str = "candidate_shaped") -> tuple[str, list[dict]]:
    """Produce a valid child recipe at out_path; returns (source, attempts_log)."""
    prompt = build_prompt(parent_source, score, env_desc, mode, training_reward)
    attempts = []
    for i in range(max_attempts):
        t0 = time.monotonic()
        entry = {"attempt": i}
        try:
            reply, call_metadata = call_model(prompt)
        except ProviderUnavailable:
            raise
        except (MutateError, subprocess.TimeoutExpired) as e:
            entry.update({"ok": False, "stage": "cli", "error": str(e)[:300],
                          "seconds": round(time.monotonic() - t0, 1)})
            attempts.append(entry)
            continue
        entry["seconds"] = round(time.monotonic() - t0, 1)
        entry["reply_chars"] = len(reply)
        entry.update(call_metadata)
        source = extract_code(reply)
        if source is None:
            entry.update({"ok": False, "stage": "parse", "error": "no python code block"})
            attempts.append(entry)
            prompt += ("\n\nYour previous proposal was rejected: no Python code block. "
                       "Return exactly one complete Python code block satisfying the contract.")
            continue
        problems = _try_candidate(source, obs_dim)
        if problems:
            entry.update({"ok": False, "stage": "validate", "error": "; ".join(problems)[:300]})
            attempts.append(entry)
            prompt += ("\n\nYour previous proposal was rejected by the validator: "
                       + "; ".join(problems)[:300]
                       + ". Correct these violations and return a different valid recipe.")
            continue
        entry["ok"] = True
        attempts.append(entry)
        Path(out_path).write_text(source)
        return source, attempts
    raise MutationExhausted(attempts)
