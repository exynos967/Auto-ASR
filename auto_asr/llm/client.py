from __future__ import annotations

import json
from urllib.parse import urlparse, urlunparse

import json_repair


def normalize_base_url(base_url: str) -> str:
    """Normalize OpenAI-compatible base url by ensuring a trailing `/v1`.

    Examples:
      - https://api.openai.com        -> https://api.openai.com/v1
      - http://127.0.0.1:8000/v1/     -> http://127.0.0.1:8000/v1
      - http://127.0.0.1:8000/custom  -> http://127.0.0.1:8000/custom/v1
    """
    url = (base_url or "").strip().rstrip("/")
    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/")

    if not path:
        path = "/v1"
    elif not path.endswith("/v1"):
        path = f"{path}/v1"

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def _as_messages(system_prompt: str, payload: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _validate_exact_keys(
    response_obj: object, payload: dict[str, str]
) -> tuple[bool, str, dict[str, str] | None]:
    if not isinstance(response_obj, dict):
        return False, "Output must be a JSON object/dict.", None

    response: dict[str, str] = {str(k): str(v) for k, v in response_obj.items()}
    expected = set(payload.keys())
    actual = set(response.keys())
    if expected != actual:
        missing = sorted(expected - actual, key=lambda x: int(x) if x.isdigit() else x)
        extra = sorted(actual - expected, key=lambda x: int(x) if x.isdigit() else x)
        parts: list[str] = []
        if missing:
            parts.append(f"Missing keys: {missing}")
        if extra:
            parts.append(f"Extra keys: {extra}")
        parts.append(f"Return ONLY a JSON dict with ALL {len(expected)} keys.")
        return False, "; ".join(parts), response

    return True, "", response


def call_chat_json_agent_loop(
    *,
    chat_fn,
    system_prompt: str,
    payload: dict[str, str],
    model: str,
    temperature: float = 0.2,
    max_steps: int = 3,
) -> dict[str, str]:
    """Call an OpenAI-compatible chat completion and require a strict JSON dict output.

    This is intentionally framework-agnostic: `chat_fn(messages, model=..., temperature=...)`
    should return the assistant content string. Tests can inject a stub.
    """
    messages = _as_messages(system_prompt, payload)
    last: dict[str, str] | None = None

    for _ in range(max_steps):
        content = str(chat_fn(messages, model=model, temperature=temperature) or "").strip()
        if not content:
            messages.append({"role": "assistant", "content": ""})
            messages.append(
                {
                    "role": "user",
                    "content": "Error: empty output. Output ONLY valid JSON.",
                }
            )
            continue

        parsed = json_repair.loads(content)
        ok, err, cleaned = _validate_exact_keys(parsed, payload)
        if cleaned is not None:
            last = cleaned
        if ok and cleaned is not None:
            return cleaned

        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Error: {err}"})

    if last is not None:
        return last
    raise RuntimeError("LLM 输出无效，且未能在重试后得到 dict 结果。")


__all__ = ["call_chat_json_agent_loop", "normalize_base_url"]
