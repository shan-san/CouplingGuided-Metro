"""One cancellable OpenAI-compatible request per synchronous dispatch call."""
import asyncio
import math
from urllib.parse import urlparse


def request_options(client, *, max_tokens, thinking_mode="disabled", reasoning_effort="low"):
    base = str(client.openai_api_base or "")
    options = dict(model=client.model_name, max_tokens=max_tokens, temperature=client.temperature)
    # Provider-specific parameters must not leak to unrelated compatible APIs.
    if urlparse(base).hostname == "api.deepseek.com":
        options["response_format"] = {"type": "json_object"}
        if thinking_mode != "auto":
            options["extra_body"] = {"thinking": {"type": thinking_mode}}
            if thinking_mode == "enabled":
                options["reasoning_effort"] = reasoning_effort
    return options


def complete(*, api_key, base_url, messages, timeout_s, options, trust_env=True):
    """Bound full-body reception, not merely the wait between socket reads.

    A fresh async SDK client is closed on every return/cancellation. This avoids
    sharing pooled connections across the event loops of synchronous decisions.
    """
    from openai import AsyncOpenAI, APITimeoutError
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Request timeout must be finite and positive")

    async def request():
        kwargs = {}
        if not trust_env:
            import httpx
            kwargs["http_client"] = httpx.AsyncClient(trust_env=False, timeout=timeout_s)
        async with AsyncOpenAI(api_key=api_key, base_url=base_url,
                               timeout=timeout_s, max_retries=0, **kwargs) as api:
            return await api.chat.completions.create(messages=messages, **options)

    async def bounded():
        return await asyncio.wait_for(request(), timeout=timeout_s)

    try:
        return asyncio.run(bounded())
    except (TimeoutError, APITimeoutError) as exc:
        raise TimeoutError(f"Model request exceeded {timeout_s:g}s wall-clock deadline; request cancelled") from exc


def invoke(client, messages, *, timeout_s=30., max_tokens=1024,
           thinking_mode="disabled", reasoning_effort="low"):
    from langchain_core.messages import AIMessage
    if thinking_mode not in ("auto", "disabled", "enabled") or reasoning_effort not in ("low", "high", "max"):
        raise ValueError("Invalid strategic thinking configuration")
    key = client.openai_api_key
    key = key.get_secret_value() if hasattr(key, "get_secret_value") else key
    settings = request_options(client, max_tokens=max_tokens, thinking_mode=thinking_mode,
                               reasoning_effort=reasoning_effort)
    roles = dict(human="user", ai="assistant", system="system")
    body = [dict(role=roles[m.type], content=m.content) for m in messages]
    result = complete(api_key=key, base_url=client.openai_api_base, messages=body,
                      timeout_s=timeout_s, options=settings)
    choice = result.choices[0]
    usage = result.usage.model_dump() if result.usage is not None else {}
    return AIMessage(content=choice.message.content or "", response_metadata=dict(
        finish_reason=choice.finish_reason, model_name=result.model, token_usage=usage,
        thinking_mode_applied=thinking_mode if urlparse(str(client.openai_api_base or "")).hostname == "api.deepseek.com" else "provider_default",
        reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        reasoning_characters=len(getattr(choice.message, "reasoning_content", None) or "")))
