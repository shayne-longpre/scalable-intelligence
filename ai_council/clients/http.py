from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

from ai_council.clients.base import ModelClientError


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: float,
    retries: int = 2,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            status_code, raw = asyncio.run(
                _post_json_once(url, headers, payload, timeout_seconds)
            )
            if status_code >= 400:
                error = ModelClientError(f"HTTP {status_code} from {url}: {raw}")
                if status_code not in RETRYABLE_STATUS_CODES or attempt == retries:
                    raise error
                last_error = error
            else:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    error = ModelClientError(
                        f"invalid JSON response from {url}: {raw[:500]}"
                    )
                    if attempt == retries:
                        raise error from exc
                    last_error = error
                else:
                    if not isinstance(parsed, dict):
                        error = ModelClientError(f"expected JSON object from {url}")
                        if attempt == retries:
                            raise error
                        last_error = error
                    elif "error" in parsed:
                        error_code = _provider_error_code(parsed["error"])
                        error = ModelClientError(
                            f"provider error {error_code or 'unknown'} from {url}: "
                            f"{json.dumps(parsed['error'], ensure_ascii=True)[:500]}"
                        )
                        if error_code not in RETRYABLE_STATUS_CODES or attempt == retries:
                            raise error
                        last_error = error
                    else:
                        return parsed
        except (httpx.HTTPError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt == retries:
                raise ModelClientError(f"request to {url} failed: {exc}") from exc
            last_error = exc
        if attempt < retries:
            time.sleep(0.5 * (2**attempt))
    else:
        raise ModelClientError(f"request to {url} failed: {last_error}")
    raise ModelClientError(f"request to {url} failed: {last_error}")


def _provider_error_code(error: Any) -> int | None:
    if not isinstance(error, dict):
        return None
    value = error.get("code", error.get("status"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def _post_json_once(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[int, str]:
    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await asyncio.wait_for(
            client.post(
                url,
                headers={**headers, "Content-Type": "application/json"},
                json=payload,
            ),
            timeout=timeout_seconds,
        )
    return response.status_code, response.text
