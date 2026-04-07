"""Calls a local Ollama model via the native HTTP API"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Optional


def ollama_server_reachable(host: str, timeout: float = 5.0) -> bool:
    base = host.rstrip("/")
    url = f"{base}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def ollama_chat(
    host: str,
    model: str,
    *,
    system: str,
    user: str,
    timeout: float = 180.0,
) -> Optional[str]:
    base = host.rstrip("/")
    url = f"{base}/api/chat"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    msg = payload.get("message") or {}
    content = msg.get("content")
    if content is None:
        return None
    return str(content).strip()


def extract_json_object(text: str) -> str:
    s = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        return s[start : end + 1]
    return s
