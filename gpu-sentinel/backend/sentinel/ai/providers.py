"""LLM providers. Local inference is the default; cloud endpoints are refused
unless ``SENTINEL_ALLOW_CLOUD_LLM=true`` is set explicitly by the customer."""
from __future__ import annotations

import abc
import ipaddress
import logging
import socket
import time
from urllib.parse import urlparse

import httpx

from sentinel.config import Settings

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


class LLMProvider(abc.ABC):
    name: str
    model: str
    local: bool = True

    @abc.abstractmethod
    def chat(self, system: str, user: str, json_mode: bool = True) -> str: ...

    def healthy(self) -> bool:
        return True


def is_private_endpoint(url: str) -> bool:
    """True if the endpoint resolves to loopback/private/link-local addresses (on-prem)."""
    host = urlparse(url).hostname or ""
    if host in ("localhost",) or host.endswith(".local") or host.endswith(".svc") or ".svc." in host or "." not in host:
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    return all(ipaddress.ip_address(i[4][0]).is_private or ipaddress.ip_address(i[4][0]).is_loopback for i in infos)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, url: str, model: str, timeout: float = 30.0):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self._health: tuple[float, bool] = (0.0, False)

    def healthy(self) -> bool:
        ts, ok = self._health
        if time.time() - ts < 60:
            return ok
        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=2)
            names = {m.get("name") for m in r.json().get("models", [])}
            ok = r.status_code == 200 and (self.model in names or any(n and n.split(":")[0] == self.model.split(":")[0] for n in names))
        except Exception:
            ok = False
        self._health = (time.time(), ok)
        return ok

    def chat(self, system: str, user: str, json_mode: bool = True) -> str:
        body = {"model": self.model, "stream": False, "options": {"temperature": 0.1},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["format"] = "json"
        try:
            r = httpx.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            raise LLMUnavailable(str(e)) from e


class OpenAICompatibleProvider(LLMProvider):
    """Any OpenAI-compatible server: vLLM, llama.cpp server, TGI, LM Studio (local) —
    or a cloud API if and only if the customer explicitly allows it."""
    name = "openai_compatible"

    def __init__(self, url: str, model: str, api_key: str | None, timeout: float, allow_cloud: bool):
        self.url, self.model, self.api_key, self.timeout = url.rstrip("/"), model, api_key, timeout
        self.local = is_private_endpoint(url)
        if not self.local and not allow_cloud:
            raise LLMUnavailable(f"refusing non-local LLM endpoint {url}: set SENTINEL_ALLOW_CLOUD_LLM=true to enable")

    def healthy(self) -> bool:
        try:
            return httpx.get(f"{self.url}/v1/models", timeout=2,
                             headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}).status_code == 200
        except Exception:
            return False

    def chat(self, system: str, user: str, json_mode: bool = True) -> str:
        body = {"model": self.model, "temperature": 0.1,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = httpx.post(f"{self.url}/v1/chat/completions", json=body, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMUnavailable(str(e)) from e


def build_provider(s: Settings) -> LLMProvider | None:
    """Returns a provider or None (deterministic template explanations only)."""
    kind = s.llm_provider
    try:
        if kind == "template":
            return None
        if kind in ("ollama", "auto"):
            p = OllamaProvider(s.llm_url, s.llm_model, s.llm_timeout_s)
            if kind == "ollama" or p.healthy():
                return p
            return None
        if kind == "openai_compatible":
            return OpenAICompatibleProvider(s.llm_url, s.llm_model, s.llm_api_key, s.llm_timeout_s, s.allow_cloud_llm)
    except LLMUnavailable as e:
        log.warning("LLM disabled: %s", e)
    return None
