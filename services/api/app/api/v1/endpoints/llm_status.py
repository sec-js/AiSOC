"""LLM provider status endpoint — Tier 3.1 (operator visibility).

Exposes a redacted snapshot of the active LLM configuration so operators
and auditors can verify the runtime LLM provider, model, and air-gap
compliance without shelling into a pod or grepping ``.env``. The
endpoint mirrors the shape of ``/api/v1/airgap/status`` and is paired
with the new "Deployment & AI" Settings panel in the web app
(WS-H2 visibility slice / WS-H4 air-gap operator UX).

Contract
--------

* **Read-only.** No knobs to flip, no secrets returned. The API key
  itself is never serialized — only ``key_set: bool``.
* **Best-effort.** When the operator hasn't set ``OPENAI_API_KEY`` /
  ``OPENAI_BASE_URL``, the endpoint reports ``provider="none"`` and
  ``effective_path="fallback"`` so the UI can clearly say
  "running on deterministic fallback".
* **Air-gap aware.** Reuses ``app.core.airgap.is_host_allowed_for_airgap``
  so the answer to "would my LLM call actually leave the pod under the
  current air-gap policy?" comes from the *same* code path that gates
  egress at request time. No drift between the indicator and reality.

Why we don't reuse ``deployment.get_airgap_status``
---------------------------------------------------

``app.api.v1.endpoints.deployment.get_airgap_status`` is an in-memory
mock for a UI that hasn't shipped yet. This endpoint deliberately reads
the *real* env-var-driven config so the indicator the operator sees in
Settings matches the policy the gateway enforces. They will eventually
converge once per-tenant overrides exist (v1.1 BYOK).
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from fastapi import APIRouter

from app.core.airgap import is_host_allowed_for_airgap
from app.core.config import settings

router = APIRouter(prefix="/llm", tags=["llm"])

# Map known hostnames → human-readable provider names so the UI doesn't
# have to do its own substring matching. Order matters only for the
# substring tier (entries earlier in the tuple win).
_HOSTED_PROVIDERS: dict[str, str] = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
}

_HOSTED_SUBSTRINGS: tuple[tuple[str, str], ...] = (
    (".openai.azure.com", "azure-openai"),
    (".azureapi.net", "azure-openai"),
    (".anthropic.com", "anthropic"),
    ("ollama", "local-ollama"),
    ("vllm", "local-vllm"),
    ("litellm", "local-litellm"),
)


def _classify_provider(base_url: str) -> str:
    """Return a stable provider id for the given base URL.

    Returns ``"none"`` when the operator hasn't configured a base URL
    *and* hasn't set ``OPENAI_API_KEY`` (i.e. the explain path is
    forced onto the deterministic fallback).
    """
    if not base_url:
        # No explicit base — treat as the OpenAI default, unless the
        # caller has also blanked the API key (handled by ``provider``
        # logic below).
        return "openai"

    host = (urlparse(base_url).hostname or "").lower()
    if not host:
        return "custom"

    if host in _HOSTED_PROVIDERS:
        return _HOSTED_PROVIDERS[host]

    for needle, provider in _HOSTED_SUBSTRINGS:
        if needle in host:
            return provider

    return "custom"


def _is_loopback_or_private_host(host: str) -> bool:
    """Cheap "is this clearly a local LLM" classifier for the UI badge.

    We don't want to claim "running locally" for a host the operator
    just happened to put in ``AISOC_AIRGAP_ALLOWLIST`` (e.g. an internal
    SaaS gateway), so this is intentionally narrower than the full
    air-gap host classifier.
    """
    if not host:
        return False
    host = host.lower().strip()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    # Bare service name with no dots (docker-compose: "ollama") is local.
    if "." not in host:
        return True
    if host.endswith(".local") or host.endswith(".internal") or host.endswith(".lan"):
        return True
    return False


def llm_status() -> dict[str, object]:
    """Return the live LLM provider snapshot.

    Shape::

        {
            "provider": "local-ollama",
            "model": "llama3.1:8b",
            "base_url": "http://ollama:11434/v1",
            "host": "ollama",
            "key_set": false,
            "airgap_enabled": true,
            "airgap_compliant": true,
            "is_local": true,
            "effective_path": "live",
            "policy_note": "...",
        }

    Notes:
        * ``base_url`` is returned verbatim (no path-stripping) because
          operators sometimes encode the model behind the path on
          single-tenant LiteLLM gateways and stripping it confuses
          troubleshooting.
        * ``key_set`` is the *only* signal we expose for the API key.
          We never return the key itself, even partially redacted.
    """
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
    model = (
        os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("AISOC_LLM_MODEL")
        or ""
    )
    key_set = bool(os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"))
    airgap_enabled = bool(settings.AISOC_AIRGAPPED)

    # If neither base_url nor key are set, the operator is running on
    # the deterministic fallback. Surface that explicitly so the
    # Settings UI can say "no LLM configured" rather than implying the
    # default OpenAI path is wired up.
    if not base_url and not key_set:
        provider = "none"
    else:
        provider = _classify_provider(base_url)

    host = (urlparse(base_url).hostname or "").lower() if base_url else ""

    # Air-gap compliance: when air-gap is OFF this is always True (the
    # check is moot). When ON, we ask the same classifier the egress
    # gate uses so the answer matches what would actually happen at
    # request time.
    if not airgap_enabled:
        airgap_compliant = True
    elif provider == "none":
        # No outbound call would happen anyway — fallback path is
        # always air-gap compliant.
        airgap_compliant = True
    elif not host:
        # base_url unset but key is set → would default to api.openai.com
        # which is never compliant under air-gap.
        airgap_compliant = False
    else:
        airgap_compliant = is_host_allowed_for_airgap(
            host, settings.AISOC_AIRGAP_ALLOWLIST
        )

    is_local = _is_loopback_or_private_host(host)

    # Effective path tells the UI which branch the explain endpoint
    # would actually take right now. Mirrors ``_llm_allowed`` in
    # ``services/agents/app/api/explain.py`` semantics so the indicator
    # cannot drift from runtime behaviour.
    if not key_set:
        effective_path = "fallback"
    elif airgap_enabled and not airgap_compliant:
        effective_path = "fallback"
    else:
        effective_path = "live"

    if provider == "none":
        policy_note = (
            "No LLM is configured. The Explain endpoint will return "
            "deterministic OCSF + MITRE summaries and skip natural-language "
            "narration. Set OPENAI_BASE_URL + OPENAI_API_KEY (or LLM_BASE_URL "
            "+ LLM_API_KEY) to enable LLM-backed summaries."
        )
    elif airgap_enabled and not airgap_compliant:
        policy_note = (
            f"Air-gapped mode is ON and host '{host or 'api.openai.com'}' is "
            "not allowed by the egress policy. The Explain endpoint will "
            "fall back to deterministic summaries until the LLM points at a "
            "host in AISOC_AIRGAP_ALLOWLIST or a private/internal endpoint."
        )
    elif effective_path == "fallback":
        # key_set is False but base_url may be set — uncommon but possible.
        policy_note = (
            "OPENAI_API_KEY (or LLM_API_KEY) is unset, so LLM calls are "
            "disabled and Explain falls back to deterministic summaries."
        )
    elif is_local:
        policy_note = (
            f"LLM calls are routed to a local provider ({provider}). No "
            "external network egress is required for the Explain path."
        )
    else:
        policy_note = (
            f"LLM calls are routed to {provider} at {host}. Outbound HTTP "
            "is required."
        )

    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "host": host,
        "key_set": key_set,
        "airgap_enabled": airgap_enabled,
        "airgap_compliant": airgap_compliant,
        "is_local": is_local,
        "effective_path": effective_path,
        "policy_note": policy_note,
    }


@router.get("/status", summary="Current LLM provider configuration")
async def get_llm_status() -> dict[str, object]:
    """Return the live LLM provider snapshot for this pod.

    The response is safe to surface in operator UIs: secrets are never
    included (we only return ``key_set: bool``) and the body is
    deterministic for a given environment so the same payload renders
    identically across pods. Pair with ``/api/v1/airgap/status`` to
    show operators a full picture of "where will my AI calls go".
    """
    return llm_status()
