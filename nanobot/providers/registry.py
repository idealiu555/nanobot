"""
Provider registry for the supported LLM backends.

The current build intentionally supports only:
  - OpenAI-compatible APIs
  - Anthropic-compatible APIs
  - OpenAI Codex OAuth
  - GitHub Copilot OAuth
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    """Metadata for one configured provider."""

    name: str
    keywords: tuple[str, ...]
    env_key: str
    display_name: str = ""
    litellm_prefix: str = ""
    skip_prefixes: tuple[str, ...] = ()
    is_oauth: bool = False
    is_direct: bool = False
    supports_prompt_caching: bool = False

    @property
    def label(self) -> str:
        return self.display_name or self.name.title()


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="openai_compatible",
        keywords=(),
        env_key="",
        display_name="OpenAI-Compatible",
        is_direct=True,
    ),
    ProviderSpec(
        name="anthropic_compatible",
        keywords=("anthropic_compatible", "anthropic-compatible", "claude"),
        env_key="",
        display_name="Anthropic-Compatible",
        litellm_prefix="anthropic",
        skip_prefixes=("anthropic/",),
        supports_prompt_caching=True,
    ),
    ProviderSpec(
        name="openai_codex",
        keywords=("openai-codex",),
        env_key="",
        display_name="OpenAI Codex",
        is_oauth=True,
    ),
    ProviderSpec(
        name="github_copilot",
        keywords=("github_copilot", "copilot"),
        env_key="",
        display_name="Github Copilot",
        litellm_prefix="github_copilot",
        skip_prefixes=("github_copilot/",),
        is_oauth=True,
    ),
)


def find_by_model(model: str) -> ProviderSpec | None:
    """Match a provider by explicit prefix or model-name keyword."""
    model_lower = model.lower()
    model_normalized = model_lower.replace("-", "_")
    model_prefix = model_lower.split("/", 1)[0] if "/" in model_lower else ""
    normalized_prefix = model_prefix.replace("-", "_")
    specs = [spec for spec in PROVIDERS if not spec.is_direct]

    for spec in specs:
        if model_prefix and normalized_prefix == spec.name:
            return spec

    for spec in specs:
        if any(kw in model_lower or kw.replace("-", "_") in model_normalized for kw in spec.keywords):
            return spec
    return None


def find_by_name(name: str) -> ProviderSpec | None:
    """Find a provider spec by config field name."""
    for spec in PROVIDERS:
        if spec.name == name:
            return spec
    return None
