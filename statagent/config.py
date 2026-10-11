"""Central configuration, loaded from environment variables.

Every LLM setting is configurable via environment variables so the same code
runs against any OpenAI-compatible provider (DeepSeek by default). Without an
API key the application runs in an offline fallback mode: a deterministic
rule-based planner replaces the LLM while all statistical computation stays
identical.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:  # load .env files when present (never required)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ENV_PREFIX = "STATAGENT_"


def _env(env: dict[str, str], name: str, default: str) -> str:
    """Read STATAGENT_<name> first, then plain <name> as a fallback alias."""
    return env.get(ENV_PREFIX + name, env.get(name, default))


@dataclass(frozen=True)
class LLMConfig:
    api_key: str | None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 120.0

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens.

    Defaults are deepseek-chat list prices at the time of writing; override
    via STATAGENT_PRICE_INPUT / STATAGENT_PRICE_OUTPUT if pricing changes.
    """

    input_per_mtok: float = 0.27
    output_per_mtok: float = 1.10


@dataclass
class AppConfig:
    llm: LLMConfig
    pricing: Pricing
    offline_fallback: bool
    seed: int
    project_root: Path
    runs_dir: Path
    reports_dir: Path
    benchmark_dir: Path
    demo_data_dir: Path

    def ensure_dirs(self) -> None:
        for d in (self.runs_dir, self.reports_dir, self.benchmark_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def llm_mode(self) -> str:
        return "live" if self.llm.is_configured else "offline"


def load_config(env: dict[str, str] | None = None) -> AppConfig:
    if env is None:
        env = dict(os.environ)
    api_key = _env(env, "DEEPSEEK_API_KEY", "") or None
    llm = LLMConfig(
        api_key=api_key,
        base_url=_env(env, "BASE_URL", "https://api.deepseek.com"),
        model=_env(env, "MODEL", "deepseek-chat"),
        temperature=float(_env(env, "TEMPERATURE", "0.0")),
        max_tokens=int(_env(env, "MAX_TOKENS", "4096")),
        timeout=float(_env(env, "TIMEOUT", "120.0")),
    )
    pricing = Pricing(
        input_per_mtok=float(_env(env, "PRICE_INPUT", "0.27")),
        output_per_mtok=float(_env(env, "PRICE_OUTPUT", "1.10")),
    )
    root = Path(__file__).resolve().parent.parent
    cfg = AppConfig(
        llm=llm,
        pricing=pricing,
        offline_fallback=_env(env, "OFFLINE_FALLBACK", "1") not in ("0", "false", "False"),
        seed=int(_env(env, "SEED", "42")),
        project_root=root,
        runs_dir=root / "runs",
        reports_dir=root / "runs" / "reports",
        benchmark_dir=root / "runs" / "benchmark",
        demo_data_dir=root / "demo_data",
    )
    cfg.ensure_dirs()
    return cfg


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return load_config()
