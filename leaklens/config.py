"""Load, validate, and gate `config.yaml` — the entry point of the data flow.

Two rules shape this module beyond plain parsing:

* **Ownership gate (SR-1).** Every module we ship touches the target — inversion
  samples its vectors, cache_poison writes canaries into it. So probing anyone's
  infrastructure requires explicit consent: if any module is enabled,
  `options.i_own_this_target` must be exactly `True`, or we refuse to run. The
  gate lives here, before the runner ever builds an adapter.

* **No silent recovery threshold (CLAUDE.md Tier 1).** The recovery threshold is
  a documented human decision, never a hidden default. If `inversion` is enabled
  we require `recovery_threshold` to be set — a missing one is an error, not 0.7.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

KNOWN_MODULES = ("inversion", "cache_poison")
KNOWN_STORE_TYPES = ("chroma", "faiss")


class ConfigError(ValueError):
    """A config that cannot be trusted to run — raised with a plain-language why."""


@dataclass
class VectorStoreConfig:
    type: str
    path: str
    collection: str
    encoder: str = "sentence-transformers/gtr-t5-base"


@dataclass
class SemanticCacheConfig:
    type: str
    similarity_threshold: float


@dataclass
class InversionConfig:
    recovery_threshold: float        # required — no silent default (Tier 1)
    sample_size: int = 100
    num_steps: int = 20


@dataclass
class Options:
    i_own_this_target: bool = False
    output_dir: str = "./results"
    seed: int = 42


@dataclass
class Config:
    vector_store: VectorStoreConfig
    modules: list[str]
    options: Options
    semantic_cache: SemanticCacheConfig | None = None
    inversion: InversionConfig | None = None


def _require(mapping: dict, key: str, where: str):
    """Return mapping[key] or raise a ConfigError naming what's missing and where."""
    if key not in mapping or mapping[key] is None:
        raise ConfigError(f"missing required '{key}' under {where}")
    return mapping[key]


def _as_mapping(value, where: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def load_config(path: str | Path) -> Config:
    """Parse `path`, validate it, enforce the ownership gate, and return a `Config`.

    Raises `ConfigError` with an actionable message on any problem — a missing
    file, a malformed section, an unknown module, a disabled ownership gate, or a
    missing recovery threshold.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"config must be a YAML mapping: {path}")

    # --- target.vector_store (required) ---
    target = _as_mapping(_require(raw, "target", "config"), "target")
    vs_raw = _as_mapping(_require(target, "vector_store", "target"), "target.vector_store")
    store_type = _require(vs_raw, "type", "target.vector_store")
    if store_type not in KNOWN_STORE_TYPES:
        raise ConfigError(
            f"target.vector_store.type '{store_type}' unknown; "
            f"expected one of {KNOWN_STORE_TYPES}"
        )
    vector_store = VectorStoreConfig(
        type=store_type,
        path=_require(vs_raw, "path", "target.vector_store"),
        collection=_require(vs_raw, "collection", "target.vector_store"),
        encoder=vs_raw.get("encoder", VectorStoreConfig.encoder),
    )

    # --- optional target.semantic_cache ---
    semantic_cache = None
    if target.get("semantic_cache") is not None:
        sc_raw = _as_mapping(target["semantic_cache"], "target.semantic_cache")
        semantic_cache = SemanticCacheConfig(
            type=_require(sc_raw, "type", "target.semantic_cache"),
            similarity_threshold=float(
                _require(sc_raw, "similarity_threshold", "target.semantic_cache")
            ),
        )

    # --- modules (non-empty; every name known) ---
    modules = _require(raw, "modules", "config")
    if not isinstance(modules, list) or not modules:
        raise ConfigError("modules must be a non-empty list of module names")
    unknown = [m for m in modules if m not in KNOWN_MODULES]
    if unknown:
        raise ConfigError(f"unknown module(s) {unknown}; expected from {KNOWN_MODULES}")

    # --- inversion section (required iff inversion enabled; threshold not silent) ---
    inversion = None
    if "inversion" in modules:
        inv_raw = _as_mapping(raw.get("inversion") or {}, "inversion")
        if inv_raw.get("recovery_threshold") is None:
            raise ConfigError(
                "inversion.recovery_threshold must be set explicitly — it is a "
                "documented human decision, never defaulted silently (CLAUDE.md Tier 1)"
            )
        inversion = InversionConfig(
            recovery_threshold=float(inv_raw["recovery_threshold"]),
            sample_size=int(inv_raw.get("sample_size", InversionConfig.sample_size)),
            num_steps=int(inv_raw.get("num_steps", InversionConfig.num_steps)),
        )

    # --- options + ownership gate ---
    opt_raw = _as_mapping(raw.get("options") or {}, "options")
    options = Options(
        i_own_this_target=bool(opt_raw.get("i_own_this_target", False)),
        output_dir=opt_raw.get("output_dir", Options.output_dir),
        seed=int(opt_raw.get("seed", Options.seed)),
    )
    # SR-1: refuse to probe a target without explicit ownership consent.
    if opt_raw.get("i_own_this_target") is not True:
        raise ConfigError(
            "ownership gate: set options.i_own_this_target: true to run modules "
            f"{modules} — LeakLens only audits infrastructure you own (SR-1)"
        )

    return Config(
        vector_store=vector_store,
        modules=list(modules),
        options=options,
        semantic_cache=semantic_cache,
        inversion=inversion,
    )
