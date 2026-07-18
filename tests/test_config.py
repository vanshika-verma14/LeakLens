"""Tests for config loading, validation, and the ownership gate — fast, no model.

The two load-bearing guards: SR-1 (no module runs without `i_own_this_target:
true`) and the Tier-1 rule that the recovery threshold is never defaulted
silently. Both get an explicit test so a future refactor can't quietly weaken them.
"""
from pathlib import Path

import pytest
import yaml

from leaklens.config import (
    Config,
    ConfigError,
    InversionConfig,
    Options,
    VectorStoreConfig,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

VALID = {
    "target": {
        "vector_store": {
            "type": "chroma",
            "path": "./results/corpus_store",
            "collection": "docs",
        }
    },
    "modules": ["inversion"],
    "inversion": {"recovery_threshold": 0.7},
    "options": {"i_own_this_target": True},
}


def write_cfg(tmp_path, data) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_valid_config_loads_with_defaults(tmp_path):
    cfg = load_config(write_cfg(tmp_path, VALID))
    assert isinstance(cfg, Config)
    assert cfg.vector_store.type == "chroma"
    # defaults applied where the user was silent
    assert cfg.vector_store.encoder == VectorStoreConfig.encoder
    assert cfg.inversion.sample_size == 100
    assert cfg.inversion.num_steps == 20
    assert cfg.inversion.recovery_threshold == 0.7
    assert cfg.options.seed == 42
    assert cfg.options.output_dir == "./results"
    assert cfg.options.i_own_this_target is True
    assert cfg.semantic_cache is None


def test_ownership_gate_blocks_when_flag_false(tmp_path):
    data = {**VALID, "options": {"i_own_this_target": False}}
    with pytest.raises(ConfigError, match="ownership gate"):
        load_config(write_cfg(tmp_path, data))


def test_ownership_gate_blocks_when_flag_missing(tmp_path):
    data = {k: v for k, v in VALID.items() if k != "options"}
    with pytest.raises(ConfigError, match="ownership gate"):
        load_config(write_cfg(tmp_path, data))


def test_recovery_threshold_is_not_silently_defaulted(tmp_path):
    data = {**VALID, "inversion": {"sample_size": 50}}  # no recovery_threshold
    with pytest.raises(ConfigError, match="recovery_threshold"):
        load_config(write_cfg(tmp_path, data))


def test_unknown_module_rejected(tmp_path):
    data = {**VALID, "modules": ["inversion", "cache_timing"]}
    with pytest.raises(ConfigError, match="unknown module"):
        load_config(write_cfg(tmp_path, data))


def test_empty_modules_rejected(tmp_path):
    data = {**VALID, "modules": []}
    with pytest.raises(ConfigError, match="non-empty"):
        load_config(write_cfg(tmp_path, data))


def test_unknown_store_type_rejected(tmp_path):
    data = {**VALID}
    data["target"] = {"vector_store": {"type": "pinecone", "path": "./s", "collection": "d"}}
    with pytest.raises(ConfigError, match="unknown"):
        load_config(write_cfg(tmp_path, data))


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(p)


def test_semantic_cache_parsed_when_present(tmp_path):
    data = {**VALID}
    data["target"] = {
        "vector_store": VALID["target"]["vector_store"],
        "semantic_cache": {"type": "gptcache", "similarity_threshold": 0.8},
    }
    cfg = load_config(write_cfg(tmp_path, data))
    assert cfg.semantic_cache.type == "gptcache"
    assert cfg.semantic_cache.similarity_threshold == 0.8


def test_shipped_example_config_is_valid():
    cfg = load_config(REPO_ROOT / "config.example.yaml")
    assert isinstance(cfg, Config)
    assert "inversion" in cfg.modules
    assert cfg.options.i_own_this_target is True
    assert isinstance(cfg.inversion, InversionConfig)
    assert isinstance(cfg.options, Options)
