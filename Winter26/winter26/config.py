"""Paths and defaults"""

from __future__ import annotations

import os
from pathlib import Path


def package_root() -> Path:
    """Directory containing ``winter26/`` (this package's parent)."""
    return Path(__file__).resolve().parent.parent


def default_paths() -> dict[str, Path]:
    root = package_root()
    sample = root / "sample_data" / "AIPHAS_raw_data_2026-2.csv"
    return {
        "input_csv": sample,
        "output_dir": root / "new_data_output",
        "mapping_dir": root / "mapping",
    }


DEFAULT_SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
ENV_OLLAMA_HOST = "OLLAMA_HOST"
ENV_OLLAMA_MODEL = "OLLAMA_MODEL"


def ollama_model_from_env(default: str) -> str:
    return os.environ.get(ENV_OLLAMA_MODEL, default)


def ollama_host_from_env(default: str) -> str:
    return os.environ.get(ENV_OLLAMA_HOST, default).rstrip("/")
