"""Tests for environment / OpenAI configuration."""

from __future__ import annotations

import os

import src.config as config


def test_get_openai_api_key_reads_env_names(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("openai_api_key", raising=False)
    config._ENV_LOADED = True
    config.get_openai_api_key.cache_clear()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert config.get_openai_api_key() == "test-key"

    config.get_openai_api_key.cache_clear()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("openai_api_key", "lower-key")
    assert config.get_openai_api_key() == "lower-key"


def test_load_env_reads_project_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")

    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("openai_api_key", raising=False)
    config._ENV_LOADED = False
    config.get_openai_api_key.cache_clear()

    config.load_env()
    assert config.get_openai_api_key() == "from-file"
