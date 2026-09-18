from cua.config import Config

_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "MODEL_NAME",
    "DEMO_APP_BASE_URL",
    "ALLOWED_DOMAINS",
    "HEADED",
    "MAX_DISCOVERY_STEPS",
    "STEP_TIMEOUT_MS",
    "ARTIFACT_DIR",
    "EVIDENCE_DIR",
    "LOG_DIR",
    "LOG_LEVEL",
]


def test_defaults(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    cfg = Config.from_env()
    assert cfg.demo_app_base_url == "http://127.0.0.1:8000"
    assert cfg.allowed_domains == ("127.0.0.1", "localhost")
    assert cfg.headed is False
    assert cfg.anthropic_api_key is None
    assert cfg.model_name == ""


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("ALLOWED_DOMAINS", "example.com, foo.bar")
    monkeypatch.setenv("HEADED", "true")
    monkeypatch.setenv("MAX_DISCOVERY_STEPS", "5")
    cfg = Config.from_env()
    assert cfg.allowed_domains == ("example.com", "foo.bar")
    assert cfg.headed is True
    assert cfg.max_discovery_steps == 5


def test_api_key_never_appears_in_repr(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret-value")
    cfg = Config.from_env()
    assert "sk-super-secret-value" not in repr(cfg)
    assert "sk-super-secret-value" not in str(cfg)
