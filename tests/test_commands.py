import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest
from typer.testing import CliRunner

from nanobot.cli.commands import _make_provider, app
from nanobot.config.schema import Config
from nanobot.providers.custom_provider import CustomProvider
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.providers.openai_codex_provider import _strip_model_prefix
from nanobot.providers.registry import find_by_model

runner = CliRunner()


class _StopGatewayError(RuntimeError):
    pass


@pytest.fixture
def mock_paths():
    """Mock config/workspace paths for test isolation."""
    with patch("nanobot.config.loader.get_config_path") as mock_cp, \
         patch("nanobot.config.loader.save_config") as mock_sc, \
         patch("nanobot.config.loader.load_config"), \
         patch("nanobot.cli.commands.get_workspace_path") as mock_ws:

        base_dir = Path("./test_onboard_data")
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir()

        config_file = base_dir / "config.json"
        workspace_dir = base_dir / "workspace"

        mock_cp.return_value = config_file
        mock_ws.return_value = workspace_dir
        mock_sc.side_effect = lambda config: config_file.write_text("{}")

        yield config_file, workspace_dir

        if base_dir.exists():
            shutil.rmtree(base_dir)


def test_onboard_fresh_install(mock_paths):
    """No existing config — should create from scratch."""
    config_file, workspace_dir = mock_paths

    result = runner.invoke(app, ["onboard"])

    assert result.exit_code == 0
    assert "Created config" in result.stdout
    assert "Created workspace" in result.stdout
    assert "nanobot is ready" in result.stdout
    assert "agents.defaults.provider" in result.stdout
    assert "provider_name" in result.stdout
    assert "agents.defaults.model" in result.stdout
    assert "model_name" in result.stdout
    assert config_file.exists()
    assert (workspace_dir / "AGENTS.md").exists()
    assert (workspace_dir / "memory" / "MEMORY.md").exists()


def test_onboard_existing_config_refresh(mock_paths):
    """Config exists, user declines overwrite — should refresh (load-merge-save)."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "existing values preserved" in result.stdout
    assert workspace_dir.exists()
    assert (workspace_dir / "AGENTS.md").exists()


def test_onboard_existing_config_overwrite(mock_paths):
    """Config exists, user confirms overwrite — should reset to defaults."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="y\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "Config reset to defaults" in result.stdout
    assert workspace_dir.exists()


def test_onboard_existing_workspace_safe_create(mock_paths):
    """Workspace exists — should not recreate, but still add missing templates."""
    config_file, workspace_dir = mock_paths
    workspace_dir.mkdir(parents=True)
    config_file.write_text("{}")

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Created workspace" not in result.stdout
    assert "Created AGENTS.md" in result.stdout
    assert (workspace_dir / "AGENTS.md").exists()


def test_config_returns_explicit_provider_name():
    config = Config()
    config.agents.defaults.provider = "openai_codex"

    assert config.get_provider_name() == "openai_codex"


def test_config_returns_explicit_provider_config():
    config = Config()
    config.agents.defaults.provider = "openai_compatible"

    assert config.get_provider() == config.providers.openai_compatible


def test_find_by_model_matches_openai_codex_keyword():
    spec = find_by_model("openai-codex/gpt-5.3-codex")

    assert spec is not None
    assert spec.name == "openai_codex"


def test_litellm_provider_canonicalizes_anthropic_compatible_hyphen_prefix():
    provider = LiteLLMProvider(default_model="anthropic-compatible/claude-3-5-sonnet")

    resolved = provider._resolve_model("anthropic-compatible/claude-3-5-sonnet")

    assert resolved == "anthropic/claude-3-5-sonnet"


def test_litellm_provider_routes_anthropic_compatible_native_model_ids() -> None:
    provider = LiteLLMProvider(
        default_model="claude-3-5-sonnet",
        provider_name="anthropic_compatible",
    )

    resolved = provider._resolve_model("claude-3-5-sonnet")

    assert resolved == "anthropic/claude-3-5-sonnet"


def test_openai_codex_strip_prefix_supports_hyphen_and_underscore():
    assert _strip_model_prefix("openai-codex/gpt-5.1-codex") == "gpt-5.1-codex"
    assert _strip_model_prefix("openai_codex/gpt-5.1-codex") == "gpt-5.1-codex"


@pytest.mark.parametrize(
    ("provider_name", "model", "api_base", "missing_field"),
    [
        ("openai_compatible", "gpt-4o", "http://localhost:8000/v1", "api_key"),
        ("openai_compatible", "gpt-4o", None, "api_base"),
        ("anthropic_compatible", "claude-3-5-sonnet", "http://localhost:8001", "api_key"),
        ("anthropic_compatible", "claude-3-5-sonnet", None, "api_base"),
    ],
)
def test_make_provider_requires_complete_compatible_provider_config(
    provider_name: str,
    model: str,
    api_base: str | None,
    missing_field: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = Config()
    config.agents.defaults.provider = provider_name
    config.agents.defaults.model = model
    provider_config = getattr(config.providers, provider_name)
    if missing_field != "api_key":
        provider_config.api_key = "test-key"
    if api_base is not None:
        provider_config.api_base = api_base

    with pytest.raises(click.exceptions.Exit) as exc:
        _make_provider(config)

    assert exc.value.exit_code == 1
    output = capsys.readouterr().out
    assert f"Incomplete {provider_name} provider configuration" in output
    assert f"providers.{provider_name}.{missing_field}" in output


@pytest.mark.parametrize(
    ("provider_name", "model", "api_base", "provider_cls"),
    [
        ("openai_compatible", "gpt-4o", "http://localhost:8000/v1", CustomProvider),
        ("anthropic_compatible", "claude-3-5-sonnet", "http://localhost:8001", LiteLLMProvider),
    ],
)
def test_make_provider_uses_compatible_providers_with_native_model_ids(
    provider_name: str,
    model: str,
    api_base: str,
    provider_cls: type,
) -> None:
    config = Config()
    config.agents.defaults.model = model
    config.agents.defaults.provider = provider_name
    provider_config = getattr(config.providers, provider_name)
    provider_config.api_key = "test-key"
    provider_config.api_base = api_base

    provider = _make_provider(config)

    assert isinstance(provider, provider_cls)
    assert provider.default_model == model
    if provider_name == "openai_compatible":
        assert provider.api_key == "test-key"
        assert provider.api_base == api_base
    else:
        assert provider.provider_name == "anthropic_compatible"
        assert provider._resolve_model(model) == "anthropic/claude-3-5-sonnet"


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-4o",
        "openai-compatible/gpt-4o",
        "anthropic/claude-opus-4-5",
    ],
)
def test_make_provider_rejects_provider_prefixed_openai_compatible_models(
    model: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = Config()
    config.agents.defaults.provider = "openai_compatible"
    config.agents.defaults.model = model
    config.providers.openai_compatible.api_key = "test-key"
    config.providers.openai_compatible.api_base = "http://localhost:8000/v1"

    with pytest.raises(click.exceptions.Exit) as exc:
        _make_provider(config)

    assert exc.value.exit_code == 1
    output = capsys.readouterr().out
    assert "Invalid model for openai_compatible provider" in output
    assert "agents.defaults.model" in output


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-opus-4-5",
        "anthropic-compatible/claude-opus-4-5",
        "openai/gpt-4o",
    ],
)
def test_make_provider_rejects_provider_prefixed_anthropic_compatible_models(
    model: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = Config()
    config.agents.defaults.provider = "anthropic_compatible"
    config.agents.defaults.model = model
    config.providers.anthropic_compatible.api_key = "test-key"
    config.providers.anthropic_compatible.api_base = "http://localhost:8001"

    with pytest.raises(click.exceptions.Exit) as exc:
        _make_provider(config)

    assert exc.value.exit_code == 1
    output = capsys.readouterr().out
    assert "Invalid model for anthropic_compatible provider" in output
    assert "agents.defaults.model" in output


@pytest.mark.parametrize("provider_name", ["openai_compatible", "anthropic_compatible"])
def test_make_provider_requires_explicit_model_for_compatible_providers(
    provider_name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = Config()
    config.agents.defaults.provider = provider_name
    getattr(config.providers, provider_name).api_key = "test-key"
    getattr(config.providers, provider_name).api_base = "http://localhost:8000/v1"

    with pytest.raises(click.exceptions.Exit) as exc:
        _make_provider(config)

    assert exc.value.exit_code == 1
    output = capsys.readouterr().out
    assert f"Missing model for {provider_name} provider" in output
    assert "agents.defaults.model" in output


def test_make_provider_requires_explicit_provider_selection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = Config()

    with pytest.raises(click.exceptions.Exit) as exc:
        _make_provider(config)

    assert exc.value.exit_code == 1
    output = capsys.readouterr().out
    assert "Missing provider selection" in output
    assert "agents.defaults.provider" in output or "openai_compatible" in output


@pytest.fixture
def mock_agent_runtime(tmp_path):
    """Mock agent command dependencies for focused CLI tests."""
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "default-workspace")
    cron_dir = tmp_path / "data" / "cron"

    with patch("nanobot.config.loader.load_config", return_value=config) as mock_load_config, \
         patch("nanobot.config.paths.get_cron_dir", return_value=cron_dir), \
         patch("nanobot.cli.commands.sync_workspace_templates") as mock_sync_templates, \
         patch("nanobot.cli.commands._make_provider", return_value=object()), \
         patch("nanobot.cli.commands._print_agent_response") as mock_print_response, \
         patch("nanobot.bus.queue.MessageBus"), \
         patch("nanobot.cron.service.CronService"), \
         patch("nanobot.agent.loop.AgentLoop") as mock_agent_loop_cls:

        agent_loop = MagicMock()
        agent_loop.channels_config = None
        agent_loop.process_direct = AsyncMock(return_value="mock-response")
        agent_loop.close_mcp = AsyncMock(return_value=None)
        mock_agent_loop_cls.return_value = agent_loop

        yield {
            "config": config,
            "load_config": mock_load_config,
            "sync_templates": mock_sync_templates,
            "agent_loop_cls": mock_agent_loop_cls,
            "agent_loop": agent_loop,
            "print_response": mock_print_response,
        }


def test_agent_help_shows_workspace_and_config_options():
    result = runner.invoke(app, ["agent", "--help"])

    assert result.exit_code == 0
    assert "--workspace" in result.stdout
    assert "-w" in result.stdout
    assert "--config" in result.stdout
    assert "-c" in result.stdout


def test_agent_uses_default_config_when_no_workspace_or_config_flags(mock_agent_runtime):
    result = runner.invoke(app, ["agent", "-m", "hello"])

    assert result.exit_code == 0
    assert mock_agent_runtime["load_config"].call_args.args == (None,)
    assert mock_agent_runtime["sync_templates"].call_args.args == (
        mock_agent_runtime["config"].workspace_path,
    )
    assert mock_agent_runtime["agent_loop_cls"].call_args.kwargs["workspace"] == (
        mock_agent_runtime["config"].workspace_path
    )
    mock_agent_runtime["agent_loop"].process_direct.assert_awaited_once()
    mock_agent_runtime["print_response"].assert_called_once_with("mock-response", render_markdown=True)


def test_agent_uses_explicit_config_path(mock_agent_runtime, tmp_path: Path):
    config_path = tmp_path / "agent-config.json"
    config_path.write_text("{}")

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_path)])

    assert result.exit_code == 0
    assert mock_agent_runtime["load_config"].call_args.args == (config_path.resolve(),)


def test_agent_config_sets_active_path(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    seen: dict[str, Path] = {}

    monkeypatch.setattr(
        "nanobot.config.loader.set_config_path",
        lambda path: seen.__setitem__("config_path", path),
    )
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.config.paths.get_cron_dir", lambda: config_file.parent / "cron")
    monkeypatch.setattr("nanobot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.cli.commands._make_provider", lambda _config: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: object())
    monkeypatch.setattr("nanobot.cron.service.CronService", lambda _store: object())

    class _FakeAgentLoop:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def process_direct(self, *_args, **_kwargs) -> str:
            return "ok"

        async def close_mcp(self) -> None:
            return None

    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr("nanobot.cli.commands._print_agent_response", lambda *_args, **_kwargs: None)

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_file)])

    assert result.exit_code == 0
    assert seen["config_path"] == config_file.resolve()


def test_agent_overrides_workspace_path(mock_agent_runtime):
    workspace_path = Path("/tmp/agent-workspace")

    result = runner.invoke(app, ["agent", "-m", "hello", "-w", str(workspace_path)])

    assert result.exit_code == 0
    assert mock_agent_runtime["config"].agents.defaults.workspace == str(workspace_path)
    assert mock_agent_runtime["sync_templates"].call_args.args == (workspace_path,)
    assert mock_agent_runtime["agent_loop_cls"].call_args.kwargs["workspace"] == workspace_path


def test_agent_workspace_override_wins_over_config_workspace(mock_agent_runtime, tmp_path: Path):
    config_path = tmp_path / "agent-config.json"
    config_path.write_text("{}")
    workspace_path = Path("/tmp/agent-workspace")

    result = runner.invoke(
        app,
        ["agent", "-m", "hello", "-c", str(config_path), "-w", str(workspace_path)],
    )

    assert result.exit_code == 0
    assert mock_agent_runtime["load_config"].call_args.args == (config_path.resolve(),)
    assert mock_agent_runtime["config"].agents.defaults.workspace == str(workspace_path)
    assert mock_agent_runtime["sync_templates"].call_args.args == (workspace_path,)
    assert mock_agent_runtime["agent_loop_cls"].call_args.kwargs["workspace"] == workspace_path


def test_gateway_uses_workspace_from_config_by_default(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    seen: dict[str, Path] = {}

    monkeypatch.setattr(
        "nanobot.config.loader.set_config_path",
        lambda path: seen.__setitem__("config_path", path),
    )
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr(
        "nanobot.cli.commands.sync_workspace_templates",
        lambda path: seen.__setitem__("workspace", path),
    )
    monkeypatch.setattr(
        "nanobot.cli.commands._make_provider",
        lambda _config: (_ for _ in ()).throw(_StopGatewayError("stop")),
    )

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["config_path"] == config_file.resolve()
    assert seen["workspace"] == Path(config.agents.defaults.workspace)


def test_gateway_workspace_option_overrides_config(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    override = tmp_path / "override-workspace"
    seen: dict[str, Path] = {}

    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr(
        "nanobot.cli.commands.sync_workspace_templates",
        lambda path: seen.__setitem__("workspace", path),
    )
    monkeypatch.setattr(
        "nanobot.cli.commands._make_provider",
        lambda _config: (_ for _ in ()).throw(_StopGatewayError("stop")),
    )

    result = runner.invoke(
        app,
        ["gateway", "--config", str(config_file), "--workspace", str(override)],
    )

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["workspace"] == override
    assert config.workspace_path == override


def test_gateway_uses_config_directory_for_cron_store(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    seen: dict[str, Path] = {}

    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.config.paths.get_cron_dir", lambda: config_file.parent / "cron")
    monkeypatch.setattr("nanobot.cli.commands.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.cli.commands._make_provider", lambda _config: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: object())
    monkeypatch.setattr("nanobot.session.manager.SessionManager", lambda _workspace: object())

    class _StopCron:
        def __init__(self, store_path: Path) -> None:
            seen["cron_store"] = store_path
            raise _StopGatewayError("stop")

    monkeypatch.setattr("nanobot.cron.service.CronService", _StopCron)

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["cron_store"] == config_file.parent / "cron" / "jobs.json"


def test_status_lists_openai_codex_as_oauth_provider(monkeypatch) -> None:
    config = Config()
    config.agents.defaults.provider = "openai_codex"
    config.agents.defaults.model = "openai_codex/gpt-5.1-codex"

    monkeypatch.setattr("nanobot.config.loader.get_config_path", lambda: Path(__file__))
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: config)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "OpenAI Codex:" in result.stdout
    assert "(OAuth)" in result.stdout
