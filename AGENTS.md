# Repository Guidelines

## Project Structure & Module Organization
`nanobot/` contains the Python package and is organized by runtime area: `agent/`, `channels/`, `providers/`, `config/`, `cli/`, `cron/`, `session/`, `skills/`, and `templates/`. Tests live in `tests/` and mostly mirror package behavior with focused unit tests such as `tests/test_commands.py`. Example media and docs assets are kept in `case/` and the root `README.md`.

## Build, Test, and Development Commands
Create a local dev environment with `pip install -e .[dev]` from the repo root. Run the Python test suite with `pytest`; narrow scope during iteration with commands like `pytest tests/test_commands.py`. Check formatting and import order with `ruff check .`. Smoke-test the CLI locally with `nanobot onboard` and `nanobot agent`.

## Coding Style & Naming Conventions
Target Python 3.11+ and follow the existing style: 4-space indentation, type hints on new or changed code, and small focused functions. Ruff enforces import sorting and the main lint set; line length is 100 characters. Use `snake_case` for modules, functions, and variables, `PascalCase` for classes, and keep CLI command names descriptive and verb-based.

## Testing Guidelines
Use `pytest` for Python tests and `pytest.mark.asyncio` for async paths. Add or update tests alongside every behavior change, especially in channel, provider, and CLI flows. Name files `test_*.py` and prefer explicit scenario names such as `test_onboard_existing_config_refresh`. There is no configured coverage gate, so maintain coverage by testing the touched paths directly.

## Commit & Pull Request Guidelines
Recent history follows short conventional prefixes such as `fix:`, `feat(qq):`, `docs:`, `refactor:`, and `chore:`. Keep commit subjects imperative and specific, for example `fix: handle CancelledError in MCP tool calls`. Pull requests should explain the behavior change, note config or migration impact, link related issues when available, and include screenshots or sample terminal/chat output for user-facing channel or CLI changes.
