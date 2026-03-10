import builtins
import json
import sys
import types

import pytest

from nanobot.agent.tools.image_search import ImageSearchTool, _search_images


@pytest.mark.asyncio
async def test_image_search_tool_normalizes_results(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDDGS:
        def __init__(self, proxy=None, timeout: int = 30):
            self.proxy = proxy
            self.timeout = timeout

        def images(self, query: str, **kwargs):
            return iter([
                {
                    "title": "Wuhan University logo",
                    "image": "https://img.example/full.jpg",
                    "thumbnail": "https://img.example/thumb.jpg",
                    "source": "example.com",
                }
            ])

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=FakeDDGS))

    tool = ImageSearchTool(max_results=3)
    payload = json.loads(await tool.execute(query="wuhan university logo"))

    assert payload["query"] == "wuhan university logo"
    assert payload["total_results"] == 1
    assert payload["results"][0]["image_url"] == "https://img.example/full.jpg"
    assert payload["results"][0]["thumbnail_url"] == "https://img.example/thumb.jpg"


def test_search_images_returns_empty_when_ddgs_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ddgs":
            raise ImportError("ddgs missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert _search_images("logo") == []


def test_search_images_forwards_proxy_to_ddgs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeDDGS:
        def __init__(self, proxy=None, timeout: int = 30):
            captured["proxy"] = proxy
            captured["timeout"] = timeout

        def images(self, query: str, **kwargs):
            return []

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=FakeDDGS))

    _search_images("logo", proxy="http://127.0.0.1:7890", timeout_seconds=12)

    assert captured["proxy"] == "http://127.0.0.1:7890"
    assert captured["timeout"] == 12


def test_image_search_tool_description_guides_usage_order() -> None:
    assert "before using exec" in ImageSearchTool.description


@pytest.mark.asyncio
async def test_image_search_tool_returns_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_search_images(*_args, **_kwargs):
        raise TimeoutError("Image search timed out")

    monkeypatch.setattr("nanobot.agent.tools.image_search._search_images", fake_search_images)

    tool = ImageSearchTool(timeout_seconds=1)
    payload = json.loads(await tool.execute(query="timeout-case"))

    assert payload["error"] == "Image search timed out"
    assert payload["query"] == "timeout-case"


@pytest.mark.asyncio
async def test_image_search_tool_execute_passes_configured_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_search_images(*_args, **kwargs):
        captured["proxy"] = kwargs.get("proxy")
        return []

    monkeypatch.setattr("nanobot.agent.tools.image_search._search_images", fake_search_images)

    tool = ImageSearchTool(proxy="socks5://127.0.0.1:1080")
    await tool.execute(query="proxy-case")

    assert captured["proxy"] == "socks5://127.0.0.1:1080"
