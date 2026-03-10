import json
from types import SimpleNamespace

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.channels.feishu import FeishuChannel, _extract_post_content
from nanobot.config.schema import FeishuConfig


def test_extract_post_content_supports_post_wrapper_shape() -> None:
    payload = {
        "post": {
            "zh_cn": {
                "title": "日报",
                "content": [[{"tag": "text", "text": "完成"}, {"tag": "img", "image_key": "img_1"}]],
            }
        }
    }

    text, image_keys = _extract_post_content(payload)

    assert text == "日报 完成"
    assert image_keys == ["img_1"]


def test_extract_post_content_keeps_direct_shape_behavior() -> None:
    payload = {
        "title": "Daily",
        "content": [[
            {"tag": "text", "text": "report"},
            {"tag": "img", "image_key": "img_a"},
            {"tag": "img", "image_key": "img_b"},
        ]],
    }

    text, image_keys = _extract_post_content(payload)

    assert text == "Daily report"
    assert image_keys == ["img_a", "img_b"]


def test_extract_post_content_preserves_link_targets() -> None:
    payload = {
        "post": {
            "zh_cn": {
                "title": "Docs",
                "content": [[
                    {"tag": "text", "text": "visit"},
                    {"tag": "a", "text": "nanobot", "href": "https://example.com"},
                    {"tag": "img", "image_key": "img_link"},
                ]],
            }
        }
    }

    text, image_keys = _extract_post_content(payload)

    assert text == "Docs visit [nanobot](https://example.com)"
    assert image_keys == ["img_link"]


def test_register_optional_event_keeps_builder_when_method_missing() -> None:
    class Builder:
        pass

    builder = Builder()
    same = FeishuChannel._register_optional_event(builder, "missing", object())
    assert same is builder


def test_register_optional_event_calls_supported_method() -> None:
    called = []

    class Builder:
        def register_event(self, handler):
            called.append(handler)
            return self

    builder = Builder()
    handler = object()
    same = FeishuChannel._register_optional_event(builder, "register_event", handler)

    assert same is builder
    assert called == [handler]


def test_split_elements_by_table_limit() -> None:
    elements = [
        {"tag": "markdown", "content": "before"},
        {"tag": "table", "rows": [{"c0": "1"}]},
        {"tag": "markdown", "content": "between"},
        {"tag": "table", "rows": [{"c0": "2"}]},
        {"tag": "markdown", "content": "after"},
    ]

    groups = FeishuChannel._split_elements_by_table_limit(elements, max_tables=1)

    assert len(groups) == 2
    assert any(el.get("tag") == "table" for el in groups[0])
    assert any(el.get("tag") == "table" for el in groups[1])


def test_detect_msg_format_prefers_interactive_for_tables() -> None:
    content = "|a|b|\n|---|---|\n|1|2|"

    assert FeishuChannel._detect_msg_format(content) == "interactive"


def test_markdown_to_post_converts_links() -> None:
    post_json = FeishuChannel._markdown_to_post("visit [nanobot](https://example.com)")
    payload = json.loads(post_json)

    first_line = payload["zh_cn"]["content"][0]
    assert any(item.get("tag") == "a" and item.get("href") == "https://example.com" for item in first_line)


@pytest.mark.asyncio
async def test_send_infers_open_id_targets() -> None:
    channel = FeishuChannel(FeishuConfig(), object())
    channel._client = object()
    calls: list[tuple[str, str, str, str]] = []

    def fake_send_message_sync(
        receive_id_type: str, receive_id: str, msg_type: str, content: str
    ) -> bool:
        calls.append((receive_id_type, receive_id, msg_type, content))
        return True

    channel._send_message_sync = fake_send_message_sync

    await channel.send(OutboundMessage(channel="feishu", chat_id="ou_user_123", content="hello"))

    assert calls == [("open_id", "ou_user_123", "text", json.dumps({"text": "hello"}, ensure_ascii=False))]


@pytest.mark.asyncio
async def test_on_message_marks_sender_fallback_as_open_id() -> None:
    channel = FeishuChannel(FeishuConfig(), object())
    captured: dict[str, object] = {}

    async def fake_handle_message(**kwargs):
        captured.update(kwargs)

    channel._handle_message = fake_handle_message

    data = SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(
                message_id="msg_1",
                message_type="text",
                content=json.dumps({"text": "ping"}, ensure_ascii=False),
            ),
            sender=SimpleNamespace(
                sender_type="user",
                sender_id=SimpleNamespace(open_id="ou_sender_123"),
            ),
        )
    )

    await channel._on_message(data)

    assert captured["chat_id"] == "ou_sender_123"
    assert captured["metadata"]["receive_id_type"] == "open_id"
