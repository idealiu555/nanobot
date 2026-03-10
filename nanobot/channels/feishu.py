"""Feishu/Lark channel implementation using lark-oapi SDK with WebSocket long connection."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.paths import get_media_dir
from nanobot.config.schema import FeishuConfig

lark = None
FEISHU_AVAILABLE = importlib.util.find_spec("lark_oapi") is not None


MSG_TYPE_MAP = {
    "image": "[image]",
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
    "media": "[media]",
}


def _extract_post_content(content_json: dict) -> tuple[str, list[str]]:
    """Extract text and image keys from Feishu post payload."""

    def _parse_block(block: dict) -> tuple[str | None, list[str]]:
        if not isinstance(block, dict) or not isinstance(block.get("content"), list):
            return None, []

        texts: list[str] = []
        images: list[str] = []
        if title := block.get("title"):
            texts.append(str(title))

        for row in block["content"]:
            if not isinstance(row, list):
                continue
            for el in row:
                if not isinstance(el, dict):
                    continue
                tag = el.get("tag")
                if tag == "text":
                    texts.append(el.get("text", ""))
                elif tag == "a":
                    link_text = str(el.get("text", "")).strip()
                    href = str(el.get("href", "")).strip()
                    if link_text and href:
                        texts.append(f"[{link_text}]({href})")
                    elif href:
                        texts.append(href)
                    elif link_text:
                        texts.append(link_text)
                elif tag == "at":
                    texts.append(f"@{el.get('user_name', 'user')}")
                elif tag == "img" and (image_key := el.get("image_key")):
                    images.append(image_key)

        text = " ".join(part for part in texts if part).strip()
        return (text or None), images

    root = content_json
    if isinstance(root, dict) and isinstance(root.get("post"), dict):
        root = root["post"]
    if not isinstance(root, dict):
        return "", []

    if "content" in root:
        text, images = _parse_block(root)
        if text or images:
            return text or "", images

    for locale in ("zh_cn", "en_us", "ja_jp"):
        if locale in root:
            text, images = _parse_block(root[locale])
            if text or images:
                return text or "", images

    for value in root.values():
        if isinstance(value, dict):
            text, images = _parse_block(value)
            if text or images:
                return text or "", images

    return "", []


class FeishuChannel(BaseChannel):
    """Feishu channel using SDK WebSocket long connection mode."""

    name = "feishu"

    def __init__(self, config: FeishuConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: FeishuConfig = config
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()

    _TABLE_RE = re.compile(
        r"((?:^[ \t]*\|.+\|[ \t]*\n)(?:^[ \t]*\|[-:\s|]+\|[ \t]*\n)(?:^[ \t]*\|.+\|[ \t]*\n?)+)",
        re.MULTILINE,
    )
    _MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
    _COMPLEX_MD_RE = re.compile(r"```|^\|.+\|.*\n\s*\|[-:\s|]+\||^#{1,6}\s+", re.MULTILINE)
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".tiff", ".tif"}
    _AUDIO_EXTS = {".opus"}
    _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    _RECEIVE_ID_PREFIX_MAP = {
        "oc_": "chat_id",
        "ou_": "open_id",
        "on_": "union_id",
    }
    _FILE_TYPE_MAP = {
        ".opus": "opus",
        ".mp4": "mp4",
        ".pdf": "pdf",
        ".doc": "doc",
        ".docx": "doc",
        ".xls": "xls",
        ".xlsx": "xls",
        ".ppt": "ppt",
        ".pptx": "ppt",
    }

    @staticmethod
    def _register_optional_event(builder: Any, method_name: str, handler: Any) -> Any:
        """Register an event handler only when the SDK supports it."""
        method = getattr(builder, method_name, None)
        return method(handler) if callable(method) else builder

    @classmethod
    def _infer_receive_id_type(cls, receive_id: str, metadata: dict[str, Any] | None = None) -> str:
        """Infer the Feishu receive_id_type from explicit metadata or common ID prefixes."""
        explicit_type = str((metadata or {}).get("receive_id_type", "")).strip()
        if explicit_type:
            return explicit_type

        normalized_id = str(receive_id or "").strip()
        for prefix, receive_id_type in cls._RECEIVE_ID_PREFIX_MAP.items():
            if normalized_id.startswith(prefix):
                return receive_id_type
        return "chat_id"

    @staticmethod
    def _split_elements_by_table_limit(elements: list[dict], max_tables: int = 1) -> list[list[dict]]:
        """Split card elements so each group contains at most *max_tables* table elements."""
        if not elements:
            return [[]]

        groups: list[list[dict]] = []
        current: list[dict] = []
        table_count = 0
        for el in elements:
            if el.get("tag") == "table":
                if table_count >= max_tables:
                    if current:
                        groups.append(current)
                    current = []
                    table_count = 0
                table_count += 1
            current.append(el)

        if current:
            groups.append(current)
        return groups or [[]]

    @staticmethod
    def _parse_md_table(table_text: str) -> dict | None:
        """Parse a markdown pipe table into a Feishu table card element."""
        lines = [_line.strip() for _line in table_text.strip().split("\n") if _line.strip()]
        if len(lines) < 3:
            return None

        def split_line(_line: str) -> list[str]:
            return [cell.strip() for cell in _line.strip("|").split("|")]

        headers = split_line(lines[0])
        rows = [split_line(_line) for _line in lines[2:]]
        columns = [
            {"tag": "column", "name": f"c{i}", "display_name": header, "width": "auto"}
            for i, header in enumerate(headers)
        ]
        return {
            "tag": "table",
            "page_size": len(rows) + 1,
            "columns": columns,
            "rows": [{f"c{i}": row[i] if i < len(row) else "" for i in range(len(headers))} for row in rows],
        }

    def _build_card_elements(self, content: str) -> list[dict]:
        """Build card elements from markdown text with dedicated table elements."""
        elements: list[dict] = []
        last_end = 0
        for match in self._TABLE_RE.finditer(content):
            before = content[last_end:match.start()].strip()
            if before:
                elements.append({"tag": "markdown", "content": before})

            table_element = self._parse_md_table(match.group(1))
            elements.append(table_element or {"tag": "markdown", "content": match.group(1)})
            last_end = match.end()

        remaining = content[last_end:].strip()
        if remaining:
            elements.append({"tag": "markdown", "content": remaining})

        return elements or [{"tag": "markdown", "content": content}]

    @classmethod
    def _detect_msg_format(cls, content: str) -> str:
        """Choose Feishu msg format: text, post, or interactive."""
        stripped = content.strip()
        if not stripped:
            return "text"
        if cls._COMPLEX_MD_RE.search(stripped):
            return "interactive"
        if len(stripped) > 2000:
            return "interactive"
        if cls._MD_LINK_RE.search(stripped):
            return "post"
        return "text"

    @classmethod
    def _markdown_to_post(cls, content: str) -> str:
        """Convert markdown links into Feishu post JSON payload."""
        lines = content.strip().split("\n")
        paragraphs: list[list[dict]] = []

        for line in lines:
            elements: list[dict] = []
            last_end = 0
            for match in cls._MD_LINK_RE.finditer(line):
                before = line[last_end:match.start()]
                if before:
                    elements.append({"tag": "text", "text": before})
                elements.append({"tag": "a", "text": match.group(1), "href": match.group(2)})
                last_end = match.end()

            remaining = line[last_end:]
            if remaining:
                elements.append({"tag": "text", "text": remaining})
            if not elements:
                elements.append({"tag": "text", "text": ""})

            paragraphs.append(elements)

        return json.dumps({"zh_cn": {"content": paragraphs}}, ensure_ascii=False)

    async def start(self) -> None:
        """Start Feishu channel and receive events through WebSocket."""
        global lark

        if not FEISHU_AVAILABLE:
            logger.error("Feishu SDK not installed. Run: pip install lark-oapi")
            return

        if lark is None:
            import lark_oapi as lark

        if not self.config.app_id or not self.config.app_secret:
            logger.error("Feishu app_id and app_secret not configured")
            return

        self._running = True
        self._loop = asyncio.get_running_loop()

        self._client = (
            lark.Client.builder()
            .app_id(self.config.app_id)
            .app_secret(self.config.app_secret)
            .log_level(lark.LogLevel.INFO)
            .build()
        )

        builder = lark.EventDispatcherHandler.builder(
            self.config.encrypt_key or "",
            self.config.verification_token or "",
        ).register_p2_im_message_receive_v1(self._on_message_sync)

        event_handler = builder.build()
        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )

        def run_ws() -> None:
            import time

            import lark_oapi.ws.client as ws_client

            ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(ws_loop)
            ws_client.loop = ws_loop
            try:
                while self._running:
                    try:
                        self._ws_client.start()
                    except Exception as e:
                        logger.warning("Feishu WebSocket error: {}", e)
                    if self._running:
                        time.sleep(5)
            finally:
                ws_loop.close()

        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()
        logger.info("Feishu channel started")

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop Feishu channel."""
        self._running = False
        logger.info("Feishu channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message to Feishu with media and rich-format support."""
        if not self._client:
            logger.warning("Feishu client not initialized")
            return

        loop = asyncio.get_running_loop()
        receive_id = msg.chat_id
        receive_id_type = self._infer_receive_id_type(receive_id, msg.metadata)

        # 1) Send media attachments first.
        for media_path in msg.media or []:
            if not os.path.isfile(media_path):
                logger.warning("Feishu media file not found: {}", media_path)
                continue

            ext = os.path.splitext(media_path)[1].lower()
            if ext in self._IMAGE_EXTS:
                image_key = await loop.run_in_executor(None, self._upload_image_sync, media_path)
                if image_key:
                    await loop.run_in_executor(
                        None,
                        self._send_message_sync,
                        receive_id_type,
                        receive_id,
                        "image",
                        json.dumps({"image_key": image_key}, ensure_ascii=False),
                    )
                continue

            file_key = await loop.run_in_executor(None, self._upload_file_sync, media_path)
            if file_key:
                msg_type = "media" if ext in self._AUDIO_EXTS or ext in self._VIDEO_EXTS else "file"
                await loop.run_in_executor(
                    None,
                    self._send_message_sync,
                    receive_id_type,
                    receive_id,
                    msg_type,
                    json.dumps({"file_key": file_key}, ensure_ascii=False),
                )

        # 2) Send text body.
        content = (msg.content or "").strip()
        if not content:
            return

        fmt = self._detect_msg_format(content)
        if fmt == "text":
            await loop.run_in_executor(
                None,
                self._send_message_sync,
                receive_id_type,
                receive_id,
                "text",
                json.dumps({"text": content}, ensure_ascii=False),
            )
            return

        if fmt == "post":
            await loop.run_in_executor(
                None,
                self._send_message_sync,
                receive_id_type,
                receive_id,
                "post",
                self._markdown_to_post(content),
            )
            return

        elements = self._build_card_elements(content)
        for chunk in self._split_elements_by_table_limit(elements):
            card = {"config": {"wide_screen_mode": True}, "elements": chunk}
            await loop.run_in_executor(
                None,
                self._send_message_sync,
                receive_id_type,
                receive_id,
                "interactive",
                json.dumps(card, ensure_ascii=False),
            )

    def _send_message_sync(self, receive_id_type: str, receive_id: str, msg_type: str, content: str) -> bool:
        """Send one Feishu message and return True on success."""
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        try:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type(receive_id_type)
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(receive_id)
                    .msg_type(msg_type)
                    .content(content)
                    .build()
                )
                .build()
            )
            response = self._client.im.v1.message.create(request)
            if not response.success():
                logger.error(
                    "Failed to send Feishu {} message: code={}, msg={}",
                    msg_type,
                    response.code,
                    response.msg,
                )
                return False
            return True
        except Exception as e:
            logger.error("Error sending Feishu {} message: {}", msg_type, e)
            return False

    def _upload_image_sync(self, file_path: str) -> str | None:
        """Upload image to Feishu and return image_key."""
        from lark_oapi.api.im.v1 import CreateImageRequest, CreateImageRequestBody

        try:
            with open(file_path, "rb") as image_file:
                request = (
                    CreateImageRequest.builder()
                    .request_body(
                        CreateImageRequestBody.builder()
                        .image_type("message")
                        .image(image_file)
                        .build()
                    )
                    .build()
                )
                response = self._client.im.v1.image.create(request)
                if response.success():
                    return response.data.image_key
                logger.error("Failed to upload image: code={}, msg={}", response.code, response.msg)
                return None
        except Exception as e:
            logger.error("Error uploading image {}: {}", file_path, e)
            return None

    def _upload_file_sync(self, file_path: str) -> str | None:
        """Upload file to Feishu and return file_key."""
        from lark_oapi.api.im.v1 import CreateFileRequest, CreateFileRequestBody

        file_name = os.path.basename(file_path)
        file_ext = Path(file_name).suffix.lower()
        file_type = self._FILE_TYPE_MAP.get(file_ext, "stream")

        try:
            with open(file_path, "rb") as stream_file:
                request = (
                    CreateFileRequest.builder()
                    .request_body(
                        CreateFileRequestBody.builder()
                        .file_type(file_type)
                        .file_name(file_name)
                        .file(stream_file)
                        .build()
                    )
                    .build()
                )
                response = self._client.im.v1.file.create(request)
                if response.success():
                    return response.data.file_key
                logger.error("Failed to upload file: code={}, msg={}", response.code, response.msg)
                return None
        except Exception as e:
            logger.error("Error uploading file {}: {}", file_path, e)
            return None

    def _download_image_sync(self, message_id: str, image_key: str) -> tuple[bytes | None, str | None]:
        """Download one Feishu image by message_id/image_key."""
        from lark_oapi.api.im.v1 import GetMessageResourceRequest

        try:
            request = (
                GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(image_key)
                .type("image")
                .build()
            )
            response = self._client.im.v1.message_resource.get(request)
            if not response.success():
                logger.error("Failed to download image: code={}, msg={}", response.code, response.msg)
                return None, None

            content = response.file.read() if hasattr(response.file, "read") else response.file
            return content, response.file_name
        except Exception as e:
            logger.error("Error downloading image {}: {}", image_key, e)
            return None, None

    def _download_file_sync(self, message_id: str, file_key: str) -> tuple[bytes | None, str | None]:
        """Download one Feishu file/media/audio by message_id/file_key."""
        from lark_oapi.api.im.v1 import GetMessageResourceRequest

        try:
            request = (
                GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(file_key)
                .type("file")
                .build()
            )
            response = self._client.im.v1.message_resource.get(request)
            if not response.success():
                logger.error("Failed to download file: code={}, msg={}", response.code, response.msg)
                return None, None

            content = response.file.read() if hasattr(response.file, "read") else response.file
            return content, response.file_name
        except Exception as e:
            logger.error("Error downloading file {}: {}", file_key, e)
            return None, None

    async def _download_and_save_media(
        self,
        msg_type: str,
        content_json: dict,
        message_id: str,
    ) -> tuple[str | None, str]:
        """Download Feishu media payload and persist to local media directory."""
        loop = asyncio.get_running_loop()
        media_dir = get_media_dir("feishu")

        data: bytes | None = None
        filename: str | None = None

        if msg_type == "image":
            image_key = content_json.get("image_key")
            if image_key:
                data, filename = await loop.run_in_executor(
                    None,
                    self._download_image_sync,
                    message_id,
                    image_key,
                )
                if not filename:
                    filename = f"{image_key[:16]}.jpg"
        elif msg_type in {"audio", "file", "media"}:
            file_key = content_json.get("file_key")
            if file_key:
                data, filename = await loop.run_in_executor(
                    None,
                    self._download_file_sync,
                    message_id,
                    file_key,
                )
                if not filename:
                    ext = {"audio": ".opus", "media": ".mp4"}.get(msg_type, "")
                    filename = f"{file_key[:16]}{ext}"

        if data and filename:
            path = media_dir / filename
            path.write_bytes(data)
            return str(path), f"[{msg_type}: {filename}]"

        return None, f"[{msg_type}: download failed]"

    def _on_message_sync(self, data: Any) -> None:
        """SDK sync callback; schedule async handler in main event loop."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)

    async def _on_message(self, data: Any) -> None:
        """Handle incoming Feishu message and forward to bus."""
        try:
            event = data.event
            message = event.message
            sender = event.sender

            sender_type = getattr(sender, "sender_type", "")
            if sender_type == "bot":
                return

            sender_id = getattr(getattr(sender, "sender_id", None), "open_id", "")
            if not sender_id:
                return

            message_id = getattr(message, "message_id", "")
            if not message_id:
                return
            if message_id in self._processed_message_ids:
                return

            self._processed_message_ids[message_id] = None
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)

            msg_type = getattr(message, "message_type", "text")
            raw_content = getattr(message, "content", "{}")
            try:
                content_json = json.loads(raw_content or "{}")
            except json.JSONDecodeError:
                content_json = {}

            content_parts: list[str] = []
            media_paths: list[str] = []

            if msg_type == "text":
                text = (content_json.get("text") or "").strip()
                if text:
                    content_parts.append(text)
            elif msg_type == "post":
                text, image_keys = _extract_post_content(content_json)
                if text:
                    content_parts.append(text)
                for image_key in image_keys:
                    file_path, content_text = await self._download_and_save_media(
                        "image",
                        {"image_key": image_key},
                        message_id,
                    )
                    if file_path:
                        media_paths.append(file_path)
                    content_parts.append(content_text)
            elif msg_type in {"image", "audio", "file", "media"}:
                file_path, content_text = await self._download_and_save_media(msg_type, content_json, message_id)
                if file_path:
                    media_paths.append(file_path)
                content_parts.append(content_text)
            else:
                content_parts.append(MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]"))

            content = "\n".join(part for part in content_parts if part).strip()
            if not content and not media_paths:
                return

            raw_chat_id = getattr(message, "chat_id", "") or ""
            chat_id = raw_chat_id or sender_id
            await self._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=content,
                media=media_paths,
                metadata={
                    "message_id": message_id,
                    "chat_type": getattr(message, "chat_type", ""),
                    "msg_type": msg_type,
                    "receive_id_type": "chat_id" if raw_chat_id else self._infer_receive_id_type(sender_id),
                },
            )
        except Exception:
            logger.exception("Error handling Feishu message")

    @staticmethod
    def _extract_content_text(msg_type: str, raw_content: str) -> str:
        """Extract plain text from Feishu message payload."""
        try:
            content_json = json.loads(raw_content or "{}")
        except json.JSONDecodeError:
            return ""

        if msg_type == "text":
            return (content_json.get("text") or "").strip()
        if msg_type == "post":
            text, _ = _extract_post_content(content_json)
            return text
        return MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]")
