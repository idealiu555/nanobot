"""Image search tool using DuckDuckGo."""

from __future__ import annotations

import asyncio
import json
from functools import partial
from math import ceil
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


def _search_images(
    query: str,
    max_results: int = 5,
    timeout_seconds: float = 20.0,
    proxy: str | None = None,
    region: str = "wt-wt",
    safesearch: str = "moderate",
    size: str | None = None,
    color: str | None = None,
    type_image: str | None = None,
    layout: str | None = None,
    license_image: str | None = None,
) -> list[dict]:
    """Execute image search using DuckDuckGo."""
    try:
        from ddgs import DDGS
    except ImportError:
        logger.error("ddgs library not installed. Run: pip install ddgs")
        return []
    try:
        from ddgs.exceptions import TimeoutException
        ddgs_timeout_exception = TimeoutException
    except Exception:
        ddgs_timeout_exception = TimeoutError

    ddgs = DDGS(proxy=proxy, timeout=max(1, ceil(timeout_seconds)))
    kwargs: dict[str, Any] = {
        "region": region,
        "safesearch": safesearch,
        "max_results": max_results,
    }
    if size:
        kwargs["size"] = size
    if color:
        kwargs["color"] = color
    if type_image:
        kwargs["type_image"] = type_image
    if layout:
        kwargs["layout"] = layout
    if license_image:
        kwargs["license_image"] = license_image

    try:
        results = ddgs.images(query, **kwargs)
        return list(results) if results else []
    except ddgs_timeout_exception as exc:
        logger.warning("Image search timed out: {}", exc)
        raise TimeoutError("Image search timed out") from exc
    except Exception as exc:
        logger.error("Failed to search images: {}", exc)
        return []


class ImageSearchTool(Tool):
    """Search for images online via DuckDuckGo."""

    name = "image_search"
    description = (
        "Search for image candidates and return direct image URLs. "
        "For logo/image download tasks, call this before using exec."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keywords for target images"},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of image results (1-20)",
                "minimum": 1,
                "maximum": 20,
            },
            "size": {
                "type": "string",
                "description": "Optional size filter",
                "enum": ["Small", "Medium", "Large", "Wallpaper"],
            },
            "type_image": {
                "type": "string",
                "description": "Optional image type filter",
                "enum": ["photo", "clipart", "gif", "transparent", "line"],
            },
            "layout": {
                "type": "string",
                "description": "Optional layout filter",
                "enum": ["Square", "Tall", "Wide"],
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        max_results: int = 5,
        timeout_seconds: float = 20.0,
        proxy: str | None = None,
    ):
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds
        self.proxy = proxy

    async def execute(
        self,
        query: str,
        max_results: int | None = None,
        size: str | None = None,
        type_image: str | None = None,
        layout: str | None = None,
        **kwargs: Any,
    ) -> str:
        n = min(max(max_results or self.max_results, 1), 20)

        search_call = partial(
            _search_images,
            query=query,
            max_results=n,
            timeout_seconds=self.timeout_seconds,
            proxy=self.proxy,
            size=size,
            type_image=type_image,
            layout=layout,
        )
        try:
            results = await asyncio.to_thread(search_call)
        except TimeoutError:
            logger.warning("Image search timed out for query: {}", query)
            return json.dumps({"error": "Image search timed out", "query": query}, ensure_ascii=False)
        if not results:
            return json.dumps({"error": "No images found", "query": query}, ensure_ascii=False)

        normalized_results = []
        for result in results:
            image_url = result.get("image") or result.get("url") or result.get("thumbnail") or ""
            thumbnail_url = result.get("thumbnail") or image_url
            normalized_results.append(
                {
                    "title": result.get("title", ""),
                    "image_url": image_url,
                    "thumbnail_url": thumbnail_url,
                    "source": result.get("source", ""),
                }
            )

        payload = {
            "query": query,
            "total_results": len(normalized_results),
            "results": normalized_results,
            "usage_hint": (
                "Prefer 'image_url' for downloads. Validate status/content-type before sending as image."
            ),
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)
