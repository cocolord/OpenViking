"""Native media views of persisted Compile evidence, without binary text evidence."""

from __future__ import annotations

import json

from openviking.utils.media_limits import MAX_INLINE_TOOL_RESULT_MEDIA_BYTES
from vikingbot.agent.tools.base import MultimodalToolResult
from vikingbot.agent.tools.ov_file import VikingMultiReadTool

MAX_IMAGES = VikingMultiReadTool._MAX_INLINE_IMAGES


def media_view(value):
    """Return a JSON view and associated images; never serialize image bytes as text."""
    images = {}

    def visit(item):
        if isinstance(item, list):
            return [visit(child) for child in item]
        if not isinstance(item, dict):
            return item
        if item.get("media_kind") == "image" and item.get("media"):
            media = item["media"]
            images.setdefault(media["image_url"]["url"], (item, media))
            return {key: visit(child) for key, child in item.items() if key != "media"}
        return {key: visit(child) for key, child in item.items()}

    return visit(value), list(images.values())


def media_fits(value):
    """Use the multi-read image count and aggregate byte budgets for one inline view."""
    _, images = media_view(value)
    return (
        len(images) <= MAX_IMAGES
        and sum(source["media_bytes"] for source, _ in images) <= MAX_INLINE_TOOL_RESULT_MEDIA_BYTES
    )


def model_content(value):
    """Attach original images alongside source metadata in a native model message."""
    view, images = media_view(value)
    text = json.dumps(view, ensure_ascii=False)
    if not images:
        return text
    if not media_fits(value):
        raise ValueError(
            "Compile image evidence exceeds the inline media budget; read fewer images"
        )
    parts = [{"type": "text", "text": text}]
    for source, media in images:
        parts.extend(
            [
                {
                    "type": "text",
                    "text": f"Original image evidence: {source['uri']} (sha256: {source['hash']}). "
                    "Use its supplied source ID; this image has no text line ranges.",
                },
                media,
            ]
        )
    return parts


def tool_result(value):
    """Preserve native media for both direct Compile readers and AgentLoop readers."""
    content = model_content(value)
    if isinstance(content, str):
        return content
    return MultimodalToolResult(text=content[0]["text"], content=content)


def request_media_bytes(value):
    """Count inline image bytes throughout a complete model request."""
    if isinstance(value, list):
        return sum(request_media_bytes(item) for item in value)
    if not isinstance(value, dict):
        return 0
    if value.get("type") == "image_url":
        url = value.get("image_url", {}).get("url", "")
        if url.startswith("data:") and ";base64," in url:
            encoded = url.split(",", 1)[1]
            return len(encoded.rstrip("=")) * 3 // 4
    return sum(request_media_bytes(item) for item in value.values())


def request_text_view(value):
    """Exclude inline image payloads from text-only size and token estimates."""
    if isinstance(value, list):
        return [request_text_view(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") == "image_url":
        return {"type": "image_url", "media_bytes": request_media_bytes(value)}
    return {key: request_text_view(item) for key, item in value.items()}
