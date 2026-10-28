"""Lossless source ranges prepared in memory for bounded compile assignments."""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import Any, Iterable
from urllib.parse import urlsplit

from openviking.parse.base import MediaType
from openviking.parse.parsers.media.utils import get_media_type
from openviking.utils.media_limits import MAX_INLINE_TOOL_RESULT_MEDIA_BYTES
from vikingbot.agent.tools.ov_file import VikingMultiReadTool
from vikingbot.compile.media import MAX_IMAGES
from vikingbot.compile.models import CompileFailure, CompileLimits

_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+\S")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_FRONTMATTER = re.compile(r"\A---\r?\n.*?\r?\n(?:---|\.\.\.)(?:\r?\n|$)", re.DOTALL)
_TABLE_SEPARATOR = re.compile(r"\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*")


def ensure_compile_source_supported(uri: str) -> None:
    """Reject known media formats that multi-read cannot consume, before any text read."""
    media_type = get_media_type(urlsplit(uri).path, None)
    if media_type in {"image", "audio", "video"} and VikingMultiReadTool._media_kind(uri) is None:
        message = (
            f"Compile cannot read this image format: {uri}. Use PNG, JPEG, GIF or WebP."
            if media_type == MediaType.IMAGE.value
            else f"Compile cannot read this media format: {uri}."
        )
        raise CompileFailure(
            "INVALID_ARGUMENT",
            message,
            stage="pipeline",
        )


@dataclass(frozen=True)
class CompileSourceRange:
    """An exact excerpt of one original source, with separately repeated reading context.

    ``start_char``/``end_char`` are zero-based, half-open Unicode character offsets;
    ``start_line``/``end_line`` are inclusive, one-based line numbers. ``content``
    preserves the original text, including line endings. ``context`` repeats source
    frontmatter, enclosing headings and a table header when they fit the repeat budget.
    Continuation flags identify cuts inside an oversized line or table row;
    these fragments require their adjacent ranges for complete interpretation.

    Original images use zero offsets, a byte hash and native media instead of
    content. Audio/video ranges explicitly identify the derived overview view.
    """

    uri: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    content: str
    context: str = ""
    continues_before: bool = False
    continues_after: bool = False
    media: dict[str, Any] | None = None
    sha256: str | None = None
    media_bytes: int = 0
    media_kind: str | None = None
    representation: str | None = None

    @property
    def input_chars(self) -> int:
        """Count source text and repeated context against the batch character budget."""
        return len(self.content) + len(self.context)


async def read_source(client, uri, limits, provider, *, offset=0, limit=-1):
    """Load Compile evidence through multi-read, keeping raw text and native images distinct."""
    ensure_compile_source_supported(uri)
    if (
        VikingMultiReadTool._media_kind(uri) == "image"
        and not provider.supports_tool_result_media()
    ):
        raise CompileFailure(
            "UNSUPPORTED_MEDIA",
            f"Compile provider cannot deliver native image evidence: {uri}",
            stage="pipeline",
        )
    result = (await VikingMultiReadTool.read_resources(client, [uri], raw_text=True))[0]
    if not result["success"]:
        raise CompileFailure("SOURCE_READ_FAILED", f"{uri}: {result['content']}", stage="pipeline")
    if result.get("image") is not None:
        return [
            CompileSourceRange(
                uri,
                0,
                0,
                0,
                0,
                "",
                media=result["image"],
                sha256=result["sha256"],
                media_bytes=result["media_bytes"],
                media_kind="image",
            )
        ]
    content = result["content"]
    lines = content.splitlines(keepends=True)
    excerpt = "".join(lines[offset:] if limit == -1 else lines[offset : offset + limit])
    start = len("".join(lines[:offset]))
    return [
        replace(
            part,
            start_char=part.start_char + start,
            end_char=part.end_char + start,
            start_line=part.start_line + offset,
            end_line=part.end_line + offset,
            media_kind=result["media_kind"],
            representation="overview" if result["media_kind"] else None,
        )
        for part in split_source(uri, excerpt, limits.source_batch_chars)
    ]


def split_source(uri: str, content: str, max_chars: int) -> list[CompileSourceRange]:
    """Partition original text by characters, preferring headings, paragraphs and whole lines.

    Complete table rows fit together with their header whenever the budget permits.
    An oversized line uses explicitly marked character continuations. Repeated
    frontmatter/headings/table headers use at most one quarter of the budget; their
    original text remains in the source ranges. Empty files retain one empty range.
    """
    if get_media_type(urlsplit(uri).path, None) == MediaType.IMAGE.value:
        raise CompileFailure(
            "INVALID_ARGUMENT", "Image evidence must not be split as text", stage="pipeline"
        )
    if max_chars < 1:
        raise ValueError("Source character budget must be positive")
    lines = content.splitlines(keepends=True)
    offsets = [0]
    contexts = []
    preferred = [0]
    headings = []
    table_header = ""
    match = _FRONTMATTER.match(content)
    frontmatter = match[0] if match else ""
    fence = ""
    for i, line in enumerate(lines):
        start = offsets[-1]
        offsets.append(start + len(line))
        marker = _FENCE.match(line)
        if marker:
            if not fence:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = ""
        if not fence and start >= len(frontmatter):
            heading = _HEADING.match(line)
            if heading:
                depth = len(heading[1])
                headings = [(level, text) for level, text in headings if level < depth]
                headings.append((depth, line.rstrip()))
                preferred.append(start)
            if _TABLE_SEPARATOR.fullmatch(line) and i:
                table_header = lines[i - 1] + line
            elif not line.lstrip().startswith("|"):
                table_header = ""
            if not line.strip():
                preferred.append(offsets[-1])
        contexts.append(
            frontmatter
            + "\n".join(text for _, text in headings)
            + ("\n" + table_header if table_header else "")
        )

    def context_at(offset: int) -> str:
        """Return enclosing source context when it fits the repeat budget."""
        if not offset or not contexts:
            return ""
        value = contexts[min(bisect_right(offsets, offset) - 1, len(contexts) - 1)]
        return value if len(value) <= max_chars // 4 else ""

    ranges = []
    start = 0
    while start < len(content) or not ranges:
        context = context_at(start)
        target = min(len(content), start + max_chars - len(context))
        end = target
        if target < len(content):
            boundary = preferred[bisect_right(preferred, target) - 1]
            line_index = bisect_right(offsets, target) - 1
            line_end = offsets[line_index]
            if boundary > start and boundary >= (start + target) // 2:
                end = boundary
            elif line_end > start:
                next_line_size = offsets[line_index + 1] - line_end
                if line_end >= (start + target) // 2 or next_line_size <= max_chars - len(
                    context_at(line_end)
                ):
                    end = line_end
        start_line = bisect_right(offsets, start)
        end_line = bisect_right(offsets, max(start, end - 1))
        ranges.append(
            CompileSourceRange(
                uri=uri,
                start_char=start,
                end_char=end,
                start_line=start_line,
                end_line=end_line,
                content=content[start:end],
                context=context,
                continues_before=start != offsets[start_line - 1],
                continues_after=end < len(content)
                and end != offsets[bisect_right(offsets, end) - 1],
            )
        )
        start = end
    return ranges


def pack_source_batches(
    sources: Iterable[tuple[str, str]], limits: CompileLimits
) -> list[list[CompileSourceRange]]:
    """Pack exact source ranges in input order, bounded only by characters.

    Adjacent ranges may share a batch when they fit. Small files share remaining
    capacity; original URIs remain the citation targets for all ranges. Source text
    and repeated context both count toward ``source_batch_chars``.
    """
    return pack_source_ranges(
        (
            part
            for uri, content in sources
            for part in split_source(uri, content, limits.source_batch_chars)
        ),
        limits,
    )


def pack_source_ranges(sources: Iterable[CompileSourceRange], limits: CompileLimits):
    """Bound text ranges and original images independently, without dropping sources."""
    batches: list[list[CompileSourceRange]] = []
    batch: list[CompileSourceRange] = []
    size = 0
    media_bytes = images = 0
    for part in sources:
        if batch and (
            size + part.input_chars > limits.source_batch_chars
            or images + int(part.media is not None) > MAX_IMAGES
            or media_bytes + part.media_bytes > MAX_INLINE_TOOL_RESULT_MEDIA_BYTES
        ):
            batches.append(batch)
            batch, size, media_bytes, images = [], 0, 0, 0
        batch.append(part)
        size += part.input_chars
        media_bytes += part.media_bytes
        images += int(part.media is not None)
    if batch:
        batches.append(batch)
    return batches
