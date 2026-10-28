# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Source-to-model regressions for text and native Compile image evidence."""

import base64
import copy
import hashlib
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image, ImageDraw
from vikingbot.agent.tools.base import MultimodalToolResult, ToolContext
from vikingbot.agent.tools.ov_file import VikingMultiReadTool
from vikingbot.compile import entry
from vikingbot.compile.media import media_fits, model_content, request_media_bytes
from vikingbot.compile.models import CompileFailure, CompileLimits, SanitizedCompileRequest
from vikingbot.compile.ops.common import payload
from vikingbot.compile.pipeline import Pipeline
from vikingbot.compile.pipeline_agent import ChildProvider, prepare_assignment
from vikingbot.compile.results import FileResponse
from vikingbot.compile.service import BotCompileService
from vikingbot.compile.skill_resources import EvidenceReader
from vikingbot.compile.sources import pack_source_ranges, read_source, split_source
from vikingbot.providers.base import LLMResponse

IMAGE_URI = "viking://resources/cedar/diagram.png"
TEXT_URI = "viking://resources/cedar/notes.md"


class MemoryFiles:
    def __init__(self):
        self.values = {}

    async def put(self, path, value):
        self.values[path] = copy.deepcopy(value)
        return path

    async def get(self, path):
        return copy.deepcopy(self.values.get(path))


@pytest.fixture
def png():
    image = Image.new("RGB", (400, 100), "white")
    ImageDraw.Draw(image).text((10, 30), "RETRY LIMIT: 7", fill="black")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def provider():
    return SimpleNamespace(supports_tool_result_media=Mock(return_value=True), chat=AsyncMock())


@pytest.fixture
def client(png):
    return SimpleNamespace(
        stat=AsyncMock(return_value={"size": len(png)}),
        download_bytes=AsyncMock(return_value=png),
        read_raw=AsyncMock(return_value="# Cedar\n\n![settings](diagram.png)\n"),
        read_content=AsyncMock(return_value="visible text"),
    )


@pytest.fixture
def pipeline(client, provider):
    runtime = Pipeline(
        client=client,
        sandbox=SimpleNamespace(sandbox_cwd="/sandbox"),
        provider=provider,
        model="vision-test",
        temperature=0,
        limits=CompileLimits(),
        request=SanitizedCompileRequest(
            **{
                "from": ["viking://resources/cedar"],
                "to": "viking://resources/wiki",
                "skill": "viking://agent/skills/llm-wiki",
                "instruction": "Extract facts present only in the diagram and cite sources.",
            }
        ),
        skill="Compile Cedar settings with citations.",
        usage={},
    )
    runtime.files = MemoryFiles()
    runtime.model.files = runtime.files
    runtime.model.resources = None
    return runtime


async def seed_image(pipeline):
    parts = await read_source(pipeline.client, IMAGE_URI, pipeline.limits, pipeline.model.provider)

    async def batches():
        yield parts

    records = await pipeline.seed(batches())
    return records[0], parts[0]


@pytest.mark.parametrize("uri", [IMAGE_URI, "viking://resources/a.JPEG?rev=1#preview"])
def test_images_can_never_be_split_as_text(uri):
    with pytest.raises(CompileFailure, match="must not be split as text"):
        split_source(uri, "\x89PNG\x00binary", max_chars=1024)


@pytest.mark.asyncio
async def test_shared_multi_read_preserves_original_image_bytes(client, png, provider):
    parts = await read_source(client, IMAGE_URI, CompileLimits(), provider)
    part = parts[0]
    assert part.content == ""
    assert (part.start_char, part.end_char, part.start_line, part.end_line) == (0, 0, 0, 0)
    assert part.sha256 == hashlib.sha256(png).hexdigest()
    assert base64.b64decode(part.media["image_url"]["url"].split(",")[1]) == png
    client.read_raw.assert_not_awaited()
    client.read_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_keeps_metadata_and_exact_absolute_ranges(client, provider):
    text = "first\n---\nprivate_metadata: retained\n---\n![image](diagram.png)\n"
    client.read_raw.return_value = text
    parts = await read_source(client, TEXT_URI, CompileLimits(), provider, offset=1, limit=3)
    assert "".join(p.content for p in parts) == "---\nprivate_metadata: retained\n---\n"
    assert parts[0].start_char == len("first\n")
    assert parts[0].start_line == 2
    assert parts[0].end_line == 4
    assert parts[0].media is None
    client.read_content.assert_not_awaited()


@pytest.mark.parametrize("name", ["notes.md", "config.ts"])
def test_normal_text_and_markdown_links_are_preserved(name):
    text = "export const retryLimit = 7;\n![diagram](diagram.png)\n"
    assert (
        "".join(p.content for p in split_source(f"viking://resources/{name}", text, 1024)) == text
    )


@pytest.mark.asyncio
async def test_prepare_batches_uses_same_reader_for_mixed_sources(client, provider):
    service = SimpleNamespace(limits=CompileLimits(), agent_loop=SimpleNamespace(provider=provider))
    batches = await BotCompileService._prepare_source_batches(
        service, client, [], files=[TEXT_URI, IMAGE_URI]
    )
    parts = [part for batch in batches for part in batch]
    assert [p.uri for p in parts] == [TEXT_URI, IMAGE_URI]
    assert parts[0].content == client.read_raw.return_value
    assert parts[1].media is not None
    client.read_raw.assert_awaited_once_with(TEXT_URI, offset=0, limit=-1)


@pytest.mark.asyncio
async def test_unsupported_provider_fails_before_loading_image(client, provider):
    provider.supports_tool_result_media.return_value = False
    with pytest.raises(CompileFailure, match="provider cannot deliver") as caught:
        await read_source(client, IMAGE_URI, CompileLimits(), provider)
    assert caught.value.code == "UNSUPPORTED_MEDIA"
    client.download_bytes.assert_not_awaited()
    client.read_raw.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["svg", "bmp", "tiff"])
async def test_unsupported_image_format_never_falls_back_to_text(client, provider, suffix):
    with pytest.raises(CompileFailure, match="image format"):
        await read_source(client, f"viking://resources/image.{suffix}", CompileLimits(), provider)
    client.read_raw.assert_not_awaited()
    client.download_bytes.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["large", "unknown-size", "wrong-format", "changed"])
async def test_image_failures_are_explicit(client, provider, problem):
    if problem == "large":
        client.stat.return_value = {"size": VikingMultiReadTool._MAX_INLINE_MEDIA_BYTES + 1}
    elif problem == "unknown-size":
        client.stat.return_value = {}
    elif problem == "wrong-format":
        client.download_bytes.return_value = b"plain text"
    else:
        client.stat.return_value = {"size": 1}
    with pytest.raises(CompileFailure) as caught:
        await read_source(client, IMAGE_URI, CompileLimits(), provider)
    assert caught.value.code == "SOURCE_READ_FAILED"
    client.read_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_audio_overview_is_marked_as_derived_evidence(client, provider):
    client.read_content.return_value = "Speaker described the retry policy."
    parts = await read_source(client, "viking://resources/a.wav", CompileLimits(), provider)
    assert parts[0].representation == "overview"
    assert parts[0].media_kind == "audio"
    assert parts[0].content == client.read_content.return_value
    client.read_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_media_overview_is_not_binary_text(client, provider):
    client.read_content.return_value = ""
    with pytest.raises(CompileFailure, match="No OpenViking textual overview"):
        await read_source(client, "viking://resources/a.mp4", CompileLimits(), provider)
    client.read_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_image_evidence_survives_seed_and_model_delivery(pipeline, png):
    record, part = await seed_image(pipeline)
    data = {"inputs": [await payload(pipeline, record)]}
    source = await pipeline.files.get(record.payload_ref)
    assert "text" not in source
    assert source["hash"] == part.sha256
    assert pipeline.evidence[record.record_id]["media_kind"] == "image"
    assert "media" not in pipeline.evidence[record.record_id]

    async def vision_stub(**kwargs):
        # This stub verifies transport and provenance, not real-model visual accuracy.
        content = kwargs["messages"][1]["content"]
        assert isinstance(content, list)
        images = [p for p in content if p["type"] == "image_url"]
        assert base64.b64decode(images[0]["image_url"]["url"].split(",")[1]) == png
        assert "RETRY LIMIT" not in content[0]["text"]
        assert IMAGE_URI in content[0]["text"]
        assert part.sha256 in content[0]["text"]
        return LLMResponse(
            content=json.dumps(
                {
                    "files": [
                        {
                            "path": "cedar.md",
                            "content": f"Retry limit: 7. [Source]({IMAGE_URI})",
                            "inputs": [record.record_id],
                        }
                    ]
                }
            )
        )

    pipeline.model.provider.chat.side_effect = vision_stub
    response = await pipeline.model.ask("compile", "Read original evidence.", data, FileResponse)
    assert "Retry limit: 7" in response.files[0].content
    assert response.files[0].inputs == [record.record_id]
    assert IMAGE_URI in response.files[0].content


@pytest.mark.asyncio
async def test_image_rereads_return_native_media_and_reject_line_spans(pipeline):
    record, _ = await seed_image(pipeline)
    reader = EvidenceReader(
        pipeline.files,
        {
            "inputs": [
                {
                    "id": "derived",
                    "payload": {"source_ranges": [record.record_id]},
                }
            ]
        },
    )
    result = await reader.execute(source_range=record.record_id)
    assert isinstance(result, MultimodalToolResult)
    assert any(p["type"] == "image_url" for p in result.content)
    assert "base64," not in result.text
    with pytest.raises(ValueError, match="no text line ranges"):
        await reader.execute(source_range=record.record_id, start_line=1, end_line=1)
    with pytest.raises(ValueError, match="outside this assignment"):
        await reader.execute(source_range="foreign-image")


@pytest.mark.asyncio
async def test_reduce_can_inline_original_image_from_derived_source_ids(pipeline):
    record, _ = await seed_image(pipeline)
    pipeline.model.provider.chat.return_value = LLMResponse(content='{"files": []}')
    data = {
        "inputs": [
            {
                "id": "derived",
                "payload": {
                    "fields": {"retry_limit": 7},
                    "source_ranges": [record.record_id],
                },
            }
        ]
    }
    await pipeline.model.ask("reduce", "Verify original evidence.", data, FileResponse)
    content = pipeline.model.provider.chat.call_args.kwargs["messages"][1]["content"]
    assert any(p["type"] == "image_url" for p in content)
    assert record.record_id in content[0]["text"]


@pytest.mark.asyncio
async def test_discovered_source_reads_use_native_images(pipeline):
    pipeline.changes = {"diff_dirs": [], "full_dirs": ["viking://resources/cedar"], "head": "test"}
    pipeline.tools = {
        "openviking_multi_read": VikingMultiReadTool(),
        "openviking_diff_paths": SimpleNamespace(name="openviking_diff_paths"),
    }
    pipeline.tool_context = ToolContext()

    async def ask(stage, system, data, schema, validate, *, readers):
        result = await readers["openviking_multi_read"].execute(uris=[IMAGE_URI])
        assert isinstance(result, MultimodalToolResult)
        assert any(p["type"] == "image_url" for p in result.content)
        assert IMAGE_URI in result.text
        return SimpleNamespace(needs_pipeline=True)

    pipeline.model.ask = AsyncMock(side_effect=ask)
    assert await entry.run(pipeline, [], {}) is None
    assert any(
        e["uri"] == IMAGE_URI and e["media_kind"] == "image" for e in pipeline.evidence.values()
    )


@pytest.mark.asyncio
async def test_many_images_are_batched_and_oversized_inline_view_is_rejected(client, provider, png):
    parts = []
    for i in range(5):
        client.download_bytes.return_value = png + bytes([i])
        client.stat.return_value = {"size": len(png) + 1}
        parts.extend(
            await read_source(client, f"viking://resources/{i}.png", CompileLimits(), provider)
        )
    assert [len(batch) for batch in pack_source_ranges(parts, CompileLimits())] == [4, 1]
    data = {
        "inputs": [
            {
                "payload": {
                    "uri": p.uri,
                    "hash": p.sha256,
                    "media": p.media,
                    "media_bytes": p.media_bytes,
                    "media_kind": "image",
                }
            }
            for p in parts
        ]
    }
    assert not media_fits(data)
    with pytest.raises(ValueError, match="media budget"):
        model_content(data)


@pytest.mark.asyncio
async def test_ordinary_multi_read_still_returns_native_images(client, png):
    tool = VikingMultiReadTool()
    tool._get_client = AsyncMock(return_value=client)
    tool._release_client = AsyncMock()
    result = await tool.execute(ToolContext(), [TEXT_URI, IMAGE_URI])
    assert isinstance(result, MultimodalToolResult)
    assert "visible text" in result.text
    assert request_media_bytes(result.content) == len(png)
    client.read_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_shared_multi_read_enforces_aggregate_image_budget(client, png, monkeypatch):
    monkeypatch.setattr(VikingMultiReadTool, "_MAX_INLINE_MEDIA_BYTES", len(png) * 2 - 1)
    results = await VikingMultiReadTool.read_resources(client, [IMAGE_URI, IMAGE_URI])
    assert results[0]["success"]
    assert not results[1]["success"]
    assert "combined inline media" in results[1]["content"]
    assert "image" not in results[1]


@pytest.mark.asyncio
async def test_agent_scratch_uses_image_handles_and_preserves_media_capability(pipeline):
    record, _ = await seed_image(pipeline)
    data = {"inputs": [await payload(pipeline, record)]}
    sandbox = SimpleNamespace(write_file=AsyncMock())
    await prepare_assignment(data, sandbox, "/job")
    saved = json.loads(sandbox.write_file.call_args.args[1])
    image = saved["inputs"][0]["payload"]
    assert "media" not in image
    assert image["source_ranges"] == [record.record_id]
    assert ChildProvider(pipeline.model, FileResponse).supports_tool_result_media()


@pytest.mark.asyncio
async def test_image_request_never_silently_uses_incapable_provider(pipeline):
    record, _ = await seed_image(pipeline)
    data = {"inputs": [await payload(pipeline, record)]}
    pipeline.model.provider.supports_tool_result_media.return_value = False
    with pytest.raises(CompileFailure) as caught:
        await pipeline.model.ask("compile", "Read image.", data, FileResponse)
    assert caught.value.code == "UNSUPPORTED_MEDIA"
    pipeline.model.provider.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_stage_reread_preserves_native_tool_result(pipeline, png):
    from vikingbot.compile.results import RecordResponse
    from vikingbot.providers.base import ToolCallRequest

    record, _ = await seed_image(pipeline)
    reads = 0

    async def respond(**kwargs):
        nonlocal reads
        reads += 1
        if reads == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="evidence",
                        name="read_evidence",
                        arguments={"source_range": record.record_id},
                        tokens=0,
                    )
                ],
            )
        result = kwargs["messages"][-1]
        assert result["role"] == "tool"
        assert request_media_bytes(result["content"]) == len(png)
        assert any(part["type"] == "image_url" for part in result["content"])
        return LLMResponse(content='{"records": []}')

    pipeline.model.provider.chat.side_effect = respond
    result = await pipeline.model.ask(
        "map",
        "Inspect assigned evidence.",
        {
            "inputs": [
                {
                    "id": "derived",
                    "payload": {"source_ranges": [record.record_id]},
                }
            ]
        },
        RecordResponse,
    )
    assert result.records == []
    assert reads == 2


@pytest.mark.asyncio
async def test_image_changes_invalidate_source_identity(pipeline, png):
    before, _ = await seed_image(pipeline)
    pipeline.client.download_bytes.return_value = png + b"new-version"
    pipeline.client.stat.return_value = {"size": len(png) + len(b"new-version")}
    after, _ = await seed_image(pipeline)
    assert before.record_id != after.record_id
    assert pipeline.evidence[before.record_id]["hash"] != pipeline.evidence[after.record_id]["hash"]


@pytest.mark.asyncio
async def test_model_request_media_budget_is_checked_before_provider_call(pipeline, monkeypatch):
    record, _ = await seed_image(pipeline)
    monkeypatch.setattr("vikingbot.compile.pipeline_io.MAX_INLINE_TOOL_RESULT_MEDIA_BYTES", 1)
    with pytest.raises(CompileFailure) as caught:
        await pipeline.model.ask(
            "compile",
            "Read image.",
            {
                "inputs": [await payload(pipeline, record)],
            },
            FileResponse,
        )
    assert caught.value.code == "MEDIA_LIMIT"
    pipeline.model.provider.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_revision_reads_still_hash_original_bytes(client):
    client.download_bytes.return_value = b"first\nsecond\nthird\n"
    results = await VikingMultiReadTool.read_resources(
        client,
        [TEXT_URI],
        offset=1,
        limit=1,
        with_revision=True,
    )
    assert results[0]["content"] == "second\n"
    assert results[0]["base_hash"] == hashlib.sha256(client.download_bytes.return_value).hexdigest()


@pytest.mark.asyncio
async def test_agent_preview_keeps_native_media_with_current_loop_signature(pipeline):
    from vikingbot.compile.pipeline_agent import complete_tool_result

    record, _ = await seed_image(pipeline)
    reader = EvidenceReader(
        pipeline.files,
        {
            "inputs": [
                {
                    "id": "derived",
                    "payload": {"source_ranges": [record.record_id]},
                }
            ]
        },
    )
    result = await reader.execute(source_range=record.record_id)
    assert await complete_tool_result(result, "compile-session", preserve_complete=True) is result
