# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Issue #6713: native images must remain readable, cited evidence in Compile."""

import base64
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image, ImageDraw
from vikingbot.agent.loop import _inline_media_bytes
from vikingbot.agent.tools.base import MultimodalToolResult, ToolContext
from vikingbot.agent.tools.ov_file import VikingMultiReadTool
from vikingbot.compile import entry
from vikingbot.compile.hashing import content_hash
from vikingbot.compile.models import CompileFailure, CompileLimits, SanitizedCompileRequest
from vikingbot.compile.ops.common import payload
from vikingbot.compile.pipeline import Pipeline
from vikingbot.compile.pipeline_agent import ChildProvider, complete_tool_result
from vikingbot.compile.results import CombineResponse, FileResponse, RecordResponse
from vikingbot.compile.service import BotCompileService
from vikingbot.compile.skill_resources import EvidenceReader
from vikingbot.compile.sources import read_source, split_source
from vikingbot.providers.base import LLMResponse, ToolCallRequest

URI = "viking://resources/cedar/diagram.png"
TEXT = "viking://resources/cedar/notes.md"


class MemoryFiles:
    def __init__(self):
        self.values = {}

    async def put(self, key, value):
        self.values[key] = value
        return key

    async def get(self, key):
        return self.values.get(key)


@pytest.fixture
def runtime():
    image = Image.new("RGB", (400, 100), "white")
    ImageDraw.Draw(image).text((10, 30), "RETRY LIMIT: 7", fill="black")
    output = BytesIO()
    image.save(output, format="PNG")
    png = output.getvalue()
    client = SimpleNamespace(
        stat=AsyncMock(return_value={"size": len(png)}),
        download_bytes=AsyncMock(return_value=png),
        read_raw=AsyncMock(return_value="# Cedar\n![settings](diagram.png)\n"),
        read_content=AsyncMock(return_value="visible text"),
    )
    provider = SimpleNamespace(supports_tool_result_media=Mock(return_value=True), chat=AsyncMock())
    pipeline = Pipeline(
        client=client,
        provider=provider,
        sandbox=SimpleNamespace(sandbox_cwd="/sandbox"),
        model="vision-test",
        temperature=0,
        limits=CompileLimits(),
        skill="Cite source facts.",
        usage={},
        request=SanitizedCompileRequest(
            **{
                "from": ["viking://resources/cedar"],
                "to": "viking://resources/wiki",
                "skill": "viking://agent/skills/llm-wiki",
                "instruction": "Use image-only facts.",
            }
        ),
    )
    pipeline.files = MemoryFiles()
    pipeline.model.files, pipeline.model.resources = pipeline.files, None
    return pipeline, png


async def seed(runtime):
    pipeline, _ = runtime
    service = SimpleNamespace(
        limits=pipeline.limits, agent_loop=SimpleNamespace(provider=pipeline.model.provider)
    )
    batches = await BotCompileService._prepare_source_batches(
        service, pipeline.client, [], files=[TEXT, URI]
    )

    async def inputs():
        for batch in batches:
            yield batch

    records = await pipeline.seed(inputs())
    return records[-1]


@pytest.mark.asyncio
async def test_preloaded_image_fact_reaches_output_with_source_identity(runtime):
    pipeline, png = runtime
    image = await seed(runtime)
    stored = await pipeline.files.get(image.payload_ref)
    assert "text" not in stored and stored["hash"] == content_hash(png)
    assert (stored["start_line"], stored["end_line"], stored["start_char"], stored["end_char"]) == (
        0,
        0,
        0,
        0,
    )
    data = {"inputs": [await payload(pipeline, image)]}
    assert "base64," not in json.dumps(data)
    pipeline.client.read_raw.assert_awaited_once_with(TEXT)

    def respond(**kwargs):
        if pipeline.model.provider.chat.await_count == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        "read",
                        "read_evidence",
                        {"source_range": image.record_id},
                        0,
                    )
                ],
            )
        content = kwargs["messages"][-1]["content"]
        attached = next(p for p in content if p["type"] == "image_url")
        assert base64.b64decode(attached["image_url"]["url"].split(",")[1]) == png
        assert "RETRY LIMIT" not in json.dumps(data)
        # Simulated visual response: verifies transport and citation, not real-model OCR.
        return LLMResponse(
            content=json.dumps(
                {
                    "files": [
                        {
                            "path": "cedar.md",
                            "content": f"Retry limit: 7. [Source]({URI})",
                            "inputs": [image.record_id],
                        }
                    ]
                }
            )
        )

    pipeline.model.provider.chat.side_effect = respond
    result = await pipeline.model.ask("compile", "Inspect original images.", data, FileResponse)
    assert "Retry limit: 7" in result.files[0].content and URI in result.files[0].content
    assert result.files[0].inputs == [image.record_id]


@pytest.mark.asyncio
async def test_discovered_images_use_existing_multi_read_wrapper(runtime):
    pipeline, png = runtime
    pipeline.changes = {"diff_dirs": [], "full_dirs": ["viking://resources/cedar"], "head": "test"}
    pipeline.tools = {
        "openviking_multi_read": VikingMultiReadTool(),
        "openviking_diff_paths": SimpleNamespace(name="openviking_diff_paths"),
    }
    pipeline.tool_context = ToolContext()

    async def ask(*args, readers, **kwargs):
        result = await readers["openviking_multi_read"].execute(uris=[URI])
        assert isinstance(result, MultimodalToolResult) and URI in result.text
        assert _inline_media_bytes([{"content": result.content}]) == len(png)
        assert "base64," not in result.text
        return SimpleNamespace(needs_pipeline=True)

    pipeline.model.ask = AsyncMock(side_effect=ask)
    assert await entry.run(pipeline, [], {}) is None
    assert next(iter(pipeline.evidence.values()))["hash"] == content_hash(png)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage,schema",
    [("map", RecordResponse), ("reduce", FileResponse), ("combine", CombineResponse)],
)
async def test_pipeline_stages_can_reread_original_images(runtime, stage, schema):
    pipeline, png = runtime
    image = await seed(runtime)
    data = {"inputs": [await payload(pipeline, image, combine=stage == "combine")]}
    result = {"files": []} if schema is FileResponse else {"records": []}
    if schema is CombineResponse:
        result = {"records": [{"inputs": [image.record_id], "content": "Retry limit: 7."}]}
    pipeline.model.provider.chat.side_effect = [
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest("read", "read_evidence", {"source_range": image.record_id}, 0)
            ],
        ),
        LLMResponse(content=json.dumps(result)),
    ]
    await pipeline.model.ask(stage, "Read the image.", data, schema)
    messages = pipeline.model.provider.chat.call_args.kwargs["messages"]
    assert _inline_media_bytes(messages) == len(png)
    reader = EvidenceReader(pipeline.files, data)
    with pytest.raises(ValueError, match="no text line ranges"):
        await reader.execute(source_range=image.record_id, start_line=1, end_line=1)
    with pytest.raises(ValueError, match="outside this assignment"):
        await reader.execute(source_range="foreign")
    result = await reader.execute(source_range=image.record_id)
    assert ChildProvider(pipeline.model, schema).supports_tool_result_media()
    assert await complete_tool_result(result, "session", preserve_complete=True) is result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "problem,code",
    [
        ("provider", "UNSUPPORTED_MEDIA"),
        ("format", "INVALID_ARGUMENT"),
        ("size", "SOURCE_READ_FAILED"),
    ],
)
async def test_image_limitations_never_fall_back_to_binary_text(runtime, problem, code):
    pipeline, _ = runtime
    uri = URI
    if problem == "provider":
        pipeline.model.provider.supports_tool_result_media.return_value = False
    elif problem == "format":
        uri = URI.replace(".png", ".svg")
    else:
        pipeline.client.stat.return_value = {
            "size": VikingMultiReadTool._MAX_INLINE_MEDIA_BYTES + 1
        }
    with pytest.raises(CompileFailure) as caught:
        await read_source(pipeline.client, uri, pipeline.model.provider)
    assert caught.value.code == code
    pipeline.client.read_raw.assert_not_awaited()
    pipeline.client.download_bytes.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["notes.md", "config.ts"])
async def test_text_reads_remain_exact_even_without_media_support(runtime, name):
    pipeline, _ = runtime
    pipeline.model.provider.supports_tool_result_media.return_value = False
    text = await read_source(pipeline.client, f"viking://resources/{name}", pipeline.model.provider)
    assert text == pipeline.client.read_raw.return_value
    assert "".join(p.content for p in split_source(TEXT, text, 1024)) == text


@pytest.mark.asyncio
async def test_multi_read_and_request_budget_still_preserve_native_images(runtime, monkeypatch):
    pipeline, png = runtime
    tool = VikingMultiReadTool()
    tool._get_client, tool._release_client = AsyncMock(return_value=pipeline.client), AsyncMock()
    result = await tool.execute(ToolContext(), [URI])
    assert isinstance(result, MultimodalToolResult)
    assert _inline_media_bytes([{"content": result.content}]) == len(png)
    monkeypatch.setattr("vikingbot.compile.pipeline_io.MAX_INLINE_TOOL_RESULT_MEDIA_BYTES", 1)
    with pytest.raises(CompileFailure, match="media budget"):
        await pipeline.model.call("compile", [{"content": result.content}], [], max_tokens=100)
    pipeline.model.provider.chat.assert_not_awaited()
