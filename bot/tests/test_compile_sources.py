# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Regression tests for Compile source preparation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from vikingbot.compile.models import CompileFailure
from vikingbot.compile.service import BotCompileService
from vikingbot.compile.sources import split_source


@pytest.mark.parametrize(
    "uri",
    [
        "viking://resources/wiki/diagram.png",
        "viking://resources/wiki/diagram.JPEG?revision=1#preview",
        "viking://resources/wiki/diagram.webp",
    ],
)
def test_split_source_rejects_images_instead_of_treating_bytes_as_text(uri):
    with pytest.raises(
        CompileFailure, match="Compile does not support image evidence sources"
    ) as caught:
        split_source(uri, "\x89PNG\r\n\x1a\n\x00binary", max_chars=1024)
    assert caught.value.code == "INVALID_ARGUMENT"
    assert caught.value.stage == "pipeline"


def test_split_source_keeps_markdown_image_references_as_text():
    content = "# Notes\n\n![diagram](diagram.png)\n"

    ranges = split_source("viking://resources/wiki/notes.md", content, max_chars=1024)

    assert [item.content for item in ranges] == [content]


def test_split_source_keeps_typescript_as_text():
    content = "export const retryLimit = 7;\n"

    ranges = split_source("viking://resources/repo/config.ts", content, max_chars=1024)

    assert [item.content for item in ranges] == [content]


@pytest.mark.asyncio
async def test_prepare_source_batches_rejects_image_before_reading_it():
    client = SimpleNamespace(read_raw=AsyncMock())

    with pytest.raises(
        CompileFailure, match="Compile does not support image evidence sources"
    ) as caught:
        await BotCompileService._prepare_source_batches(
            SimpleNamespace(),
            client,
            [],
            files=["viking://resources/wiki/diagram.png"],
        )

    assert caught.value.code == "INVALID_ARGUMENT"
    client.read_raw.assert_not_awaited()
