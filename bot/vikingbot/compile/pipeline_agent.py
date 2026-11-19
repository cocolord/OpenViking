"""Private file-tool workers; natural completion returns their validated Markdown files."""

from __future__ import annotations

import asyncio
import json
import shlex
import uuid
from collections import Counter
from copy import copy
from typing import Any

from openviking.utils.path_safety import sanitize_relative_viking_path
from vikingbot.agent.loop import _PlainTextDelivered, _PlainTextFinal
from vikingbot.agent.tools.base import Tool, ToolContext
from vikingbot.agent.tools.compile import CompileChildTool
from vikingbot.agent.tools.registry import ToolRegistry
from vikingbot.compile import file_ops
from vikingbot.compile.models import COMPILE_DRAFT_ROOT
from vikingbot.compile.pipeline_io import retry_allowed
from vikingbot.compile.renderer import output_path_key, validate_relative_file_path
from vikingbot.compile.results import Document, IncompleteWorkerOutput, InvalidWorkerOutput
from vikingbot.compile.skill_resources import EvidenceReader
from vikingbot.providers.base import LLMProvider


class ChildProvider(LLMProvider):
    """Route each child request through the task's bounded, metered provider calls."""

    def __init__(self, model, stage="agent"):
        super().__init__()
        self.model = model
        self.stage = stage
        self.failures = Counter()

    def get_default_model(self):
        return self.model.model

    def should_inline_tool_result_media(self, model=None):
        """Send original evidence; model capability is decided by the endpoint."""
        return True

    async def chat(self, messages, tools=None, **kwargs):
        """Retry truncated output without executing incomplete tool calls."""
        while True:
            response = await self.model.call(
                self.stage, messages, tools, max_tokens=self.model.max_tokens
            )
            if response.finish_reason not in {"length", "max_tokens"}:
                return response
            error = "Agent output was truncated; use smaller writes and complete tool calls"
            if not retry_allowed(self.failures, error):
                raise ValueError(error)
            self.model.metrics["repairs"] += 1
            messages = [*messages, {"role": "user", "content": error}]


async def complete_tool_result(result, session_key, *, preserve_complete: bool = False):
    """Keep scoped tool evidence complete for subsequent model requests.

    Child file tools cannot read shared preview spill files, so head/tail previews
    are not a lossless delivery mechanism for these isolated assignments.
    """
    return result


class DraftDeleteTool(Tool):
    """Remove one file at the workspace path bound by CompileChildTool."""

    name = "delete_file"
    description = "Delete an obsolete output file."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, sandbox, root):
        self.sandbox, self.root = sandbox, root

    async def execute(self, tool_context: ToolContext, **kwargs: Any) -> str:
        path = kwargs["path"]
        try:
            local = self.sandbox.local_file_path(path)
        except FileNotFoundError:
            return "File is already absent."
        if local is not None:
            if not local.resolve().is_relative_to((self.sandbox.workspace / self.root).resolve()):
                raise ValueError("Output file resolves outside this worker's directory")
            await asyncio.to_thread(local.unlink)
        else:
            result = await self.sandbox.execute(f"rm -f -- {shlex.quote(path)}")
            self.sandbox._ensure_command_succeeded(result, "draft file removal")
        return "Removed obsolete output file."


class TargetReader(Tool):
    """Read a known target path and remember its revision before allowing replacement."""

    name = "read_target_file"
    description = "Read a target file you need to update. Use read_file for your drafts."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to request.to."}
        },
        "required": ["path"],
    }

    def __init__(self, runtime):
        self.runtime = runtime

    async def execute(self, tool_context=None, path=""):
        """Return existing UTF-8 Markdown or an explicit missing-file result; errors propagate."""
        path = validate_relative_file_path(path)
        content = await file_ops.load_old(self.runtime, path)
        if isinstance(content, bytes):
            raise ValueError("Target is not UTF-8 text")
        return json.dumps({"path": path, "content": content}, ensure_ascii=False)


async def retry_drafts(sandbox, output_root: str) -> list[Document]:
    """Select reusable Markdown from the previous private worker's output directory."""
    root = sanitize_relative_viking_path(output_root)
    if not root.startswith(COMPILE_DRAFT_ROOT + "/") or not root.endswith("/output"):
        raise ValueError("Invalid private worker output directory")
    prefix = root + "/"
    drafts = []
    for entry in await sandbox.list_files(root, max_entries=None):
        if not entry.path.startswith(prefix):
            raise ValueError("Worker output is outside its private output directory")
        if not entry.size or not entry.path.lower().endswith(".md"):
            continue
        try:
            path = validate_relative_file_path(entry.path[len(prefix) :])
        except ValueError:
            continue
        payload = await sandbox.read_file_bytes(entry.path)
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if text.strip():
            drafts.append(Document(path, entry.path))
    return sorted(drafts, key=lambda draft: draft.path)


class CopyDrafts(Tool):
    """Copy explicitly selected read-only drafts into the current output directory."""

    name = "copy_drafts"
    description = (
        "Copy reviewed previous-attempt drafts to their original output-relative paths. "
        "Pass their content_file aliases from assignment.json. Existing outputs are kept."
    )
    parameters = {
        "type": "object",
        "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
        "required": ["paths"],
        "additionalProperties": False,
    }

    def __init__(self, sandbox, output_root, drafts: dict[str, Document], write_lock: asyncio.Lock):
        self.sandbox, self.output_root = sandbox, output_root
        self.drafts, self.write_lock = drafts, write_lock

    async def execute(self, tool_context: ToolContext, **kwargs: Any) -> str:
        selected = list(dict.fromkeys(kwargs["paths"]))
        if any(alias not in self.drafts for alias in selected):
            raise ValueError("Copy paths must be previous-attempt content_file aliases")
        prefix = self.output_root + "/"
        copied, skipped = [], []
        async with self.write_lock:
            existing = set()
            for entry in await self.sandbox.list_files(self.output_root, max_entries=None):
                if not entry.path.startswith(prefix):
                    raise ValueError("Worker output is outside its private output directory")
                existing.add(output_path_key(entry.path[len(prefix) :]))
            for alias in selected:
                draft = self.drafts[alias]
                key = output_path_key(draft.path)
                if key in existing:
                    skipped.append(draft.path)
                    continue
                payload = await self.sandbox.read_file_bytes(draft.workspace_path)
                await self.sandbox.write_file_bytes(prefix + draft.path, payload)
                existing.add(key)
                copied.append(draft.path)
        return json.dumps({"copied": copied, "skipped": skipped}, ensure_ascii=False)


def agent_runner(loop, session_key, connection, limits):
    """Bind metered AgentLoops with isolated file tools and no submission protocol."""

    async def run(system, data, model, *, stage, runtime, previous_attempt=None) -> list[Document]:
        """Return validated files only after natural completion."""
        child_id = uuid.uuid4().hex
        root = f"{COMPILE_DRAFT_ROOT}/{child_id}"
        output_root = f"{root}/output"
        sandbox = model.files.sandbox
        assignment = dict(data)
        assignment["documents"] = []
        readonly_paths = {"assignment.json": f"{root}/assignment.json"}
        evidence_instructions = "Read assignment.json and all assigned evidence. "
        for index, document in enumerate(data.get("documents", [])):
            path = f"inputs/{index}.md"
            await sandbox.write_file(
                f"{root}/{path}", await sandbox.read_file(document["workspace_path"])
            )
            assignment["documents"].append({"path": document["path"], "content_file": path})
            readonly_paths[path] = f"{root}/{path}"
        drafts = {}
        if previous_attempt is not None and previous_attempt.output_root is not None:
            for index, draft in enumerate(
                await retry_drafts(sandbox, previous_attempt.output_root)
            ):
                alias = f"inputs/previous/{index}.md"
                drafts[alias] = draft
                readonly_paths[alias] = draft.workspace_path
        if drafts:
            assignment["previous_attempt"] = {
                "reason": str(previous_attempt),
                "drafts": [
                    {"path": draft.path, "content_file": alias} for alias, draft in drafts.items()
                ],
            }
            evidence_instructions = (
                "Read assignment.json and review its previous_attempt drafts first. "
                "Use copy_drafts to reuse reviewed pages at their listed output paths. "
                "Treat the old drafts as read-only, and write or edit only in the new output. "
                "Use the current assignment and evidence to verify and fill specific gaps. "
            )
        await sandbox.write_file(
            f"{root}/assignment.json", json.dumps(assignment, ensure_ascii=False)
        )
        registry = ToolRegistry(config=loop.config)
        write_lock = asyncio.Lock() if drafts else None
        if write_lock is not None:
            registry.register(CopyDrafts(sandbox, output_root, drafts, write_lock))
        for name in ("read_file", "write_file", "edit_file", "delete_file"):
            tool = (
                DraftDeleteTool(sandbox, output_root)
                if name == "delete_file"
                else loop.tools.get(name)
            )
            if tool is None:
                raise ValueError(f"Compile requires {name}")
            registry.register(
                CompileChildTool(
                    tool,
                    output_root,
                    merge_only=True,
                    readonly_paths=readonly_paths,
                    write_lock=write_lock,
                )
            )
        evidence = EvidenceReader(model.files, data)
        if evidence.allowed:
            registry.register(evidence)
        registry.register(runtime.resources)
        can_update_target = runtime.target_has_content is not False or bool(runtime.changes)
        if can_update_target:
            registry.register(TargetReader(runtime))
        search = runtime.tools.get("openviking_search") if can_update_target else None
        if search is not None and runtime.tool_context is not None:

            async def search_target(tool_context=None, **arguments):
                arguments["target_uri"] = runtime.target
                return await search.execute(runtime.tool_context, **arguments)

            scoped = copy(search)
            scoped.execute = search_target
            registry.register(scoped)
        child = copy(loop)
        child.provider = ChildProvider(model, stage)
        child._preview_tool_result = complete_tool_result

        completed_outputs: list[Document] | None = None
        validation_error: InvalidWorkerOutput | None = None
        completion_error: Exception | None = None

        async def check_completion(context):
            nonlocal completed_outputs, validation_error, completion_error
            try:
                completed_outputs = await runtime.collect_outputs(output_root, stage)
            except InvalidWorkerOutput as error:
                validation_error = error
                messages = child.context.add_assistant_message(context.messages, context.text, [])
                return _PlainTextDelivered(
                    messages=messages,
                    tools_used=[],
                    next_prompt=(
                        f"Fix only this output issue: {error}. Keep valid Markdown and its facts. "
                        "Use delete_file for obsolete drafts. Once fixed, reply Done."
                    ),
                )
            except Exception as error:
                # AgentLoop swallows callback exceptions for ordinary chat replies.
                # Carry infrastructure failures out without regenerating the files.
                completion_error = error
            return _PlainTextFinal()

        async def remind_completion(iteration):
            remaining = limits.subagent_iterations - iteration + 1
            if remaining in (20, 5, 1):
                return (
                    f"{remaining} model/tool rounds remain, including this one. "
                    "Reply Done as soon as the requested files are ready."
                )
            return None

        final, _, _, _, _ = await child._run_agent_loop(
            messages=[
                {
                    "role": "system",
                    "content": system
                    + "\n"
                    + evidence_instructions
                    + "Write only deliverable Markdown at publication-relative paths. "
                    "Use delete_file for obsolete drafts. Reply Done once required writes succeed. "
                    "Reopen files only to resolve a specific issue. "
                    f"Budget: {limits.subagent_iterations} rounds.",
                },
                {
                    "role": "user",
                    "content": "Process assignment.json and write the requested Markdown files.",
                },
            ],
            session_key=session_key,
            publish_events=False,
            tool_registry=registry,
            openviking_tool_names=set(registry.tool_names),
            openviking_connection=connection,
            allow_final_fallback=False,
            require_natural_completion=True,
            on_plain_text=check_completion,
            status_note_provider=remind_completion,
            inject_write_experience=False,
            agent_id=child_id,
            context_compact_budget=None,
            max_iterations=limits.subagent_iterations,
        )
        if completion_error is not None:
            raise completion_error
        if final is None or completed_outputs is None:
            message = "Compile worker exhausted its tool rounds before completion"
            if validation_error is not None:
                message += f". Last output validation error: {validation_error}"
            raise IncompleteWorkerOutput(message, output_root=output_root) from validation_error
        return completed_outputs

    return run
