"""Preserve .env line endings through the external provider's setup writer."""

import os
import stat

import pytest


def test_lf_file_keeps_lf_on_every_platform(external_provider):
    """Writing one variable must not retype the whole file's line endings.

    Text mode on Windows translates "\n" to CRLF, so an LF .env came back
    with every untouched line rewritten.
    """
    home, _, module, _ = external_provider("env-lf")
    env = home / ".env"
    env.write_bytes(b"A=1\nOPENAI_API_KEY=old\nB=2\n")

    module._write_env_vars(env, {"OPENAI_API_KEY": "new"})

    assert env.read_bytes() == b"A=1\nOPENAI_API_KEY=new\nB=2\n"


def test_crlf_file_keeps_crlf(external_provider):
    """A file saved with CRLF keeps its own line ending."""
    home, _, module, _ = external_provider("env-crlf")
    env = home / ".env"
    env.write_bytes(b"A=1\r\nOPENAI_API_KEY=old\r\nB=2\r\n")

    module._write_env_vars(env, {"OPENAI_API_KEY": "new"})

    assert env.read_bytes() == b"A=1\r\nOPENAI_API_KEY=new\r\nB=2\r\n"


@pytest.mark.parametrize("eol", [b"\n", b"\r\n"], ids=["lf", "crlf"])
def test_env_updates_preserve_bytes_and_safe_credentials(external_provider, eol):
    home, _, module, _ = external_provider("env-credentials")
    env = home / ".env"
    env.write_bytes(
        b"\xef\xbb\xbf" + eol.join([b"OPENAI_API_KEY=old", b"NAME=caf\xe9", b"REMOVE=old"])
        + eol
    )

    module._write_env_vars(
        env, {"OPENAI_API_KEY": "new", "ADDED": "safe\r\nvalue\x00"},
        remove_keys=("REMOVE", "OPENAI_API_KEY"),
    )

    assert env.read_bytes() == eol.join(
        [b"OPENAI_API_KEY=new", b"NAME=caf\xe9", b"ADDED=safevalue"]
    ) + eol
    if os.name == "posix":
        assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_new_env_file_uses_lf_and_can_be_emptied(external_provider):
    home, _, module, _ = external_provider("env-new")
    env = home / "nested" / ".env"

    module._write_env_vars(env, {"OPENAI_API_KEY": "new"})
    assert env.read_bytes() == b"OPENAI_API_KEY=new\n"
    if os.name == "posix":
        assert stat.S_IMODE(env.stat().st_mode) == 0o600
    module._write_env_vars(env, {}, remove_keys=("OPENAI_API_KEY",))
    assert env.read_bytes() == b""
