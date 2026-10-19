# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-block payload transport between the stage workers and the router.

A pipelined Causal-Forcing stream moves two kinds of intermediate payload once
per block: the DiT stage's ``StageHandoff`` (pickled latents plus the fields the
decoder needs to rebuild a ``Req``) and the VAE stage's decoded pixels. Both are
large and both are point-to-point, which rules out the obvious channel: the
Dynamo request plane is JSON, so a block would have to be base64'd through it --
about 6.7 MB per 250 ms chunk at 480x832, encoded and decoded on the event loop
of every hop it passes through.

So the payload travels through ``/dev/shm`` and only a *reference* -- a small
JSON dict -- goes over the request plane. This is the same trick used on both
sides of the seam already: SGLang's own scheduler protocol spills arrays over
32 MB to ``/dev/shm`` as ``.npy`` files (``ipc_array.py``), and vLLM-Omni's
shared-memory connector does the same for its inter-stage edges. Reusing the
convention rather than inventing one means the payload never has to be copied
into a different representation on the way past.

Two properties are load-bearing:

*Writes are atomic.* The payload is written to a temporary name in the same
directory and then ``os.replace``d into place, so a reader can never observe a
half-written block. A partial read would not raise -- it would decode into
garbage pixels, or into a truncated pickle, which is far harder to attribute.

*Reads consume.* :func:`take` unlinks the file it just read, exactly as
``NumpyArrayFileRef.materialize`` does. A block has exactly one consumer, and
``/dev/shm`` is RAM: leaking one segment per block would grow with the stream
rather than with its concurrency, so a long session would eventually fill the
node's shared memory. :func:`sweep` covers what a crash leaves behind.

Single node only, and deliberately so. Multi-node needs a real connector (as the
vLLM-Omni path has), which is a transport change and not a protocol one: nothing
outside this module knows how a ref is fulfilled.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ``/dev/shm`` is tmpfs on Linux, so this is RAM: a block never touches a disk on
# its way between two processes on the same node.
_DEFAULT_BLOCK_DIR = "/dev/shm/dynamo_cf_blocks"

KIND_HANDOFF = "handoff"
KIND_FRAMES = "frames"

# Age past which :func:`sweep` will delete a block file it finds at startup.
# Long enough that it cannot race a live stream's in-flight block (a block is
# consumed within seconds of being written), short enough that a crashed run's
# leftovers do not survive into the next one.
_STALE_AGE_S = 3600.0


def block_dir() -> str:
    """Directory holding in-flight block payloads (env: ``DYN_CF_BLOCK_DIR``)."""
    return os.environ.get("DYN_CF_BLOCK_DIR", _DEFAULT_BLOCK_DIR)


def block_key(request_id: str, block: int, kind: str) -> str:
    """Filesystem-safe name for one block's payload.

    Keyed by ``kind`` as well as by block index because both stages of a
    pipelined stream write block N: the DiT stage a handoff, the VAE stage the
    pixels it decoded from that handoff. Without the kind they would collide on
    one name, and the collision would look like a corrupt payload rather than
    like two writers.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in request_id)
    return f"{safe}_b{block}_{kind}"


def _path(key: str) -> str:
    return os.path.join(block_dir(), key)


def _write_atomic(key: str, payload: bytes) -> str:
    directory = block_dir()
    os.makedirs(directory, exist_ok=True)
    # Same directory as the destination so ``os.replace`` is a rename within one
    # filesystem, which is what makes it atomic.
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{key}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        target = _path(key)
        os.replace(tmp, target)
    except BaseException:
        # A tmp file left behind would never be read by anyone and never swept
        # (sweep only matches block refs), so it would pin RAM until reboot.
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return target


def put_handoff(key: str, payload: bytes) -> dict[str, Any]:
    """Publish one pickled ``StageHandoff`` and return its ref."""
    path = _write_atomic(key, payload)
    return {"kind": KIND_HANDOFF, "path": path, "nbytes": len(payload)}


def put_frames(key: str, frames: np.ndarray) -> dict[str, Any]:
    """Publish one block of canonical ``(T, H, W, 3) uint8`` pixels and return its ref.

    Shape and dtype travel in the ref rather than in a header on the payload, so
    the file is exactly the pixel bytes: the reader reconstructs the array with
    one read and no parse. That also keeps the file readable by anything that
    knows the geometry, which is what makes a stuck block debuggable by hand.
    """
    if not isinstance(frames, np.ndarray):
        raise TypeError(f"frames must be an np.ndarray, got {type(frames).__name__}")
    if not frames.flags["C_CONTIGUOUS"]:
        frames = np.ascontiguousarray(frames)
    path = _write_atomic(key, frames.tobytes())
    return {
        "kind": KIND_FRAMES,
        "path": path,
        "nbytes": int(frames.nbytes),
        "shape": [int(d) for d in frames.shape],
        "dtype": str(frames.dtype),
    }


def _take_bytes(ref: dict[str, Any]) -> bytes:
    path = ref.get("path")
    if not path:
        raise ValueError(f"block ref carries no path: {ref!r}")
    try:
        with open(path, "rb") as f:
            payload = f.read()
    except FileNotFoundError as e:
        raise RuntimeError(
            f"block payload {path} is gone. Either it was already taken -- a block "
            "has exactly one consumer -- or the writer and the reader are not on the "
            "same node, which this transport does not support."
        ) from e
    finally:
        # Unlinked whether or not the read succeeded: a payload that could not be
        # read will not become readable, and the stream is over either way.
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    expected = ref.get("nbytes")
    if expected is not None and len(payload) != int(expected):
        raise RuntimeError(
            f"block payload {path} is {len(payload)} bytes, ref says {expected}; "
            "the write was not atomic or the file was truncated"
        )
    return payload


def take_handoff(ref: dict[str, Any]) -> bytes:
    """Read and consume a handoff payload published by :func:`put_handoff`."""
    return _take_bytes(ref)


def take_frames(ref: dict[str, Any]) -> np.ndarray:
    """Read and consume a pixel block published by :func:`put_frames`."""
    payload = _take_bytes(ref)
    shape = tuple(int(d) for d in ref.get("shape") or ())
    dtype = np.dtype(ref.get("dtype") or "uint8")
    if not shape:
        raise ValueError(f"frame ref carries no shape: {ref!r}")
    # ``frombuffer`` would return a read-only view of an immutable ``bytes``, and
    # the encoder downstream is entitled to write into what it is handed.
    return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()


def sweep(max_age_s: float = _STALE_AGE_S) -> int:
    """Delete block files older than ``max_age_s``; return how many went.

    Called once at worker startup. Every block is normally unlinked by its
    consumer, so anything left is the residue of a run that died between the put
    and the take -- and in ``/dev/shm`` that residue is resident memory, held
    until someone deletes it or the node reboots.
    """
    directory = block_dir()
    if not os.path.isdir(directory):
        return 0
    now = time.time()
    removed = 0
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        try:
            if now - os.path.getmtime(path) < max_age_s:
                continue
            os.unlink(path)
            removed += 1
        except (FileNotFoundError, PermissionError):
            # Gone already, or somebody else's: both fine to skip. Another
            # worker sweeping the same directory is the expected case.
            continue
    if removed:
        logger.info("Swept %d stale block payload(s) from %s", removed, directory)
    return removed
