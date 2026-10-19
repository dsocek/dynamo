# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

try:
    from dynamo.vllm.omni.connectors import nixl_connector as nixl_module
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)


pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
]


class _FakeDescriptor:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor


class _FakeMeta:
    def __init__(self, token: str):
        self._token = token

    def model_dump(self):
        return {"token": self._token}


class _FakeReadableOp:
    def __init__(self, token: str, registry: dict[str, dict]):
        self._token = token
        self._registry = registry

    def metadata(self):
        return _FakeMeta(self._token)

    async def wait_for_completion(self):
        await self._registry[self._token]["read_done"].wait()


class _FakeReadOp:
    async def wait_for_completion(self):
        return None


class _FakeConnector:
    def __init__(self):
        self._registry: dict[str, dict] = {}
        self._counter = 0

    async def create_readable(self, descriptors):
        desc_list = descriptors if isinstance(descriptors, list) else [descriptors]
        self._counter += 1
        token = f"tok-{self._counter}"
        self._registry[token] = {
            "src": [d.tensor for d in desc_list],
            "read_done": nixl_module.asyncio.Event(),
        }
        return _FakeReadableOp(token, self._registry)

    async def begin_read(self, rdma_meta, local_descriptors):
        token = rdma_meta["token"]
        local_list = (
            local_descriptors
            if isinstance(local_descriptors, list)
            else [local_descriptors]
        )
        src_list = self._registry[token]["src"]
        for src, dst in zip(src_list, [d.tensor for d in local_list], strict=False):
            dst.copy_(src)
        self._registry[token]["read_done"].set()
        return _FakeReadOp()


class _FakeRdmaMetadata:
    @classmethod
    def model_validate(cls, value):
        return value


@pytest.fixture
def fake_nixl(monkeypatch):
    fake_mod = SimpleNamespace(
        Connector=_FakeConnector,
        Descriptor=_FakeDescriptor,
        RdmaMetadata=_FakeRdmaMetadata,
    )
    monkeypatch.setattr(nixl_module, "_nixl_connect", fake_mod)
    monkeypatch.setattr(nixl_module, "_NIXL_AVAILABLE", True)
    return fake_mod


def test_nixl_connector_tensor_roundtrip(fake_nixl):
    connector = nixl_module.DynamoOmniNixlConnector(config={})
    try:
        payload = torch.arange(8, dtype=torch.float32)
        ok, size, metadata = connector.put("0", "1", "req-tensor", payload)

        assert ok is True
        assert metadata is not None
        assert size == payload.numel() * payload.element_size()
        assert metadata["kind"] == "tensor_list"

        out = connector.get("0", "1", "req-tensor", metadata=metadata)
        assert out is not None
        received, rx_size = out
        assert rx_size == size
        assert isinstance(received, torch.Tensor)
        assert torch.equal(received, payload)
    finally:
        connector.close()


def test_nixl_connector_object_roundtrip(fake_nixl):
    connector = nixl_module.DynamoOmniNixlConnector(config={})
    try:
        payload = {
            "kind": "text",
            "outputs": [{"token_ids": [1, 2], "text": "hi", "finish_reason": "stop"}],
            "multimodal_output": {"sr": 16000},
        }
        ok, size, metadata = connector.put("1", "router", "req-obj", payload)

        assert ok is True
        assert metadata is not None
        assert size > 0
        assert metadata["kind"] == "serialized_obj"

        out = connector.get("1", "router", "req-obj", metadata=metadata)
        assert out is not None
        received, _ = out
        assert received == payload
    finally:
        connector.close()


def test_nixl_connector_cleanup_clears_pending(fake_nixl):
    connector = nixl_module.DynamoOmniNixlConnector(config={"pending_timeout_s": 60})
    try:
        ok, _, _ = connector.put(
            "0", "1", "req-clean", torch.tensor([1], dtype=torch.int64)
        )
        assert ok is True
        assert "req-clean" in connector._pending

        connector.cleanup("req-clean")
        assert "req-clean" not in connector._pending
    finally:
        connector.close()


# -- send_device -----------------------------------------------------------
#
# receive_device governs only where the receiver lands the data. put() registers its
# descriptors on whatever device the tensors already occupy, so on a heterogeneous edge
# whose transports cannot read remote device memory the SENDER has to be told too. The
# failure this covers: a CUDA-resident latent block on a TCP-only edge, where the
# receiving host's UCX reports "cannot find remote protocol for: get(multi) into host
# memory from cuda" and the router reports only "empty payload for block 0" -- nothing on
# either side naming the sender's buffer. Under RoCE cuda_copy hid the asymmetry.
#
# "meta" stands in for a foreign device throughout, so these run without a second vendor's
# runtime present; the mechanism under test is device-agnostic.


@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, None),
        ({"send_device": "cpu"}, "cpu"),
        ({"source_device": "cpu"}, "cpu"),
    ],
    ids=["unset", "send_device", "source_device-alias"],
)
def test_send_device_is_resolved_from_config(config, expected):
    resolved = nixl_module._resolve_configured_send_device(config)
    assert (None if resolved is None else resolved.type) == expected


def test_an_unset_send_device_leaves_the_payload_where_it_is():
    """The default must stay "expose it in place": every homogeneous deployment relies on
    it, and a staging copy nobody asked for would be pure overhead."""
    payload = torch.arange(4, dtype=torch.float32)
    tensors, specs = nixl_module._normalize_tensor_payload(payload)
    assert tensors[0].device == payload.device
    assert specs[0]["device"] == str(payload.device)


def test_send_device_moves_the_payload_before_the_spec_is_written():
    """The invariant that makes this safe: the receiver reads spec["device"] back as
    source_device, so the move has to happen before the spec is built. Moving afterwards
    would leave the two disagreeing and the receiver allocating for a device the sender
    no longer exposes."""
    tensors, specs = nixl_module._normalize_tensor_payload(
        torch.arange(4, dtype=torch.float32),
        send_device=torch.device("meta"),
    )
    assert tensors[0].device.type == "meta"
    assert specs[0]["device"] == str(tensors[0].device)


def test_every_tensor_in_a_multi_tensor_payload_is_moved():
    """A latent block arrives as a list, and one straggler left on the source device is
    enough to fail the whole read."""
    tensors, specs = nixl_module._normalize_tensor_payload(
        [torch.zeros(2), torch.ones(3)],
        send_device=torch.device("meta"),
    )
    assert [t.device.type for t in tensors] == ["meta", "meta"]
    assert {s["device"] for s in specs} == {"meta"}


def test_put_registers_and_advertises_the_send_device(fake_nixl):
    """End to end through the config: what put retains for the remote read and what it
    puts on the wire must both be the configured device, not the payload's original."""
    connector = nixl_module.DynamoOmniNixlConnector(config={"send_device": "meta"})
    try:
        ok, _, metadata = connector.put(
            "0", "1", "req-send", torch.arange(4, dtype=torch.float32)
        )
        assert ok is True
        assert metadata["tensor_specs"][0]["device"] == "meta"
        retained = connector._pending["req-send"][2]
        assert [t.device.type for t in retained] == ["meta"]
    finally:
        connector.close()


def test_a_send_device_does_not_disturb_the_serialized_path(fake_nixl):
    """Serialized objects are built on the host already. They must keep round-tripping
    whatever the tensor edge is configured to do, or setting send_device for the latent
    hop would break the router's own object payloads."""
    connector = nixl_module.DynamoOmniNixlConnector(config={"send_device": "cpu"})
    try:
        ok, _, metadata = connector.put("1", "router", "req-obj", {"kind": "text"})
        assert ok is True
        assert metadata["kind"] == "serialized_obj"
        received, _ = connector.get("1", "router", "req-obj", metadata=metadata)
        assert received == {"kind": "text"}
    finally:
        connector.close()


def test_register_dynamoomni_nixl_connector_registers_when_missing(monkeypatch):
    factory = MagicMock()
    factory.list_registered_connectors.return_value = []
    monkeypatch.setattr(nixl_module, "OmniConnectorFactory", factory)
    monkeypatch.setattr(nixl_module, "_OMNI_FACTORY_IMPORT_ERROR", None)

    nixl_module.register_dynamoomni_nixl_connector()

    factory.register_connector.assert_called_once_with(
        "NixlConnector",
        nixl_module.create_dynamoomni_nixl_connector,
    )
