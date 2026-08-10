from .helpers import make_graph_from_asm
from pcode_graph.torch_utils import BinaryConverter
from pcode_graph.gnn_exporter import (
    graph_to_data,
    map_calling_convention_registers,
)
from .fixtures import (
    outdir,
    ArchContext,
    x86_64,
    log,
)
from pathlib import Path


def test_value_to_bits_negative_long(outdir: Path):

    c = BinaryConverter(64)
    v = -(2**63)
    t = c.value_to_bits(v)
    assert t[0] == 1
    assert all(t[i] == 0 for i in range(1, 64))

    v2 = c.bits_to_value(t)
    assert v == v2

    t = c.value_to_bits(-1)
    assert all(t[i] == 1 for i in range(0, 64))

    v2 = c.bits_to_value(t)
    assert -1 == v2


def test_value_to_bits_diff(outdir: Path):
    c = BinaryConverter(64)

    b1 = c.value_to_bits(5281380235)
    b2 = c.value_to_bits(1632657620)
    assert any(b1 != b2)


def test_div_tensor(
    outdir: Path,
    x86_64: ArchContext,
    log,
):
    g = make_graph_from_asm("div rbx", x86_64, outdir)
    data = graph_to_data(
        g,
        map_calling_convention_registers(x86_64.arch),
        max_value_bytes=16,
        include_size=True,
    )
    log.info(data)
