from pcode_graph.graph import (
    EdgeKinds,
    NodeKinds,
)
from .helpers import make_graph_from_asm
from pcode_graph.pcode import OpCodes
from pcode_graph.gnn_exporter import (
    graph_to_data,
    MAX_OPERAND_NUMBER,
    map_calling_convention_registers,
)
from .fixtures import (
    ArchContext,
    arm64,
    outdir,
    log
)
from pathlib import Path
from pcode_graph.arch import Arch


def test_calling_conv_mapping(outdir: Path, log):

    x86_64 = map_calling_convention_registers(Arch.x86_64)
    arm_64 = map_calling_convention_registers(Arch.arm_64)

    assert (x86_64["rax"] * arm_64["x0"]).sum().item() == 1
    assert x86_64["rax"].sum().item() == 1
    assert arm_64["x0"].sum().item() == 2
    assert (x86_64["rcx"] == arm_64["x3"]).all()


def test_graph_to_data(outdir: Path, arm64: ArchContext, log):
    g = make_graph_from_asm(
        """
mul x0, x1, x2
br x16
""",
        arm64,
        outdir,
    )

    regs = map_calling_convention_registers(arm64.arch)

    data = graph_to_data(
        g,
        regs,
        max_value_bytes=8,
        include_size=True,
    )
    log.info(data)
    assert data.x is not None

    node_features = (
        len(NodeKinds)  # kind, hot-encoded
        + 4  # possible values for sizes, hot-encoded
        + len(OpCodes)  # opcode, hot-encoded
        + len(next(iter(regs.values())))  # register mask
        + 8 * 8  # value bits, hot-encoded
    )
    assert data.x.shape == (len(g.nodes), node_features)  # number of nodes

    num_edges = len(g.edges)
    assert data.edge_index is not None
    assert data.edge_index.shape == (2, num_edges)

    assert data.edge_attr is not None
    assert data.edge_attr.shape == (num_edges, len(EdgeKinds) + MAX_OPERAND_NUMBER)
