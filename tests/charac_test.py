from csv import get_dialect

from pcode_graph.charac import (
    FunctionCharacteristics,
    get_function_characteristics,
    get_graph_diameters,
)
from pathlib import Path
from .fixtures import outdir, log, arm64, x86_64, ArchContext
from tests.helpers import make_graph_from_asm


def test_graph_type(outdir, arm64, log):

    g = make_graph_from_asm(
        """
add x0, x1, x2
mul x1, x0, x0
""",
        arm64,
        outdir,
    )

    f_type = get_function_characteristics(g)
    assert f_type == FunctionCharacteristics.data


def test_loop(outdir: Path, arm64: ArchContext):

    g = make_graph_from_asm(
        "\n".join(
            [
                "mov w2, #0",
                "loop: cmp w0, #0",
                "b.eq exit",
                "add w2, w2, w1",
                "mul w2, w2, w1",
                "sub w0, w0, #1",
                "b loop",
                "exit: ldr x30, [sp]",
                "ret",
            ]
        ),
        arm64,
        outdir,
    )

    assert get_function_characteristics(g).loop


def test_diameters(outdir: Path, arm64: ArchContext):

    g = make_graph_from_asm(
        """
mov x0, 1
b.eq 0x0c
add x0, x0, 1
mul x1, x1, x3
mul x2, x2, x3
mul x3, x1, x2
""",
        arm64,
        outdir,
    )

    d = get_graph_diameters(g)
    assert d.control_only == 5
    assert d.data_only == 3
    assert d.both == 4
