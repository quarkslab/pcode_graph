from pathlib import Path
from pcode_graph.analysis import RichPcodeList
from pcode_graph.graph import CDG
from pcode_graph.pcode import dump_operation
from pcode_graph.view import draw_graph
from pcode_graph.maker import MakerFlags, make_graph
from tests.fixtures import ArchContext


def assemble(
    asm: str, arch: ArchContext, outdir: Path | None = None, base_address: int = 0
) -> bytes:
    code = arch.assembler.assemble(asm, base_address)
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "input.asm", "w") as f:
            f.write(arch.translator.dump_asm(code, base_address))
    return code


def load_pcode_ops_from_binary(
    code: bytes,
    arch: ArchContext,
    outdir: Path | None,
    base_address: int,
) -> RichPcodeList:
    operations = arch.translator.translate(code, base_address)
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "input.pcode", "w") as f:
            for o in operations:
                f.write(f"{dump_operation(o)}\n")

    rpl = RichPcodeList(arch.arch, operations)
    if outdir is not None:
        with open(outdir / "analysis.md", "w") as f:
            f.write(rpl.dump())

    return rpl


def make_program_from_asm(
    asm: str,
    arch: ArchContext,
    outdir: Path | None = None,
    base_address: int = 0,
) -> RichPcodeList:

    return load_pcode_ops_from_binary(
        assemble(asm, arch, outdir, base_address),
        arch,
        outdir,
        base_address,
    )


def make_graph_from_binary(
    code: bytes,
    arch: ArchContext,
    outdir: Path | None = None,
    base_address: int = 0,
    flags: MakerFlags | None = None,
) -> CDG:

    prg = load_pcode_ops_from_binary(
        code,
        arch,
        outdir,
        base_address,
    )
    graph = make_graph(prg, flags)

    if outdir is not None:
        # Dump in HTML format
        draw_graph(graph, outdir / "graph.html")

        # Dump in markdown format
        with open(outdir / "graph.md", "w") as f:
            f.write(str(graph))

    return graph


def make_graph_from_asm(
    asm: str,
    arch: ArchContext,
    outdir: Path | None = None,
    base_address: int = 0,
    flags: MakerFlags | None = None,
) -> CDG:

    g = make_graph_from_binary(
        assemble(asm, arch, outdir, base_address),
        arch=arch,
        outdir=outdir,
        base_address=base_address,
        flags=flags,
    )

    return g
