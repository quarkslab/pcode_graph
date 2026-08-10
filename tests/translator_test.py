from pathlib import Path

import pypcode
from pytest import mark
import pytest
from pcode_graph.pcode import dump_operation
from pcode_graph.translator import ChunkOps, LiftStats, Translator
from tests.fixtures import x86_64, outdir, ArchContext, arm64


def test_assembling_x86(outdir: Path, x86_64: ArchContext):
    code = x86_64.assembler.assemble(
        """
mov    eax,esi
imul   eax,edi
xor    ecx,ecx
test   edi,edi
cmovg  ecx,esi
sub    eax,ecx
ret
""",
        0,
    )

    ops = x86_64.translator.translate(code, 0)
    with open(outdir / "input.pcode", "w") as f:
        for o in ops:
            f.write(f"{dump_operation(o)}\n")


def imark_addresses(chunks: list[ChunkOps]) -> list[int]:    
    return [
        input.offset
        for chunk in chunks
        for op in chunk.operations
        if op.opcode == pypcode.OpCode.IMARK
        for input in op.inputs
    ]


X86_64_PUSH_RBP = b"\x55"
X86_64_MOV_RBP_RSP = b"\x48\x89\xe5"
X86_64_MOV_EAX_1 = b"\xb8\x01\x00\x00\x00"
X86_64_RET = b"\xc3"
X86_64_GARBAGE = b"\xff\xff"
X86_64_NOP = b"\x90"

X86_64_FUNCTION = X86_64_PUSH_RBP + X86_64_MOV_RBP_RSP + X86_64_MOV_EAX_1 + X86_64_RET


@mark.parametrize("window", [2, 3, 7, 10, 16, 4096])
def test_window_smaller_than_code(x86_64: ArchContext, window: int):

    base = 0x400000
    code = X86_64_FUNCTION * 5
    expected_addresses = [
        base + i * len(X86_64_FUNCTION) + delta for i in range(5) for delta in (0, 1, 4, 9)
    ]

    st = LiftStats()
    chunks = list(
        x86_64.translator.iter_operations(code, base, window=window, stats=st)
    )

    assert imark_addresses(chunks) == expected_addresses
    assert sum(c.num_instructions for c in chunks) == 20
    assert st.num_instructions == 20
    assert st.lifted_bytes == len(code)
    assert st.coverage == 1.0
    assert st.skipped_bytes == st.truncated_bytes == 0
    assert st.resyncs == 0

    assert chunks[0].address == base
    assert [c.address for c in chunks] == sorted({c.address for c in chunks})

    if window < len(code):
        assert st.translate_calls > 1


def test_window_boundary_does_not_split_an_instruction(x86_64: ArchContext):

    base = 0x1000
    code = X86_64_PUSH_RBP + X86_64_MOV_EAX_1 + X86_64_RET  # window=3 cuts the MOV
    st = LiftStats()
    chunks = list(x86_64.translator.iter_operations(code, base, window=3, stats=st))

    assert imark_addresses(chunks) == [base, base + 1, base + 6]
    assert st.lifted_bytes == len(code)
    assert st.truncated_bytes == 0


def test_stats(outdir: Path, x86_64: ArchContext):

    code = (X86_64_PUSH_RBP + X86_64_MOV_RBP_RSP + X86_64_MOV_EAX_1 + X86_64_RET) * 200
    st = LiftStats()
    chunks = list(
        x86_64.translator.iter_operations(code, 0x400000, window=64, stats=st)
    )

    assert st.lifted_bytes == len(code)
    assert st.num_instructions == 800
    assert st.skipped_bytes == 0 and st.truncated_bytes == 0


def test_garbage_management(outdir: Path, x86_64: ArchContext):

    prefix = X86_64_PUSH_RBP + X86_64_MOV_RBP_RSP
    suffix = X86_64_MOV_EAX_1 + X86_64_RET
    code = prefix + X86_64_GARBAGE + suffix
    st = LiftStats()

    chunks = list(
        x86_64.translator.iter_operations(code, 0x1000, window=4096, stats=st)
    )
    num_instructions = sum(c.num_instructions for c in chunks)
    assert num_instructions == 4
    assert st.lifted_bytes + st.skipped_bytes + st.truncated_bytes == len(code)


def test_truncated_instruction(outdir: Path, x86_64: ArchContext):

    code = X86_64_PUSH_RBP + b"\xb8\x01\x00"  # two missing bytes
    st = LiftStats()
    chunks = list(x86_64.translator.iter_operations(code, 0x2000, stats=st))
    num_instructions = sum(c.num_instructions for c in chunks)

    assert num_instructions == 1
    assert st.truncated_bytes == 3
    assert st.lifted_bytes == 1


def test_all_invalid(outdir: Path, x86_64: ArchContext):
    code = X86_64_GARBAGE * 500
    st = LiftStats()
    chunks = list(x86_64.translator.iter_operations(code, 0x3000, window=128, stats=st))
    assert len(chunks) == 0
    assert st.lifted_bytes == 0
    assert st.skipped_bytes + st.truncated_bytes == len(code)
    assert st.coverage == 0.0


A64_STP = b"\xfd\x7b\xbf\xa9"  # stp x29, x30, [sp, #-0x10]!
A64_MOV_X0_1 = b"\x20\x00\x80\xd2"  # mov x0, #1
A64_ADD_X0_1 = b"\x00\x04\x00\x91"  # add x0, x0, #1
A64_NOP = b"\x1f\x20\x03\xd5"
A64_RET = b"\xc0\x03\x5f\xd6"
A64_GARBAGE = b"\xff\xff\xff\xff"  # "Unable to resolve constructor" -> BadDataError


def test_arm64_resync_and_truncation(arm64: ArchContext):
    """AArch64 : resynchronisation par pas de 4, et section coupée en plein vol."""

    base = 0x8000
    code = (
        A64_STP
        + A64_MOV_X0_1
        + A64_GARBAGE
        + A64_ADD_X0_1
        + A64_RET
        + A64_STP[:2]  # truncated section
    )
    assert len(code) == 22

    st = LiftStats()
    chunks = list(arm64.translator.iter_operations(code, base, stats=st))

    assert imark_addresses(chunks) == [base, base + 4, base + 12, base + 16]
    assert sum(c.num_instructions for c in chunks) == 4
    assert all(addr % 4 == 0 for addr in imark_addresses(chunks))

    assert st.lifted_bytes == 16
    assert st.skipped_bytes == 4
    assert st.resyncs == 1
    assert st.truncated_bytes == 2

    # Comptabilité complète : chaque octet est lifté, sauté ou tronqué une fois.
    assert st.lifted_bytes + st.skipped_bytes + st.truncated_bytes == st.total_bytes
    assert st.coverage == pytest.approx(16 / 22)


@mark.parametrize("window", [4, 8, 12, 4096])
def test_arm64_window_smaller_than_code(arm64: ArchContext, window: int):
    base = 0x8000
    code = (A64_STP + A64_MOV_X0_1 + A64_ADD_X0_1 + A64_NOP + A64_RET) * 4

    st = LiftStats()
    chunks = list(
        arm64.translator.iter_operations(code, base, window=window, stats=st)
    )

    assert imark_addresses(chunks) == [base + 4 * i for i in range(len(code) // 4)]
    assert st.lifted_bytes == len(code)
    assert st.skipped_bytes == st.truncated_bytes == 0
    assert st.coverage == 1.0
