from pathlib import Path

from pypcode import OpCode
from pcode_graph.analysis import (
    MEMORY,
    EntryIndex,
    ExitIndex,
    RichPcodeList,
    UnknownCodeIndex,
    VarKinds,
    wrap,
)
from tests.helpers import make_program_from_asm
from .fixtures import outdir, arm64, ArchContext, log, x86_64, mips_64


def dump_exit_defs(prg: RichPcodeList):
    return {str(v): s for v, s in prg.get_exit_defs().items()}


def test_basic(outdir: Path, arm64: ArchContext, log):

    prg = make_program_from_asm(
        """
mul x0, x1, x2
mul x1, x0, x0
""",
        arm64,
        outdir,
    )

    _, mul1_op, __, mul2_op = prg.operations
    i1, i2 = mul2_op.inputs
    assert i1.getRegisterName() == i2.getRegisterName() == "x0"

    assert prg.get_defs(3, wrap(i1)) == {1}

    assert [prg.successors.get(i, []) for i in range(len(prg.operations))] == [
        [1],
        [2],
        [3],
        [ExitIndex],
    ]

    assert dump_exit_defs(prg) == dict(x0={1}, x1={3})

    assert [(str(v), s) for v, s in prg.iter_input_defs(0)] == []
    assert [(str(v), s) for v, s in prg.iter_input_defs(1)] == [
        ("x1", {EntryIndex}),
        ("x2", {EntryIndex}),
    ]
    assert [(str(v), s) for v, s in prg.iter_input_defs(3)] == [
        ("x0", {1}),
        ("x0", {1}),
    ]


def test_ud_chain(outdir: Path, arm64: ArchContext, log):

    prg = make_program_from_asm(
        """
mov x2, x0
mov x1, x2
mul x1, x1, x3
""",
        arm64,
        outdir,
    )

    _, mov1, __, mov2, ___, mul = prg.operations
    assert mul.inputs[0].getRegisterName() == "x1"
    assert mul.inputs[1].getRegisterName() == "x3"

    assert prg.get_defs(3, wrap(mov2.inputs[0])) == {1}
    assert prg.get_defs(5, wrap(mul.inputs[0])) == {3}
    assert prg.get_defs(5, wrap(mul.inputs[1])) == {EntryIndex}


def test_ret(outdir: Path, arm64: ArchContext, log):

    prg = make_program_from_asm(
        """
mul x0, x1, x2
mul x1, x0, x0
ret
""",
        arm64,
        outdir,
    )

    assert [o.opcode.__name__ for o in prg.operations] == [
        "IMARK",
        "INT_MULT",
        "IMARK",
        "INT_MULT",
        "IMARK",
        "COPY",
        "RETURN",
    ]
    assert dump_exit_defs(prg) == {
        "x0": {1},
        "x1": {3},
        "pc": {5},
    }
    assert prg.reachable == [True] * 7
    assert [prg.predecessors.get(i, set()) for i in range(7)] == [
        {EntryIndex},
        {0},
        {1},
        {2},
        {3},
        {4},
        {5},
    ]


def test_cbranch_x86(outdir: Path, x86_64: ArchContext):

    prg = make_program_from_asm(
        """
            mov rax, 0x1
            jne label
            mov rax, 0x2
label:      ret
""",
        x86_64,
        outdir,
    )

    assert prg.reachable == [True] * len(prg.operations)

    assert [
        (
            i,
            o.opcode.__name__,
            prg.predecessors.get(i, set()),
            prg.successors.get(i, []),
        )
        for i, o in enumerate(prg.operations)
    ] == [
        (0, "IMARK", {EntryIndex}, [1]),
        (1, "COPY", {0}, [2]),
        (2, "IMARK", {1}, [3]),
        (3, "BOOL_NEGATE", {2}, [4]),
        (4, "CBRANCH", {3}, [7, 5]),
        (5, "IMARK", {4}, [6]),
        (6, "COPY", {5}, [7]),
        (7, "IMARK", {6, 4}, [8]),
        (8, "LOAD", {7}, [9]),
        (9, "INT_ADD", {8}, [10]),
        (10, "RETURN", {9}, [UnknownCodeIndex]),
    ]

    assert prg.operations[1].output is not None
    rax_var = wrap(prg.operations[1].output)
    assert rax_var.kind == VarKinds.Register and rax_var.pretty_name == "RAX"
    rax_defs = prg.get_exit_defs().get(rax_var, set())
    assert rax_defs == {1, 6}


def test_simple_load_store(outdir: Path, arm64: ArchContext, log):
    prg = make_program_from_asm(
        """
ldr x0, [x9]
str x0, [x8]
""",
        arm64,
        outdir,
    )

    assert [(i, o.opcode.__name__) for i, o in enumerate(prg.operations)] == [
        (0, "IMARK"),
        (1, "COPY"),
        (2, "LOAD"),
        (3, "IMARK"),
        (4, "COPY"),
        (5, "STORE"),
    ]

    load_inputs = {str(var): defs for var, defs in prg.iter_input_defs(2)}

    assert load_inputs["MEMORY"] == {EntryIndex}
    assert prg.get_exit_defs()[MEMORY] == {5}


def test_store_load(outdir: Path, arm64: ArchContext, log):

    prg = make_program_from_asm(
        """
str x0, [sp, #0x16]
ldr x2, [sp, #0x10]
""",
        arm64,
        outdir,
    )

    assert [(i, o.opcode.__name__) for i, o in enumerate(prg.operations)] == [
        (0, "IMARK"),
        (1, "COPY"),
        (2, "INT_ADD"),
        (3, "STORE"),
        (4, "IMARK"),
        (5, "INT_ADD"),
        (6, "LOAD"),
    ]
    assert prg.reaching_defs[6][MEMORY] == {EntryIndex, 3}
    assert prg.get_exit_defs()[MEMORY] == {3}


def test_load_store2(outdir: Path, arm64: ArchContext, log):

    prg = make_program_from_asm(
        """
ldr x1, [x0]
str x0, [x1]
str x0, [x2]
ldr x2, [x2]
""",
        arm64,
        outdir,
    )

    assert [(i, o.opcode.__name__) for (i, o) in enumerate(prg.operations)] == [
        (0, "IMARK"),
        (1, "COPY"),
        (2, "LOAD"),
        (3, "IMARK"),
        (4, "COPY"),
        (5, "STORE"),
        (6, "IMARK"),
        (7, "COPY"),
        (8, "STORE"),
        (9, "IMARK"),
        (10, "COPY"),
        (11, "LOAD"),
    ]
    assert prg.reaching_defs[11][MEMORY] == {EntryIndex, 5, 8}


def test_add_rsp_ret(outdir: Path, x86_64: ArchContext):
    prg = make_program_from_asm(
        """
add  rsp, 0x8
ret
""",
        x86_64,
        outdir,
    )

    assert {
        str(var)
        for var in prg.get_exit_defs()
        if var.size != 1 and var.kind != VarKinds.Unique
    } == {"RSP", "RIP"}


def test_implicit_dependencies(outdir: Path, arm64: ArchContext, log):

    prg = make_program_from_asm(
        """
mov x8, x1
mov w6, w8
""",
        arm64,
        outdir,
    )

    assert [(i, o.opcode.__name__) for i, o in enumerate(prg.operations)] == [
        (0, "IMARK"),
        (1, "COPY"),
        (2, "IMARK"),
        (3, "INT_ZEXT"),
    ]

    var, defs = list(prg.iter_input_defs(3))[0]
    assert var.pretty_name == "w8"
    assert var.size == 4
    assert defs == {1}


def test_cbranch_to_nop(outdir: Path, x86_64: ArchContext):

    prg = make_program_from_asm(
        """
            je label
            mov rax, 0x2
label:      nop
""",
        x86_64,
        outdir,
    )

    assert [(i, o.opcode.__name__) for i, o in enumerate(prg.operations)] == [
        (0, "IMARK"),
        (1, "CBRANCH"),
        (2, "IMARK"),
        (3, "COPY"),
        (4, "IMARK"),
    ]

    assert prg.predecessors[4] == {1, 3}
    assert prg.successors[1] == [4, 2]


def test_unreachable(outdir: Path, arm64: ArchContext):

    prg = make_program_from_asm(
        """
mul x0, x1, x2
ret
mul x1, x0, x0
""",
        arm64,
        outdir,
    )

    assert [(i, o.opcode.__name__) for i, o in enumerate(prg.operations)] == [
        (0, "IMARK"),
        (1, "INT_MULT"),
        (2, "IMARK"),
        (3, "COPY"),
        (4, "RETURN"),
        (5, "IMARK"),
        (6, "INT_MULT"),
    ]

    assert prg.reachable[0]
    assert not prg.reachable[-1]


def test_cfg_bug(outdir: Path, x86_64: ArchContext):
    asm = """
mov     rax, rdi
mov     edi, esi
call    rax
add     rax, 1
ret
"""
    make_program_from_asm(asm, x86_64, outdir)


def test_pc_relative_accesses(outdir, x86_64, log):

    asm = """    
    push	rbx
    mov	rbx, qword ptr [rip + 0x4cdb]
    mov	rdi, qword ptr [rbx + 0x8]
    call	0x134b
    add	rbx, 0x10
    mov	qword ptr [rip + 0x4cc7], rbx
    pop	rbx
    ret
"""

    prg = make_program_from_asm(
        asm,
        x86_64,
        outdir,
        base_address=0x1379,
    )

    assert prg.operations[5].inputs[0].space.name == "ram"
    assert prg.operations[25].output is not None
    assert prg.operations[25].output.space.name == "ram"
    assert prg.operations[5].inputs[0].offset == prg.operations[25].output.offset


def test_mul_in_loop(outdir, arm64, log):
    asm = """
loop:
mul x1, x1, x2
bne loop
"""
    make_program_from_asm(asm, arm64, outdir, base_address=0x54AE0)


def test_call_in_loop(outdir, x86_64, log):
    asm = """
loop:
call 0x6060
je loop
"""
    p = make_program_from_asm(asm, x86_64, outdir, base_address=0x54AE0)

    assert len(p.predecessors[0]) == 2
    sub_rsp = p.operations[1]
    assert sub_rsp.opcode == OpCode.INT_SUB
    assert sub_rsp.output is not None
    assert sub_rsp.output.getRegisterName() == "RSP"

    call = p.operations[3]
    assert call.opcode == OpCode.CALL
    assert len(p.successors[3]) == 2
    inputs = {var.pretty_name: defs for var, defs in p.iter_input_defs(1)}
    assert inputs["RSP"] == {EntryIndex, 1}
