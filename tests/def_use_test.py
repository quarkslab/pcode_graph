from pcode_graph.analysis.program import (
    EntryIndex,
    OpIndex,
    Program,
    Var,
    VarKinds,
    split,
    wrap,
)
from tests.helpers import make_program_from_asm
from .fixtures import ArchContext, arm64, x86_64


def reference_reaching_defs(program: Program) -> list[dict[Var, set[OpIndex]]]:
    """Naive operation-level fixed point, storing all definitions for each operation."""

    entry = {
        byte: {EntryIndex}
        for index in range(len(program.operations))
        if program.reachable(index)
        for var in program.iter_inputs(index)
        if var.kind == VarKinds.Register
        for byte in split(var)
    }
    inputs: list[dict[Var, set[OpIndex]]] = [{} for _ in program.operations]
    changed = True
    while changed:
        changed = False
        for index, op in enumerate(program.operations):
            if not program.reachable(index):
                continue
            merged: dict[Var, set[OpIndex]] = {}
            for pred in program.pred_ops(index):
                if pred == EntryIndex:
                    pred_outputs = entry
                else:
                    pred_outputs = dict(inputs[pred])
                    output = program.operations[pred].output
                    if output:
                        pred_outputs.update(
                            {byte: {pred} for byte in split(wrap(output))}
                        )
                for byte, defs in pred_outputs.items():
                    merged.setdefault(byte, set()).update(defs)
            if merged != inputs[index]:
                inputs[index] = merged
                changed = True
    return inputs


def check_against_reference(program: Program) -> None:
    reference = reference_reaching_defs(program)
    for index in range(len(program.operations)):
        for var in program.iter_inputs(index):
            expected = set().union(
                *(reference[index].get(byte, set()) for byte in split(var))
            )
            assert program.get_defs(index, var) == expected, (index, var)


def test_matches_reference_with_shared_temporaries(x86_64: ArchContext):
    program = make_program_from_asm(
        """
loop:
cmovz rax, rbx
bsf rcx, rax
mov qword ptr [rsp + 8], rcx
mov dl, byte ptr [rsp + 8]
sub rbx, 1
jnz loop
ret
""",
        x86_64,
    )
    # cmovz and bsf use P-Code temporaries across their internal blocks
    assert any(
        byte.kind == VarKinds.Unique
        for inputs in program.block_inputs
        for byte in inputs
    )
    check_against_reference(program)


def test_matches_reference_on_arm64(arm64: ArchContext):
    program = make_program_from_asm(
        """
loop:
csel x0, x1, x2, eq
add w1, w1, w0
cbz x0, done
subs x2, x2, #1
b.ne loop
done:
ret
""",
        arm64,
    )
    check_against_reference(program)
