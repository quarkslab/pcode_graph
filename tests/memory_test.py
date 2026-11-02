from pathlib import Path

import pytest


from pcode_graph.graph import CDG, NodeIndex, NodeKinds
from pcode_graph.maker import MakerFlags
from pcode_graph.analysis.program import wrap
from pcode_graph.analysis.alias_analysis import STACK_POINTERS
from pcode_graph.pcode import OpCodes
from .helpers import make_graph_from_asm, make_program_from_asm
from .fixtures import *

X86_ADD_O0 = """
push rbp
mov rbp, rsp
mov dword ptr [rbp-0x14], edi
mov dword ptr [rbp-0x18], esi
mov eax, dword ptr [rbp-0x14]
add eax, dword ptr [rbp-0x18]
mov dword ptr [rbp-4], eax
mov eax, dword ptr [rbp-4]
pop rbp
ret
"""


def find_nodes(graph: CDG, opcode: OpCodes) -> list[NodeIndex]:
    return [index for index, node in enumerate(graph.nodes) if node.opcode == opcode]


def data_predecessor_names(graph: CDG, index: NodeIndex) -> set[str]:
    proxy = graph.compute_edge_proxy()
    return {str(graph.nodes[pred]) for pred in proxy[index].get_data_predecessors()}


def test_spilled_arguments_x86(outdir: Path, x86_64: ArchContext, log):
    graph = make_graph_from_asm(X86_ADD_O0, x86_64, outdir)

    # Only the return address is still loaded
    (load,) = find_nodes(graph, OpCodes.LOAD)
    assert data_predecessor_names(graph, load) == {"INT_ADD"}
    check_outputs(graph, "RAX", "RBP")

    (output_rax,) = [i for i, n in enumerate(graph.nodes) if str(n) == "RAX"]
    (add,) = graph.compute_edge_proxy()[output_rax].get_data_predecessors()
    assert data_predecessor_names(graph, add) == {"EDI", "ESI"}


def test_spilled_arguments_arm64(outdir: Path, arm64: ArchContext, log):
    graph = make_graph_from_asm(
        """
sub sp, sp, #0x10
str w0, [sp, #0xc]
str w1, [sp, #8]
ldr w8, [sp, #0xc]
ldr w9, [sp, #8]
add w0, w8, w9
add sp, sp, #0x10
ret
""",
        arm64,
        outdir,        
    )

    assert not find_nodes(graph, OpCodes.LOAD)
    (output_w0,) = [i for i, n in enumerate(graph.nodes) if str(n) == "x0"]
    (add,) = graph.compute_edge_proxy()[output_w0].get_data_predecessors()
    assert data_predecessor_names(graph, add) == {"w0", "w1"}


def test_enabled_by_default(outdir: Path, x86_64: ArchContext, log):
    graph = make_graph_from_asm(X86_ADD_O0, x86_64, outdir)

    assert len(find_nodes(graph, OpCodes.LOAD)) == 1


def test_loop_counter(outdir: Path, x86_64: ArchContext, log):
    graph = make_graph_from_asm(
        """
push rbp
mov rbp, rsp
mov dword ptr [rbp-4], 0
mov dword ptr [rbp-8], 0
jmp cond
body:
mov eax, dword ptr [rbp-4]
add dword ptr [rbp-8], eax
add dword ptr [rbp-4], 1
cond:
cmp dword ptr [rbp-4], edi
jl body
mov eax, dword ptr [rbp-8]
pop rbp
ret
""",
        x86_64,
        outdir,        
    )

    assert len(find_nodes(graph, OpCodes.LOAD)) == 1

    # Counter and sum are merged with their initial value
    value_phis = [
        index
        for index, node in enumerate(graph.nodes)
        if node.kind == NodeKinds.Phi
        and data_predecessor_names(graph, index) == {"#0x0", "INT_ADD"}
    ]
    assert len(value_phis) == 2


def test_partial_overlap(outdir: Path, arm64: ArchContext, log):
    graph = make_graph_from_asm(
        """
str x0, [sp, #-0x10]
ldr w1, [sp, #-0xc]
""",
        arm64,
        outdir,        
    )

    # Not forwarded, but linked to the stored value
    (load,) = find_nodes(graph, OpCodes.LOAD)
    assert data_predecessor_names(graph, load) == {"x0", "INT_ADD"}
    (store,) = find_nodes(graph, OpCodes.STORE)
    assert not list(graph.compute_edge_proxy()[store].get_data_successors())


def test_load_in_loop(outdir: Path, arm64: ArchContext, log):
    graph = make_graph_from_asm(
        """
loop:
ldr w1, [sp, #-0xc]
str x0, [sp, #-0x10]
cbnz x2, loop
""",
        arm64,
        outdir,        
    )

    # Initial memory is implicit, the load still depends on the stored value
    (load,) = find_nodes(graph, OpCodes.LOAD)
    assert data_predecessor_names(graph, load) == {"x0", "INT_ADD"}


def test_no_alias(outdir: Path, arm64: ArchContext, log):
    graph = make_graph_from_asm(
        """
str x0, [sp, #0x40]
ldp x29, x30, [sp], #0x10
""",
        arm64,
        outdir,        
    )

    for load in find_nodes(graph, OpCodes.LOAD):
        assert len(list(graph.compute_edge_proxy()[load].get_data_predecessors())) == 1


def test_unresolved_store_may_alias(outdir: Path, x86_64: ArchContext, log):
    # RDI may point to RSP-8, as in a ROP gadget
    graph = make_graph_from_asm(
        """
mov qword ptr [rsp-8], rax
mov qword ptr [rdi], rcx
mov rbx, qword ptr [rsp-8]
ret
""",
        x86_64,
        outdir,        
    )

    rbx_load = [
        load
        for load in find_nodes(graph, OpCodes.LOAD)
        if any(
            str(graph.nodes[s]) == "RBX"
            for s in graph.compute_edge_proxy()[load].get_data_successors()
        )
    ]
    assert len(rbx_load) == 1


def test_store_below_stack_pointer_is_kept(outdir: Path, x86_64: ArchContext, log):
    # The next gadget of a chain may read it
    graph = make_graph_from_asm(
        "mov qword ptr [rsp-0x10], rax; ret", x86_64, outdir,
    )

    assert len(find_nodes(graph, OpCodes.STORE)) == 1
    check_outputs(graph)


def test_unresolved_load(outdir: Path, arm64: ArchContext, log):
    graph = make_graph_from_asm(
        """
str x0, [sp, #-0x10]
ldr x1, [x2]
""",
        arm64,
        outdir,        
    )

    (load,) = find_nodes(graph, OpCodes.LOAD)
    assert data_predecessor_names(graph, load) == {"x2"}


def test_call_clobbers_stack(outdir: Path, x86_64: ArchContext, log):
    program = make_program_from_asm(
        """
mov qword ptr [rsp-0x10], rdi
mov rax, qword ptr [rsp-0x10]
call 0x1000
mov rbx, qword ptr [rsp-0x10]
""",
        x86_64,
        outdir,
    )
    memory = program.memory

    # Only the load before the call is resolved
    (call,) = [i for i, op in enumerate(program.operations) if op.opcode.name == "CALL"]
    (forwarded_load,) = memory.forwarded_loads
    assert forwarded_load < call
    assert list(memory.load_defs) == [forwarded_load]


def test_internal_call_does_not_clobber(outdir: Path, x86_64: ArchContext, log):
    # Call used as push + jump, as in obfuscated code
    graph = make_graph_from_asm(
        "call target; nop; target: pop rax; ret", x86_64, outdir,
    )

    # Only the return address of ret is still loaded
    assert len(find_nodes(graph, OpCodes.LOAD)) == 1


def test_address_wraps_around(outdir: Path, x86_64: ArchContext, log):
    # RAX = RSP + 2^64, so the second store overwrites the first one
    program = make_program_from_asm(
        """
mov qword ptr [rsp-8], rcx
mov rax, 0x4000000000000000
add rax, rax
add rax, rax
add rax, rsp
mov qword ptr [rax-8], rdx
mov rbx, qword ptr [rsp-8]
""",
        x86_64,
        outdir,
    )

    stores = [
        index
        for index, op in enumerate(program.operations)
        if op.opcode.name == "STORE"
    ]
    assert list(program.memory.forwarded_loads.values()) == [[stores[1]]]


def test_long_stack_pointer_chain(outdir: Path, x86_64: ArchContext, log):
    program = make_program_from_asm(
        "mov qword ptr [rsp], rax\n" + "add rsp, 8\n" * 600 + "mov rbx, [rsp - 4800]",
        x86_64,
        outdir,
    )

    assert len(program.memory.forwarded_loads) == 1


def test_long_chain_defined_after_use(outdir: Path, x86_64: ArchContext, log):
    # The load precedes the stack pointer definitions in program order
    program = make_program_from_asm(
        "jmp body; use: mov rbx, [rsp]; ret; body: mov [rsp], rax;"
        + "add rsp, 8;" * 2000
        + "sub rsp, 16000; jmp use",
        x86_64,
        outdir,
    )

    # Both the mov and the ret read the single store
    assert len(set(map(tuple, program.memory.forwarded_loads.values()))) == 1
    assert len(program.memory.forwarded_loads) == 2


@pytest.mark.parametrize(
    "arch_fixture", ["x86_64", "x86_32", "arm64", "arm32", "mips_64", "mips_32"]
)
def test_stack_pointers_match_pypcode(
    arch_fixture: str, request: pytest.FixtureRequest
):
    arch: ArchContext = request.getfixturevalue(arch_fixture)
    stack_pointer = STACK_POINTERS[arch.arch]
    registers = arch.translator.context.registers
    assert wrap(registers[stack_pointer.pretty_name]) == stack_pointer


def test_global_variable(outdir: Path, x86_64: ArchContext, log):
    graph = make_graph_from_asm(
        """
mov qword ptr [rip + 0x100], rax
mov rbx, qword ptr [rip + 0xf9]
add qword ptr [rip + 0x200], 1
ret
""",
        x86_64,
        outdir,
    )

    # The written global is forwarded: remain the incremented global and the return address
    assert len(find_nodes(graph, OpCodes.LOAD)) == 2
    assert len(find_nodes(graph, OpCodes.STORE)) == 2
    (output_rbx,) = [i for i, n in enumerate(graph.nodes) if str(n) == "RBX"]
    assert data_predecessor_names(graph, output_rbx) == {"RAX"}


def test_stack_and_globals_do_not_alias(outdir: Path, x86_64: ArchContext, log):
    program = make_program_from_asm(
        """
mov qword ptr [rsp - 8], rcx
mov qword ptr [rip + 0x100], rdx
mov rbx, qword ptr [rsp - 8]
""",
        x86_64,
        outdir,
    )

    assert len(program.memory.forwarded_loads) == 1
