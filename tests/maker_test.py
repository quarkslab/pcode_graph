from pytest import mark
from pcode_graph.graph import CDG, EdgeKinds, NodeKinds
from pcode_graph.maker import MakerFlags
from pcode_graph.pcode import OpCodes
from pcode_graph.registers import SPECIAL_REGISTERS
from .helpers import make_graph_from_asm
from .fixtures import log, outdir, arm64, x86_64, ArchContext
from pathlib import Path


def check_inputs(g: CDG, *expected_names: str):
    names = []
    for n in g.nodes:
        match n.kind:
            case NodeKinds.ReadMemory | NodeKinds.InputRegister | NodeKinds.Constant:
                names.append(str(n))
    assert set(names) == set(expected_names)


def check_outputs(g: CDG, *expected_names: str):
    names = []
    for n in g.nodes:
        match n.kind:
            case NodeKinds.WrittenMemory | NodeKinds.OutputRegister:
                names.append(str(n))
    assert set(names) == set(expected_names)


def check_ops(g: CDG, *expected_opcodes: str):
    assert sorted(
        [
            (n.opcode.name if n.opcode is not None else "NOP")
            for n in g.nodes
            if n.kind == NodeKinds.Operation
        ]
    ) == sorted(expected_opcodes)


def check_kinds(g: CDG, *expected_kinds: NodeKinds):
    assert {n.kind for n in g.nodes} == set(expected_kinds)


def test_simple_copy(log, outdir: Path, arm64: ArchContext):
    g = make_graph_from_asm(
        """
mov x2, x0
""",
        arm64,
        outdir,
    )

    check_inputs(g, "x0")
    check_outputs(g, "x2")
    check_ops(g)


def test_basic(outdir: Path, arm64: ArchContext, log):

    g = make_graph_from_asm(
        """
add x0, x1, x2
mul x1, x0, x0
""",
        arm64,
        outdir,
    )

    check_inputs(g, "x1", "x2")
    check_outputs(g, "x0", "x1")
    check_ops(g, "INT_ADD", "INT_MULT")


def test_ud_chain(outdir: Path, arm64: ArchContext, log):

    g = make_graph_from_asm(
        """
mov x2, x0
mov x1, x2
mul x1, x1, x3
""",
        arm64,
        outdir,
    )

    check_inputs(g, "x0", "x3")
    check_outputs(g, "x2", "x1")
    check_ops(g, "INT_MULT")


def test_add_rsp_x86(outdir: Path, x86_64: ArchContext):

    special_regs = SPECIAL_REGISTERS[x86_64.arch] - {"RSP"}

    g = make_graph_from_asm(
        """
add        rsp, 0x8
ret
""",
        x86_64,
        outdir,
        flags=MakerFlags(reg_outputs_black_list=special_regs),
    )

    check_inputs(g, "#0x8", "RSP", "MEMORY")
    check_outputs(g, "RSP")


def test_simple_load_store(outdir: Path, arm64: ArchContext, log):
    g = make_graph_from_asm(
        """
ldr x0, [x9]
str x0, [x8]
""",
        arm64,
        outdir,
    )

    check_inputs(g, "MEMORY", "x9", "x8")
    check_outputs(g, "x0", "MEMORY")
    check_ops(g, "LOAD", "STORE")


def test_load_store(outdir: Path, arm64: ArchContext, log):

    special_regs = SPECIAL_REGISTERS[arm64.arch] - {"sp"}

    g = make_graph_from_asm(
        """
str x0, [sp, #0x40]
ldp x29, x30, [sp], #0x10
""",
        arm64,
        outdir,
        flags=MakerFlags(reg_outputs_black_list=special_regs),
    )

    load1, load2 = [n for n, node in enumerate(g.nodes) if node.opcode == OpCodes.LOAD]
    (store,) = [n for n, node in enumerate(g.nodes) if node.opcode == OpCodes.STORE]

    # So far, we treat memory as a single space, not maxing aliasing analysis at all
    proxy = g.compute_edge_proxy()
    common_preds = set(proxy[load1].get_data_predecessors()).intersection(
        proxy[load2].get_data_predecessors()
    )
    (phi_index,) = tuple(common_preds)
    assert g.nodes[phi_index].kind == NodeKinds.Phi
    assert {str(g.nodes[p]) for p in proxy[phi_index].get_data_predecessors()} == {
        "MEMORY",
        "STORE",
    }
    check_outputs(g, "x29", "x30", "MEMORY", "sp")
    check_inputs(g, "x0", "MEMORY", "sp", "#0x40", "#0x10", "#0x8")


def test_consecutive_stores(outdir: Path, arm64: ArchContext, log):

    g = make_graph_from_asm(
        """
str x0, [sp, 0x10]
str x1, [sp, 0x20]
ldr x2, [sp, 0x10]
""",
        arm64,
        outdir,
    )

    (load_index,) = [n for n, node in enumerate(g.nodes) if node.opcode == OpCodes.LOAD]
    proxy = g.compute_edge_proxy()
    phi_index, add_index = list(proxy[load_index].get_data_predecessors())
    assert g.nodes[phi_index].kind == NodeKinds.Phi
    assert g.nodes[add_index].opcode == OpCodes.INT_ADD
    assert len(list(proxy[phi_index].get_data_predecessors())) == 3


def test_call_ind_arm(outdir: Path, arm64: ArchContext, log):
    g = make_graph_from_asm(
        """
add x0, x1, x2
blr x16
mul x1, x0, x0
""",
        arm64,
        outdir,
        base_address=0x100,
    )

    # This library is designed to be used for exotic code:
    # We do not resolve call convention to express that x0 could come
    # from the called function.
    check_ops(g, "INT_ADD", "INT_ADD", "CALLIND", "INT_MULT")
    check_inputs(g, "#0x4", "x1", "x2", "x16", "#0x104")
    check_outputs(g, "x0", "x1", "x30")

    call = -1
    for i, n in enumerate(g.nodes):
        if n.opcode == OpCodes.CALLIND:
            call = i
            break
    else:
        assert False, "Call not found"

    assert len(list(g.compute_edge_proxy()[call].get_control_successors())) == 2


def test_div(outdir: Path, x86_64: ArchContext, log):

    make_graph_from_asm("div rbx", x86_64, outdir)


def test_implicit_dependencies(outdir: Path, arm64: ArchContext, log):

    g = make_graph_from_asm(
        """
mov x8, x1
mov w6, w8
mov x0, x6
""",
        arm64,
        outdir,
    )

    check_inputs(g, "x1")
    check_outputs(g, "x8", "x6", "x0")

    x0_output = None
    for i, n in enumerate(g.nodes):
        if n.kind == NodeKinds.OutputRegister and n.register_name == "x0":
            x0_output = i
            break

    assert x0_output is not None

    x1_input = None
    for i, n in enumerate(g.nodes):
        if n.kind == NodeKinds.InputRegister and n.register_name == "x1":
            x1_input = i
            break

    assert x1_input is not None

    proxy = g.compute_edge_proxy()

    assert list(proxy[x0_output].get_data_predecessors()) == [x1_input]


def test_various_opcodes(outdir: Path, arm64: ArchContext, log):

    # Just to check all of theses are supported
    # Notice that code after ret is unreachable
    make_graph_from_asm(
        """
bl 0x1ece0cb78
ldr x8, [sp, 0x20]
mov x0, x20
blr x8
str x0, [x8, #0x6e8]
ldp x29, x30, [sp], #0x10
ret
adrp x8, 0x1000
blr x8
add x9, x9, #0x89
brk #0x5519
ret
""",
        arm64,
        outdir,
    )


def test_various_opcodes_x86(outdir: Path, x86_64: ArchContext):
    # Just to check all of theses are supported:
    make_graph_from_asm(
        """
main: test    rdi, rdi
      jz      loc_5580
      mov     rax, [rdi+38h]
      test    rax, rax
      mov     edx, [rax+8]
      mov     qword ptr [rax+20h], 0
      mov     qword ptr [rdi+28h], 0
      mov     qword ptr [rdi+10h], 0
      mov     qword ptr [rdi+30h], 0
      test    edx, edx
      and     edx, 1
      mov     [rdi+60h], rdx

loc_5580: test rdi, rdi

""",
        x86_64,
        outdir,
    )


def test_load_x86(outdir: Path, x86_64: ArchContext):
    g = make_graph_from_asm(
        """
mov rax, [rdi+38h]
""",
        x86_64,
        outdir,
    )

    check_inputs(g, "RDI", "#0x38", "MEMORY")
    check_outputs(g, "RAX")
    check_ops(g, "INT_ADD", "LOAD")


def test_flag_tmp_removal(outdir: Path, arm64: ArchContext):
    g = make_graph_from_asm(
        "cmp w0, #0; b.eq 0x1c",
        arm64,
        outdir,
    )

    check_inputs(g, "w0", "#0x0")
    check_outputs(g)
    check_ops(g, "INT_SUB", "INT_EQUAL", "CBRANCH")


def test_loop_end_jnz(outdir: Path, x86_64: ArchContext, log):
    asm = """
mov rax, 1
loop:
dec rax
jnz loop
"""
    g = make_graph_from_asm(asm, x86_64, outdir)

    check_ops(g, "INT_SUB", "INT_EQUAL", "BOOL_NEGATE", "CBRANCH")


def test_simple_loop_arm(outdir: Path, arm64: ArchContext, log):
    asm = """
mov x0, 1
loop:
mul x0, x0, x0
b.eq loop
"""
    g = make_graph_from_asm(asm, arm64, outdir)

    # check_ops(g, "INT_SUB", "INT_EQUAL", "BOOL_NEGATE", "CBRANCH")


def test_cbranch_to_nop(outdir: Path, x86_64: ArchContext):

    g = make_graph_from_asm(
        """
            mov rax, 0x1
            jne label
            mov rax, rdi
label:      nop
""",
        x86_64,
        outdir,
    )

    # Check that the second COPY is kept to express the CBRANCH semantics
    # but the first one is removed.
    check_ops(g, "BOOL_NEGATE", "CBRANCH", "COPY")


def test_cbranch_to_nop2(outdir: Path, x86_64: ArchContext):

    g = make_graph_from_asm(
        """
            mov rax, 0x1
            jne label
            mov rax, 0x2
            add rax, rax
label:      nop
""",
        x86_64,
        outdir,
    )

    # Check that the second COPY is removed, as it is not alone in the conditional path
    check_ops(g, "BOOL_NEGATE", "CBRANCH", "INT_ADD")


def test_branch_to_implicit_copy_node(outdir: Path, arm64: ArchContext):

    g = make_graph_from_asm(
        "\n".join(
            [
                "b 0x4",  # 0
                "mov x1, x2",  # 4
            ]
        ),
        arm64,
        outdir,
    )

    check_ops(g)


def test_jump_x86(outdir: Path, x86_64: ArchContext, log):
    g = make_graph_from_asm(
        """mov   rax, rdi
sub   rdi, 0x1
je   skip
add   rax, rdi
skip:
ret
""",
        x86_64,
        outdir,
    )

    proxy = g.compute_edge_proxy()
    (load,) = [i for i, n in enumerate(g.nodes) if n.opcode == OpCodes.LOAD]
    assert len(list(proxy[load].get_control_predecessors())) == 2

    (rax,) = [
        i
        for i, n in enumerate(g.nodes)
        if n.kind == NodeKinds.OutputRegister and n.register_name == "RAX"
    ]
    (rax_pred,) = list(proxy[rax].get_data_predecessors())
    assert g.nodes[rax_pred].kind == NodeKinds.Phi


@mark.timeout(5)
def test_infinite_loop(log, outdir: Path, x86_64: ArchContext):
    # Infinite loop without side effects: the branch instruction is not added into the graph

    asm = """
label: jmp label
"""

    g = make_graph_from_asm(asm, x86_64, outdir)

    assert len(g.nodes) == 2
    assert len(g.edges) == 1
    assert g.edges[0].kind == EdgeKinds.Control
    check_kinds(g, NodeKinds.Begin, NodeKinds.End)


def test_jmp_copy(outdir: Path, x86_64: ArchContext, log):
    g = make_graph_from_asm(
        """
mov        rbx, 0x2
xor        rbx, rcx
mov        rax, 0x1
jmp rax
""",
        x86_64,
        outdir,
    )

    (end,) = [i for i, n in enumerate(g.nodes) if n.kind == NodeKinds.External]
    (branchind,) = [i for i, n in enumerate(g.nodes) if n.opcode == OpCodes.BRANCHIND]
    assert list(g.compute_edge_proxy()[end].get_control_predecessors()) == [branchind]


def test_imul(outdir: Path, x86_64: ArchContext):
    g = make_graph_from_asm("imul rax, rbx", x86_64, outdir)

    # Check that the operations needed to compute the flags has been removed
    num_mult = 0
    for node in g.nodes:
        if node.opcode == OpCodes.INT_MULT:
            num_mult += 1
    assert num_mult == 1


def test_call_ind_x86(outdir: Path, x86_64: ArchContext):
    asm = """
mov     rax, rdi
mov     edi, esi
call    rax
add     rax, 1
ret
"""
    g = make_graph_from_asm(asm, x86_64, outdir)


def test_ropchain(outdir: Path, x86_64: ArchContext):

    # The code following the first ret is considered as unreachable

    asm = """
mov rcx, 9
ret
add rax, rbx
ret
"""

    g = make_graph_from_asm(asm, x86_64, outdir)

    input_regs = set(
        str(n).lower() for n in g.nodes if n.kind == NodeKinds.InputRegister
    )
    assert "rax" not in input_regs
    assert "rbx" not in input_regs

    output_regs = set(
        str(n).lower() for n in g.nodes if n.kind == NodeKinds.OutputRegister
    )
    assert "rcx" in output_regs
    assert "rax" not in output_regs


def test_pc_relative_accesses(outdir: Path, x86_64: ArchContext):

    asm = """
    push rbx
    mov rbx, qword ptr [rip + 0x4cdb]
    mov rdi, qword ptr [rbx + 0x8]
    call 0x134b
    add rbx, 0x10
    mov qword ptr [rip + 0x4cc7], rbx
    pop rbx
    ret
"""
    make_graph_from_asm(asm, x86_64, outdir)


def test_remove_useless_phi(outdir: Path, x86_64: ArchContext):
    asm = """
    mov esi, 0x2
    call 0x1200
    mov esi, 0x2
    call 0x1200
    """

    g = make_graph_from_asm(asm, x86_64, outdir, base_address=0x3B60)

    proxy = g.compute_edge_proxy()
    for i, node in enumerate(g.nodes):
        if node.kind == NodeKinds.Phi:
            assert len(set(proxy[i].get_data_predecessors())) > 1


def test_loop_in_dataflow_error(outdir: Path, x86_64: ArchContext, log):
    # Removed beginning of function:
    # mov rcx, qword ptr [rip + 0x1af962]
    # lea rsi, [rip + 0x1a5ca3]
    asm = """
loop:
    mov rax, qword ptr [rip + 0x1af8e4]
    cmp rcx, rax
    jb end
    lea rdx, [rax + 0x1]
    mov qword ptr [rip + 0x1af8d4], rdx
    movzx eax, byte ptr [rax]
    cmp byte ptr [rsi + rax], 0x0
    je loop
end:
    ret
    """

    make_graph_from_asm(asm, x86_64, outdir, base_address=0x5142F)


def test_simple_call(outdir: Path, x86_64: ArchContext, log):

    asm = """
call 0x546fb
"""
    g1 = make_graph_from_asm(asm, x86_64, outdir / "simple")

    check_inputs(g1, "RSP", "#0x8", "#0x5")  # 0x5 is return address
    check_outputs(g1, "MEMORY")

    ignored_regs = SPECIAL_REGISTERS[x86_64.arch] - {"RSP"}

    g2 = make_graph_from_asm(
        asm,
        x86_64,
        outdir / "full",
        flags=MakerFlags(reg_outputs_black_list=ignored_regs),
    )
    check_inputs(g2, "RSP", "#0x8", "#0x5")  # 0x5 is return address
    check_outputs(g2, "RSP", "MEMORY")


def test_call_in_loop(outdir: Path, x86_64: ArchContext, log):
    asm = """
loop:
call 0x546fb
je loop
"""
    g = make_graph_from_asm(asm, x86_64, outdir)

    check_inputs(g, "RSP", "#0x8", "#0x5", "ZF")


def test_loop_in_dataflow_error2(outdir: Path, x86_64: ArchContext, log):
    asm = """
loop:
call 0x546fb
mov	rax, qword ptr [rip + 0x1ac20a]
lea	rdx, [rax + 0x1]
mov	qword ptr [rip + 0x1ac1ff], rdx
cmp	byte ptr [rax], 0x2c
je loop
"""
    make_graph_from_asm(asm, x86_64, outdir, base_address=0x54AE0)


def test_binutils_operatorf(outdir: Path, x86_64: ArchContext, log):
    # Torture-test with loops and unreachable code
    asm = """
   l44330: push	r12
   l44332: push	rbp
   l44333: push	rbx
   l44334: sub	rsp, 0x20
   l44338: mov	rax, qword ptr fs:[0x28]
   l44341: mov	qword ptr [rsp + 0x18], rax
   l44346: xor	eax, eax
   l44348: mov	rax, qword ptr [rip + 0x1bc9d9]
   l4434f: movzx	eax, byte ptr [rax]
   l44352: mov	dword ptr [rdi], 0x1
   l44358: movzx	ecx, al
   l4435b: mov	edx, 0x0
   l44360: lea	rsi, [rip + 0x1b2d79]
   l44367: cmp	byte ptr [rsi + rcx], 0x0
   l4436b: jne	l44435
   l44371: mov	rbx, rdi
   l44374: movzx	ebp, al
   l44377: lea	rdx, [rip + 0x1b2e62]
   l4437e: test	byte ptr [rdx + rcx], 0x2
   l44382: jne	l443b5
   l44384: cmp	ebp, 0x3e
   l44387: jg	l4446e
   l4438d: cmp	ebp, 0x20
   l44390: jle	l44492
   l44396: lea	eax, [rbp - 0x21]
   l44399: cmp	eax, 0x1d
   l4439c: ja	l44492
   l443a2: mov	eax, eax
   l443a4: lea	rdx, [rip + 0x132a99]
   l443ab: movsxd	rax, dword ptr [rdx + 4*rax]
   l443af: add	rax, rdx
   l443b2: jmp	rax
   l443b5: lea	rdi, [rsp + 0x10]
   l443ba: call	0x44171 #<get_symbol_name>
   l443bf: mov	byte ptr [rsp + 0xf], al
   l443c3: mov	r12, qword ptr [rsp + 0x10]
   l443c8: lea	rdx, [rsp + 0xf]
   l443cd: mov	esi, 0x2
   l443d2: mov	rdi, r12
   l443d5: call	0x80223 #<i386_operator>
   l443da: mov	edx, eax
   l443dc: cmp	eax, 0x1
   l443df: je	l44454
   l443e1: lea	eax, [rax - 0x8]
   l443e4: cmp	eax, 0x2
   l443e7: ja	l44414
   l443e9: mov	edx, 0x5
   l443ee: lea	rsi, [rip + 0x10c620]   # 0x150a15 <_IO_stdin_used+0xa15>
   l443f5: mov	edi, 0x0
   l443fa: call	0x35a40 #<.plt.sec+0x210>
   l443ff: mov	rdi, rax
   l44402: mov	rsi, r12
   l44405: mov	eax, 0x0
   l4440a: call	0x4f4fc #<as_bad>
   l4440f: mov	edx, 0x0
   l44414: mov	rax, qword ptr [rip + 0x1bc90d] # 0x200d28 <input_line_pointer>
   l4441b: movzx	ecx, byte ptr [rsp + 0xf]
   l44420: mov	byte ptr [rax], cl
   l44422: mov	rax, qword ptr [rip + 0x1bc8ff] # 0x200d28 <input_line_pointer>
   l44429: sub	rax, r12
   l4442c: mov	dword ptr [rbx], eax
   l4442e: mov	qword ptr [rip + 0x1bc8f3], r12 # 0x200d28 <input_line_pointer>
   l44435: mov	rax, qword ptr [rsp + 0x18]
   l4443a: sub	rax, qword ptr fs:[0x28]
   l44443: jne	l445e7 #<operatorf+0x2b7>
   l44449: mov	eax, edx
   l4444b: add	rsp, 0x20
   l4444f: pop	rbx
   l44450: pop	rbp
   l44451: pop	r12
   l44453: ret
   l44454: mov	rax, qword ptr [rip + 0x1bc8cd] # 0x200d28 <input_line_pointer>
   l4445b: movzx	edx, byte ptr [rsp + 0xf]
   l44460: mov	byte ptr [rax], dl
   l44462: mov	qword ptr [rip + 0x1bc8bf], r12 # 0x200d28 <input_line_pointer>
   l44469: jmp	l44384 #<operatorf+0x54>
   l4446e: cmp	ebp, 0x7c
   l44471: jne	l44492 #<operatorf+0x162>
   l44473: mov	edx, 0x10
   l44478: mov	rax, qword ptr [rip + 0x1bc8a9] # 0x200d28 <input_line_pointer>
   l4447f: cmp	byte ptr [rax + 0x1], 0x7c
   l44483: jne	l44435 #<operatorf+0x105>
   l44485: mov	dword ptr [rbx], 0x2
   l4448b: mov	edx, 0x1d
   l44490: jmp	l44435 #<operatorf+0x105>
   l44492: movsxd	rbp, ebp
   l44495: lea	rax, [rip + 0x132d84]   # 0x177220 <op_encoding>
   l4449c: mov	edx, dword ptr [rax + 4*rbp]
   l4449f: test	edx, edx
   l444a1: jne	l44435 #<operatorf+0x105>
   l444a3: mov	rbp, qword ptr [rip + 0x1bc87e] # 0x200d28 <input_line_pointer>
   l444aa: mov	edx, 0x0
   l444af: mov	esi, 0x2
   l444b4: mov	edi, 0x0
   l444b9: call	0x80223 #<i386_operator>
   l444be: mov	edx, eax
   l444c0: test	eax, eax
   l444c2: je	l444d0 #<operatorf+0x1a0>
   l444c4: mov	rax, qword ptr [rip + 0x1bc85d] # 0x200d28 <input_line_pointer>
   l444cb: sub	rax, rbp
   l444ce: mov	dword ptr [rbx], eax
   l444d0: mov	qword ptr [rip + 0x1bc851], rbp # 0x200d28 <input_line_pointer>
   l444d7: jmp	l44435 #<operatorf+0x105>
   l444dc: movsxd	rbp, ebp
   l444df: lea	rax, [rip + 0x132d3a]   # 0x177220 <op_encoding>
   l444e6: mov	edx, dword ptr [rax + 4*rbp]
   l444e9: jmp	l44435 #<operatorf+0x105>
   l444ee: mov	rax, qword ptr [rip + 0x1bc833] # 0x200d28 <input_line_pointer>
   l444f5: movzx	eax, byte ptr [rax + 0x1]
   l444f9: cmp	al, 0x3d
   l444fb: je	l4451e #<operatorf+0x1ee>
   l444fd: cmp	al, 0x3e
   l444ff: je	l44525 #<operatorf+0x1f5>
   l44501: mov	edx, 0x18
   l44506: cmp	al, 0x3c
   l44508: jne	l44435 #<operatorf+0x105>
   l4450e: mov	edx, 0xe
   l44513: mov	dword ptr [rbx], 0x2
   l44519: jmp	l44435 #<operatorf+0x105>
   l4451e: mov	edx, 0x19
   l44523: jmp	l44513 #<operatorf+0x1e3>
   l44525: mov	edx, 0x17
   l4452a: jmp	l44513 #<operatorf+0x1e3>
   l4452c: mov	edx, 0x0
   l44531: mov	rax, qword ptr [rip + 0x1bc7f0] # 0x200d28 <input_line_pointer>
   l44538: cmp	byte ptr [rax + 0x1], 0x3d
   l4453c: jne	l44435 #<operatorf+0x105>
   l44542: mov	dword ptr [rbx], 0x2
   l44548: mov	edx, 0x16
   l4454d: jmp	l44435 #<operatorf+0x105>
   l44552: mov	rax, qword ptr [rip + 0x1bc7cf] # 0x200d28 <input_line_pointer>
   l44559: movzx	eax, byte ptr [rax + 0x1]
   l4455d: cmp	al, 0x3d
   l4455f: je	l4457e #<operatorf+0x24e>
   l44561: mov	edx, 0x1b
   l44566: cmp	al, 0x3e
   l44568: jne	l44435 #<operatorf+0x105>
   l4456e: mov	edx, 0xf
   l44573: mov	dword ptr [rbx], 0x2
   l44579: jmp	l44435 #<operatorf+0x105>
   l4457e: mov	edx, 0x1a
   l44583: jmp	l44573 #<operatorf+0x243>
   l44585: mov	rax, qword ptr [rip + 0x1bc79c] # 0x200d28 <input_line_pointer>
   l4458c: movzx	eax, byte ptr [rax + 0x1]
   l44590: cmp	al, 0x21
   l44592: je	l445b1 #<operatorf+0x281>
   l44594: mov	edx, 0x11
   l44599: cmp	al, 0x3d
   l4459b: jne	l44435 #<operatorf+0x105>
   l445a1: mov	dword ptr [rbx], 0x2
   l445a7: mov	edx, 0x17
   l445ac: jmp	l44435 #<operatorf+0x105>
   l445b1: mov	dword ptr [rbx], 0x2
   l445b7: mov	edx, 0x12
   l445bc: jmp	l44435 #<operatorf+0x105>
   l445c1: mov	edx, 0x13
   l445c6: mov	rax, qword ptr [rip + 0x1bc75b] # 0x200d28 <input_line_pointer>
   l445cd: cmp	byte ptr [rax + 0x1], 0x26
   l445d1: jne	l44435 #<operatorf+0x105>
   l445d7: mov	dword ptr [rbx], 0x2
   l445dd: mov	edx, 0x1c
   l445e2: jmp	l44435 #<operatorf+0x105>
   l445e7: call	0x35a60 #<.plt.sec+0x230>
   """

    make_graph_from_asm(asm, x86_64, outdir, base_address=0x44330)


def test_binutils_bfd_log2(outdir: Path, x86_64: ArchContext, log):
    """A test with a negative offset in a branch operation"""

    asm = """
xor	eax, eax
cmp	rdi, 0x1
jbe	l8d1f7 #<bfd_log2+0x17>
sub	rdi, 0x1
bsr	rdi, rdi
lea	eax, [rdi + 0x1]
l8d1f7: ret
nop	dword ptr [rax + rax]
    """

    make_graph_from_asm(asm, x86_64, outdir, base_address=0x8D1E4)


def test_unreachable_loop(outdir: Path, x86_64: ArchContext, log):

    asm = """
    jmp end
loop:
    add rax, rax
    jmp loop
end:
    mov rax, 0x1
"""

    g1 = make_graph_from_asm(asm, x86_64, outdir / "skip")

    check_inputs(g1, "#0x1")
    check_outputs(g1, "RAX")
    check_ops(g1)


def test_reachable_useless_loop(outdir: Path, x86_64: ArchContext, log):

    asm = """
loop:
    jz loop
end:
    mov rax, 0x1
"""

    g = make_graph_from_asm(asm, x86_64, outdir)

    # TODO: add checks
    # check_inputs(g, "#0x1")
    # check_outputs(g, "RAX")
    # check_ops(g, "COPY")


def test_reachable_useless_loop2(outdir: Path, x86_64: ArchContext, log):

    asm = """
loop:
    add rax, rax
    jz loop
end:
    mov rax, 0x1
"""

    g = make_graph_from_asm(asm, x86_64, outdir)

    # TODO: add checks
    # check_inputs(g, "#0x1")
    # check_outputs(g, "RAX")
    # check_ops(g, "COPY")


def test_binutils_as_new_parse_ident(outdir: Path, x86_64: ArchContext, log):
    # Non-reg test: was presented an infinite loop in make_graph when building CFG

    asm = """
l1403c0: push	r13
l1403c2: mov	r8, rdi
l1403c5: mov	rax, rsi
l1403c8: push	r12
l1403ca: push	rbp
l1403cb: xor	ebp, ebp
l1403cd: push	rbx
l1403ce: mov	r10d, dword ptr [rsi + 0x34]
l1403d2: mov	rdx, qword ptr [rsi + 0x20]
l1403d6: mov	r9, qword ptr [rsi + 0x8]
l1403da: cmp	r10d, -0x1
l1403de: je	l1404f0 #<parse_ident+0x130>
l1403e4: cmp	rdx, r9
l1403e7: jb	l1404d0 #<parse_ident+0x110>
l1403ed: mov	dword ptr [rax + 0x28], 0x1
l1403f4: test	byte ptr [rip + 0x73145], 0x4 # 0x1b3540 <_sch_istable>
l1403fb: je	l140490 #<parse_ident+0xd0>
l140401: mov	rdi, qword ptr [rax + 0x20]
l140405: mov	ebx, 0x4
l14040a: mov	rdx, -0x30
l140411: lea	r11, [rip + 0x73128]    # 0x1b3540 <_sch_istable>
l140418: cmp	rdi, r9
l14041b: jb	l140430 #<parse_ident+0x70>
l14041d: test	bx, bx
l140420: je	l140480 #<parse_ident+0xc0>
l140422: mov	dword ptr [rax + 0x28], 0x1
l140429: jmp	l140422 #<parse_ident+0x62>
l14042b: nop	dword ptr [rax + rax]
l140430: mov	r12, qword ptr [rax]
l140433: movsx	ecx, byte ptr [r12 + rdi]
l140438: movzx	r13d, cl
l14043c: mov	esi, ecx
l14043e: test	byte ptr [r11 + 2*r13], 0x4
l140443: je	l140468 #<parse_ident+0xa8>
l140445: lea	rdx, [rdx + 4*rdx]
l140449: add	rdx, rdx
l14044c: test	sil, sil
l14044f: jne	l1404c0 #<parse_ident+0x100>
l140451: mov	dword ptr [rax + 0x28], 0x1
l140458: mov	rcx, -0x30
l14045f: add	rdx, rcx
l140462: jmp	l140418 #<parse_ident+0x58>
l140464: nop	dword ptr [rax]
l140468: cmp	r10d, -0x1
l14046c: je	l140480 #<parse_ident+0xc0>
l14046e: cmp	byte ptr [r12 + rdi], 0x5f
l140473: je	l1405d0 #<parse_ident+0x210>
l140479: mov	rdi, qword ptr [rax + 0x20]
l14047d: nop	dword ptr [rax]
l140480: mov	rcx, rdx
l140483: add	rcx, rdi
l140486: mov	qword ptr [rax + 0x20], rcx
l14048a: jae	l140548 #<parse_ident+0x188>
l140490: mov	dword ptr [rax + 0x28], 0x1
l140497: xor	edi, edi
l140499: xor	edx, edx
l14049b: xor	eax, eax
l14049d: xor	esi, esi
l14049f: pop	rbx
l1404a0: mov	qword ptr [r8 + 0x10], rax
l1404a4: mov	rax, r8
l1404a7: pop	rbp
l1404a8: pop	r12
l1404aa: mov	qword ptr [r8], rdi
l1404ad: mov	qword ptr [r8 + 0x8], rdx
l1404b1: pop	r13
l1404b3: mov	qword ptr [r8 + 0x18], rsi
l1404b7: ret
l1404b8: nop	dword ptr [rax + rax]
l1404c0: add	rdi, 0x1
l1404c4: sub	ecx, 0x30
l1404c7: mov	qword ptr [rax + 0x20], rdi
l1404cb: movsxd	rcx, ecx
l1404ce: jmp	l14045f #<parse_ident+0x9f>
l1404d0: mov	r12, qword ptr [rsi]
l1404d3: movzx	esi, byte ptr [r12 + rdx]
l1404d8: cmp	sil, 0x75
l1404dc: jne	l140501 #<parse_ident+0x141>
l1404de: add	rdx, 0x1
l1404e2: mov	ebp, 0x1
l1404e7: mov	qword ptr [rax + 0x20], rdx
l1404eb: nop	dword ptr [rax + rax]
l1404f0: cmp	rdx, r9
l1404f3: jae	l1403ed #<parse_ident+0x2d>
l1404f9: mov	r12, qword ptr [rax]
l1404fc: movzx	esi, byte ptr [r12 + rdx]
l140501: test	sil, sil
l140504: je	l1403ed #<parse_ident+0x2d>
l14050a: lea	rcx, [rdx + 0x1]
l14050e: lea	rdi, [rip + 0x7302b]    # 0x1b3540 <_sch_istable>
l140515: movzx	edx, sil
l140519: movsx	r11d, sil
l14051d: mov	qword ptr [rax + 0x20], rcx
l140521: test	byte ptr [rdi + 2*rdx], 0x4
l140525: je	l140490 #<parse_ident+0xd0>
l14052b: cmp	sil, 0x30
l14052f: jne	l140606 #<parse_ident+0x246>
l140535: cmp	r10d, -0x1
l140539: jne	l1405e0 #<parse_ident+0x220>
l14053f: mov	rdi, rcx
l140542: xor	edx, edx
l140544: nop	dword ptr [rax]
l140548: cmp	r9, rcx
l14054b: jb	l140490 #<parse_ident+0xd0>
l140551: add	rdi, qword ptr [rax]
l140554: test	ebp, ebp
l140556: je	l1405a0 #<parse_ident+0x1e0>
l140558: test	rdx, rdx
l14055b: je	l1405f6 #<parse_ident+0x236>
l140561: mov	rcx, rdx
l140564: xor	esi, esi
l140566: jmp	l140579 #<parse_ident+0x1b9>
l140568: nop	dword ptr [rax + rax]
l140570: add	rsi, 0x1
l140574: test	rcx, rcx
l140577: je	l1405b8 #<parse_ident+0x1f8>
l140579: sub	rcx, 0x1
l14057d: cmp	byte ptr [rdi + rcx], 0x5f
l140581: jne	l140570 #<parse_ident+0x1b0>
l140583: test	rsi, rsi
l140586: je	l1405f3 #<parse_ident+0x233>
l140588: mov	rax, rdx
l14058b: mov	rdx, rcx
l14058e: sub	rax, rsi
l140591: add	rax, rdi
l140594: jmp	l1405a4 #<parse_ident+0x1e4>
l140596: nop	word ptr cs:[rax + rax]
l1405a0: xor	esi, esi
l1405a2: xor	eax, eax
l1405a4: xor	ecx, ecx
l1405a6: test	rdx, rdx
l1405a9: cmove	rdi, rcx
l1405ad: jmp	l14049f #<parse_ident+0xdf>
l1405b2: nop	word ptr [rax + rax]
l1405b8: mov	rax, rdx
l1405bb: xor	edx, edx
l1405bd: sub	rax, rsi
l1405c0: add	rax, rdi
l1405c3: xor	edi, edi
l1405c5: jmp	l14049f #<parse_ident+0xdf>
l1405ca: nop	word ptr [rax + rax]
l1405d0: add	rdi, 0x1
l1405d4: jmp	l140480 #<parse_ident+0xc0>
l1405d9: nop	dword ptr [rax]
l1405e0: cmp	rcx, r9
l1405e3: jae	l14053f #<parse_ident+0x17f>
l1405e9: mov	rdi, rcx
l1405ec: xor	edx, edx
l1405ee: jmp	l14046e #<parse_ident+0xae>
l1405f3: mov	rdx, rcx
l1405f6: mov	dword ptr [rax + 0x28], 0x1
l1405fd: xor	esi, esi
l1405ff: xor	eax, eax
l140601: jmp	l14049f #<parse_ident+0xdf>
l140606: movzx	ebx, word ptr [rip + 0x72f33] # 0x1b3540 <_sch_istable>
l14060d: lea	edx, [r11 - 0x30]
l140611: mov	rdi, rcx
l140614: movsxd	rdx, edx
l140617: and	ebx, 0x4
l14061a: jmp	l140411 #<parse_ident+0x51>
l14061f: nop
"""
    make_graph_from_asm(asm, x86_64, outdir, base_address=0x1403C0)


def test_copy_removal_in_branch(outdir: Path, arm64: ArchContext, log):

    g = make_graph_from_asm(
        """
mov x0, 1
b.eq 0x0c
add x0, x0, 1
mul x1, x1, x3
""",
        arm64,
        outdir,
    )

    check_ops(g, "CBRANCH", "INT_ADD", "INT_MULT")


def test_useless_cbranch(outdir: Path, x86_64: ArchContext, log):

    g = make_graph_from_asm(
        """
            mov rax, 0x1
            jne label
            nop
label:      nop
""",
        x86_64,
        outdir,
    )

    # Everything should be simpified
    check_ops(g, "BOOL_NEGATE", "CBRANCH")


def test_invariance_by_reordering(outdir: Path, arm64: ArchContext, log):

    g1 = make_graph_from_asm(
        """
mov x0, 1
b.eq 0x0c
add x0, x0, 1
mul x1, x1, x3
mul x2, x2, x3
mul x3, x1, x2
""",
        arm64,
        outdir / "g1",
    )

    g2 = make_graph_from_asm(
        """
mov x0, 1
b.eq 0x0c
add x0, x0, 1
mul x2, x2, x3
mul x1, x1, x3
mul x3, x1, x2
""",
        arm64,
        outdir / "g2",
    )

    # TODO: add checks after switching to BB version of graph
    # Computing graph equality is complicated,
    # rather we could check by hand some simple properties


def test_mba_dataflow(outdir: Path, x86_64: ArchContext, log):
    asm = """
mov r11, r10
neg r11
mov r12, r10
mov r13, r10
not r13
neg r13
add r13, r14
neg r13
add r12, r13
add r11, r12
not r11
"""

    g = make_graph_from_asm(asm, x86_64, outdir, flags=MakerFlags(build_cfg=False))

    check_ops(
        g,
        "INT_NEGATE",
        "INT_2COMP",
        "INT_ADD",
        "INT_2COMP",
        "INT_ADD",
        "INT_2COMP",
        "INT_ADD",
        "INT_NEGATE",
    )

    check_inputs(g, "R10", "R14")
    check_outputs(g, "R11", "R12", "R13")

    # Check that no begin/end node is generated
    check_kinds(
        g, NodeKinds.InputRegister, NodeKinds.OutputRegister, NodeKinds.Operation
    )


def test_constant_simplification(outdir: Path, x86_64: ArchContext, log):

    asm = """
LEA ECX,[RSI + RDI*0x1]
XOR EAX,EAX
CMP EDI,ESI
CMOVZ EAX,ECX
RET
"""

    g = make_graph_from_asm(asm, x86_64, outdir, flags=MakerFlags(build_cfg=False))

    # Check no orphan node

    proxy = g.compute_edge_proxy()

    for n, node in enumerate(g.nodes):
        if node.kind == NodeKinds.Constant:
            assert list(proxy[n].get_data_successors()), f"Orphan constant {node}"


def test_lea_simplification(outdir: Path, x86_64: ArchContext, log):

    asm = """
LEA ECX,[RSI + RDI*0x1]
XOR EAX,EAX
CMP EDI,ESI
CMOVZ EAX,ECX
RET
"""

    g = make_graph_from_asm(asm, x86_64, outdir, flags=MakerFlags(build_cfg=True))

    proxy = g.compute_edge_proxy()

    for n, node in enumerate(g.nodes):
        if node.kind == NodeKinds.Operation and node.opcode == OpCodes.INT_MULT:
            for n2 in proxy[n].get_data_predecessors():
                node2 = g.nodes[n2]
                if node2.kind == NodeKinds.Constant:
                    assert node2.value != 1, "Multiplication by 1 not simplified"
