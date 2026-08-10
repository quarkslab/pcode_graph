from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, override
from pypcode import OpCode, PcodeOp, Varnode
from pcode_graph.arch import Arch
from pcode_graph.pcode import OpCodes, dump_operation, from_native_opcode
from pcode_graph.visitor import PcodeVisitor

OpIndex = int
EntryIndex = -1
UnknownCodeIndex = -2
ExitIndex = -3

Address = int


def dump_index(i: OpIndex) -> str:
    match i:
        case -1:
            return "entry"
        case -2:
            return "elsewhere"
        case -3:
            return "exit"
        case _:
            assert i >= 0
            return str(i)


class VarKinds(Enum):
    # Register or named P-Code temporary
    Register = 0
    # P-Code temporary
    Unique = 1
    # Numerical constant
    Constant = 2
    # PC-relative address value
    Ram = 3
    # Dummy access to the memory
    IndirectAccess = 4


@dataclass(frozen=True, order=True)
class Var:
    """Hashable and comparable wrapper for temporaries, registers, constants and memory."""

    kind: VarKinds
    size: int = field(compare=False)
    value: int
    pretty_name: str = field(compare=False)

    def __str__(self) -> str:
        return self.pretty_name


def wrap(var: Varnode) -> Var:
    match var.space.name:
        case "register":
            return Var(VarKinds.Register, var.size, var.offset, var.getRegisterName())
        case "unique":
            return Var(VarKinds.Unique, var.size, var.offset, f"${var.offset}")
        case "const":
            return Var(VarKinds.Constant, var.size, var.offset, f"${var.offset}")
        case "ram":
            return Var(VarKinds.Ram, var.size, var.offset, f"[0x{var.offset:x}]")
        case _ as name:
            raise ValueError(f"Unexpected space for input variable: {name}")


# Special variable singleton
MEMORY = Var(VarKinds.IndirectAccess, 0, 0, "MEMORY")


@dataclass
class RichPcodeList:
    """A class to run static analysis on a list of P-Code operations."""

    # Source architecture
    arch: Arch

    # List of P-Code operations
    operations: list[PcodeOp]

    # CFG edges: src/dst indexes on the previous list.
    # Order of successors matters for cbranch, call, etc.
    predecessors: dict[OpIndex, set[OpIndex]] = field(default_factory=dict)
    successors: dict[OpIndex, list[OpIndex]] = field(default_factory=dict)

    # First P-Code operation for each instruction address
    addresses: dict[Address, OpIndex] = field(default_factory=dict)

    # Reachable operation flags
    reachable: list[bool] = field(default_factory=list)

    # Reaching definitions. For each operation, by variable, list of:
    #   - definitions (= writing operation index) that reach this operation,
    #   - or EntryIndex (for initial definitions of inputs).
    reaching_defs: list[dict[Var, set[OpIndex]]] = field(default_factory=list)

    def __post_init__(self):
        InstructionAddressCollector(self)()
        CFGBuilder(self)()
        UnreachableCodeFinder(self)()
        ReachingDefBuilder(self)()

    def __str__(self) -> str:
        return self.dump()

    def dump(self) -> str:
        """Dumps the program and analysis results in a markdown table."""

        s = "|index|op|preds|succs|input defs|reachable|exit def|\n"
        exit_defs = self.get_exit_defs()

        s += "|--|--|--|--|--|--|--|\n"
        for i, op in enumerate(self.operations):
            s += f"|{i}|{dump_operation(op)}|"
            s += ", ".join(dump_index(p) for p in self.predecessors.get(i, [])) + "|"
            s += ", ".join(dump_index(p) for p in self.successors.get(i, [])) + "|"
            var_defs = []
            for var, defs in self.iter_input_defs(i):
                if defs:
                    var_defs.append(
                        str(var) + " from " + " or ".join(dump_index(d) for d in defs)
                    )

            s += ", ".join(var_defs) + "|"
            if self.reachable[i]:
                s += "x"
            s += "|"
            exit_def_here = []
            for var, defs in exit_defs.items():
                for j in defs:
                    if i == j:
                        exit_def_here.append(str(var))
            if exit_def_here:
                s += ", ".join(exit_def_here)
            s += "|\n"

        return s

    def get_defs(self, index: OpIndex, var: Var) -> set[OpIndex]:
        """Returns all the operations defining the value at given operation of the given variable."""

        return self.reaching_defs[index].get(var, set())

    def get_exit_defs(self) -> dict[Var, set[OpIndex]]:
        """Computes side effects of the program."""

        exit_defs: dict[Var, set[OpIndex]] = {}

        for index, op in enumerate(self.operations):
            if not self.reachable[index]:
                continue

            if not {UnknownCodeIndex, ExitIndex}.intersection(
                self.successors.get(index, [])
            ):
                # Not an exit
                continue

            for var, defs in self.reaching_defs[index].items():
                if var.kind == VarKinds.Unique:
                    # Ignore P-Code temporary
                    continue
                write_defs = defs - {EntryIndex}
                if write_defs:
                    # logger.debug(f"Add exit defs for var {var} from op {index}: {defs}")
                    exit_defs.setdefault(var, set()).update(write_defs)
            if op.output:
                var = wrap(op.output)
                if var.kind != VarKinds.Unique:
                    exit_defs[var] = {index}
            elif op.opcode == OpCode.STORE:
                # Do not kill other memory defs
                exit_defs.setdefault(MEMORY, set()).add(index)
        return exit_defs

    def iter_inputs(self, index: OpIndex) -> Iterator[Var]:
        """Iterates operation inputs, skipping **special** ones (see P-Code documentation)."""

        op = self.operations[index]
        if op.opcode == OpCode.LOAD:
            yield MEMORY

        for i, var in enumerate(op.inputs):
            if i != 0 or op.opcode not in {
                OpCode.IMARK,
                OpCode.CBRANCH,
                OpCode.BRANCH,
                OpCode.CALL,
                OpCode.LOAD,
                OpCode.STORE,
            }:
                yield wrap(var)

    def iter_input_defs(self, index: OpIndex) -> Iterator[tuple[Var, set[OpIndex]]]:
        """For each operand, returns the written variable and the operation(s) defining it."""

        for var in self.iter_inputs(index):
            yield var, self.get_defs(index, var)


class PassBase(PcodeVisitor):

    def __init__(self, program: RichPcodeList) -> None:
        super().__init__()
        self.program = program


class InstructionAddressCollector(PassBase):
    """Collects instruction addresses removing IMARK operations"""

    def __call__(self):
        for i, op in enumerate(self.program.operations):
            if op.opcode == OpCode.IMARK:
                self.program.addresses[op.inputs[0].offset] = i


class CFGBuilder(PassBase):
    """List the CFG edges"""

    def __call__(self):

        # Collected src/dst pairs
        self._edges: list[tuple[OpIndex, OpIndex]] = []

        for o, op in enumerate(self.program.operations):
            opcode = from_native_opcode(op.opcode)
            self.visit(opcode, o, op)

        for src, dst in self._edges:
            assert src >= 0 and src < len(self.program.operations)
            self.program.successors.setdefault(src, []).append(dst)
            if dst not in {UnknownCodeIndex, ExitIndex}:
                self.program.predecessors.setdefault(dst, set()).add(src)

        self.program.predecessors.setdefault(0, set()).add(EntryIndex)

    @override
    def visit_default(self, index: int, op: PcodeOp):
        # Edge to next operation
        next_index = index + 1
        if next_index == len(self.program.operations):
            next_index = ExitIndex
        self._edges.append((index, next_index))

    def _get_jump_edge(self, index: int, op: PcodeOp):
        # Edge to branched operation
        address_op = op.inputs[0]
        match address_op.space.name:
            case "ram":
                target_index = self.program.addresses.get(
                    address_op.offset, UnknownCodeIndex
                )
                self._edges.append((index, target_index))
            case "const":
                # Detect negative offsets
                bits = address_op.size * 8
                sign_bit = 1 << (bits - 1)
                if (address_op.offset & sign_bit) != 0:
                    offset = address_op.offset - (1 << bits)
                else:
                    offset = address_op.offset
                self._edges.append((index, index + offset))
            case "register" | "unique":
                self._edges.append((index, UnknownCodeIndex))
            case _ as name:
                raise ValueError(f"Unexpected address space for jump address: {name}")

    @override
    def visit_branch(self, index: int, op: PcodeOp):
        # Unconditional branching without return
        self._get_jump_edge(index, op)

    @override
    def visit_branchind(self, index: int, op: PcodeOp):
        # Unconditional branching without return
        self._get_jump_edge(index, op)

    @override
    def visit_return(self, index: int, op: PcodeOp):
        # Unconditional branching without return
        self._get_jump_edge(index, op)

    @override
    def visit_call(self, index: int, op: PcodeOp):
        self._get_jump_edge(index, op)
        if self._edges[-1][-1] is UnknownCodeIndex:
            # Calling an unseen function: we suppose it returns here after
            self.visit_default(index, op)

    @override
    def visit_callind(self, index: int, op: PcodeOp):
        self._get_jump_edge(index, op)
        if self._edges[-1][-1] is UnknownCodeIndex:
            # Calling an unseen function: we suppose it returns here after
            self.visit_default(index, op)

    @override
    def visit_cbranch(self, index: int, op: PcodeOp):
        # Conditional branching
        self._get_jump_edge(index, op)
        self.visit_default(index, op)


class UnreachableCodeFinder(PassBase):
    def __call__(self):
        self.program.reachable = [False] * len(self.program.operations)
        active: set[int] = {0}
        while active:
            index = active.pop()
            self.program.reachable[index] = True
            for s in self.program.successors.get(index, []):
                if s >= 0 and not self.program.reachable[s]:
                    active.add(s)


class ReachingDefBuilder(PassBase):
    def __call__(self):

        # logger.debug("Initialize defs")
        # Initialize empty defs for each operation
        self.program.reaching_defs = [{} for i in range(len(self.program.operations))]

        # To detect the end of propagation
        changed: bool = True

        def add_def(var: Var, to_op: OpIndex, from_op: OpIndex):
            nonlocal changed
            assert to_op >= 0

            defs = self.program.reaching_defs[to_op].setdefault(var, set())
            if from_op not in defs:
                # logger.debug(f" {var} from {from_op} reaching {to_op}")
                defs.add(from_op)
                changed = True

        # Initialize with start defs
        for index, op in enumerate(self.program.operations):

            if not self.program.reachable[index]:
                continue

            if op.output:
                var = wrap(op.output)
                for succ in self.program.successors.get(index, []):
                    if succ >= 0:
                        add_def(var, succ, index)

            elif op.opcode == OpCode.STORE:
                for succ in self.program.successors.get(index, []):
                    if succ >= 0:
                        add_def(MEMORY, succ, index)

            for var in self.program.iter_inputs(index):
                if var.kind in {
                    VarKinds.IndirectAccess,
                    VarKinds.Ram,
                    VarKinds.Register,
                }:
                    add_def(var, 0, EntryIndex)

        # Propagate defs to predecessors
        while changed:
            changed = False

            # logger.debug("Now reaching defs are:")
            # for index, op in enumerate(self.program.operations):
            #     for var, defs in self.program.reaching_defs[index].items():
            #         logger.debug(
            #             f"{index:>5}: {dump_operation(op)}: {var} from {', '.join(dump_index(d) for d in defs)}"
            #         )

            for index, op in enumerate(self.program.operations):

                if not self.program.reachable[index]:
                    continue

                for p in self.program.predecessors.get(index, set()):

                    if p < 0:
                        continue

                    pred_op = self.program.operations[p]
                    for var, defs in self.program.reaching_defs[p].items():

                        if pred_op.output and wrap(pred_op.output) == var:
                            # Kill previous defs of output variable.
                            # As we do not analyze which part of memory is read or written,
                            # a write to memory won't kill previous definitions of MEMORY.
                            continue

                        for writer_index in defs:
                            add_def(var, index, writer_index)
