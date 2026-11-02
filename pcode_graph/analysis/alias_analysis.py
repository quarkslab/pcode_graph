"""Summary analysis of stack and static memory accesses.

LOAD and STORE addresses are resolved relatively to the stack pointer value at the entry
of the code, or as absolute constants, so that loads can be linked to the stores they read.
This mainly targets spilled variables, as found in code compiled without optimization (-O0),
and global variables.

No assumption is made on the use of the stack: calls to unknown code and stores to
unresolved addresses may write anywhere, so the loads they reach remain unresolved.
Stack and absolute addresses are supposed not to alias each other.
"""

from dataclasses import dataclass, field
import heapq

from pypcode import OpCode, PcodeOp

from pcode_graph.analysis.interning import Interner
from pcode_graph.analysis.pass_base import PassBase
from pcode_graph.arch import Arch
from pcode_graph.analysis.program import (
    EntryIndex,
    Memory,
    OpIndex,
    Program,
    UnknownCodeIndex,
    Var,
    VarKinds,
    wrap,
)
from pcode_graph.bit_utils import unsigned_to_signed

# Stack pointer register of each architecture, as identified by pypcode
STACK_POINTERS: dict[Arch, Var] = {
    Arch.x86_64: Var(VarKinds.Register, 8, 0x20, "RSP"),
    Arch.x86_32: Var(VarKinds.Register, 4, 0x10, "ESP"),
    Arch.arm_64: Var(VarKinds.Register, 8, 0x8, "sp"),
    Arch.arm_32: Var(VarKinds.Register, 4, 0x54, "sp"),
    Arch.mips_64: Var(VarKinds.Register, 8, 0xE8, "sp"),
    Arch.mips_32: Var(VarKinds.Register, 4, 0x74, "sp"),
}

# Pseudo definition of memory written by not analyzed code or an unresolved store
UnknownWriteIndex = -4


@dataclass(frozen=True, slots=True)
class StackAddress:
    """Address relative to the stack pointer value at entry."""

    offset: int


# Known constant, stack address or unknown value
AbstractValue = int | StackAddress | None

# Tracked memory byte: (is relative to the stack, offset or absolute address)
MemoryByte = tuple[bool, int]


@dataclass(frozen=True, slots=True)
class MemoryAccess:
    in_stack: bool
    offset: int
    size: int

    @property
    def bytes(self) -> list[MemoryByte]:
        return [
            (self.in_stack, offset)
            for offset in range(self.offset, self.offset + self.size)
        ]


DefinitionSet = frozenset[OpIndex]


@dataclass(slots=True)
class MemoryState:
    """Definitions of memory bytes, updated in place."""

    # Definitions of tracked bytes
    tracked: dict[MemoryByte, DefinitionSet]

    # Definitions of untracked bytes
    default: DefinitionSet

    def copy(self) -> "MemoryState":
        return MemoryState(dict(self.tracked), self.default)

    def read(self, access: MemoryAccess) -> DefinitionSet:
        return frozenset().union(
            *(self.tracked.get(byte, self.default) for byte in access.bytes)
        )

    def write(self, access: MemoryAccess, written: DefinitionSet) -> None:
        for byte in access.bytes:
            self.tracked[byte] = written

    def clobber(self, intern: Interner[DefinitionSet]) -> None:
        unknown = {UnknownWriteIndex}
        for byte, defs in self.tracked.items():
            self.tracked[byte] = intern(defs | unknown)
        self.default = intern(self.default | unknown)

    def merge(self, other: "MemoryState", intern: Interner[DefinitionSet]) -> bool:
        """Adds the definitions of the other state, returns True if this state changed."""

        default_grows = not other.default <= self.default
        changed = default_grows
        for byte, theirs in other.tracked.items():
            mine = self.tracked.get(byte)
            if mine is None:
                if not theirs <= self.default:
                    changed = True
                elif not default_grows:
                    continue
                self.tracked[byte] = intern(self.default | theirs)
            elif mine is not theirs and not theirs <= mine:
                self.tracked[byte] = intern(mine | theirs)
                changed = True
        for byte, mine in self.tracked.items():
            if byte not in other.tracked and not other.default <= mine:
                self.tracked[byte] = intern(mine | other.default)
                changed = True
        if default_grows:
            self.default = intern(self.default | other.default)
        return changed


ADDRESS_OPCODES = {OpCode.COPY, OpCode.INT_ADD, OpCode.INT_SUB, OpCode.INT_OR}
CALL_OPCODES = {OpCode.CALL, OpCode.CALLIND, OpCode.CALLOTHER}


class MemoryAnalyzer(PassBase):

    def __init__(self, program: Program) -> None:
        super().__init__(program)

        self.stack_pointer = STACK_POINTERS.get(program.arch)
        self._output_values: dict[OpIndex, AbstractValue] = {}
        self._loads: dict[OpIndex, MemoryAccess] = {}
        self._stores: dict[OpIndex, MemoryAccess | None] = {}
        self._intern: Interner[DefinitionSet] = Interner()

    @property
    def result(self) -> Memory:
        return self.program.memory

    def __call__(self):
        if self.stack_pointer is None or not self.program.basic_blocks:
            return

        for index, op in enumerate(self.program.operations):
            if not self.program.reachable(index):
                continue
            if op.output is not None:
                # Evaluating in program order bounds the recursion depth
                self._output_value(index, wrap(op.output))
                if op.opcode == OpCode.LOAD:
                    access = self._compute_access(index, op.output.size)
                    if access is not None:
                        self._loads[index] = access
            elif op.opcode == OpCode.STORE:
                self._stores[index] = self._compute_access(index, op.inputs[2].size)

        self._collect_load_defs(self._compute_reads())
        del self._intern

    def value_of(self, index: OpIndex, var: Var) -> AbstractValue:
        """Evaluates the given variable when read by the given operation."""

        match var.kind:
            case VarKinds.Constant:
                return unsigned_to_signed(var.value, var.size)
            case VarKinds.Register | VarKinds.Unique:
                pass
            case _:
                return None

        definitions = self.program.get_defs(index, var)
        if definitions == {EntryIndex}:
            return StackAddress(0) if var == self.stack_pointer else None
        if not definitions or EntryIndex in definitions:
            return None
        values = {self._output_value(definition, var) for definition in definitions}
        return values.pop() if len(values) == 1 else None

    def _output_value(self, index: OpIndex, var: Var) -> AbstractValue:
        op = self.program.operations[index]
        if op.output is None or wrap(op.output) != var:
            return None
        if index not in self._output_values:
            # Breaks cycles in loops
            self._output_values[index] = None
            self._output_values[index] = self._evaluate(index, op)
        return self._output_values[index]

    def _evaluate(self, index: OpIndex, op: PcodeOp) -> AbstractValue:
        if op.opcode not in ADDRESS_OPCODES:
            return None
        assert op.output is not None
        operands = [self.value_of(index, wrap(varnode)) for varnode in op.inputs]
        size = op.output.size
        match op.opcode, operands:
            case OpCode.COPY, [value]:
                return value
            case OpCode.INT_ADD, [StackAddress(offset), int(constant)] | [
                int(constant),
                StackAddress(offset),
            ]:
                return StackAddress(truncate(offset + constant, size))
            case OpCode.INT_ADD, [int(left), int(right)]:
                return truncate(left + right, size)
            case OpCode.INT_SUB, [StackAddress(offset), int(constant)]:
                return StackAddress(truncate(offset - constant, size))
            case OpCode.INT_SUB, [StackAddress(left), StackAddress(right)]:
                return truncate(left - right, size)
            case OpCode.INT_SUB, [int(left), int(right)]:
                return truncate(left - right, size)
            case OpCode.INT_OR, [value, 0] | [0, value]:
                return value
        return None

    def _compute_access(self, index: OpIndex, size: int) -> MemoryAccess | None:
        address = self.value_of(index, wrap(self.program.operations[index].inputs[1]))
        match address:
            case StackAddress(offset):
                return MemoryAccess(True, offset, size)
            case int(absolute):
                return MemoryAccess(False, absolute, size)
        return None

    def _apply(self, index: OpIndex, state: MemoryState) -> None:
        """Updates the memory state with the effect of the given operation."""

        op = self.program.operations[index]
        if op.opcode == OpCode.STORE:
            access = self._stores[index]
            if access is None:
                state.clobber(self._intern)
            else:
                state.write(access, self._intern(frozenset({index})))
        elif op.opcode in CALL_OPCODES and UnknownCodeIndex in self.program.succ_ops(
            index
        ):
            state.clobber(self._intern)

    def _compute_reads(self) -> dict[OpIndex, DefinitionSet]:
        """Computes the definitions of the bytes read by each resolved load."""

        blocks = self.program.basic_blocks
        block_inputs: list[MemoryState | None] = [None] * len(blocks)
        block_inputs[0] = MemoryState({}, self._intern(frozenset({EntryIndex})))
        reads: dict[OpIndex, DefinitionSet] = {}
        pending = [0]
        queued = {0}

        while pending:
            block_index = heapq.heappop(pending)
            queued.remove(block_index)
            block_input = block_inputs[block_index]
            assert block_input is not None
            state = block_input.copy()
            for index in blocks[block_index].op_indexes:
                if index in self._loads:
                    reads[index] = self._intern(state.read(self._loads[index]))
                self._apply(index, state)

            for successor in blocks[block_index].succ_blocks:
                if successor < 0:
                    continue
                previous = block_inputs[successor]
                if previous is None:
                    block_inputs[successor] = state.copy()
                    changed = True
                else:
                    changed = previous.merge(state, self._intern)
                if changed and successor not in queued:
                    heapq.heappush(pending, successor)
                    queued.add(successor)
        return reads

    def _collect_load_defs(self, reads: dict[OpIndex, DefinitionSet]) -> None:
        for index, definitions in reads.items():
            if UnknownWriteIndex in definitions:
                continue
            self.result.load_defs[index] = set(definitions)
            if EntryIndex not in definitions and all(
                self._stores[definition] == self._loads[index]
                for definition in definitions
            ):
                self.result.forwarded_loads[index] = sorted(definitions)


def truncate(value: int, size: int) -> int:
    """Wraps the given value as a signed integer of the given size in bytes."""

    return unsigned_to_signed(value & ((1 << (size * 8)) - 1), size)
