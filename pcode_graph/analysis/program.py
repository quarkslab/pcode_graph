from bisect import bisect_left
from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from typing import Iterator
from pypcode import PcodeOp, Varnode
from pcode_graph.arch import Arch
from pcode_graph.pcode import dump_operation
from .native_pcode import (
    is_data_input,
)

OpIndex = int
EntryIndex = -1
UnknownCodeIndex = -2
ExitIndex = -3

Address = int

BlockIndex = int


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
    # Code access as memory accesses are made explicit
    Ram = 3


@dataclass(frozen=True, order=True)
class Var:
    """Hashable and comparable wrapper for temporaries, registers and constants."""

    kind: VarKinds
    size: int
    value: int
    pretty_name: str = field(compare=False)

    def __str__(self) -> str:
        return self.pretty_name


def wrap(var: Varnode) -> Var:
    match var.space.name:
        case "register":
            name = var.getRegisterName() or "UNKNOWN"
            return Var(VarKinds.Register, var.size, var.offset, name)
        case "unique":
            return Var(VarKinds.Unique, var.size, var.offset, f"${var.offset}")
        case "const":
            return Var(VarKinds.Constant, var.size, var.offset, f"${var.offset}")
        case "ram":
            return Var(VarKinds.Ram, var.size, var.offset, f"[0x{var.offset:x}]")
        case _ as name:
            raise ValueError(f"Unexpected space for input variable: {name}")


# Writing operations by variable byte (see split)
Definitions = dict[Var, frozenset[OpIndex]]

# Writing operations inside a block by variable byte, sorted by index
LocalWriters = dict[Var, list[OpIndex]]


def split(var: Var) -> tuple[Var, ...]:
    """Splits a register into its bytes to track definitions.
    Overlapping registers share bytes (RAX, AX, AH) so that a partial write only kills the definitions of the bytes it writes.
    """

    if var.kind == VarKinds.Constant:
        return (var,)
    return _split_bytes(var.kind, var.value, var.size)


@cache
def _split_bytes(kind: VarKinds, value: int, size: int) -> tuple[Var, ...]:
    return tuple(
        Var(kind, 1, value + offset, f"{kind.name}:{value + offset:#x}")
        for offset in range(size)
    )


@dataclass(slots=True)
class BasicBlock:
    """Maximal sequence of operations with a single entry and a single exit."""

    # Operation indexes range: start included, end excluded
    start: OpIndex
    end: OpIndex

    # Block-level CFG edges, special indexes (EntryIndex, ExitIndex...) are kept as is.
    # Order of successors matters, as for operations.
    pred_blocks: set[BlockIndex] = field(default_factory=set)
    succ_blocks: list[BlockIndex] = field(default_factory=list)

    reachable: bool = False

    @property
    def last_op_index(self) -> OpIndex:
        return self.end - 1

    @property
    def op_indexes(self) -> range:
        return range(self.start, self.end)


@dataclass
class Memory:
    # Maps a resolved LOAD to possible STORE definitions
    load_defs: dict[OpIndex, set[OpIndex]] = field(default_factory=dict)

    # Maps a load to defining stores if reading exactly the values written by them
    forwarded_loads: dict[OpIndex, list[OpIndex]] = field(default_factory=dict)


@dataclass
class Program:
    """A class to run static analysis on a list of P-Code operations."""

    # Source architecture
    arch: Arch

    # List of P-Code operations
    operations: list[PcodeOp]

    # Basic blocks sorted by operation index, and block of each operation
    basic_blocks: list[BasicBlock] = field(default_factory=list)
    block_indexes: list[BlockIndex] = field(default_factory=list)

    # Reaching definitions at the entry of each block. By variable byte, set of:
    #   - definitions (= writing operation index) that reach the block,
    #   - or EntryIndex (for initial definitions of inputs).
    # P-Code temporaries are only kept if read in a block before being written in it.
    # Definitions inside a block are derived on demand from block_writers,
    # which avoids storing a full copy of the definitions for each operation.
    # Both are empty for unreachable blocks.
    block_inputs: list[Definitions] = field(default_factory=list)
    block_writers: list[LocalWriters] = field(default_factory=list)

    # Memory analysis results
    memory: Memory = field(default_factory=Memory)

    def __str__(self) -> str:
        return self.dump()

    def block_of(self, index: OpIndex) -> BasicBlock:
        return self.basic_blocks[self.block_indexes[index]]

    def reachable(self, index: OpIndex) -> bool:
        return self.block_of(index).reachable

    def pred_ops(self, index: OpIndex) -> set[OpIndex]:
        """Operation-level CFG predecessors, derived from the block-level CFG."""

        block = self.block_of(index)
        if index != block.start:
            return {index - 1}
        return {
            self.basic_blocks[pred].last_op_index if pred >= 0 else pred
            for pred in block.pred_blocks
        }

    def succ_ops(self, index: OpIndex) -> list[OpIndex]:
        """Operation-level CFG successors, derived from the block-level CFG."""

        block = self.block_of(index)
        if index != block.last_op_index:
            return [index + 1]
        return [
            self.basic_blocks[successor].start if successor >= 0 else successor
            for successor in block.succ_blocks
        ]

    def dump(self) -> str:
        """Dumps the program and analysis results in a markdown table."""

        s = "|index|op|preds|succs|input defs|reachable|exit def|\n"
        exit_defs = self.get_exit_defs()

        s += "|--|--|--|--|--|--|--|\n"
        for i, op in enumerate(self.operations):
            s += f"|{i}|{dump_operation(op)}|"
            s += ", ".join(dump_index(p) for p in self.pred_ops(i)) + "|"
            s += ", ".join(dump_index(p) for p in self.succ_ops(i)) + "|"
            var_defs = []
            for var, defs in self.iter_input_defs(i):
                if defs:
                    var_defs.append(
                        str(var) + " from " + " or ".join(dump_index(d) for d in defs)
                    )

            s += ", ".join(var_defs) + "|"
            if self.reachable(i):
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

        block_index = self.block_indexes[index]
        inputs = self.block_inputs[block_index]
        writers = self.block_writers[block_index]
        defs: set[OpIndex] = set()
        for byte in split(var):
            byte_writers = writers.get(byte)
            if byte_writers:
                position = bisect_left(byte_writers, index)
                if position:
                    defs.add(byte_writers[position - 1])
                    continue
            defs.update(inputs.get(byte, ()))
        return defs

    def get_reaching_defs(self, index: OpIndex) -> Definitions:
        """Returns all the definitions reaching the given operation.
        P-Code temporaries written in other blocks are omitted if not read before being
        written in this block.
        """

        return self._definitions_before(self.block_indexes[index], index)

    def _definitions_before(self, block_index: BlockIndex, end: OpIndex) -> Definitions:
        definitions = dict(self.block_inputs[block_index])
        for byte, writers in self.block_writers[block_index].items():
            position = bisect_left(writers, end)
            if position:
                definitions[byte] = frozenset((writers[position - 1],))
        return definitions

    def get_exit_defs(self) -> dict[Var, set[OpIndex]]:
        """Computes side effects of the program."""

        exit_defs: dict[Var, set[OpIndex]] = {}
        for block_index, block in enumerate(self.basic_blocks):
            if not block.reachable or not {UnknownCodeIndex, ExitIndex}.intersection(
                block.succ_blocks
            ):
                continue
            outputs = self._definitions_before(block_index, block.end)
            for byte, defs in outputs.items():
                if byte.kind == VarKinds.Unique:
                    # Ignore P-Code temporary
                    continue
                for writer in defs - {EntryIndex}:
                    # Group bytes back by written variable
                    output = self.operations[writer].output
                    assert output is not None
                    exit_defs.setdefault(wrap(output), set()).add(writer)
        return exit_defs

    def iter_inputs(self, index: OpIndex) -> Iterator[Var]:
        """Iterates operation inputs, skipping **special** ones
        that have no semantic interest to put in the graph (see P-Code documentation).
        Accessed memory is implicit.
        """

        op = self.operations[index]
        for position, var in enumerate(op.inputs):
            # TODO: move this helper in a separate file
            if is_data_input(op.opcode, position):
                yield wrap(var)

    def iter_input_defs(self, index: OpIndex) -> Iterator[tuple[Var, set[OpIndex]]]:
        """For each operand, returns the written variable and the operation(s) defining it."""

        for var in self.iter_inputs(index):
            yield var, self.get_defs(index, var)
