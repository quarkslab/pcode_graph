import heapq

from .interning import Interner
from .pass_base import PassBase
from .program import (
    BasicBlock,
    Definitions,
    EntryIndex,
    LocalWriters,
    OpIndex,
    Var,
    VarKinds,
    wrap,
    split,
)


class ReachingDefBuilder(PassBase):
    """Computes reaching definitions at the entry of each block.

    Identical definition sets are shared (interned) to bound memory on large functions.
    """

    def __call__(self):
        block_count = len(self.program.basic_blocks)
        block_inputs: list[Definitions] = [{} for _ in range(block_count)]
        block_writers: list[LocalWriters] = [{} for _ in range(block_count)]
        self.program.block_inputs = block_inputs
        self.program.block_writers = block_writers
        if not block_count:
            return

        self._intern: Interner[frozenset[OpIndex]] = Interner()

        # Initialize
        entry = self._intern(frozenset({EntryIndex}))
        for index in range(len(self.program.operations)):
            if not self.program.reachable(index):
                continue
            for var in self.program.iter_inputs(index):
                if var.kind == VarKinds.Register:
                    for byte in split(var):
                        block_inputs[0][byte] = entry

        # Block effects: last writer of each byte written in the block
        for block_index, block in enumerate(self.program.basic_blocks):
            if block.reachable:
                block_writers[block_index] = self._collect_writers(block)
        shared_temporaries = self._collect_shared_temporaries()
        generated_by_block: list[Definitions] = [
            {
                byte: self._intern(frozenset((byte_writers[-1],)))
                for byte, byte_writers in writers.items()
                if byte.kind != VarKinds.Unique or byte in shared_temporaries
            }
            for writers in block_writers
        ]

        # Each reachable block is processed at least once, lowest index first:
        # follows the code layout, close to a reverse post-order.
        # The list is sorted, so it is already a valid heap.
        pending = [
            index
            for index, block in enumerate(self.program.basic_blocks)
            if block.reachable
        ]
        queued = set(pending)

        while pending:
            block_index = heapq.heappop(pending)
            queued.remove(block_index)

            outputs = block_inputs[block_index] | generated_by_block[block_index]
            for successor in self.program.basic_blocks[block_index].succ_blocks:
                if successor < 0:
                    continue
                changed = self._merge_definitions(block_inputs[successor], outputs)
                if changed and successor not in queued:
                    heapq.heappush(pending, successor)
                    queued.add(successor)

        del self._intern

    def _collect_writers(self, block: BasicBlock) -> LocalWriters:
        writers: LocalWriters = {}
        for index in block.op_indexes:
            output = self.program.operations[index].output
            if output:
                for byte in split(wrap(output)):
                    writers.setdefault(byte, []).append(index)
        return writers

    def _collect_shared_temporaries(self) -> set[Var]:
        """Returns P-Code temporary bytes read in a block before being written in it."""

        shared: set[Var] = set()
        for block, writers in zip(self.program.basic_blocks, self.program.block_writers):
            if not block.reachable:
                continue
            for index in block.op_indexes:
                for var in self.program.iter_inputs(index):
                    if var.kind != VarKinds.Unique:
                        continue
                    for byte in split(var):
                        byte_writers = writers.get(byte)
                        if not byte_writers or byte_writers[0] >= index:
                            shared.add(byte)
        return shared

    def _merge_definitions(self, target: Definitions, source: Definitions) -> bool:
        """Adds source definitions to target ones, returns True if target changed."""

        changed = False
        for var, defs in source.items():
            target_defs = target.get(var)
            if target_defs is None:
                target[var] = defs
                changed = True
            elif target_defs is not defs and not defs <= target_defs:
                target[var] = self._intern(target_defs | defs)
                changed = True
        return changed
