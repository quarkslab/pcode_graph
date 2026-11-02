import random

from pcode_graph.analysis.alias_analysis import MemoryByte, MemoryState
from pcode_graph.analysis.interning import Interner

ALL_BYTES: list[MemoryByte] = [(in_stack, offset) for in_stack in (True, False) for offset in range(8)]


def random_state(generator: random.Random) -> MemoryState:
    def random_definitions() -> frozenset[int]:
        return frozenset(generator.sample(range(6), generator.randint(0, 3)))

    tracked = {
        (generator.random() < 0.5, generator.randint(0, 5)): random_definitions()
        for _ in range(generator.randint(0, 6))
    }
    return MemoryState(tracked, random_definitions())


def definitions_of(state: MemoryState) -> dict[MemoryByte, frozenset[int]]:
    return {byte: state.tracked.get(byte, state.default) for byte in ALL_BYTES}


def test_merge_matches_union_of_definitions():
    generator = random.Random(0)
    for _ in range(20000):
        state, other = random_state(generator), random_state(generator)
        before = definitions_of(state)
        expected = {
            byte: definitions | definitions_of(other)[byte]
            for byte, definitions in before.items()
        }

        changed = state.merge(other, Interner())

        assert definitions_of(state) == expected
        assert changed == (before != expected)
