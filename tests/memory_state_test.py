import random

from pcode_graph.analysis.alias_analysis import (
    DefinitionSet,
    MemoryAccess,
    MemoryByte,
    MemoryState,
    UnknownWriteIndex,
)
from pcode_graph.analysis.interning import Interner

ALL_BYTES: list[MemoryByte] = [
    (in_stack, offset) for in_stack in (True, False) for offset in range(8)
]


def random_state(
    generator: random.Random, intern: Interner[DefinitionSet]
) -> MemoryState:
    def random_definitions() -> DefinitionSet:
        return intern(frozenset(generator.sample(range(6), generator.randint(0, 3))))

    tracked = {
        (generator.random() < 0.5, generator.randint(0, 5)): random_definitions()
        for _ in range(generator.randint(0, 6))
    }
    return MemoryState(tracked, random_definitions())


def definitions_of(state: MemoryState) -> dict[MemoryByte, DefinitionSet]:
    return {byte: state.tracked.get(byte, state.default) for byte in ALL_BYTES}


def test_merge_matches_union_of_definitions():
    generator = random.Random(0)
    intern: Interner[DefinitionSet] = Interner()
    for _ in range(20000):
        state = random_state(generator, intern)
        other = random_state(generator, intern)
        before = definitions_of(state)
        other_before = definitions_of(other)
        expected = {
            byte: definitions | other_before[byte]
            for byte, definitions in before.items()
        }

        changed = state.merge(other, intern)

        assert definitions_of(state) == expected
        assert definitions_of(other) == other_before
        assert changed == (before != expected)


def test_write_and_clobber():
    intern: Interner[DefinitionSet] = Interner()
    state = MemoryState({}, intern(frozenset({-1})))
    access = MemoryAccess(True, 2, 4)
    state.write(access, 7, intern)
    copy = state.copy()
    copy.clobber(intern)

    assert state.read(access) == {7}
    assert state.read(MemoryAccess(True, 0, 4)) == {-1, 7}
    assert copy.read(access) == {7, UnknownWriteIndex}
    assert copy.read(MemoryAccess(False, 0, 1)) == {-1, UnknownWriteIndex}
