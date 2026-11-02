from collections.abc import Hashable


class Interner[T: Hashable]:
    """Stores equal, unmutable shared instance to save memory, like identical strings."""

    def __init__(self) -> None:
        self._values: dict[T, T] = {}

    def __call__(self, value: T) -> T:
        return self._values.setdefault(value, value)
