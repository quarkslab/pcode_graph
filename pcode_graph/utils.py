def unsigned_to_signed(value: int, size_bytes: int) -> int:
    """Converts the given unsigned value into a signed one."""

    assert size_bytes > 0

    mask = 1 << (size_bytes * 8 - 1)
    return (value ^ mask) - mask


def signed_to_unsigned(value: int, size_bytes: int) -> int:
    """Converts the given signed value into an unsigned one."""

    return value + (value < 0) * (2 ** (size_bytes * 8))


