from enum import StrEnum


class Arch(StrEnum):
    arm_32 = "arm_32"
    arm_64 = "arm_64"
    x86_64 = "x86_64"
    x86_32 = "x86_32"
    mips_32 = "mips_32"  # Big endian
    mips_64 = "mips_64"
