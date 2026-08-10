import keystone

from pcode_graph.arch import Arch

KEYSTONE_MODES = {
    Arch.arm_32: (
        keystone.KS_ARCH_ARM,
        keystone.KS_MODE_ARM | keystone.KS_MODE_V8 | keystone.KS_MODE_LITTLE_ENDIAN,
    ),
    Arch.arm_64: (keystone.KS_ARCH_ARM64, keystone.KS_MODE_LITTLE_ENDIAN),
    Arch.x86_32: (keystone.KS_ARCH_X86, keystone.KS_MODE_32),
    Arch.x86_64: (keystone.KS_ARCH_X86, keystone.KS_MODE_64),
    Arch.mips_32: (
        keystone.KS_ARCH_MIPS,
        keystone.KS_MODE_MIPS32 | keystone.KS_MODE_BIG_ENDIAN,
    ),
    Arch.mips_64: (
        keystone.KS_ARCH_MIPS,
        keystone.KS_MODE_MIPS64 | keystone.KS_MODE_BIG_ENDIAN,
    ),
}


# To ease diagnostic when writing assembly by hand...
def split_instructions(asm: str):

    asm = asm.replace("\t", " ")
    instructions: list[str] = []

    for line in asm.splitlines():
        for ins in line.split(";"):
            ins = ins.strip()
            if ins:
                instructions.append(ins)

    return instructions


class Assembler:

    def __init__(self, arch: Arch) -> None:
        self.arch = arch
        self._keystone = keystone.Ks(*KEYSTONE_MODES[arch])

    def assemble(self, instructions: str, base_address: int) -> bytes:
        try:
            data, count = self._keystone.asm(
                instructions, addr=base_address, as_bytes=True
            )
        except keystone.KsError:
            for i, ins in enumerate(split_instructions(instructions)):
                try:
                    self._keystone.asm(ins)
                except keystone.KsError as e:
                    raise ValueError(f"Error in instruction {i}: {ins}: {e}")
            raise

        assert count > 0
        assert isinstance(data, bytes)
        return data
