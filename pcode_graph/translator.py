from dataclasses import dataclass
from typing import Iterator, NamedTuple
from loguru import logger
import pypcode
from pcode_graph.arch import Arch


@dataclass(slots=True)
class ChunkOps:
    """Result of lifting a piece of binary."""

    num_instructions: int
    address: int
    operations: list[pypcode.PcodeOp]


@dataclass
class Instruction:
    """Result of disassembling."""

    address: int
    length: int
    mnem: str
    body: str

    def __str__(self) -> str:
        return self.dump(True)

    def dump(self, print_address: bool):
        s = ""
        if print_address:
            s += f"{self.address:05x} "
        s += self.mnem
        if self.body:
            s += " " + self.body
        return s


@dataclass(slots=True)
class LiftStats:
    total_bytes: int = 0
    lifted_bytes: int = 0
    skipped_bytes: int = 0
    truncated_bytes: int = 0
    num_instructions: int = 0
    resyncs: int = 0
    translate_calls: int = 0

    @property
    def coverage(self) -> float:
        return self.lifted_bytes / self.total_bytes if self.total_bytes else 1.0


class ArchParams(NamedTuple):
    #: Resynchronisation step, in bytes, after an undecodable instruction.
    #: Must divide every valid instruction address, otherwise recovery lands
    #: mid-instruction.
    alignment: int
    #: Longest single instruction encoding, in bytes.
    max_instruction_size: int


ARCH_PARAMETERS: dict[Arch, ArchParams] = {
    Arch.x86_32: ArchParams(1, 15),
    Arch.x86_64: ArchParams(1, 15),
    Arch.arm_32: ArchParams(4, 4),
    Arch.arm_64: ArchParams(4, 4),
    Arch.mips_32: ArchParams(4, 4),
    Arch.mips_64: ArchParams(4, 4),
}


def create_pcode_context(arch: Arch) -> pypcode.Context:
    match arch:
        case Arch.arm_32:
            return pypcode.Context("ARM:LE:32:v8")
        case Arch.arm_64:
            return pypcode.Context("AARCH64:LE:64:v8A")
            # return pypcode.Context("AARCH64:LE:64:AppleSilicon")
        case Arch.x86_32:
            return pypcode.Context("x86:LE:32:default")
        case Arch.x86_64:
            return pypcode.Context("x86:LE:64:default")
        case Arch.mips_32:
            return pypcode.Context("MIPS:BE:32:default")
        case Arch.mips_64:
            return pypcode.Context("MIPS:BE:64:default")
        case _:
            raise ValueError(f"Unknown architecture {arch}")


def get_pcode_context_arch(context: pypcode.Context) -> Arch:
    size = int(context.language.size)
    processor = context.language.processor
    match processor:
        case "ARM":
            if size == 32:
                return Arch.arm_32
        case "AARCH64":
            if size == 64:
                return Arch.arm_64
        case "MIPS":
            match size:
                case 32:
                    return Arch.mips_32
                case 64:
                    return Arch.mips_64
        case "x86":
            match size:
                case 32:
                    return Arch.x86_32
                case 64:
                    return Arch.x86_64

    raise ValueError(f"Unsupported context with {processor=} and {size=}")


class Translator:

    def __init__(self, arch_context: pypcode.Context | Arch):
        if isinstance(arch_context, Arch):
            self.arch = arch_context
            self.context = create_pcode_context(self.arch)
        else:
            self.context = arch_context
            self.arch = get_pcode_context_arch(arch_context)

        self.alignment, self.max_instr_size = ARCH_PARAMETERS[self.arch]

    def translate(self, data: bytes, base_address: int) -> list[pypcode.PcodeOp]:
        """Translates a piece of code into P-Code operations."""

        ops = []
        for chunk in self.iter_operations(data, base_address):
            ops += chunk.operations
        return ops

    def iter_operations(
        self,
        data: bytes,
        base_address: int,
        window: int = 4096,
        max_instructions: int = 1024,
        stats: LiftStats | None = None,
    ) -> Iterator[ChunkOps]:
        """Translates a piece of code into P-Code, handling errors and avoiding
        out-of-memory for big sections.

        Each ChunkOps covers a contiguous range: undecodable bytes are skipped,
        ``instruction_alignment`` at a time. ``window`` and ``max_instructions``
        both bound the memory of a single call.
        """

        assert window > 1 and max_instructions > 1
        if stats is None:
            stats = LiftStats()
        stats.total_bytes += len(data)

        offset = 0
        padded_data = data
        if self.arch in {Arch.mips_32, Arch.mips_64}:
            # Avoid delay slot exception
            padded_data = data + b"\x00\x00\x00\x00"

        while offset < len(data):
            window_base = base_address + offset

            try:
                stats.translate_calls += 1
                ops = self.context.translate(
                    padded_data,
                    window_base,
                    max_instructions=max_instructions,
                    offset=offset,
                    max_bytes=window,
                ).ops
            except (IndexError, pypcode.BadDataError, pypcode.UnimplError, pypcode.LowlevelError):
                ops = []

            chunk: list[pypcode.PcodeOp] = []
            num_instructions = 0
            consumed_bytes = 0
            truncated = False

            for op in ops:
                if op.opcode == pypcode.OpCode.IMARK:
                    # A single IMARK may cover several instructions (grouped delay slots)
                    # Do not assume the varnodes are sorted.
                    end = max(i.offset + i.size for i in op.inputs) - base_address

                    # pypcode silently zero-pads past the end of the buffer
                    # and reports the instruction's full length, so the
                    # last instruction of a section can be entirely
                    # fabricated. Any instruction consuming a padding byte
                    # necessarily ends past `size`, which makes this test
                    # both sufficient and exact.
                    if end > len(data):
                        truncated = True
                        break

                    num_instructions += len(op.inputs)
                    consumed_bytes = max(consumed_bytes, end - offset)

                chunk.append(op)

            if consumed_bytes > 0:
                stats.lifted_bytes += consumed_bytes
                stats.num_instructions += num_instructions
                yield ChunkOps(num_instructions, window_base, chunk)
                offset += consumed_bytes                

            if truncated:
                stats.truncated_bytes += len(data) - offset
                return

            if consumed_bytes == 0:                    
                # Nothing decodable at `offset`: resynchronise.
                stats.skipped_bytes += min(self.alignment, len(data) - offset)
                stats.resyncs += 1
                offset += self.alignment

    def disassemble(self, code: bytes, base_address: int) -> Iterator[Instruction]:
        data = self.context.disassemble(code, base_address=base_address)
        for i in data.instructions:
            yield Instruction(i.addr.offset, i.length, i.mnem, i.body)

    def dump_asm(
        self,
        code: bytes,
        base_address: int,
        print_address: bool = True,
    ) -> str:
        return "\n".join(
            i.dump(print_address) for i in self.disassemble(code, base_address)
        )
