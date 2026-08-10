from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from loguru import logger
from pypcode import OpCode
from pcode_graph.arch import Arch
import lief
from pcode_graph.translator import ChunkOps, Translator

lief.disable_leak_warning()


@dataclass
class ChunkCode:
    address: int
    content: bytes


@dataclass
class FunctionCode:
    name: str
    address: int
    content: bytes


@dataclass
class FunctionOps(ChunkOps):
    function_name: str


def parse_binary(binary_path: Path) -> lief.Binary:
    assert binary_path.exists(), f"{binary_path} not found"
    logger.debug(f"Parse {binary_path}")
    b = lief.parse(binary_path.as_posix())
    assert isinstance(b, lief.Binary)
    return b


def get_architecture(binary: Path | lief.Binary) -> Arch:
    """Guesses the architecture of given binary."""

    if isinstance(binary, Path):
        binary = parse_binary(binary)

    # ELF
    if isinstance(binary, lief.ELF.Binary):

        match binary.header.machine_type:
            case lief.ELF.ARCH.ARM:
                arch = Arch.arm_32
            case lief.ELF.ARCH.AARCH64:
                arch = Arch.arm_64
            case lief.ELF.ARCH.X86_64:
                arch = Arch.x86_64
            case lief.ELF.ARCH.I386:
                arch = Arch.x86_32
            case lief.ELF.ARCH.MIPS:
                if binary.header.identity_data != lief.ELF.Header.ELF_DATA.MSB:
                    raise ValueError("Unsupported edianness")
                if binary.header.identity_class == lief.ELF.Header.CLASS.ELF64:
                    arch = Arch.mips_64
                else:
                    flags = set(binary.header.flags_list)
                    _P = lief.ELF.PROCESSOR_FLAGS
                    if _P.MIPS_ABI2 in flags or bool(
                        flags
                        & {
                            _P.MIPS_ARCH_3, _P.MIPS_ARCH_4, _P.MIPS_ARCH_5, _P.MIPS_ARCH_64, _P.MIPS_ARCH_64R2, _P.MIPS_ARCH_64R6, # type: ignore  # fmt: skip
                        }
                    ):
                        arch = Arch.mips_64
                    else:
                        arch = Arch.mips_32
            case _ as v:
                raise ValueError(f"Unsupported CPU type {v}")

    # Mach-O
    elif isinstance(binary, lief.MachO.FatBinary):
        arch = None
        for sub_binary in binary:
            match sub_binary.header.cpu_type:
                case lief.MachO.Header.CPU_TYPE.ARM64:
                    arch = Arch.arm_64
                case lief.MachO.Header.CPU_TYPE.X86_64:
                    arch = Arch.x86_64
                case _:
                    continue
            break
        else:
            raise ValueError(f"No sub binary found with supported architecture")

    elif isinstance(binary, lief.MachO.Binary):
        match binary.header.cpu_type:
            case lief.MachO.Header.CPU_TYPE.ARM64:
                arch = Arch.arm_64
            case lief.MachO.Header.CPU_TYPE.X86_64:
                arch = Arch.x86_64
            case _ as v:
                raise ValueError(f"Unsupported CPU type {v}")

    # PE
    elif isinstance(binary, lief.PE.Binary):
        match binary.header.machine:
            case lief.PE.Header.MACHINE_TYPES.ARM64:
                arch = Arch.arm_64
            case lief.PE.Header.MACHINE_TYPES.AMD64:
                arch = Arch.x86_64
            case _ as v:
                raise ValueError(f"Unsupported CPU type {v}")

    else:
        raise ValueError("Unsupported binary format")

    return arch


def iter_code_sections(binary: Path | lief.Binary) -> Iterator[ChunkCode]:

    if isinstance(binary, Path):
        binary = parse_binary(binary)

    if binary.format == lief.Binary.FORMATS.MACHO:
        flags = {
            lief.MachO.Section.FLAGS.PURE_INSTRUCTIONS,  # type: ignore
            lief.MachO.Section.FLAGS.SOME_INSTRUCTIONS,  # type: ignore
        }
        for section in binary.sections:
            if flags.intersection(section.flags_list):  # type: ignore
                yield ChunkCode(section.virtual_address, section.content.tobytes())

    elif binary.format == lief.Binary.FORMATS.PE:
        for section in binary.sections:
            if section.has_characteristic(  # type: ignore
                lief.PE.Section.CHARACTERISTICS.CNT_CODE
            ) and section.has_characteristic(  # type: ignore
                lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE
            ):
                yield ChunkCode(section.virtual_address, section.content.tobytes())

    elif binary.format == lief.Binary.FORMATS.ELF:
        for section in binary.sections:
            if section.has(lief.ELF.Section.FLAGS.EXECINSTR):  # type: ignore
                yield ChunkCode(section.virtual_address, section.content.tobytes())


def get_function(binary: Path | lief.Binary, name: str) -> FunctionCode:

    if isinstance(binary, Path):
        binary = parse_binary(binary)

    for f in iter_functions(binary):
        if f.name == name:
            return f

    raise LookupError(f"Function {name} not found in {binary}")


def iter_functions(binary: Path | lief.Binary) -> Iterator[FunctionCode]:

    if isinstance(binary, Path):
        binary = parse_binary(binary)

    cache: dict[Any, tuple[bytes, int]] = {}
    sym: lief.ELF.Symbol

    for sym in binary.symtab_symbols:  # type: ignore
        if sym.type != lief.ELF.Symbol.TYPE.FUNC or sym.size == 0:
            continue
        section: lief.ELF.Section | None = sym.section
        if section is None:
            continue

        section_id = section.name
        if section_id not in cache:
            cache[section_id] = (bytes(section.content), section.virtual_address)

        section_content, section_address = cache[section_id]

        assert isinstance(sym.name, str)
        yield FunctionCode(
            sym.name,
            section_address + sym.value,
            section_content[sym.value : sym.value + sym.size],
        )


def lookup_chunk(
    binary: Path | lief.Binary,
    start_address: int,
    end_address: int,
) -> bytes:
    """Returns the code at given address from the given binary.

    Args:
        binary (Path | lief.Binary): Binary to read.
        start_address (int): Address of the first instruction to return.
        end_address (int): Address of the first instruction after the returned chunk.

    Returns:
        ChunkCode: A piece of binary code.
    """

    for section in iter_code_sections(binary):
        if (
            section.address <= start_address
            and section.address + len(section.content) >= end_address
        ):
            offset = start_address - section.address
            size = end_address - start_address
            return section.content[offset : offset + size]

    raise LookupError("Cannot find a section containing given range")


def collect_functions(
    binary: Path | lief.Binary,
    translator: Translator,
) -> Iterator[FunctionOps]:

    logger.debug(f"Extract functions")

    for function in iter_functions(binary):

        instructions = list(translator.disassemble(function.content, function.address))
        operations = translator.translate(function.content, function.address)
        yield FunctionOps(
            len(instructions),
            function.address,
            operations,
            function.name,
        )


def collect_basic_blocks(
    binary: Path | lief.Binary,
    translator: Translator,
    min_instructions: int,
    max_instructions: int,
) -> Iterator[ChunkOps]:
    """Extracts the P-Code operations of basic blocks found in a binary, using LIEF library."""

    logger.debug(f"Extract basic blocks")

    # Extract P-Code operations of executable sections
    for section in iter_code_sections(binary):

        # Slice into basic blocks
        num_instructions = 0
        operations = []
        bb_start_offset = 0

        # First disassemble to translate into P-Code one instruction at a time,
        # in order to avoid consuming too much RAM.
        for instr in translator.disassemble(section.content, section.address):

            instr_start_offset = instr.address - section.address
            instr_end_offset = instr_start_offset + instr.length
            instr_code = section.content[instr_start_offset:instr_end_offset]

            instr_ops = translator.translate(instr_code, instr.address)

            num_instructions += 1

            if num_instructions <= max_instructions:
                operations += instr_ops

            last_op = instr_ops[-1]
            if last_op.opcode not in {
                OpCode.BRANCH,
                OpCode.BRANCHIND,
                OpCode.CBRANCH,
                OpCode.RETURN,
            }:
                # Not at the end of a basic block
                continue

            if min_instructions <= num_instructions <= max_instructions:
                yield ChunkOps(
                    num_instructions,
                    section.address + bb_start_offset,
                    # section.content[bb_start_offset:instr_end_offset],
                    operations,
                )

            bb_start_offset = instr_end_offset
            num_instructions = 0
            operations = []


def collect_chunks(
    binary: Path | lief.Binary,
    translator: Translator,    
    
) -> Iterator[ChunkOps]:
    """Extracts the P-Code operations of the sections found in a binary, using LIEF library."""

    for section in iter_code_sections(binary):
        yield from translator.iter_operations(section.content, section.address)        
        