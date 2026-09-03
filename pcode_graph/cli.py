#!/usr/bin/env python3
"""
Dumps Control and Data Graph (CDG) and various data from a binary or assembly file.

Usage:
    cdg ( asm | functions | html | md | pcode | table | stats ) <input_path> [options]

Dump commands:
    asm        Assembly dump
    functions  Functions from symbol table (using LIEF, binary must not be stripped)
    html       CDG in HTML format
    md         CDG in markdown/mermaid format
    pcode      P-Code operations
    table      Markdown table of the P-Code enriched by static analysis results
    stats      Prints graph statistics

Options:
    -f, --function <name>       Take only the function with given symbol name.
    -c, --chunk <begin:end>     Pass "0xbeaf:0xdead" to lift only the chunk started at 0xbeaf and ending before 0xdead.
    -a, --arch <arch>           Use the given architecture (x86|arm|mips)_(64|32) or auto [Default: auto].
    -b, --base-address <int>    Consider the given assembly starts at given address [Default: 0].
    -D, --dataflow-only         Only extract dataflow graph (and no control flow).
    -o, --output <path>         Where to write the output.
    -h, --help                  This help.
    -d, --debug                 Debug mode: do no catch exceptions.
"""

from pathlib import Path
import re
import sys
from docopt import docopt
from lief import Binary
import lief
from loguru import logger
import pypcode
from pcode_graph.arch import Arch
from pcode_graph.asm import Assembler
from pcode_graph.charac import count_edges, get_graph_diameters
from pcode_graph.lief_importer import (
    get_function,
    iter_code_sections,
    iter_functions,
    get_architecture,
    lookup_chunk,
    parse_binary,
)
from pcode_graph.log import setup_logger
from pcode_graph.translator import Translator
from pcode_graph.pcode import dump_operation
from pcode_graph.analysis import RichPcodeList
from pcode_graph.maker import MakerFlags, make_graph
from pcode_graph.view import draw_graph
from magika import Magika


def main():
    args = docopt(__doc__)

    try:
        if args["--debug"]:
            setup_logger(stderr_verbosity="DEBUG")
        else:
            lief.logging.disable()
            setup_logger()

        input_path = Path(args["<input_path>"])
        assert input_path.exists()

        binary: Binary | None = None
        assert input_path.stat().st_size > 0, "Input file is empty"
        if not Magika().identify_path(input_path).output.is_text:
            binary = parse_binary(input_path)

        if args["--arch"] != "auto":
            arch: Arch = Arch(args["--arch"])
        else:
            assert binary, "Assembly file requires --arch"
            arch = get_architecture(binary)

        output_path = args["--output"]
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            output = open(output_path, "w")
        else:
            output = sys.stdout

        if args["functions"]:
            assert binary, "Command 'functions' requires a binary file"
            for f in iter_functions(binary):
                print(
                    f"{hex(f.address)}: {f.name}: {len(f.content)} bytes", file=output
                )
            sys.exit(0)

        translator = Translator(arch)

        function_name = args["--function"]
        chunk_begin_end = args["--chunk"]
        operations: list[pypcode.PcodeOp] = []

        if binary:
            if function_name or chunk_begin_end:
                if function_name:
                    # Lift function
                    function = get_function(binary, function_name)
                    start_address = function.address
                    code = function.content
                else:
                    # Lift piece of code
                    if not re.search("[a-fA-FxX]", chunk_begin_end):
                        logger.warning(
                            f"This does not look like hexadecimal addresses: {chunk_begin_end}"
                        )
                    start_address, end_address = [
                        int(a, 16) for a in chunk_begin_end.split(":")
                    ]
                    code = lookup_chunk(binary, start_address, end_address)
                if args["asm"]:
                    print(translator.dump_asm(code, start_address, False), file=output)
                    sys.exit(0)
                operations = translator.translate(code, start_address)
            else:
                # Lift everything
                for section in iter_code_sections(binary):
                    if args["asm"]:
                        print(f"; Section at {hex(section.address)}", file=output)
                        print(
                            translator.dump_asm(
                                section.content, section.address, False
                            ),
                            file=output,
                        )
                    else:
                        operations += translator.translate(
                            section.content, section.address
                        )
                if args["asm"]:
                    sys.exit(0)
        else:
            assert (
                not function_name
            ), "Option --function not supported for assembly files."
            assert (
                not chunk_begin_end
            ), "Option --chunk not supported for assembly files."
            assembly = input_path.read_text()
            assembler = Assembler(arch)
            start_address = int(args["--base-address"])
            code = assembler.assemble(assembly, start_address)
            if args["asm"]:
                print(translator.dump_asm(code, start_address, False), file=output)
                sys.exit(0)
            operations = translator.translate(code, start_address)

        if args["pcode"]:
            for op in operations:
                print(dump_operation(op), file=output)
            sys.exit(0)

        logger.info(f"Analyze {len(operations)} P-Code operations")
        rpl = RichPcodeList(arch, operations)

        if args["table"]:
            print(rpl.dump())
            sys.exit(0)

        flags = MakerFlags(build_cfg=not args["--dataflow-only"])
        cdg = make_graph(rpl, flags)

        if args["md"]:
            print(str(cdg), file=output)
            sys.exit(0)

        if args["stats"]:
            stats = dict(
                diameter=get_graph_diameters(cdg)._asdict(),
                edges=count_edges(cdg)._asdict(),
                nodes=len(cdg.nodes),
            )

            print(repr(stats), file=output)
            sys.exit(0)

        assert args["html"], "Unknown command"
        assert output_path, "HTML output file is mandatory"
        output.close()
        draw_graph(cdg, Path(output_path))

    except Exception as e:
        if args["--debug"]:
            raise
        print(str(e), file=sys.stderr)
        sys.exit(1)
