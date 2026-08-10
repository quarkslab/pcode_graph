from pathlib import Path
import re
import shutil
from pcode_graph.arch import Arch
from pytest import fixture
from loguru import logger
from pcode_graph.asm import Assembler
from pcode_graph.log import setup_logger
from pcode_graph.translator import Translator


class ArchContext:
    """To avoid initializing keystone and pypcode at each test."""

    def __init__(self, arch: Arch):
        self.arch = arch
        self.assembler = Assembler(arch)
        self.translator = Translator(arch)


@fixture(scope="session")
def arm64() -> ArchContext:
    return ArchContext(Arch.arm_64)


@fixture(scope="session")
def x86_64() -> ArchContext:
    return ArchContext(Arch.x86_64)


@fixture(scope="session")
def mips_64() -> ArchContext:
    return ArchContext(Arch.mips_64)


@fixture
def outdir(request) -> Path:
    """Creates a fresh test unique directory for generated files,
    and setups a catch-all logfile in it.
    """

    suite, test = request.node.nodeid.split(".py::")
    if suite.startswith("tests/"):
        suite = suite[6:]
    suite = Path("out/tests/pcode_graph") / suite
    if test.find("[") != -1:
        # Handle parameterized tests
        test_base, parameters = test.split("[", maxsplit=1)
        suite = suite / test_base
        test = re.sub(r"[^a-zA-Z0-9]", "_", parameters[:-1])

    suite.mkdir(parents=True, exist_ok=True)
    output_path: Path = suite / test
    if output_path.is_dir():
        # Remove old content but keep directory to avoid losing the path to the open files in editor
        for filepath in output_path.glob("*"):
            if not filepath.is_dir():
                filepath.unlink()
            else:
                shutil.rmtree(filepath)
    else:
        output_path.mkdir()

    return output_path


@fixture
def log(outdir: Path):
    setup_logger(outdir / "test.log")
    return logger
