from enum import Enum, auto
import pypcode
from pcode_graph.arch import Arch
from pcode_graph.utils import unsigned_to_signed


# Replacement for native pypcode opcode (to ease pickling and dumping)
# Raw P-Code only
class OpCodes(Enum):
    BOOL_AND = 0
    BOOL_NEGATE = auto()
    BOOL_OR = auto()
    BOOL_XOR = auto()
    BRANCH = auto()
    BRANCHIND = auto()
    CALL = auto()
    CALLIND = auto()
    CALLOTHER = auto()
    CBRANCH = auto()
    COPY = auto()
    FLOAT_ABS = auto()
    FLOAT_ADD = auto()
    FLOAT_CEIL = auto()
    FLOAT_DIV = auto()
    FLOAT_EQUAL = auto()
    FLOAT_FLOAT2FLOAT = auto()
    FLOAT_FLOOR = auto()
    FLOAT_INT2FLOAT = auto()
    FLOAT_LESS = auto()
    FLOAT_LESSEQUAL = auto()
    FLOAT_MULT = auto()
    FLOAT_NAN = auto()
    FLOAT_NEG = auto()
    FLOAT_NOTEQUAL = auto()
    FLOAT_ROUND = auto()
    FLOAT_SQRT = auto()
    FLOAT_SUB = auto()
    FLOAT_TRUNC = auto()
    INT_2COMP = auto()
    INT_ADD = auto()
    INT_AND = auto()
    INT_CARRY = auto()
    INT_DIV = auto()
    INT_EQUAL = auto()
    INT_LEFT = auto()
    INT_LESS = auto()
    INT_LESSEQUAL = auto()
    INT_MULT = auto()
    INT_NEGATE = auto()
    INT_NOTEQUAL = auto()
    INT_OR = auto()
    INT_REM = auto()
    INT_RIGHT = auto()
    INT_SBORROW = auto()
    INT_SCARRY = auto()
    INT_SDIV = auto()
    INT_SEXT = auto()
    INT_SLESS = auto()
    INT_SLESSEQUAL = auto()
    INT_SREM = auto()
    INT_SRIGHT = auto()
    INT_SUB = auto()
    INT_XOR = auto()
    INT_ZEXT = auto()
    LOAD = auto()
    LZCOUNT = auto()
    PIECE = auto()
    POPCOUNT = auto()
    RETURN = auto()
    SEGMENTOP = auto()
    STORE = auto()
    SUBPIECE = auto()


def from_native_opcode(opcode: pypcode.OpCode) -> OpCodes | None:
    if opcode.name == "IMARK":
        return None
    return OpCodes[opcode.name]


JUMP_OPCODES = {
    OpCodes.BRANCH,
    OpCodes.BRANCHIND,
    OpCodes.CALL,
    OpCodes.CALLIND,
    OpCodes.CBRANCH,
    OpCodes.RETURN,
    OpCodes.CALLOTHER,
}


NUM_OPERANDS: dict[OpCodes, int] = {
    OpCodes.BOOL_AND: 2,
    OpCodes.BOOL_NEGATE: 1,
    OpCodes.BOOL_OR: 2,
    OpCodes.BOOL_XOR: 2,
    OpCodes.BRANCH: 0,
    OpCodes.BRANCHIND: 1,
    OpCodes.CALL: 0,
    OpCodes.CALLIND: 1,
    OpCodes.CALLOTHER: -1,
    OpCodes.CBRANCH: 1,
    OpCodes.COPY: 1,
    OpCodes.FLOAT_ABS: 1,
    OpCodes.FLOAT_ADD: 2,
    OpCodes.FLOAT_CEIL: 1,
    OpCodes.FLOAT_DIV: 2,
    OpCodes.FLOAT_EQUAL: 2,
    OpCodes.FLOAT_FLOAT2FLOAT: 1,
    OpCodes.FLOAT_FLOOR: 1,
    OpCodes.FLOAT_INT2FLOAT: 1,
    OpCodes.FLOAT_LESS: 2,
    OpCodes.FLOAT_LESSEQUAL: 2,
    OpCodes.FLOAT_MULT: 2,
    OpCodes.FLOAT_NAN: 1,
    OpCodes.FLOAT_NEG: 1,
    OpCodes.FLOAT_NOTEQUAL: 2,
    OpCodes.FLOAT_ROUND: 1,
    OpCodes.FLOAT_SQRT: 1,
    OpCodes.FLOAT_SUB: 2,
    OpCodes.FLOAT_TRUNC: 1,
    OpCodes.INT_2COMP: 1,
    OpCodes.INT_ADD: 2,
    OpCodes.INT_AND: 2,
    OpCodes.INT_CARRY: 2,
    OpCodes.INT_DIV: 2,
    OpCodes.INT_EQUAL: 2,
    OpCodes.INT_LEFT: 2,
    OpCodes.INT_LESS: 2,
    OpCodes.INT_LESSEQUAL: 2,
    OpCodes.INT_MULT: 2,
    OpCodes.INT_NEGATE: 1,
    OpCodes.INT_NOTEQUAL: 2,
    OpCodes.INT_OR: 2,
    OpCodes.INT_REM: 2,
    OpCodes.INT_RIGHT: 2,
    OpCodes.INT_SBORROW: 2,
    OpCodes.INT_SCARRY: 2,
    OpCodes.INT_SDIV: 2,
    OpCodes.INT_SEXT: 1,
    OpCodes.INT_SLESS: 2,
    OpCodes.INT_SLESSEQUAL: 2,
    OpCodes.INT_SREM: 2,
    OpCodes.INT_SRIGHT: 2,
    OpCodes.INT_SUB: 2,
    OpCodes.INT_XOR: 2,
    OpCodes.INT_ZEXT: 1,
    OpCodes.LOAD: 2,  # Memory node added
    OpCodes.LZCOUNT: 1,
    OpCodes.PIECE: 2,
    OpCodes.POPCOUNT: 1,
    OpCodes.RETURN: 1,
    OpCodes.SEGMENTOP: 2,
    OpCodes.STORE: 2,
    OpCodes.SUBPIECE: 2,
}


def dump_variable(var: pypcode.Varnode | None):
    if var is None:
        return "none"
    reg = var.getRegisterName()
    if reg:
        assert var.space.name == "register"
        return reg
    match var.space.name:
        case "unique":
            return f"${var.offset}"
        case "const":
            # Consider all constants as signed to reduce id length and improve
            # readability
            value = unsigned_to_signed(var.offset, var.size)
            return "#" + hex(value)
        case "ram":
            return f"[0x{var.offset:x}]"
    return f"{var.space.name}:{var.offset}"


def dump_operation(op: pypcode.PcodeOp) -> str:
    s = ""
    if op.output is not None:
        s += dump_variable(op.output) + " = "
    if op.opcode != pypcode.OpCode.COPY:
        s += op.opcode.name.lower() + " "
    s += ", ".join(dump_variable(v) for v in op.inputs)
    return s

