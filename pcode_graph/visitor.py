from typing import Callable

from pcode_graph.pcode import OpCodes


class PcodeVisitor:
    """A hierarchical visitor to P-Code operations."""

    def __init__(self) -> None:
        self._pcode_callbacks: dict[OpCodes | None, Callable] = {
            opcode: getattr(self, "visit_" + opcode.name.lower()) for opcode in OpCodes
        }
        self._pcode_callbacks[None] = self.visit_nop

    def visit(self, opcode: OpCodes | None, *args, **kwargs):
        "Visitor main entry point."
        return self._pcode_callbacks[opcode](*args, **kwargs)

    def visit_nop(self, *args, **kwargs):
        return self.visit_default(*args, **kwargs)

    def visit_binary(self, *args, **kwargs):
        "A binary logical or arithmetical operation."
        return self.visit_default(*args, **kwargs)

    def visit_unary(self, *args, **kwargs):
        "An unary logical or arithmetical operation."
        return self.visit_default(*args, **kwargs)

    def visit_jump(self, *args, **kwargs):
        "Any operation impacting the usual instruction flow."
        return self.visit_default(*args, **kwargs)

    def visit_default(self, *args, **kwargs):
        "Catch-all callback"
        raise NotImplementedError

    # fmt: off
    def visit_invalid(self, *args, **kwargs): return self.visit_default(*args, **kwargs)
    def visit_bool_and(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_bool_negate(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_bool_or(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_bool_xor(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_branch(self, *args, **kwargs): return self.visit_jump(*args, **kwargs)
    def visit_branchind(self, *args, **kwargs): return self.visit_jump(*args, **kwargs)
    def visit_call(self, *args, **kwargs): return self.visit_branch(*args, **kwargs)
    def visit_callind(self, *args, **kwargs): return self.visit_branchind(*args, **kwargs)
    def visit_callother(self, *args, **kwargs): return self.visit_default(*args, **kwargs)  # Note: callother is not a implemented as a branching instruction but is followed by one
    def visit_cbranch(self, *args, **kwargs): return self.visit_jump(*args, **kwargs)
    def visit_copy(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_float_abs(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_float_add(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_float_ceil(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_float_div(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_float_equal(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_float_float2float(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_float_floor(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_float_int2float(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_float_less(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_float_lessequal(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_float_mult(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_float_nan(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_float_neg(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_float_notequal(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_float_round(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_float_sqrt(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_float_sub(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_float_trunc(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_int_2comp(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_int_add(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_and(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_carry(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_div(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_equal(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_left(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_less(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_lessequal(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_mult(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_negate(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_int_notequal(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_or(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_rem(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_right(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_sborrow(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_scarry(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_sdiv(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_sext(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_int_sless(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_slessequal(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_srem(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_sright(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_sub(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)
    def visit_int_xor(self, *args, **kwargs): return self.visit_binary(*args, **kwargs)    
    def visit_int_zext(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_load(self, *args, **kwargs): return self.visit_default(*args, **kwargs)
    def visit_lzcount(self, *args, **kwargs): return self.visit_unary(*args, **kwargs)
    def visit_piece(self, *args, **kwargs): return self.visit_default(*args, **kwargs)
    def visit_popcount(self, *args, **kwargs): return self.visit_default(*args, **kwargs)
    def visit_return(self, *args, **kwargs): return self.visit_branchind(*args, **kwargs)
    def visit_segmentop(self, *args, **kwargs): return self.visit_default(*args, **kwargs)
    def visit_store(self, *args, **kwargs): return self.visit_default(*args, **kwargs)
    def visit_subpiece(self, *args, **kwargs): return self.visit_default(*args, **kwargs)
    # fmt: on
