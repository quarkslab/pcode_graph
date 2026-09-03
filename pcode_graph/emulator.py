from typing import override
from pcode_graph.graph import (
    CDG,
    EdgeKinds,
    Node,
    NodeIndex,
    NodeKinds,
    OperandNumber,
    RegisterId,
    Value,
)
from pcode_graph.utils import signed_to_unsigned, unsigned_to_signed
from pcode_graph.visitor import PcodeVisitor


class Emulator(PcodeVisitor):
    """Emulate the execution of an homogeneous graph"""

    def __init__(self, graph: CDG) -> None:
        super().__init__()
        self.graph = graph

        # Known value of every node
        self.values: dict[NodeIndex, int] = {}

        # Inputs and outputs node indexes sorted by register id
        self.inputs: dict[RegisterId, NodeIndex] = {}
        self.outputs: dict[RegisterId, NodeIndex] = {}

        for index, node in enumerate(self.graph.nodes):
            if node.kind == NodeKinds.InputRegister:
                self.inputs[node.register_id] = index
            elif node.kind == NodeKinds.OutputRegister:
                self.outputs[node.register_id] = index
            elif node.kind == NodeKinds.Constant:
                self.values[index] = node.value

    def clear(self):
        self.values.clear()

    def set_input(self, index: NodeIndex, value: Value):
        self.values[index] = value

    def set_input_register(self, register_id: RegisterId, value: Value):
        index = self.inputs[register_id]
        self.set_input(index, value)

    def set_input_registers(self, values_by_reg_id: dict[RegisterId, Value]):
        for rid, value in values_by_reg_id.items():
            self.set_input_register(rid, value)

    def emulate(self):

        # TODO: support CFG and Phi nodes
        proxy = self.graph.compute_edge_proxy()
        propagated = True
        while propagated:
            propagated = False

            for index, node in enumerate(self.graph.nodes):
                if index in self.values:
                    # Already computed
                    continue
                p = proxy[index]
                match node.kind:

                    case NodeKinds.Operation:
                        operands: dict[
                            OperandNumber | None, tuple[NodeIndex, Value]
                        ] = {}

                        for in_edge in p.get_in_edges():
                            if in_edge.kind != EdgeKinds.Data:
                                continue

                            value = self.values.get(in_edge.source_node)
                            if value is None:
                                break

                            assert in_edge.operand_number not in operands
                            operands[in_edge.operand_number] = (
                                in_edge.source_node,
                                value,
                            )
                        else:
                            # We can evaluate this node
                            self.values[index] = self.visit(node.opcode, node, operands)
                            propagated = True

                    case NodeKinds.OutputRegister:
                        (in_edge_index,) = p.get_input_edges()
                        in_edge = self.graph.edges[in_edge_index]
                        value = self.values.get(in_edge.source_node)
                        if value is not None:
                            self.values[index] = value
                            # Do not set propagated as OutputRegister nodes are never used as inputs

    def get_output(self, register_id) -> Value:
        index = self.outputs[register_id]
        return self.values[index]

    def get_outputs(self) -> dict[RegisterId, Value]:
        out_values = {}
        for register_id, index in self.outputs.items():
            out_values[register_id] = self.values[index]
        return out_values

    @override
    def visit_bool_and(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        return int((i1 != 0) and (i2 != 0))

    @override
    def visit_bool_negate(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        return int(values[0] == 0)

    @override
    def visit_bool_or(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        return int((i1 != 0) or (i2 != 0))

    @override
    def visit_bool_xor(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        return int((i1 != 0) ^ (i2 != 0))

    @override
    def visit_int_2comp(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i = values[0][1]
        mask = 2**node.size_bits - 1
        return (mask ^ i) + 1

    @override
    def visit_int_add(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        return i1 + i2

    @override
    def visit_int_and(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        return i1 & i2

    @override
    def visit_int_carry(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        return int((i1 + i2) >= 2**node.size_bits)

    @override
    def visit_int_div(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        if i2 == 0:
            return 0
        return i1 // i2

    @override
    def visit_int_equal(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        return int(i1 == i2)

    @override
    def visit_int_left(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        return i1 << i2

    @override
    def visit_int_less(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        return int(i1 < i2)

    @override
    def visit_int_lessequal(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        return int(i1 <= i2)

    @override
    def visit_int_mult(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        return i1 * i2

    @override
    def visit_int_negate(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i = values[0][1]
        mask = 2 ** (node.size * 8) - 1
        return mask ^ i

    @override
    def visit_int_notequal(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        return int(i1 != i2)

    @override
    def visit_int_or(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        return i1 | i2

    @override
    def visit_int_rem(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        if i2 == 0:
            return 0
        q = i1 // i2
        return i1 - q * i2

    @override
    def visit_int_right(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        return i1 >> i2

    @override
    def visit_int_sborrow(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        input_size = self.graph.nodes[values[0][0]].size
        max = 2 ** (input_size * 8 - 1)
        return int(
            not (
                -max
                <= (
                    unsigned_to_signed(i1, input_size)
                    - unsigned_to_signed(i2, input_size)
                    < max
                )
            )
        )

    @override
    def visit_int_scarry(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        input_size = self.graph.nodes[values[0][0]].size
        max = 2 ** (input_size * 8 - 1)
        return int(
            not (
                -max
                <= (
                    unsigned_to_signed(i1, input_size)
                    + unsigned_to_signed(i2, input_size)
                    < max
                )
            )
        )

    @override
    def visit_int_sdiv(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        if i2 == 0:
            return 0
        s1, s2 = unsigned_to_signed(i1, node.size), unsigned_to_signed(i2, node.size)
        return signed_to_unsigned(int(s1 / s2), node.size)

    @override
    def visit_int_sext(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i = values[0][1]
        input_size = self.graph.nodes[values[0][0]].size
        s = unsigned_to_signed(i, input_size)
        return signed_to_unsigned(s, node.size)

    @override
    def visit_int_sless(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        input_size = self.graph.nodes[values[0][0]].size
        return int(
            unsigned_to_signed(i1, input_size) < unsigned_to_signed(i2, input_size)
        )

    @override
    def visit_int_slessequal(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        input_size = self.graph.nodes[values[0][0]].size
        return int(
            unsigned_to_signed(i1, input_size) <= unsigned_to_signed(i2, input_size)
        )

    @override
    def visit_int_srem(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        if i2 == 0:
            return 0
        s1, s2 = unsigned_to_signed(i1, node.size), unsigned_to_signed(i2, node.size)
        q = int(s1 / s2)
        return signed_to_unsigned(s1 - q * s2, node.size)

    @override
    def visit_int_sright(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        s1, s2 = unsigned_to_signed(i1, node.size), unsigned_to_signed(i2, node.size)
        return signed_to_unsigned(s1 >> s2, node.size)

    @override
    def visit_int_sub(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        return i1 - i2

    @override
    def visit_int_xor(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i1, i2 = values[0][1], values[1][1]
        return i1 ^ i2

    @override
    def visit_int_zext(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        return values[0][1]

    @override
    def visit_load(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> None:
        raise NotImplementedError

    @override
    def visit_lzcount(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        value = values[0][1]
        max_bits = node.size * 8
        top_bit = 1 << (max_bits - 1)
        count = 0
        while not (value & top_bit):
            count += 1
            value <<= 1
        return count

    @override
    def visit_piece(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        # Consider the virtual machine as big endian
        i1, i2 = values[0][1], values[1][1]
        return (i1 << self.graph.nodes[values[1][0]].size_bits) | i2

    @override
    def visit_popcount(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        i = values[0][1]
        return i.bit_count()

    @override
    def visit_segmentop(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        # TODO: opcode not documented in Pcode spec
        raise NotImplementedError

    @override
    def visit_store(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> None:
        raise NotImplementedError

    @override
    def visit_subpiece(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        # Consider the machine as big endian (see piece)
        i1, i2 = values[0][1], values[1][1]
        return (i1 >> (i2 * 8)) & (2**node.size_bits - 1)

    @override
    def visit_copy(
        self, node: Node, values: dict[OperandNumber, tuple[NodeIndex, Value]]
    ) -> int:
        return values[0][1]
