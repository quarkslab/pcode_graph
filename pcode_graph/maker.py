from dataclasses import dataclass, field

from pcode_graph.analysis import (
    EntryIndex,
    OpIndex,
    RichPcodeList,
    UnknownCodeIndex,
    ExitIndex,
    Var,
    VarKinds,
)
from pcode_graph.asm import Assembler
from pcode_graph.checker import check_graph
from pcode_graph.graph import CDG, Edge, EdgeKinds, NodeIndex, NodeKinds, Node
from pcode_graph.pcode import from_native_opcode
from pcode_graph.registers import SPECIAL_REGISTERS
from pcode_graph.simplifier import GraphSimplifier
from pcode_graph.translator import Translator

from pypcode import OpCode
from loguru import logger


@dataclass
class MakerFlags:
    build_cfg: bool = True
    """Add control flow links in addition to the data flow. Defaults to True."""

    reg_outputs_black_list: set[str] | None = None
    """Do not consider the given registers as graph outputs to simplify the graphs.
    Arch-dependent register names.
    If white-list is not None, the value of black-list is ignored.
    If both black-list and white-list are None, the content of SPECIAL_REGISTERS is used
    for black-list.
    """

    reg_outputs_white_list: set[str] | None = None
    """Consider only the given registers as graph outputs to simplify the graphs.
    Arch-dependent register names.
    If white-list is not None, the value of black-list is ignored.
    """


def make_graph(prg: RichPcodeList, flags: MakerFlags | None = None) -> CDG:
    """Makes a graph from a list of P-Code operations.

    Args:
        prg (RichPcodeList): the input list of operations, enriched by analyses.
        flags (MakerFlags): determines what to put in the graph.

    Returns:
        CDG: the graph built.
    """

    if flags is None:
        flags = MakerFlags()    

    black_list, white_list = flags.reg_outputs_black_list, flags.reg_outputs_white_list
    if black_list is None and white_list is None:
        black_list = SPECIAL_REGISTERS[prg.arch]    

    # Already created input nodes, by related input variable
    input_node_indexes: dict[Var, NodeIndex] = {}

    # Already created operation nodes, by related P-Code operation
    operation_node_indexes: dict[OpIndex, NodeIndex] = {}

    # Already created phi nodes, by (sorted) possible alternatives
    # to avoid duplicate phi nodes
    phi_node_indexes: dict[tuple[NodeIndex, NodeIndex], NodeIndex] = {}

    # Created nodes of all kinds
    nodes: list[Node] = []

    # Created edges
    edges: list[Edge] = []

    # To build the data graph recursively
    newly_taken_ops: list[tuple[OpIndex, NodeIndex]] = []

    # External code node if created
    external_node_index: NodeIndex | None = None

    # End node if created
    end_node_index: NodeIndex | None = None

    # begin node has zero index
    if flags.build_cfg:
        nodes.append(Node(NodeKinds.Begin))
        begin_index = 0
    else:
        begin_index = -1

    def get_or_create_terminal(op_index: OpIndex) -> NodeIndex:
        nonlocal external_node_index, end_node_index

        if op_index == ExitIndex:
            if end_node_index is None:
                end_node_index = len(nodes)
                nodes.append(Node(NodeKinds.End))
            return end_node_index
        assert op_index == UnknownCodeIndex
        if external_node_index is None:
            external_node_index = len(nodes)
            nodes.append(Node(NodeKinds.External))
        return external_node_index

    def get_or_create_operation(op_index: OpIndex) -> NodeIndex:
        node_index = operation_node_indexes.get(op_index)
        if node_index is None:
            op = prg.operations[op_index]
            node_index = len(nodes)
            newly_taken_ops.append((op_index, node_index))
            nodes.append(
                Node(
                    NodeKinds.Operation,
                    op.output.size if op.output is not None else 0,
                    opcode=from_native_opcode(op.opcode),
                )
            )
            operation_node_indexes[op_index] = node_index

            # logger.debug(f"Create node {node_index}: {nodes[-1]} from operation {op_index}: {dump_operation(op)}")
        return node_index

    def get_or_create_input(var: Var) -> NodeIndex:
        node_index = input_node_indexes.get(var)
        if node_index is None:
            match var.kind:
                case VarKinds.Constant:
                    node = Node(NodeKinds.Constant, var.size, var.value)
                case VarKinds.Register:
                    node = Node(
                        NodeKinds.InputRegister,
                        var.size,
                        register_id=var.value,
                        register_name=var.pretty_name,
                    )
                case VarKinds.Ram:
                    node = Node(NodeKinds.ReadMemory, var.size)
                case VarKinds.IndirectAccess:
                    node = Node(NodeKinds.ReadMemory, var.size)
                case _:
                    raise ValueError(f"Unexpected kind for input {var.kind}")
            node_index = len(nodes)
            nodes.append(node)
            input_node_indexes[var] = node_index
        return node_index

    def add_control_edge(
        from_node: NodeIndex, to_node: NodeIndex, operand_number: int | None
    ):
        edges.append(Edge(from_node, to_node, EdgeKinds.Control, operand_number))

    def add_data_edge(
        from_node: NodeIndex, to_node: NodeIndex, operand_number: int | None
    ):
        edges.append(Edge(from_node, to_node, EdgeKinds.Data, operand_number))

    def link_defs(
        var: Var,
        def_op_indexes: set[OpIndex],
        user_node_index: NodeIndex,
        operand_number: int | None,
    ):

        # Collect data preds to see if we need a phi node
        # Sort defs to remain determinist
        pred_node_indexes = []
        for def_op_index in sorted(list(def_op_indexes)):
            if def_op_index == EntryIndex:
                # Input variable
                pred_node_indexes.append(get_or_create_input(var))
            else:
                pred_node_indexes.append(get_or_create_operation(def_op_index))

        if len(pred_node_indexes) == 1:
            # Simple def, direct link
            add_data_edge(pred_node_indexes[0], user_node_index, operand_number)
        else:
            # Create phi node
            assert pred_node_indexes
            pred_node_indexes.sort()
            key = tuple(pred_node_indexes)
            phi_node_index = phi_node_indexes.get(key)
            if phi_node_index is None:
                phi_node = Node(NodeKinds.Phi, var.size)
                phi_node_index = len(nodes)
                nodes.append(phi_node)
                phi_node_indexes[key] = phi_node_index
                for pred_node_index in pred_node_indexes:
                    add_data_edge(pred_node_index, phi_node_index, None)
            add_data_edge(phi_node_index, user_node_index, operand_number)

    # Build the data flow

    # Start from writes to outputs registers and memory
    for var, def_op_indexes in prg.get_exit_defs().items():
        assert def_op_indexes
        match var.kind:
            case VarKinds.Register:                
                if white_list is not None:
                    if var.pretty_name not in white_list:
                        continue
                else:
                    assert black_list is not None
                    if var.pretty_name.lower().startswith("tmp"):
                        # Ignore writes to temporaries
                        continue
                    if var.pretty_name in black_list:
                        # Ignore writes to special registers
                        continue
                    if var.pretty_name == "UNKNOWN":
                        # Register name not provided in P-Code
                        continue
                assert var.pretty_name
                output_node = Node(
                    NodeKinds.OutputRegister,
                    var.size,
                    register_id=var.value,
                    register_name=var.pretty_name,
                )

            case VarKinds.IndirectAccess | VarKinds.Ram:
                output_node = Node(NodeKinds.WrittenMemory, var.size)

            case VarKinds.Unique:
                # Ignore writes to temporaries
                continue

            case VarKinds.Constant:
                raise ValueError(f"Unexpected output kind {var.kind}")

        output_node_index = len(nodes)
        nodes.append(output_node)
        # logger.debug(f"Start from output {output_node_index} {output_node}")
        link_defs(var, def_op_indexes, output_node_index, None)

    if flags.build_cfg:
        # Also add conditional branching or branching to unknown address
        for op_index, op in enumerate(prg.operations):
            if not prg.reachable[op_index]:
                continue
            if op.opcode == OpCode.CBRANCH or (
                UnknownCodeIndex in prg.successors.get(op_index, set())
                and op.opcode != OpCode.IMARK
            ):
                get_or_create_operation(op_index)

    # Follow predecessors recursively until the whole dataflow graph is built
    while newly_taken_ops:
        op_index, node_index = newly_taken_ops.pop()

        for number, (var, defs) in enumerate(prg.iter_input_defs(op_index)):

            if not defs:
                # No definition means this is a constant
                pred_node_index = get_or_create_input(var)
                add_data_edge(pred_node_index, node_index, number)
                continue

            link_defs(var, defs, node_index, number)

    if flags.build_cfg:
        # Build the control flow graph, jumping over nodes which
        # do not already belong to the data flow graph.

        for op_index, node_index in operation_node_indexes.items():

            if not prg.reachable[op_index]:
                continue

            successors = prg.successors.get(op_index, [])

            for number, successor_op_index in enumerate(successors):
                if successor_op_index < 0:  # ie. in {UnknownCodeIndex, ExitIndex}
                    add_control_edge(
                        node_index, get_or_create_terminal(successor_op_index), number
                    )
                    continue

                successor_node_index = operation_node_indexes.get(successor_op_index)
                seen_ops: set[OpIndex] = set()

                while successor_node_index is None:

                    # This successor is not part of the dataflow graph, skip it.

                    # Having more than one successor would imply being a CBRANCH,
                    # but CBRANCH are always put in the dataflow, see above.
                    (successor_op_index,) = prg.successors[successor_op_index]

                    if successor_op_index in seen_ops:
                        logger.warning("Looping CFG without side effects")
                        add_control_edge(node_index, node_index, number)
                        break

                    seen_ops.add(successor_op_index)

                    if successor_op_index < 0:  # ie. in {UnknownCodeIndex, ExitIndex}
                        successor_node_index = get_or_create_terminal(
                            successor_op_index
                        )
                        continue

                    successor_node_index = operation_node_indexes.get(
                        successor_op_index
                    )

                else:
                    assert successor_node_index >= 0
                    add_control_edge(node_index, successor_node_index, number)

        if len(nodes) == 1:
            # Empty graph
            add_control_edge(begin_index, get_or_create_terminal(ExitIndex), None)
        else:
            # Link the entry node to the first taken operation
            # TODO: specify entry operation
            entry_op_index = 0
            while entry_op_index not in operation_node_indexes:
                successor_op_indexes = tuple(prg.successors.get(entry_op_index, ()))
                assert len(successor_op_indexes) == 1
                # Advance to next operation
                entry_op_index = successor_op_indexes[0]
            add_control_edge(begin_index, operation_node_indexes[entry_op_index], None)

    graph = CDG(nodes, edges)

    # Uncomment to locate a bug: graph structure vs simplifier
    # check_graph(graph)

    GraphSimplifier(graph).process()
    check_graph(graph)

    return graph


def make_graph_from_binary(
    translator: Translator,
    code: bytes,
    base_address: int,
    flags: MakerFlags | None = None,
) -> CDG:
    """Makes a CDG graph from a piece of binary code.

    Args:
        translator (Translator): The P-Code translator context.
        code (bytes): The binary code to translate.
        base_address (int): The address of the first instruction of the given code.
        flags (MakerFlags): determines what to put in the graph.

    Returns:
        CDG: the graph built.
    """

    operations = translator.translate(code, base_address)
    prg = RichPcodeList(translator.arch, operations=operations)
    return make_graph(prg, flags)


def make_graph_from_asm(
    assembler: Assembler,
    translator: Translator,
    asm: str,
    base_address: int = 0,
    flags: MakerFlags | None = None,
) -> CDG:
    """Makes a CDG graph from a piece of binary code. Mainly for testing purpose.

    Args:
        translator (Translator): The P-Code translator context.
        asm (str): The assembly code to translate.
        base_address (int): The address of the first instruction of the given code.
        flags (MakerFlags): determines what to put in the graph.

    Returns:
        CDG: the graph built.
    """

    return make_graph_from_binary(
        code=assembler.assemble(asm, base_address),
        base_address=base_address,
        translator=translator,
        flags=flags,
    )
