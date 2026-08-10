from torch import Tensor
import torch
from torch_geometric.data import Data
from pcode_graph.arch import Arch
from pcode_graph.pcode import OpCodes
from pcode_graph.registers import ALIAS, CALLING_CONVENTIONS, PARENTS
from pcode_graph.torch_utils import BinaryConverter
from pcode_graph.graph import EdgeKinds, NodeKinds, Node, CDG


def map_registers(registers: set[str], arch: Arch) -> dict[str, Tensor]:
    """Maps the given register names to a one-hot encoded mask,
    handling registers aliases."""

    kept_regs: set[str] = set()

    for r in registers:
        kept_regs.add(PARENTS[arch].get(r, r))

    masks = {}

    zeros = torch.zeros((len(kept_regs),), dtype=torch.float32)
    for i, r in enumerate(kept_regs):
        mask = zeros.clone()
        mask[i] = 1

    return masks


def map_calling_convention_registers(arch: Arch) -> dict[str, Tensor]:
    """Maps the register involved in parameter passing and value-return to a bit mask."""

    lengths: dict[str, int] = {}

    for arch_cc in CALLING_CONVENTIONS.values():
        for k, regs in arch_cc.items():
            lengths[k] = max(len(regs), lengths.get(k, 0))

    arch_cc = CALLING_CONVENTIONS[arch]
    registers = sum(arch_cc.values(), start=[])
    zeros = torch.zeros((sum(lengths.values()),), dtype=torch.float)
    masks: dict[str, Tensor] = {r: zeros.clone() for r in registers}

    i = 0
    for key, l in lengths.items():
        for j, r in enumerate(arch_cc.get(key, [])):
            masks[r][i + j] = 1
        i += l

    for r in registers:
        for a in ALIAS[arch].get(r, set()):
            masks[a] = masks[r]

    return masks


def node_to_tensor(
    node: Node,
    include_size: bool,
    value_converter: BinaryConverter,
    registers_emb: dict[str, Tensor] | None,
    opcodes_mapper: dict[OpCodes, int],
    node_kinds_mapper: dict[NodeKinds, int],
    encode_registers_in_all_nodes: bool,
) -> Tensor:
    """Computes the initial feature for the given node."""

    features = []

    node_kind = torch.zeros((len(node_kinds_mapper),), dtype=torch.float32)
    if node.kind in node_kinds_mapper:
        node_kind[node_kinds_mapper[node.kind]] = 1
    features.append(node_kind)

    max_bytes = value_converter.num_bits // 8
    if include_size:
        # Use hot-encoding for possible value sizes,
        # which are necessarily powers of two
        size_indexes = {
            1: 0,
            2: 1,
            4: 2,
            8: 3,
            16: 4,
        }
        node_size = torch.zeros((size_indexes[max_bytes] + 1,), dtype=torch.float32)
        if node.size:
            node_size[size_indexes[node.size]] = 1
        features.append(node_size)

    node_opcode = torch.zeros((len(opcodes_mapper),), dtype=torch.float32)
    if node.opcode is not None:
        node_opcode[opcodes_mapper[node.opcode]] = 1
    features.append(node_opcode)

    if registers_emb is not None:
        mask = unknown_reg = torch.zeros_like(next(iter(registers_emb.values())))
        if (
            node.kind in {NodeKinds.InputRegister, NodeKinds.OutputRegister}
            or encode_registers_in_all_nodes
        ):
            mask = registers_emb.get(node.register_name, unknown_reg)
        features.append(mask)

    node_val = value_converter.value_to_bits(
        node.signed_value() if node.kind == NodeKinds.Constant else 0
    )

    features.append(node_val)

    return torch.cat(features)


# Used to encode the order of operation inputs,
# (except for OutputRegister, Phi and CALLOTHER Operation,
# which can have more inputs) and the special successor
# for CBRANCH, CALL and CALLIND.
MAX_OPERAND_NUMBER = 2


def graph_to_data(
    graph: CDG,
    registers_emb: dict[str, Tensor] | None = None,
    opcodes_mapper: dict[OpCodes, int] | None = None,
    node_kinds_mapper: dict[NodeKinds, int] | None = None,
    max_value_bytes: int = 4,
    include_size: bool = False,
    include_output_registers_for_all_node: bool = False,
) -> Data:
    """Converts this graph into pytorch geometric Data format.

    Args:
        graph (CDG): the graph to export.
        registers_emb (dict[str, Tensor]): Dictionary giving the embedding to use for a set of registers
            to encode in node features. Registers not found in the dict will be encoded as full-zero.
            You can pass None to avoid any register encoding in node features.
        opcodes_mapper (dict[OpCodes, int]): A dict mapping Operation OpCodes into zero-based indexes.
          OpCodes are one-hot encoded. OpCodes not found in the dict are converted to zero.
          Pass None to use the whole list of supported opcodes.
        node_kinds_mapper (dict[NodeKinds, int]): A dict mapping node kinds into zero-based indexes.
          Node kinds are one-hot encoded. Node kinds not found in the dict are converted to zero.
          Pass None to use the whole list of node kinds.
        max_value_bytes (int): Maximum bytes to encode when putting constant values in the graph.
        include_size (bool): Whether to add sizes in node features.
        include_output_registers_for_all_node (bool): Whether to include output register for all node
            else, only InputRegister and OutputRegister node include them.

    Returns:
        Data: the graph in pytorch_geometric format.
    """

    value_converter = BinaryConverter(max_value_bytes * 8)

    features = []
    edge_src = []
    edge_dst = []
    edge_attr = []

    if opcodes_mapper is None:
        opcodes_mapper = {o: o.value for o in OpCodes}

    if node_kinds_mapper is None:
        node_kinds_mapper = {k: k.value for k in NodeKinds}

    for index, node in enumerate(graph.nodes):
        node_features = node_to_tensor(
            node,
            include_size,
            value_converter,
            registers_emb,
            opcodes_mapper,
            node_kinds_mapper,
            include_output_registers_for_all_node
            or node.kind in {NodeKinds.InputRegister, NodeKinds.OutputRegister},
        )
        features.append(node_features)

    # Gather edges by node
    assert len(EdgeKinds) == 2
    for edge in graph.edges:
        attr = torch.zeros((2 + MAX_OPERAND_NUMBER,))
        if edge.kind == EdgeKinds.Data:
            attr[0] = 1
        else:
            attr[1] = 1
        if edge.operand_number is not None and edge.operand_number < MAX_OPERAND_NUMBER:
            attr[edge.operand_number + 2] = 1

        edge_src.append(edge.source_node)
        edge_dst.append(edge.destination_node)
        edge_attr.append(attr)

    if len(edge_attr) == 0:
        # Support for a graph without any edge
        edges = torch.zeros((2, 2 + MAX_OPERAND_NUMBER, 0))
    else:
        edges = torch.stack(edge_attr)

    return Data(
        x=torch.stack(features),
        edge_index=torch.stack(
            (
                torch.tensor(edge_src, dtype=torch.int64),
                torch.tensor(edge_dst, dtype=torch.int64),
            )
        ),
        edge_attr=edges,
    )
