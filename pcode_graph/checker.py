from pcode_graph.graph import CDG, EdgeKinds, NodeKinds

from pcode_graph.pcode import NUM_OPERANDS, OpCodes


def check_graph(graph: CDG):
    """Checks the coherency of this graph.

    Can be used to verify that the graph remains valid after a set of transformations
    made by a neural network for instance.

    Raises AssertionError if invalid.
    """

    # Check edges are valid
    has_cfg = False
    for edge in graph.edges:
        # assert isinstance(edge.source_node, int)
        # assert isinstance(edge.destination_node, int)
        has_cfg |= edge.kind == EdgeKinds.Control
        assert 0 <= edge.source_node < len(graph.nodes)
        assert 0 <= edge.destination_node < len(graph.nodes)

    # Check nodes attributes are coherent with node kind
    proxy_dict = graph.compute_edge_proxy()

    for index, node in enumerate(graph.nodes):

        if node.value:
            assert node.kind == NodeKinds.Constant

        assert (not node.register_name) ^ (
            node.kind in {NodeKinds.InputRegister, NodeKinds.OutputRegister}
        )

        proxy = proxy_dict[index]
        control_succs = list(proxy.get_control_successors())
        data_preds = list(proxy.get_data_predecessors())

        if node.kind != NodeKinds.Operation:
            assert node.opcode is None
        else:
            assert node.opcode is not None
            # Check operands numbering
            assert NUM_OPERANDS[node.opcode] in {-1, len(data_preds)}

            data_numbers: list[int] = []
            control_numbers: list[int] = []
            for edge in proxy.get_in_edges():
                if edge.kind == EdgeKinds.Data:
                    assert edge.operand_number is not None
                    data_numbers.append(edge.operand_number)
            for edge in proxy.get_out_edges():
                if edge.kind == EdgeKinds.Control:
                    assert edge.operand_number is not None
                    control_numbers.append(edge.operand_number)
            assert sorted(data_numbers) == list(range(len(data_preds)))
            assert sorted(control_numbers) == list(range(len(control_succs)))

        if has_cfg:
            # Check CFG coherency
            match len(control_succs):
                case 0:
                    assert node.kind in {
                        NodeKinds.External,
                        NodeKinds.End,
                        NodeKinds.InputRegister,
                        NodeKinds.OutputRegister,
                        NodeKinds.Phi,
                        NodeKinds.Constant,
                        NodeKinds.ReadMemory,
                        NodeKinds.WrittenMemory,
                    }, f"Node {index} {node} without successor"
                case 1:
                    assert node.kind in {NodeKinds.Begin, NodeKinds.Operation}
                    if node.kind == NodeKinds.Operation:
                        assert node.opcode not in {
                            OpCodes.CBRANCH,
                            OpCodes.CALLIND,
                            OpCodes.CALL,
                        }, f"Node {index} {node} with only 1 successors"
                case 2:
                    assert node.kind == NodeKinds.Operation and node.opcode in {
                        OpCodes.CBRANCH,
                        OpCodes.CALLIND,
                        OpCodes.CALL,
                    }, f"Node {index} {node} with 2 successors"
                case _:
                    raise AssertionError("Unexpected number of successors")

        # Check dataflow coeherency
        match len(data_preds):
            case 0:
                assert node.kind in {
                    NodeKinds.Begin,
                    NodeKinds.InputRegister,
                    NodeKinds.OutputRegister,
                    NodeKinds.Constant,
                    NodeKinds.ReadMemory,
                    NodeKinds.External,
                    NodeKinds.End,
                    NodeKinds.Operation,  # For example the CALL operation
                }, f"Node {index} {node} without data predecessors"
            case 1 | 2:
                assert node.kind in {
                    NodeKinds.Operation,
                    NodeKinds.Phi,
                    NodeKinds.OutputRegister,
                    NodeKinds.WrittenMemory,
                }, f"Node {index} {node} with {len(data_preds)} data predecessors"
            case _:
                if node.kind not in {
                    NodeKinds.Phi,
                    NodeKinds.OutputRegister,
                    NodeKinds.WrittenMemory,
                }:
                    assert (
                        node.kind == NodeKinds.Operation
                        and node.opcode == OpCodes.CALLOTHER
                    ), f"Node {index} {node} with {len(data_preds)} data predecessors"
