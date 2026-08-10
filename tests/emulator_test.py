from pcode_graph.pcode import OpCodes
from pcode_graph.graph import CDG, Edge, EdgeKinds, Node, NodeKinds
from pcode_graph.emulator import Emulator


def test_basic_dataflow_emulation():

    graph = CDG([], [])
    r1 = graph.add_node(
        Node(NodeKinds.InputRegister, register_id=0, register_name="R1")
    )
    r2 = graph.add_node(
        Node(NodeKinds.InputRegister, register_id=1, register_name="R2")
    )
    add = graph.add_node(Node(NodeKinds.Operation, opcode=OpCodes.INT_ADD, size=4))
    r3 = graph.add_node(
        Node(NodeKinds.OutputRegister, register_id=2, register_name="R3")
    )

    graph.add_edge(Edge(r1, add, EdgeKinds.Data, 0))
    graph.add_edge(Edge(r2, add, EdgeKinds.Data, 1))
    graph.add_edge(Edge(add, r3, EdgeKinds.Data, None))

    e = Emulator(graph)
    e.set_input_registers({0: 112, 1: 23})
    e.emulate()
    assert e.get_outputs() == {2: 135}
