from enum import IntFlag, auto
from typing import Callable, NamedTuple
import igraph
from pcode_graph.nx_exporter import export_to_networkx
from pcode_graph.graph import CDG, Edge, EdgeKinds, NodeKinds
from pcode_graph.pcode import OpCodes


from collections import deque
from itertools import chain


class GraphCharacteristics(NamedTuple):
    data_only: int
    control_only: int
    both: int


def count_edges(cdg: CDG) -> GraphCharacteristics:

    data_only: int = 0

    for edge in cdg.edges:
        if edge.kind == EdgeKinds.Data:
            data_only += 1

    return GraphCharacteristics(data_only, len(cdg.edges) - data_only, len(cdg.edges))


def get_graph_diameter(
    cdg: CDG,
    edge_filter: Callable[[Edge], bool] = lambda e: True,
) -> int:
    """Computes the undirected diameter of the given graph."""

    undirected = igraph.Graph(
        n=len(cdg.nodes),
        edges=[
            (edge.source_node, edge.destination_node)
            for edge in cdg.edges
            if edge_filter(edge)
        ],
        directed=True,
    )
    return undirected.diameter(directed=True, unconn=True)


def get_graph_diameters(
    cdg: CDG,
) -> GraphCharacteristics:
    return GraphCharacteristics(
        data_only=get_graph_diameter(cdg, lambda e: e.kind == EdgeKinds.Data),
        control_only=get_graph_diameter(cdg, lambda e: e.kind != EdgeKinds.Data),
        both=get_graph_diameter(cdg),
    )


class FunctionCharacteristics(IntFlag):
    """Helps evaluating the level of difficulty of a function,
    regarding miscellaneous ML tasks."""

    data = auto()
    condition = auto()
    memory = auto()
    phi = auto()
    call = auto()
    loop = auto()


def has_loops(cdg: CDG) -> bool:
    """Returns True if the given graph has loops."""
    edges = []
    for edge in cdg.edges:
        if edge.kind == EdgeKinds.Control:
            edges.append((edge.source_node, edge.destination_node))
        elif edge.kind == EdgeKinds.BasicBlockMembership:
            # Reverse membership to create cycles
            edges.append((edge.destination_node, edge.source_node))
    return not igraph.Graph(n=len(cdg.nodes), edges=edges, directed=True).is_dag()


def get_function_characteristics(graph: CDG) -> FunctionCharacteristics:
    """Characterizes a function regarding control flow features."""

    # Every function contains data-flow easiest level
    function_characteristics = FunctionCharacteristics.data

    if has_loops(graph):
        function_characteristics |= FunctionCharacteristics.loop

    for n in graph.nodes:
        match n.kind:
            case NodeKinds.Operation:
                if n.opcode in {OpCodes.CALL, OpCodes.CALLIND}:
                    function_characteristics |= FunctionCharacteristics.call
                elif n.opcode == OpCodes.CBRANCH:
                    function_characteristics |= FunctionCharacteristics.condition
                elif n.opcode in {OpCodes.LOAD, OpCodes.STORE}:
                    function_characteristics |= FunctionCharacteristics.memory

            case NodeKinds.Phi:
                function_characteristics |= FunctionCharacteristics.phi

    return function_characteristics
