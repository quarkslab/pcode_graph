from enum import IntFlag, auto
from typing import NamedTuple
import networkx as nx


from pcode_graph.nx_exporter import export_to_networkx
from pcode_graph.graph import CDG, EdgeKinds, NodeKinds
from pcode_graph.pcode import OpCodes


from collections import deque
from itertools import chain


class Stats(NamedTuple):
    data_only: int
    control_only: int
    both: int


def count_edges(cdg: CDG) -> Stats:

    data_only: int = 0

    for edge in cdg.edges:
        if edge.kind == EdgeKinds.Data:
            data_only += 1

    return Stats(data_only, len(cdg.edges) - data_only, len(cdg.edges))


def get_graph_diameters(cdg: CDG) -> Stats:
    """Computes the diameter of the given graph."""

    n = len(cdg.nodes)

    def build_adjacency(edges):
        adj = [[] for _ in range(n)]
        for e in edges:
            adj[e.source_node].append(e.destination_node)
        return adj

    data_edges = []
    ctrl_edges = []
    for e in cdg.edges:
        if e.kind == EdgeKinds.Data:
            data_edges.append(e)
        else:
            ctrl_edges.append(e)

    def diameter(adjacency) -> int:
        # Compute graph diameter with BFS
        d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            front = deque((src,))
            while front:
                current = front.popleft()
                for neigbor in adjacency[current]:
                    if dist[neigbor] == -1:
                        dist[neigbor] = dist[current] + 1
                        front.append(neigbor)
            d = max(d, max(dist))
        return d

    return Stats(
        diameter(build_adjacency(data_edges)),
        diameter(build_adjacency(ctrl_edges)),
        diameter(build_adjacency(chain(data_edges, ctrl_edges))),
    )


class FunctionCharacteristics(IntFlag):
    """Helps evaluating the level of difficulty of a function,
    regarding miscalleneous ML tasks."""

    data = auto()
    condition = auto()
    memory = auto()
    phi = auto()
    call = auto()
    loop = auto()


def get_function_characteristics(graph: CDG) -> FunctionCharacteristics:
    """Characterizes a function regarding control flow features."""

    # Every function contains data-flow easiest level
    function_characteristics = FunctionCharacteristics.data

    ng = export_to_networkx(graph, False)
    try:
        nx.find_cycle(ng)
    except nx.NetworkXNoCycle:
        pass
    else:
        function_characteristics |= FunctionCharacteristics.loop

    for n in graph.nodes:
        match n.kind:
            case NodeKinds.Operation:
                if n.opcode in {OpCodes.CALL, OpCodes.CALLIND}:
                    function_characteristics |= FunctionCharacteristics.call
                elif n.opcode == OpCodes.CBRANCH:
                    function_characteristics |= FunctionCharacteristics.condition

            case NodeKinds.Phi:
                function_characteristics |= FunctionCharacteristics.phi

            case NodeKinds.ReadMemory | NodeKinds.WrittenMemory:
                function_characteristics |= FunctionCharacteristics.memory

    return function_characteristics
