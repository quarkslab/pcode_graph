from dataclasses import asdict
from pcode_graph.graph import CDG, EdgeKinds
import networkx as nx


def export_to_networkx(graph: CDG, ignore_cfg: bool) -> nx.MultiDiGraph:
    """Export the graph to network X format, keeping attributes."""

    ng = nx.MultiDiGraph()
    for n, node in enumerate(graph.nodes):
        ng.add_node(n, **asdict(node))
    for e in graph.edges:
        if ignore_cfg and e.kind == EdgeKinds.Control:
            continue
        ng.add_edge(e.source_node, e.destination_node, kind=e.kind.name, operand_number=e.operand_number,)
    return ng




