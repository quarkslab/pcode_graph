
from pcode_graph.checker import check_graph
from pcode_graph.graph import (
    CDG,
    EdgeIndex,
    EdgeKinds,
    NodeIndex,
    OpCodes,
    NodeKinds,
)

from loguru import logger
from pathlib import Path


class GraphSimplifier:

    def __init__(
        self,
        graph: CDG,
    ) -> None:
        self.graph = graph

        self._proxy = self.graph.compute_edge_proxy()

    def _finalize_removals(self):

        remap: dict[int, int] = {}
        shift = 0
        kept_nodes = []
        for index, node in enumerate(self.graph.nodes):
            if node is None:
                shift += 1
            else:
                remap[index] = index - shift
                kept_nodes.append(node)

        self.graph.nodes = kept_nodes

        kept_edges = []
        for edge in self.graph.edges:
            if edge is None:
                continue
            edge.destination_node = remap[edge.destination_node]
            edge.source_node = remap[edge.source_node]
            kept_edges.append(edge)

        self.graph.edges = kept_edges

    def _remove_node(self, index: NodeIndex):

        # logger.debug(f"Remove {self.graph.nodes[index]} at {index}")

        proxy = self._proxy[index]

        control_succs = list(proxy.get_control_successors())
        assert (
            len(control_succs) <= 1
        ), "More than one control successor to the node to remove"

        data_preds = list(proxy.get_data_predecessors())

        # Convert to set to handle particular case of Phi node
        assert len(data_preds) <= 1 or (
            self.graph.nodes[index].kind == NodeKinds.Phi and len(set(data_preds)) == 1
        ), "More than one data predecessor to the node to remove"

        for i, e in enumerate(self.graph.edges):
            if e is None:
                continue
            remove = False
            if e.source_node == index:
                if e.kind == EdgeKinds.Data:
                    self._proxy[e.source_node].out_edge_indexes.remove(i)
                    e.source_node = data_preds[0]
                    self._proxy[e.source_node].out_edge_indexes.append(i)
                else:
                    remove = True

            if e.destination_node == index:
                if e.kind == EdgeKinds.Control:
                    self._proxy[e.destination_node].in_edge_indexes.remove(i)
                    e.destination_node = control_succs[0]
                    self._proxy[e.destination_node].in_edge_indexes.append(i)
                else:
                    remove = True

            if remove:
                self._remove_edge(i)

        # Patch the graph
        self.graph.nodes[index] = None  # type: ignore
        self._graph_changed = True

        # self.debug_helper()

    def _remove_edge(self, edge_index: int):

        edge = self.graph.edges[edge_index]
        self._proxy[edge.source_node].out_edge_indexes.remove(edge_index)
        self._proxy[edge.destination_node].in_edge_indexes.remove(edge_index)
        self.graph.edges[edge_index] = None  # type: ignore
        self._graph_changed = True

    def debug_helper(self):
        try:
            self.rounds += 1
        except AttributeError:
            self.rounds = 0
        path = Path(f"round{self.rounds}.md")
        path.write_text(str(self.graph))
        logger.debug(f"Written {path}")
        check_graph(self.graph)

    def process(self):

        self._graph_changed = True

        while self._graph_changed:
            self._graph_changed = False

            # Remove useless phi nodes
            for index, node in enumerate(self.graph.nodes):

                if node is None:
                    continue

                if node.kind == NodeKinds.Phi:

                    phi_data_preds = set(self._proxy[index].get_data_predecessors())
                    if len(phi_data_preds) > 1:
                        # Keep this Phi node
                        continue

                    self._remove_node(index)                    

                elif node.opcode == OpCodes.COPY:

                    ctrl_preds = set(self._proxy[index].get_control_predecessors())
                    ctrl_succs = set(self._proxy[index].get_control_successors())

                    remove = True
                    if ctrl_preds and ctrl_succs:
                        (ctrl_succ,) = ctrl_succs

                        remove = False
                        # Remove copies that dominates their successor...
                        if list(self._proxy[ctrl_succ].get_control_predecessors()) == [
                            index
                        ]:
                            remove = True
                        elif len(ctrl_preds) == 1:
                            # ...or are dominated by their unique predecessor
                            (ctrl_pred,) = ctrl_preds
                            if list(
                                self._proxy[ctrl_pred].get_control_successors()
                            ) == [index]:
                                remove = True

                    if remove:
                        self._remove_node(index)

                elif node.kind == NodeKinds.Constant:
                    if not list(self._proxy[index].get_data_successors()):
                        # Remove constants let orphan by other simplifications
                        self._remove_node(index)

                elif node.opcode in {
                    OpCodes.INT_ZEXT,
                    OpCodes.INT_SEXT,
                }:
                    self._remove_node(index)

                elif node.opcode == OpCodes.SUBPIECE:

                    operands: list[EdgeIndex] = self._proxy[index].get_input_edges()
                    assert len(operands) == 2
                    source_node_index = self.graph.edges[operands[1]].source_node
                    num_trunc_bytes = self.graph.nodes[source_node_index]
                    assert num_trunc_bytes.kind == NodeKinds.Constant
                    if num_trunc_bytes.value == 0:
                        self._remove_edge(operands[1])
                        self._remove_node(index)

                elif node.opcode == OpCodes.INT_MULT:
                    # Remove mulitiplication by one
                    for op_edge_index in self._proxy[index].get_input_edges():
                        op_edge = self.graph.edges[op_edge_index]
                        o = op_edge.source_node
                        op = self.graph.nodes[o]
                        if op.kind == NodeKinds.Constant and op.value == 1:
                            self._remove_edge(op_edge_index)
                            self._remove_node(index)
                            break

        self._finalize_removals()
