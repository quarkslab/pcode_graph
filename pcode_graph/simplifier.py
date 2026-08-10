from pcode_graph.checker import check_graph
from pcode_graph.graph import CDG, Edge, EdgeKinds, NodeIndex, OpCodes, NodeKinds

from loguru import logger
from pathlib import Path


class GraphSimplifier:

    def __init__(
        self,
        graph: CDG,
    ) -> None:
        self.graph = graph

    def _update_proxy(self):
        self._proxy = self.graph.compute_edge_proxy()

    def _remove_node(self, index: NodeIndex):
        def remap_index(n: NodeIndex):
            assert n != index
            return n - int(n > index)

        # logger.debug(f"Remove {self.graph.nodes[index]} at {index}")

        # Remap edges
        proxy = self._proxy[index]

        control_succs = proxy.get_control_successors()
        assert (
            len(control_succs) <= 1
        ), "More than one control successor to the node to remove"

        data_preds = proxy.get_data_predecessors()

        # Convert to set to handle particular case of Phi node
        assert len(data_preds) <= 1 or (
            self.graph.nodes[index].kind == NodeKinds.Phi and len(set(data_preds)) == 1
        ), "More than one data predecessor to the node to remove"

        kept_edges: list[Edge] = []

        def remap_edge(
            kind: EdgeKinds, source: NodeIndex, dest: NodeIndex, operand: int | None
        ):
            # logger.debug(
            #     f"Remap {self.graph.dump_edge(Edge(kind=kind, source_node=source, destination_node=dest, operand_number=operand))}"
            # )
            edge = Edge(
                source_node=remap_index(source),
                destination_node=remap_index(dest),
                kind=kind,
                operand_number=operand,
            )
            kept_edges.append(edge)

        for e in self.graph.edges:
            if e.source_node == index:
                if e.kind == EdgeKinds.Data:
                    remap_edge(
                        e.kind,
                        data_preds[0],
                        e.destination_node,
                        e.operand_number,
                    )
            elif e.destination_node == index:
                if e.kind == EdgeKinds.Control:
                    remap_edge(
                        e.kind,
                        e.source_node,
                        control_succs[0],
                        e.operand_number,
                    )
            else:
                remap_edge(e.kind, e.source_node, e.destination_node, e.operand_number)

        # Patch the graph
        self.graph.edges = kept_edges
        self.graph.nodes.pop(index)

        self._graph_changed = True
        self._update_proxy()

        # self.debug_helper()

    def _remove_edge(self, edge: Edge):

        self.graph.remove_edge(edge)
        self._graph_changed = True
        self._update_proxy()

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
        self._update_proxy()

        while self._graph_changed:
            self._graph_changed = False

            # Remove useless phi nodes
            for index, node in enumerate(self.graph.nodes):

                if node.kind == NodeKinds.Phi:

                    phi_data_preds = set(self._proxy[index].get_data_predecessors())
                    if len(phi_data_preds) > 1:
                        # Keep this Phi node
                        continue

                    self._remove_node(index)
                    break

                elif node.opcode == OpCodes.COPY:

                    ctrl_preds = set(self._proxy[index].get_control_predecessors())
                    ctrl_succs = self._proxy[index].get_control_successors()

                    remove = True
                    if ctrl_preds and ctrl_succs:
                        (ctrl_succ,) = ctrl_succs

                        remove = False
                        # Remove copies that dominates their successor...
                        if self._proxy[ctrl_succ].get_control_predecessors() == [index]:
                            remove = True
                        elif len(ctrl_preds) == 1:
                            # ...or are dominated by their unique predecessor
                            (ctrl_pred,) = ctrl_preds
                            if self._proxy[ctrl_pred].get_control_successors() == [
                                index
                            ]:
                                remove = True

                    if remove:
                        self._remove_node(index)
                        break

                elif node.kind == NodeKinds.Constant:                    
                    if not self._proxy[index].get_data_successors():
                        # Remove constants let orphan by other simplifications
                        self._remove_node(index)
                        break

                elif node.opcode in {
                    OpCodes.INT_ZEXT,
                    OpCodes.INT_SEXT,
                }:
                    self._remove_node(index)
                    break

                elif node.opcode == OpCodes.SUBPIECE:

                    operands: list[Edge] = self._proxy[index].get_inputs_edges()

                    assert len(operands) == 2
                    num_trunc_bytes = self.graph.nodes[operands[1].source_node]
                    assert num_trunc_bytes.kind == NodeKinds.Constant
                    if num_trunc_bytes.value == 0:
                        self._remove_edge(operands[1])
                        self._remove_node(index)                        
                        break

                elif node.opcode == OpCodes.INT_MULT:
                    # Remove mulitiplication by one
                    operands: list[Edge] = self._proxy[index].get_inputs_edges()
                    for op_edge in operands:
                        o = op_edge.source_node
                        op = self.graph.nodes[o]
                        if op.kind == NodeKinds.Constant and op.value == 1:
                            self._remove_edge(op_edge)
                            self._remove_node(index)                            
                            break
                    else:
                        continue
                    break
