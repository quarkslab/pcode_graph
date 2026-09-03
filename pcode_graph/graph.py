from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Iterator
from pcode_graph.pcode import OpCodes
from pcode_graph.utils import unsigned_to_signed

NodeIndex = int
EdgeIndex = int
OperandNumber = int
RegisterId = int
Value = int


class NodeKinds(Enum):

    InputRegister = 0
    OutputRegister = 1
    Constant = 2

    Operation = 3
    Phi = 4

    ReadMemory = 5
    WrittenMemory = 6

    Begin = 7
    External = 8
    End = 9


class EdgeKinds(Enum):
    Data = 0
    Control = 1


@dataclass(slots=True)
class Edge:
    source_node: NodeIndex

    destination_node: NodeIndex

    # Kind of edge
    kind: EdgeKinds

    # Operand rank if has meaning
    operand_number: OperandNumber | None


@dataclass(slots=True)
class Node:
    # Kind of node
    kind: NodeKinds

    # Number of bytes of the input or operation represented by this node
    size: int = 0

    # Value for a constant node
    value: Value = 0

    # Id of the register whatever its size version (ex: w0 and x0 will have same id in arm64)
    # Warning: this value is unique in the current pcode context only.
    register_id: RegisterId = -1

    # Name of the register, for display purpose
    register_name: str = ""

    # Operation code
    opcode: OpCodes | None = None

    def __post_init__(self):
        assert (self.kind == NodeKinds.Operation) ^ (self.opcode is None)

    def signed_value(self) -> int:
        assert self.kind == NodeKinds.Constant
        return unsigned_to_signed(self.value, self.size)

    @property
    def size_bits(self) -> int:
        return self.size * 8

    def __str__(self):
        match self.kind:
            case NodeKinds.ReadMemory | NodeKinds.WrittenMemory:
                return "MEMORY"
            case NodeKinds.InputRegister | NodeKinds.OutputRegister:
                return self.register_name
            case NodeKinds.Constant:
                return f"#{hex(self.signed_value())}"
            case NodeKinds.Operation:
                assert self.opcode is not None
                return self.opcode.name
            case NodeKinds.Phi:
                return "ϕ"
            case NodeKinds.Begin:
                return "BEGIN"
            case NodeKinds.External:
                return "EXTERNAL"
            case NodeKinds.End:
                return "END"
            case _:
                raise ValueError(self.kind)


@dataclass(slots=True)
class EdgeProxy:
    """Easy access to edges for one node."""

    graph: CDG
    in_edge_indexes: list[EdgeIndex]
    out_edge_indexes: list[EdgeIndex]

    def get_in_edges(self) -> Iterator[Edge]:
        for index in self.in_edge_indexes:
            yield self.graph.edges[index]

    def get_out_edges(self) -> Iterator[Edge]:
        for index in self.out_edge_indexes:
            yield self.graph.edges[index]

    def get_data_predecessors(self) -> Iterator[NodeIndex]:
        for e in self.in_edge_indexes:
            edge = self.graph.edges[e]
            if edge.kind == EdgeKinds.Data:
                yield edge.source_node

    def get_control_predecessors(self) -> Iterator[NodeIndex]:
        for e in self.in_edge_indexes:
            edge = self.graph.edges[e]
            if edge.kind == EdgeKinds.Control:
                yield edge.source_node

    def get_data_successors(self) -> Iterator[NodeIndex]:
        for e in self.out_edge_indexes:
            edge = self.graph.edges[e]
            if edge.kind == EdgeKinds.Data:
                yield edge.destination_node

    def get_control_successors(self) -> Iterator[NodeIndex]:
        for e in self.out_edge_indexes:
            edge = self.graph.edges[e]
            if edge.kind == EdgeKinds.Control:
                yield edge.destination_node

    def get_input_edges(self) -> list[EdgeIndex]:
        """Returns data input operands, by number."""

        operands: list[tuple[OperandNumber | None, EdgeIndex]] = []
        hasNone = False
        hasNumber = False
        for in_edge_index in self.in_edge_indexes:
            in_edge = self.graph.edges[in_edge_index]
            if in_edge.kind == EdgeKinds.Control:
                continue
            n = in_edge.operand_number
            if n is None:
                hasNone = True
            else:
                hasNumber = True
            operands.append((n, in_edge_index))
        assert hasNone ^ hasNumber
        if hasNumber:
            operands.sort()
        return [o[1] for o in operands]


@dataclass(slots=True)
class CDG:
    """
    Graph of P-Code operations with Control and Data flow edges.
    """

    # Graph nodes
    # Order in this list gives the NodeIndex used in Edge class
    nodes: list[Node]

    # Edges, order does not matter
    edges: list[Edge]

    def clone(self) -> "CDG":
        return CDG([replace(n) for n in self.nodes], [replace(e) for e in self.edges])

    def compute_edge_proxy(self) -> dict[NodeIndex, EdgeProxy]:
        cache = {index: EdgeProxy(self, [], []) for index in range(len(self.nodes))}
        for e, edge in enumerate(self.edges):
            cache[edge.source_node].out_edge_indexes.append(e)
            cache[edge.destination_node].in_edge_indexes.append(e)
        return cache

    def add_node(self, node: Node) -> NodeIndex:
        """Adds a node into the graph, without any edge. Returns the index of the added node."""
        index = len(self.nodes)
        self.nodes.append(node)
        return index

    def remove_node(self, index: NodeIndex):
        """Removes a node from the graph and its immediate edges, former linked nodes can remain unconnected."""

        def remap(i: NodeIndex) -> NodeIndex:
            return i - int(i > index)

        self.nodes.pop(index)
        self.edges = [
            replace(
                e,
                source_node=remap(e.source_node),
                destination_node=remap(e.destination_node),
            )
            for e in self.edges
            if e.source_node != index and e.destination_node != index
        ]

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def remove_edge(self, edge: Edge):
        self.edges.remove(edge)

    def dump(self) -> str:
        """Dumps the graph in mermaid format."""
        s = ""
        for index, node in enumerate(self.nodes):
            match node.kind:
                case NodeKinds.Begin | NodeKinds.End | NodeKinds.External:
                    s += f"n{index}[[{node}]]\n"
                case NodeKinds.Operation:
                    s += f"n{index}[{node}]\n"
                case NodeKinds.Phi:
                    s += f"n{index}" + "{{" + f"{node}" + "}}\n"
                case _:
                    s += f"n{index}(({node}))\n"

        for edge in self.edges:
            link = "-->" if edge.kind == EdgeKinds.Data else "-.->"
            rank = f"|{edge.operand_number}|" if edge.operand_number is not None else ""
            s += f"n{edge.source_node} {link}{rank} n{edge.destination_node}\n"

        return s

    def dump_edge(self, edge: Edge) -> str:
        link = "-->" if edge.kind == EdgeKinds.Data else "-.->"
        rank = f"|{edge.operand_number}|" if edge.operand_number is not None else ""
        return f"{self.nodes[edge.source_node]} {link}{rank} {self.nodes[edge.destination_node]}"

    def __str__(self) -> str:
        """Dumps the graph in markdown + mermaid format."""

        return "```mermaid\nflowchart\n" + self.dump() + "```\n"
