from typing import List
import networkx as nx
import matplotlib.pyplot as plt
from pddlgym.structs import Literal, LiteralDisjunction, LiteralConjunction
from pddlgym.core import PDDLEnv
from pddlgym_planners.fd import FD


class GoalNetwork:
    """
    A directed acyclic graph representing a partial order of subgoals.
    """

    def __init__(
        self,
        dag_or_env: nx.DiGraph | PDDLEnv,
    ) -> None:
        """
        A directed acyclic graph representing a partial order of landmarks. Uses the
        LM_RHW landmark extraction algorithm (Richter, Helmert, and Westphal 2008),
        without reasonable orderings.

        Parameters
        ----------
        dag_or_env : nx.DiGraph | PDDLEnv
            If a PDDLEnv is provided, uses FastDownward to invoke the LM_RHW
            algorithm to extract landmarks from the environment.

            If a nx.DiGraph is provided, it must be a valid landmark graph.
        """
        if isinstance(dag_or_env, nx.DiGraph):
            for node in dag_or_env.nodes:
                assert isinstance(
                    dag_or_env.nodes[node].get("landmark", None),
                    (Literal, LiteralConjunction, LiteralDisjunction),
                ), "must be a valid landmark graph"
            self.dag = dag_or_env
        else:
            fd_planner = FD()
            det_env = dag_or_env.determinized()
            state, _ = det_env.reset()
            self.dag = fd_planner.get_landmarks(det_env.domain, state)

    def __hash__(self) -> int:
        """Hash this goal network"""
        return hash(frozenset(self.dag.nodes))  # checking nodes is sufficient

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GoalNetwork) and set(self.dag.nodes) == set(
            other.dag.nodes
        )

    def is_empty(self) -> bool:
        """Whether this goal network contains any nodes"""
        return len(self.dag.nodes) == 0

    def next_subgoals(self) -> List[str]:
        """Return a list of subgoals with no predecessors"""
        has_no_predecessors = []
        for node in self.dag.nodes:
            if self.dag.in_degree(node) == 0:
                has_no_predecessors.append(node)
        return has_no_predecessors

    def get_literal(self, label: str) -> Literal:
        """Return Literal for this subgoal label"""
        return self.dag._node[label]["landmark"]

    def remove_subgoal(self, label: str) -> None:
        """Remove this subgoal from the goal network"""
        new_dag = self.dag.copy()
        new_dag.remove_node(label)
        return GoalNetwork(new_dag)

    def goal_distance(self, node: str = "goal") -> int:
        """Return the number of ancestors of this subgoal"""
        return len(nx.ancestors(self.dag, node))

    def show_goal_network(self) -> None:
        """
        Render the landmark graph using matplotlib.
        """
        height = 0
        width = 0
        for layer, nodes in enumerate(nx.topological_generations(self.dag)):
            for node in nodes:
                self.dag.nodes[node]["layer"] = layer
            height += 1
            width = max(width, len(nodes))
        pos = nx.multipartite_layout(
            self.dag, scale=max(1, height, width), subset_key="layer", align="horizontal"
        )
        nx.draw(self.dag, pos=pos, with_labels=True, node_size=1000)
        x_pos, y_pos = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
        plt.xlim(min(x_pos) - 1, max(x_pos) + 1)
        plt.ylim(min(y_pos) - 1, max(y_pos) + 1)
        plt.annotate(
            "\n".join([f"{n}: {self.get_literal(n)}" for n in self.dag.nodes]),
            xy=(0, 0),
            xytext=(min(x_pos), min(y_pos) - 2),
            va="top",
        )
        plt.show()
