from __future__ import annotations
import numpy as np
import logging
from math import sqrt, log
from pddlgym.structs import Literal
from lamp.goal_network import GoalNetwork
from lamp.action_space import ActionSpace
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Generic, Hashable, TypeVar, Callable, override

A = TypeVar("A", bound=Hashable)  # Actions
S = TypeVar("S", bound=Hashable)  # States


class TreeNodeFactory:
    """
    A factory for TreeNodes, ensuring that there is only one node created for each
    underlying state.
    """

    def __init__(self):
        self._state_nodes = {}
        self._landmark_nodes = {}

    def new_node(self, state: S) -> StateNode | LandmarkNode:
        """
        Create a new StateNode or LandmarkNode only if the underlying state has not
        yet been encountered. If it has, return the existing instance.
        """
        if isinstance(state, GoalNetwork):
            if state not in self._landmark_nodes:
                self._landmark_nodes[state] = LandmarkNode(state)
            return self._landmark_nodes[state]
        elif isinstance(state, frozenset):
            if state not in self._state_nodes:
                self._state_nodes[state] = StateNode(state)
            return self._state_nodes[state]
        raise Exception("Invalid state")


class TreeNode(ABC, Generic[S, A]):
    """
    A decision node in a Monte Carlo search tree
    """

    def __init__(self, state: S) -> None:
        self.state: S = state
        self._appliable_actions: frozenset[A] | None = None
        self.visits: int = 0
        self.Q: Dict[A, float] = {}
        self.N: Dict[A, int] = {}
        self._expanded: bool = True

    def is_expanded(self) -> bool:
        """
        Whether this decision node has been expanded. Only expanded nodes will have
        statisics updated after a rollout.
        """
        return self._expanded

    def is_deadend(self) -> bool:
        """
        Whether there are any applicable actions at this node
        """
        return len(self.get_applicable_actions()) == 0

    def is_leaf(self) -> bool:
        """
        Whether this node's children have been initialized yet
        """
        return not hasattr(self, "children")

    def expand(self) -> None:
        """
        Expand this decision node, allowing this node to update UCT statistics
        """
        self._expanded = True

    def get_applicable_actions(self) -> List[A]:
        """
        Generate and return the set of applicable actions at this decision node,
        while caching the result for future queries.
        """
        assert hasattr(self, "children")
        return list(self.children.keys())

    def select_action(
        self, policy: TreePolicy, rng: np.random.RandomState = np.random.RandomState()
    ) -> A:
        """
        Select an applicable action at this decision node according to the `policy`.
        If `max` is `True`, an action which maximizes `policy` is selected.
        Otherwise, an action that minimizes `policy` is selected.
        """
        return policy(self, rng)

    def update(
        self, action: A, reward: float, has_goal: bool, goal_utility: float
    ) -> None:
        """
        Update `Q` and `N` values for `action` at this decision node given the
        observed `reward`.
        """
        assert self._expanded, "Cannot update unexpanded node"
        k = goal_utility if has_goal else 0
        self.Q[action] = (self.N[action] * self.Q[action] + reward + k) / (
            1 + self.N[action]
        )
        self.N[action] += 1
        self.visits += 1

    @abstractmethod
    def initialize_children(**kwargs):
        """Initialize self.children"""
        raise NotImplementedError()

    def sample_next_node(
        self, a: A, rng: np.random.RandomState = np.random.RandomState()
    ):
        """Sample the outcomes of executing action a from this DecisionNode"""
        assert hasattr(self, "children") and a in self.children
        probs = []
        outcomes = []
        for outcome, prob in self.children[a]:
            probs.append(prob)
            outcomes.append(outcome)
        return rng.choice(outcomes, p=probs)


class StateNode(TreeNode[frozenset[Literal], Literal]):
    """
    A node in a Monte Carlo search tree representing a state of the environment
    """

    def __init__(self, state: frozenset[Literal]) -> None:
        super().__init__(state)
        self.Q_LM: Dict[str, Dict[Literal, float]] = defaultdict(
            lambda: defaultdict(float)  ## WHAT SHOULD THE DEFAULT VALUE OF Q_LM BE?
        )
        self.N_LM: Dict[str, Dict[Literal, int]] = defaultdict(lambda: defaultdict(int))
        self.visits_LM: Dict[str, int] = defaultdict(int)

    @override
    def update(
        self,
        action: Literal,
        subgoal: str,
        total_reward: float,
        subgoal_reward: float,
        has_goal: bool,
        has_subgoal: bool,
        goal_utility: float,
    ) -> None:
        """
        Perform a UCB update on this node
        """
        super().update(action, total_reward, has_goal, goal_utility)
        k = goal_utility if has_subgoal else 0
        self.Q_LM[subgoal][action] = (
            self.N_LM[subgoal][action] * self.Q_LM[subgoal][action] + subgoal_reward + k
        ) / (1 + self.N_LM[subgoal][action])
        self.N_LM[subgoal][action] += 1
        self.visits_LM[subgoal] += 1

    @override
    def initialize_children(
        self,
        node_factory: TreeNodeFactory,
        action_space: ActionSpace,
        q_init: Callable[[frozenset[Literal], Literal], float] = lambda s, a: 0,
        n_init: int = 0,
    ) -> None:
        """
        Find this node's children, and initialize Q values and N values.
        """
        initial_valid_actions_outcomes = action_space.get_valid_actions_and_outcomes(
            self.state
        )

        # remove redundant/superfluous actions
        reasonable_actions_outcomes = action_space.filter_superfluous_actions(
            self.state, initial_valid_actions_outcomes
        )

        if (
            redundant_actions := set(reasonable_actions_outcomes)
            - set(initial_valid_actions_outcomes)
        ) != set({}):
            logging.debug(
                f"detected {len(redundant_actions)} actions:" + f" {redundant_actions}"
            )

        self.children = {}
        for a, outcomes in reasonable_actions_outcomes.items():
            # initialize q-value with heuristic for current state
            self.Q[a] = q_init(self.state, a)  # h_u(state) + h_p(state) * k_g
            self.N[a] = n_init  # c.f. PROST numberOfInitialVisits
            self.visits += n_init

            # initialize each child's subtree
            self.children[a] = []
            total_probs = 0
            for outcome in outcomes:
                self.children[a].append(
                    (node_factory.new_node(outcome.literals), outcome.prob)
                )
                total_probs += outcome.prob
            if total_probs != 1:
                self.children[a].append(
                    (node_factory.new_node(self.state), 1 - total_probs)
                )

    def expand(self) -> None:
        """
        Expand this decision node
        """
        self._expanded = True


class LandmarkNode(TreeNode[GoalNetwork, str]):
    """
    A node in a Monte Carlo search tree representing a state of the goal network
    """

    def __init__(self, goal_network: GoalNetwork) -> None:
        super().__init__(goal_network)

    @override
    def initialize_children(self, node_factory: TreeNodeFactory) -> None:
        """Generate the list of landmarks in the goal network with no predecessors"""
        self.children = {}
        for a in self.state.next_subgoals():
            self.Q[a] = 0
            self.N[a] = 0
            self.children[a] = [
                (node_factory.new_node(self.state.remove_subgoal(a)), 1)
            ]


class TreePolicy(ABC, Generic[A]):
    """
    A policy used to select among applicable actions at a TreeNode
    """

    @abstractmethod
    def __call__(
        self, node: TreeNode, rng: np.random.RandomState = np.random.RandomState()
    ) -> A:
        raise NotImplementedError()


class UCBPolicy(TreePolicy):
    """
    A TreePolicy based using the UCB1 formula
    """

    def __init__(
        self,
        rng: np.random.RandomState = np.random.RandomState(),
        c: float = sqrt(2),
        normalize: bool = True,
        alpha: float = 0,
    ):
        self.normalize = normalize
        self.alpha = alpha
        self.c = c
        self.rng = rng

    def __call__(self, node: TreeNode, subgoal=None) -> A:
        if self.alpha > 0:
            assert subgoal and isinstance(node, StateNode), "greedy policy used"
        actions = np.array(list(sorted(node.get_applicable_actions())))
        c = c_lm = self.c
        if self.normalize:
            c *= max(node.Q.values())
            if self.alpha > 0 and subgoal in node.Q_LM:
                c_lm *= max(node.Q_LM[subgoal].values())
        if self.alpha < 1:
            goal_vals = np.array(
                [self._ucb_value(node.Q[a], node.N[a], node.visits, c) for a in actions]
            )
        else:
            goal_vals = np.zeros(len(actions))
        if self.alpha > 0:
            subgoal_vals = np.array(
                [
                    self._ucb_value(
                        node.Q_LM[subgoal][a],
                        node.N_LM[subgoal][a],
                        node.visits,
                        c_lm,
                    )
                    for a in actions
                ]
            )
        else:
            subgoal_vals = np.zeros(len(actions))
        vals = (1 - self.alpha) * goal_vals + self.alpha * subgoal_vals
        max_val = np.max(vals)
        max_actions = actions[vals == max_val]
        return self.rng.choice(max_actions)

    def _ucb_value(self, q_a: float, n_a: int, n: int, c: float) -> float:
        if n_a == 0:
            return np.inf
        return q_a + c * sqrt(log(n) / n_a)


class MaxPolicy(TreePolicy):
    """
    A purely exploitative TreePolicy which selects the action with the maximum Q value
    """

    def __init__(
        self, rng: np.random.RandomState = np.random.RandomState(), alpha: float = 0
    ):
        self.alpha = alpha
        self.rng = rng

    def __call__(self, node: TreeNode, subgoal=None) -> A:
        if self.alpha > 0:
            assert subgoal and isinstance(node, StateNode), "greedy policy used"
        actions = np.array(list(sorted(node.get_applicable_actions())))
        if self.alpha < 1:
            goal_vals = np.array([node.Q[a] for a in actions])
        else:
            goal_vals = np.zeros(len(actions))
        if self.alpha > 0:
            subgoal_vals = np.array([node.Q_LM[subgoal][a] for a in actions])
        else:
            subgoal_vals = np.zeros(len(actions))
        vals = (1 - self.alpha) * goal_vals + self.alpha * subgoal_vals
        max_val = np.max(vals)
        max_actions = actions[vals == max_val]
        return self.rng.choice(max_actions)


class RobustPolicy(TreePolicy):
    """
    A TreePolicy which selects the action with the maximum N value
    """

    def __init__(
        self, rng: np.random.RandomState = np.random.RandomState(), alpha: float = 0
    ) -> None:
        self.alpha = alpha
        self.rng = rng

    def __call__(self, node: TreeNode, subgoal=None) -> A:
        if self.alpha > 0:
            assert subgoal and isinstance(node, StateNode), "greedy policy used"
        actions = np.array(list(sorted(node.get_applicable_actions())))
        if self.alpha < 1:
            goal_vals = np.array([node.N[a] for a in actions])
        else:
            goal_vals = np.zeros(len(actions))
        if self.alpha > 0:
            subgoal_vals = np.array([node.N_LM[subgoal][a] for a in actions])
        else:
            subgoal_vals = np.zeros(len(actions))
        vals = (1 - self.alpha) * goal_vals + self.alpha * subgoal_vals
        max_val = np.max(vals)
        max_actions = actions[vals == max_val]
        return self.rng.choice(max_actions)


class DefaultPolicy(TreePolicy):
    """
    A TreePolicy which selects the action at random
    """

    def __init__(self, rng: np.random.RandomState = np.random.RandomState()) -> None:
        self.rng = rng

    def __call__(self, node: TreeNode) -> A:
        actions = np.array(list(sorted(node.get_applicable_actions())))
        return self.rng.choice(actions)
