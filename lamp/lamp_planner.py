from __future__ import annotations
from typing import Self, Callable, Optional
from enum import Enum, auto
from dataclasses import dataclass, replace
import numpy as np
from pddlgym.core import PDDLEnv
from pddlgym.structs import Literal, LiteralConjunction, LiteralDisjunction, State
from pddlgym.inference import check_goal
from lamp.config import LAMPConfig
from lamp.action_space import ActionSpace
from lamp.goal_network import GoalNetwork
from lamp.tree import (
    StateNode,
    LandmarkNode,
    TreeNodeFactory,
    DefaultPolicy,
    UCBPolicy,
    MaxPolicy,
)
from tqdm import tqdm


class PlanningResult(Enum):
    """
    Enum representing the possible outcomes of LAMPPlanner.run():

    SUCCESS         -> Successfully reached the goal
    DEADLOCKED      -> Reached a deadlocking state
    EXCEEDED_BUDGET -> Unable to reach goal within cost budget
    """

    SUCCESS = auto()  # Successfully reached the goal
    FAILURE_DEADLOCKED = auto()  # Reached a deadlocking state
    FAILURE_BUDGET = auto()  # Unable to reach goal within cost budget

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"


@dataclass
class PlanningContext:
    env: PDDLEnv
    initial_state: frozenset[Literal]
    goal: Literal | LiteralConjunction | LiteralDisjunction
    action_space: ActionSpace
    node_factory: TreeNodeFactory
    goal_network: GoalNetwork
    n_rollouts: int
    horizon: int
    budget: float
    exploration_const: float
    normalize_exploration_const: bool
    goal_utility: float
    risk_factor: float
    greediness: float
    cost_fn: Callable[[frozenset[Literal], Literal], float]
    utility_fn: Callable[[float], float]
    h_util: Callable[[frozenset[Literal]], float]
    h_ptg: Callable[[frozenset[Literal]], float]
    q_init: Callable[[frozenset[Literal], Literal], float]
    n_init: float
    default_policy: DefaultPolicy
    ucb_policy: UCBPolicy
    greedy_ucb_policy: UCBPolicy
    max_policy: MaxPolicy
    greedy_max_policy: MaxPolicy
    rng: np.random.RandomState


@dataclass
class RolloutResult:
    """
    The result of a LAMP rollout containing the cost or reward for both the
    current subgoal and the final goal
    """

    final_goal: float
    has_goal: bool
    subgoal: float
    has_subgoal: bool

    def discount(self, gamma: float) -> Self:
        """
        Discount this the rewards by a discount factor `gamma`.
        Returns this `RolloutResult` for chaining.
        """
        self.subgoal *= gamma
        self.final_goal *= gamma
        return self

    def increment(self, cost: float | int) -> Self:
        """
        Increment this the rewards by `cost`.
        Returns this `RolloutResult` for chaining.
        """
        self.subgoal += cost
        self.final_goal += cost
        return self


class LAMPPlanner:
    """
    The Landmark-Aided Monte Carlo Planning (LAMP) algorithm
    """

    def __init__(self, cfg: Optional[LAMPConfig] = LAMPConfig()):
        """
        The Landmark-Aided Monte Carlo Planning (LAMP) algorithm

        Parameters
        ----------

        cfg : Optional[LAMPConfig]
            The configuration parameters for this LAMPPlanner. If none is provided,
            default values will be used.
        """
        self.cfg = cfg

    def _setup(
        self, env: PDDLEnv, goal_network: GoalNetwork | None, cfg: LAMPConfig
    ) -> PlanningContext:
        """
        Setup the PlanningContext for a run of LAMPPlanner
        """
        # assert (
        #     env.env._problem_index_fixed
        # ), "env problem index must be fixed using env.fix_problem_index(idx)"
        initial_state, _ = env.reset()
        rng = np.random.RandomState(seed=cfg.seed)
        ctx = PlanningContext(
            env=env,
            initial_state=initial_state,
            goal=initial_state.goal,
            action_space=ActionSpace(env, initial_state),
            node_factory=TreeNodeFactory(),
            goal_network=goal_network or GoalNetwork(env),
            n_rollouts=cfg.n_rollouts,
            horizon=cfg.horizon,
            budget=cfg.budget,
            exploration_const=cfg.exploration_const,
            normalize_exploration_const=cfg.normalize_exploration_const,
            goal_utility=cfg.goal_utility,
            risk_factor=cfg.risk_factor,
            greediness=cfg.greediness,
            cost_fn=lambda s, a: 1,  # lambda s, a: 0 if check_goal(s, goal) else 1
            utility_fn=lambda cost: np.exp(cfg.risk_factor * cost),
            h_util=cfg.h_util,
            h_ptg=cfg.h_ptg,
            q_init=lambda s, a: cfg.h_util(s) + cfg.h_ptg(s) * cfg.goal_utility,
            n_init=cfg.n_init,
            default_policy=DefaultPolicy(rng),
            ucb_policy=UCBPolicy(
                rng, cfg.exploration_const, cfg.normalize_exploration_const
            ),
            greedy_ucb_policy=UCBPolicy(
                rng,
                cfg.exploration_const,
                cfg.normalize_exploration_const,
                cfg.greediness,
            ),
            max_policy=MaxPolicy(rng),
            greedy_max_policy=MaxPolicy(rng, cfg.greediness),
            rng=rng,
        )
        return ctx

    def run(
        self,
        env: PDDLEnv,
        goal_network: Optional[GoalNetwork] = None,
        **override_config,
    ):
        """
        Iteratively select and apply the best action in each state according to the
        LAMP algorithm, until the problem's goal is achieved.

        Parameters
        ----------
        env : PDDLEnv
            The environment to run the planner in. Note: this environment must have a
            fixed problem index.

        goal_network : Optional[GoalNetwork]
            Optionally provide a goal network for LAMP to use. If none is provided,
            LAMP will use automatically generate a landmark graph using LM_RHW to use.

        **override_config : Any
            Configuration parameters (overrides any default parameters, or parameters
            set during the initialization of this LAMPPlanner)
        """
        cfg = replace(self.cfg, **override_config)
        ctx = self._setup(env, goal_network, cfg)

        state = ctx.initial_state
        state_node: StateNode = StateNode(state.literals)
        landmark_node = LandmarkNode(ctx.goal_network)
        subgoal_label = None
        if ctx.greediness == 0:
            subgoal_label = "goal"
            subgoal = ctx.goal
            landmark_node = LandmarkNode(
                GoalNetwork(ctx.goal_network.dag.subgraph("goal"))
            )
        budget = ctx.budget
        total_cumulative_cost = 0
        subgoal_cumulative_cost = 0
        while True:
            if state_node.is_leaf():
                state_node.initialize_children(
                    ctx.node_factory, ctx.action_space, ctx.q_init, ctx.n_init
                )
            if landmark_node.is_leaf():
                landmark_node.initialize_children(ctx.node_factory)
            if landmark_node.state.is_empty() or self._check_goal(state_node, ctx.goal):
                return PlanningResult.SUCCESS, total_cumulative_cost
            if total_cumulative_cost >= budget:
                return PlanningResult.FAILURE_BUDGET, total_cumulative_cost
            if len(state_node.get_applicable_actions()) == 0:
                return PlanningResult.FAILURE_DEADLOCKED, total_cumulative_cost
            action = self._plan(
                ctx,
                state_node,
                subgoal_label,
                landmark_node,
                total_cumulative_cost,
                subgoal_cumulative_cost,
                cfg.show_progress,
            )
            if isinstance(action, str):
                subgoal_label = action
                subgoal = ctx.goal_network.get_literal(subgoal_label)
                if cfg.show_progress:
                    print(f"\rSelected subgoal {subgoal}")
            else:
                action_cost = ctx.cost_fn(state_node.state, action)
                state_node = state_node.sample_next_node(action, ctx.rng)
                total_cumulative_cost += action_cost
                subgoal_cumulative_cost += action_cost
                if cfg.show_progress:
                    print(f"\r\tSelected action {action}")
            if self._check_goal(state_node, subgoal):
                landmark_node = landmark_node.sample_next_node(subgoal_label, ctx.rng)
                subgoal_label = None
                subgoal_cumulative_cost = 0

    def _plan(
        self,
        ctx: PlanningContext,
        state_node: StateNode,
        subgoal_label: str | None,
        landmark_node: LandmarkNode,
        total_cumulative_cost: float = 0,
        subgoal_cumulative_cost: float = 0,
        show_progress: bool = False,
    ) -> Literal:
        """
        Perform iterations of LAMP, then return the best
        action or landmark.
        """
        if self._check_goal(state_node, ctx.goal):
            return
        for _ in tqdm(
            range(ctx.n_rollouts),
            leave=False,
            position=0,
            disable=not show_progress,
            desc=f"{"\tSelecting action" if subgoal_label else "Selecting subgoal"}",
        ):
            self._simulate(
                ctx,
                state_node,
                subgoal_label,
                landmark_node,
                0,
                total_cumulative_cost,
                subgoal_cumulative_cost,
            )
        if subgoal_label is not None:
            subgoal = ctx.goal_network.get_literal(subgoal_label)
            return ctx.greedy_max_policy(state_node, subgoal)
        else:
            return ctx.max_policy(landmark_node)

    def _simulate(
        self,
        ctx: PlanningContext,
        state_node: StateNode,
        subgoal_label: str | None,
        landmark_node: LandmarkNode,
        depth: int,
        total_cumulative_cost: float,
        subgoal_cumulative_cost: float,
    ) -> RolloutResult:
        """
        Perform one rollout of LAMP and backpropagate costs
        """
        # Base Case: Goal network is empty or reached final goal
        if landmark_node.state.is_empty() or self._check_goal(state_node, ctx.goal):
            return RolloutResult(
                final_goal=0, has_goal=True, subgoal=0, has_subgoal=True
            )

        if landmark_node.is_leaf():
            landmark_node.initialize_children(ctx.node_factory)

        # Base Case: No subgoal has been selected. Select a new subgoal from the goal network.
        if subgoal_label is None:
            if not landmark_node.is_expanded():
                landmark_node.expand()
                subgoal_label = ctx.default_policy(landmark_node)
                result = self._rollout(
                    ctx,
                    state_node,
                    subgoal_label,
                    landmark_node,
                    depth,
                    total_cumulative_cost,
                    0,
                )
            else:
                subgoal_label = ctx.ucb_policy(landmark_node)
                result = self._simulate(
                    ctx,
                    state_node,
                    subgoal_label,
                    landmark_node,
                    depth,
                    total_cumulative_cost,
                    0,
                )
            landmark_node.update(
                subgoal_label,
                ctx.utility_fn(total_cumulative_cost + result.final_goal),
                result.has_goal,
                ctx.goal_utility,
            )
            return RolloutResult(
                final_goal=result.final_goal,
                has_goal=result.has_goal,
                subgoal=0,
                has_subgoal=True,
            )

        # Base Case: The current subgoal has been achieved. Remove it from the goal network.
        subgoal = ctx.goal_network.get_literal(subgoal_label)
        assert subgoal is not None
        if self._check_goal(state_node, subgoal):
            return self._simulate(
                ctx,
                state_node,
                None,
                landmark_node.sample_next_node(subgoal_label, ctx.rng),
                depth,
                total_cumulative_cost,
                0,
            )

        if state_node.is_leaf():
            state_node.initialize_children(
                ctx.node_factory, ctx.action_space, ctx.q_init, ctx.n_init
            )

        # Base Case: Reached deadend
        if state_node.is_deadend():
            future_cost = self._get_remaining_cost_at_deadend(depth, ctx.horizon)
            return RolloutResult(
                final_goal=future_cost,
                has_goal=False,
                subgoal=future_cost,
                has_subgoal=False,
            )

        # Base Case: Reached maximum rollout depth
        if depth == ctx.horizon - 1:
            return RolloutResult(
                final_goal=0, has_goal=False, subgoal=0, has_subgoal=False
            )

        # The current subgoal has not yet been achieved. Select an action to execute.
        if not state_node.is_expanded():
            state_node.expand()
            action = ctx.default_policy(state_node)
            action_cost = ctx.cost_fn(state_node.state, action)
            next_state = state_node.sample_next_node(action, ctx.rng)
            result = self._rollout(
                ctx,
                next_state,
                subgoal_label,
                landmark_node,
                depth + 1,
                total_cumulative_cost + action_cost,
                subgoal_cumulative_cost + action_cost,
            )
        else:
            action = ctx.greedy_ucb_policy(state_node, subgoal)
            action_cost = ctx.cost_fn(state_node.state, action)
            next_state = state_node.sample_next_node(action, ctx.rng)
            result = self._simulate(
                ctx,
                next_state,
                subgoal_label,
                landmark_node,
                depth + 1,
                total_cumulative_cost + action_cost,
                subgoal_cumulative_cost + action_cost,
            )
        state_node.update(
            action,
            subgoal,
            ctx.utility_fn(total_cumulative_cost + result.final_goal + action_cost),
            ctx.utility_fn(subgoal_cumulative_cost + result.subgoal + action_cost),
            result.has_goal,
            result.has_subgoal,
            ctx.goal_utility,
        )
        return result.increment(action_cost)

    def _rollout(
        self,
        ctx: PlanningContext,
        state_node: StateNode,
        subgoal_label: Optional[str],
        landmark_node: LandmarkNode,
        depth: int,
        total_cumulative_cost: float,
        subgoal_cumulative_cost: float,
    ):
        """
        Perform one rollout of LAMP without backpropagating costs
        """
        # Base Case: Goal Network is empty (i.e. all goals have been achieved)
        if landmark_node.state.is_empty() or self._check_goal(state_node, ctx.goal):
            return RolloutResult(
                final_goal=0, has_goal=True, subgoal=0, has_subgoal=True
            )

        # Base Case: No subgoal has been selected. Select a new subgoal from the goal network.
        if landmark_node.is_leaf():
            landmark_node.initialize_children(ctx.node_factory)
        if subgoal_label is None:
            subgoal_label = ctx.default_policy(landmark_node)
            result = self._rollout(
                ctx,
                state_node,
                subgoal_label,
                landmark_node,
                depth,
                total_cumulative_cost,
                0,
            )
            return RolloutResult(
                final_goal=result.final_goal,
                has_goal=False,
                subgoal=0,
                has_subgoal=False,
            )

        # Base Case: The current subgoal has been achieved. Remove it from the goal network
        subgoal = ctx.goal_network.get_literal(subgoal_label)
        assert subgoal is not None
        if self._check_goal(state_node, subgoal):
            return self._rollout(
                ctx,
                state_node,
                None,
                landmark_node.sample_next_node(subgoal_label, ctx.rng),
                depth,
                total_cumulative_cost,
                0,
            )

        if state_node.is_leaf():
            state_node.initialize_children(
                ctx.node_factory, ctx.action_space, ctx.q_init, ctx.n_init
            )

        # Base Case: Reached deadend
        if len(state_node.get_applicable_actions()) == 0:
            return RolloutResult(
                final_goal=depth, has_goal=False, subgoal=depth, has_subgoal=False
            )
        # Base Case: Reached maximum rollout depth
        if depth == ctx.horizon - 1:
            return RolloutResult(
                final_goal=0, has_goal=False, subgoal=0, has_subgoal=False
            )

        # The current subgoal has not yet been achieved. Select an action to execute.
        action = ctx.default_policy(state_node)
        action_cost = ctx.cost_fn(state_node.state, action)
        next_state = state_node.sample_next_node(action, ctx.rng)
        result = self._rollout(
            ctx,
            next_state,
            subgoal_label,
            landmark_node,
            depth + 1,
            total_cumulative_cost + 1,
            subgoal_cumulative_cost + 1,
        )
        return result.increment(action_cost)

    def _get_remaining_cost_at_deadend(self, depth, horizon):
        # cost = cost_fn(s, next(iter(actions)))
        return horizon - 1 - depth

    def _check_goal(
        self,
        state_node: StateNode,
        goal: Literal | LiteralConjunction | LiteralDisjunction,
    ) -> bool:
        empty_set = frozenset()
        return check_goal(State(state_node.state, empty_set, empty_set), goal)
