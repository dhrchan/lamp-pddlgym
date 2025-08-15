from dataclasses import dataclass
from typing import Callable, Optional
from pddlgym.structs import Literal

@dataclass
class LAMPConfig:
    """
    Configuration for a LAMPPlanner. 

    Parameters
    ----------
    n_rollouts : int
        number of rollouts to perform at each step (default = 100)
    horizon : int
        maximum depth of each rollout (default = 10)
    budget : float
        the maximum cost budget (maximum number of steps) for a single run (default = 100)
    exploration_const : float
        exploration-exploitation tradeoff (c) value for UCB (default = sqrt(2))
    normalize_exploration_const : bool
        whether to normalize c in UCB (default = True)
    n_init : int
        initial visit count (delta) (default = 5)
    greediness : float
        greediness value (alpha) for LAMP (default = 0.5)
    risk_factor : float
        risk factor (lambda) for GUBS criterion (default = -0.1)
    goal_utility : float
        goal utility constant for GUBS criterion (default = 1)
    h_util : Callable[[frozenset[Literal]], float]
        utility heuristic (default = lambda _: 1)
    h_ptg : Callable[[frozenset[Literal]], float]
        probability-to-goal heuristic (default = lambda _: 1)
    seed : Optional[int]
        random seed (default = None)
    show_progress : bool
        whether to print planning progress to stdout (default = False)
    """
    n_rollouts: int = 100  # number of rollouts to perform
    horizon: int = 20  # maximum depth of each rollout
    budget: float = 100  # the maximum cost budget for a single run
    exploration_const: float = 2 ** 0.5  # exploration-exploitation tradeoff (c) value for UCB
    normalize_exploration_const: bool = True  # whether to normalize c in UCB
    n_init: int = 5  # initial visit count (delta)
    greediness: float = 0.5  # greediness value (alpha) for LAMP
    risk_factor: float = -0.1  # risk factor (lambda) for GUBS criterion
    goal_utility: float = 1  # goal utility constant for GUBS criterion
    h_util: Callable[[frozenset[Literal]], float] = lambda _: 1  # utility heuristic
    h_ptg: Callable[[frozenset[Literal]], float] = lambda _: 1  # probability-to-goal heuristic
    seed: Optional[int] = None  # random seed
    show_progress: bool = False  # whether to print planning progress to stdout