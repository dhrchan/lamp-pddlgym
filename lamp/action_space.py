from typing import NamedTuple
from pddlgym.core import get_successor_states, InvalidAction
from pddlgym.structs import Literal, State
from pddlgym.spaces import LiteralActionSpace, LiteralSpace


class Outcome(NamedTuple):
    prob: float
    literals: frozenset[Literal]


class ActionSpace:
    """
    The collection of actions for a PDDLEnv.
    """

    def __init__(self, env, initial_state):
        self.env = env
        self.initial_state = initial_state
        if isinstance(env.action_space, LiteralActionSpace):#self.env.env._dynamic_action_space:
            self.all_actions = None
        else:
            self.all_actions = frozenset(
                env.action_space.all_ground_literals(
                    self.initial_state, valid_only=False
                )
            )
            

    def get_valid_actions_and_outcomes(
        self, s: frozenset[Literal]
    ) -> dict[Literal, frozenset[Outcome]]:
        """
        Find all applicable actions and their respective outcomes at state `s`
        """
        empty_set = frozenset()
        state = State(s, empty_set, empty_set)
        actions = self.all_actions or self.env.action_space.all_ground_literals(
            State(s, self.initial_state.objects, self.initial_state.goal)
        )
        valid_actions_successors = {}
        for a in actions:
            successors = None
            try:
                successors = frozenset(
                    {
                        Outcome(prob=prob, literals=s_.literals)
                        for s_, prob in get_successor_states(
                            state,
                            a,
                            self.env.domain,
                            return_probs=True,
                            raise_error_on_invalid_action=True,
                        ).items()
                    }
                )
            except InvalidAction:
                continue

            assert successors is not None
            valid_actions_successors[a] = successors

        return valid_actions_successors

    def filter_superfluous_actions(
        self, s: frozenset[Literal], actions_outcomes: dict[Literal, frozenset[Outcome]]
    ) -> dict[Literal, frozenset[Outcome]]:
        """
        Prune superfluous actions. (PROST: Keller and Eyerich, ICAPS 2012)
        """

        # find actions with equal outcomes
        reversed_outcomes: dict[frozenset[Outcome], Literal] = {}
        superfluous_actions = set()
        for a, outcomes in actions_outcomes.items():
            if outcomes in reversed_outcomes:
                # if outcome is equal to one of other action, then current action can be pruned
                superfluous_actions.add(a)
            reversed_outcomes[outcomes] = a

        # actions whose single outcome is the current state can also be ignored
        cur_state_outcome_set = frozenset({Outcome(prob=1, literals=s)})
        superfluous_actions.update(
            set(
                {
                    a
                    for a in actions_outcomes
                    if actions_outcomes[a] == cur_state_outcome_set
                }
            )
        )

        # reasonable actions are actions that are not superfluous
        reasonable_actions = set(actions_outcomes) - superfluous_actions

        reasonable_actions_outcomes = {
            a: actions_outcomes[a] for a in reasonable_actions
        }

        return reasonable_actions_outcomes
