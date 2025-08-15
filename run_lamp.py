import argparse
import pickle
import os
import numpy as np
import pddlgym
from lamp import LAMPPlanner, LAMPConfig, GoalNetwork

def main():
    parser = argparse.ArgumentParser(description="Run LAMP planner on a PDDLgym environment.")

    # Environment arguments
    parser.add_argument("env_name", type=str, help="PDDLGym environment name (e.g., PDDLEnvProb_blocks-v0)")
    parser.add_argument("--problem_index",type=int, default=0, help="Problem index within the environment")
    parser.add_argument("--goal_network_pkl", type=str, default=None, help="Goal network")

    # LAMP config arguments
    parser.add_argument("--n_rollouts", type=int, default=20, help="Number of rollouts")
    parser.add_argument("--horizon", type=int, default=20, help="Planning horizon")
    parser.add_argument("--budget", type=int, default=200, help="Planning budget")
    parser.add_argument("--exploration_const", type=float, default=2**0.5, help="Exploration constant")
    parser.add_argument("--normalize_exploration_const", action="store_true", default=True, help="Normalize exploration constant")
    parser.add_argument("--n_init", type=int, default=0, help="Initial value of N")
    parser.add_argument("--greediness", type=float, default=0.5, help="Greediness factor")
    parser.add_argument("--risk_factor", type=float, default=-0.1, help="Risk factor")
    parser.add_argument("--goal_utility", type=int, default=1, help="Goal utility")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--show_progress", action="store_true", default=False, help="Show progress bar")

    args = parser.parse_args()

    # Create PDDLgym environment
    env = pddlgym.make(args.env_name)
    env.fix_problem_index(args.problem_index)

    if args.goal_network_pkl is not None:
        if os.path.exists(args.goal_network_pkl):
            with open(args.goal_network_pkl, 'rb') as pkl:
                gn = pickle.load(pkl)
        else:
            gn = GoalNetwork(env)
            with open(args.goal_network_pkl, 'wb') as pkl:
                pickle.dump(gn, pkl, pickle.HIGHEST_PROTOCOL)
    else:
        gn = None

    args.seed = args.seed or np.random.randint(2 ** 32)

    # Create LAMP config
    cfg = LAMPConfig(
        n_rollouts=args.n_rollouts,
        horizon=args.horizon,
        budget=args.budget,
        exploration_const=args.exploration_const,
        normalize_exploration_const=args.normalize_exploration_const,
        n_init=args.n_init,
        greediness=args.greediness,
        risk_factor=args.risk_factor,
        goal_utility=args.goal_utility,
        h_util=lambda _: 1,  # Default value
        h_ptg=lambda _: 1,  # Default value
        seed=args.seed,
        show_progress=args.show_progress,
    )

    # Create and run LAMP planner
    planner = LAMPPlanner(cfg)
    result, cost = planner.run(env, gn)
    print(
        args.greediness,
        args.n_rollouts,
        args.seed,
        result.name,
        cost,
        args.horizon,
        args.budget,
        args.exploration_const,
        args.normalize_exploration_const,
        args.n_init,
        args.risk_factor,
        args.goal_utility,
        sep=","
    )


if __name__ == "__main__":
    main()
