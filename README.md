# Landmark-Assisted Monte Carlo Planning

An implementation of the Landmark-Assisted Monte Carlo Planning algorithm, using [PDDLGym](https://github.com/tomsilver/pddlgym). 

## Installation
1. Clone this repository
2. `pip install . -e`

## Example Usage

```python
import pddlgym
from lamp import LAMPPlanner

env = pddlgym.make("PDDLEnvBlocks-v0")
env.fix_problem_index(0)

planner = LAMPPlanner()

planner.run(env, show_progress=True)
```
