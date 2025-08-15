"""Fast-downward planner.
http://www.fast-downward.org/ObtainingAndRunningFastDownward
"""

import re
import os
import sys
import subprocess
import tempfile
import io
import networkx as nx
from pddlgym.inference import check_goal
from pddlgym.structs import LiteralDisjunction, Not
from pddlgym.spaces import LiteralSpace
from pddlgym.parser import PDDLProblemParser
from pddlgym_planners.pddl_planner import PDDLPlanner
from pddlgym_planners.planner import PlanningFailure

FD_URL = "https://github.com/aibasel/downward.git"


class FD(PDDLPlanner):
    """Fast-downward planner.
    """
    def __init__(self, alias_flag="--alias seq-opt-lmcut", final_flags=""):
        super().__init__()
        dirname = os.path.dirname(os.path.realpath(__file__))
        self._exec = os.path.join(dirname, "FD/fast-downward.py")
        # print("Instantiating FD", end=' ')
        # if alias_flag:
        #     print("with", alias_flag, end=' ')
        # if final_flags:
        #     print("with", final_flags, end=' ')
        # print()
        self._alias_flag = alias_flag
        self._final_flags = final_flags
        if not os.path.exists(self._exec):
            self._install_fd()
    
    def get_landmarks(self, domain, state):
        act_predicates = [domain.predicates[a] for a in list(domain.actions)]
        act_space = LiteralSpace(
            act_predicates, type_to_parent_types=domain.type_to_parent_types)
        dom_file = tempfile.NamedTemporaryFile(delete=False).name
        prob_file = tempfile.NamedTemporaryFile(delete=False).name
        domain.write(dom_file)
        lits = set(state.literals)
        if not domain.operators_as_actions:
            lits |= set(act_space.all_ground_literals(state, valid_only=False))
        PDDLProblemParser.create_pddl_file(
            prob_file, state.objects, lits, "myproblem",
            domain.domain_name, state.goal, fast_downward_order=True)
        gn = self.landmarks_from_pddl(dom_file, prob_file)
        for n in gn._node:
            gn._node[n]["landmark"] = self._atom_to_literal(domain, gn._node[n]["label"])
        gn.add_node("goal", landmark=state.goal)
        for node in list(gn.nodes):
            if gn.out_degree(node) == 0 and node != "goal":
                gn.add_edge(node, "goal")
        for node in list(gn.nodes):
            if gn.in_degree(node) == 0 and check_goal(state, gn._node[node]["landmark"]):
                gn.remove_node(node)
        return gn

    def landmarks_from_pddl(self, dom_file, prob_file):
        """PDDL-specific landmark generation method.
        """
        sas_file = tempfile.NamedTemporaryFile(delete=False).name
        config = '--search "let(hlm,landmark_sum(lm_rhw(verbosity=debug),pref=true),'
        config += 'let(hff,ff(),lazy_greedy([hff,hlm],preferred=[hff,hlm])))"'
        cmd_str = "{} --log-level debug --sas-file {} {} {} {} {}".format(
            self._exec, sas_file,
            dom_file, prob_file, config, self._final_flags)
        output = subprocess.getoutput(cmd_str)
        os.remove(dom_file)
        os.remove(prob_file)
        self._cleanup()
        return self._output_to_landmark_graph(output)

    def _get_cmd_str(self, dom_file, prob_file, timeout):
        sas_file = tempfile.NamedTemporaryFile(delete=False).name
        timeout_cmd = "gtimeout" if sys.platform == "darwin" else "timeout"
        cmd_str = "{} {} {} {} --sas-file {} {} {} {}".format(
            timeout_cmd, timeout, self._exec, self._alias_flag, sas_file,
            dom_file, prob_file, self._final_flags)
        return cmd_str

    def _get_cmd_str_searchonly(self, sas_file, timeout):
        timeout_cmd = "gtimeout" if sys.platform == "darwin" else "timeout"
        cmd_str = "{} {} {} {} --search {} {}".format(
            timeout_cmd, timeout, self._exec, self._alias_flag,
            sas_file, self._final_flags)
        return cmd_str

    def _cleanup(self):
        """Run FD cleanup
        """
        cmd_str = "{} --cleanup".format(self._exec)
        subprocess.getoutput(cmd_str)

    def _output_to_plan(self, output):
        # Technically this is number of evaluated states which is always
        # 1+number of expanded states, but we report evaluated for consistency
        # with FF.
        num_node_expansions = re.findall(r"evaluated (\d+) state", output.lower())
        plan_length = re.findall(r"plan length: (\d+) step", output.lower())
        plan_cost = re.findall(r"] plan cost: (\d+)", output.lower())
        search_time = re.findall(r"] search time: (\d+\.\d+)", output.lower())
        total_time = re.findall(r"] total time: (\d+\.\d+)", output.lower())
        if "num_node_expansions" not in self._statistics:
            self._statistics["num_node_expansions"] = 0
        if len(num_node_expansions) == 1:
            assert int(num_node_expansions[0]) == float(num_node_expansions[0])
            self._statistics["num_node_expansions"] += int(
                num_node_expansions[0])
        if len(search_time) == 1:
            try:
                search_time_float = float(search_time[0])
                self._statistics["search_time"] = search_time_float
            except:
                raise PlanningFailure("Error on output's search time format: {}".format(search_time[0]))
        if len(search_time) == 1:
            try:
                total_time_float = float(total_time[0])
                self._statistics["total_time"] = total_time_float
            except:
                raise PlanningFailure("Error on output's total time format: {}".format(total_time[0]))
        if len(plan_length) == 1:
            try:
                plan_length_int = int(plan_length[0])
                self._statistics["plan_length"] = plan_length_int
            except:
                raise PlanningFailure("Error on output's plan length format: {}".format(plan_length[0]))
        if len(plan_cost) == 1:
            try:
                plan_cost_int = int(plan_cost[0])
                self._statistics["plan_cost"] = plan_cost_int
            except:
                raise PlanningFailure("Error on output's plan cost format: {}".format(plan_cost[0]))
        if "Solution found!" not in output:
            raise PlanningFailure("Plan not found with FD! Error: {}".format(
                output))
        if "Plan length: 0 step" in output:
            return []
        
        fd_plan = re.findall(r"(.+) \(\d+?\)", output.lower())
        if not fd_plan:
            raise PlanningFailure("Plan not found with FD! Error: {}".format(
                output))
        return fd_plan
    
    def _output_to_landmark_graph(self, output):
        start, end = "Dumping landmark graph:", "}"
        txt = str(output).split(start)
        if len(txt) <= 1:
            return nx.DiGraph()
        txt = txt[1]
        ind = txt.index(end)
        text_io = io.StringIO(txt[:ind + 1])
        di = nx.nx_pydot.read_dot(text_io)
        return nx.DiGraph(di)
    
    def _atom_to_literal(self, domain, label) -> LiteralDisjunction:
        """Convert Fast Downward's atom notation to a pddlgym Literal"""
        # TODO: Add support for conjunctive landmarks
        disj = label.strip('"').split(" | ")
        literals = []
        for g in disj:
            pred = g[g.find(" ") + 1 : g.find("(")].strip("_")
            params = []
            for name in g[g.find("(") + 1 : g.find(")")].split(", "):
                if name:
                    params.append(name)
            if g.startswith("Atom"):
                literals.append(domain.predicates[pred](*params))
            elif g.startswith("NegatedAtom"):
                literals.append(Not(domain.predicates[pred](*params)))
        if len(literals) > 1:
            return LiteralDisjunction(literals)
        else:
            return literals[0]

    def _install_fd(self):
        loc = os.path.dirname(self._exec)
        # Install and compile FD.
        os.system("git clone {} {}".format(FD_URL, loc))
        os.system("cd {} && ./build.py && cd -".format(loc))
        assert os.path.exists(self._exec)
