# -*- coding: utf-8 -*-
# @Time    : 2025/9/26 15:50
# @Author  : Dong Z.
# @File    : treeNode.py

from imports import *
from DQN import DQN
from clusterTool import ClusterTool


class TreeNode:
    """
    Search tree node with lower-bound information for rollout evaluation.
    """

    def __init__(self,
                 cluster_tool: ClusterTool,
                 parent: Optional["TreeNode"] = None,
                 action_from_parent: int = -1):
        self.cluster_tool: ClusterTool = cluster_tool

        self.parent: Optional["TreeNode"] = parent
        self.children: Dict[int, "TreeNode"] = {}
        self.action_from_parent: int = action_from_parent

        self.num_visited: int = 0
        self.estimated_obj: float = float("inf")
        self.immediate_obj_from_parent: float = 0.0

        self.lower_bound: float = float("inf")

        self._unexpanded_actions: Optional[List[int]] = None
        self.is_terminal: bool = self.cluster_tool.is_done

        self.ucb_value: float = 0.0
        self.exploit: float = 0.0
        self.explore: float = 0.0
        self.lb_term: float = 0.0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def set_as_root(self):
        """Set the current node as the new root node."""
        self.parent = None

    def is_leaf(self) -> bool:
        return (not self.is_terminal) and (len(self.children) == 0)

    def fully_expanded(self) -> bool:
        if self.is_terminal:
            return True
        if self._unexpanded_actions is None:
            return False
        return len(self._unexpanded_actions) == 0

    def compute_lower_bound(self, pre_state_lb) -> float:
        """
        Compute and cache the lower bound of the current node.
        """
        if self.lower_bound == float('inf'):
            raw_lb = self.cluster_tool.compute_lower_bound(pre_state_lb=pre_state_lb)
            self.lower_bound = max(pre_state_lb, raw_lb)
        return self.lower_bound

    def _init_unexpanded_actions(self, agent: DQN):
        """Initialize unexpanded actions and sort them by descending Q-values."""
        if self._unexpanded_actions is not None:
            return

        action_mask = self.cluster_tool.get_action_mask()
        feasible_actions = np.nonzero(action_mask)[0].tolist()
        if not feasible_actions:
            self._unexpanded_actions = []
            return

        chamber_state, robot_state = self.cluster_tool.state
        chamber_state = np.copy(chamber_state)
        robot_state = np.copy(robot_state)

        with torch.no_grad():
            device = next(agent.q_net.parameters()).device
            cs = torch.as_tensor(chamber_state[None, ...], dtype=torch.float32, device=device)
            rs = torch.as_tensor(robot_state[None, ...], dtype=torch.long, device=device)
            a_mask = torch.as_tensor(action_mask[None, ...], dtype=torch.bool, device=device)

            q_values, _, _ = agent.q_net(cs, rs, a_mask)
            q_values = q_values[0].detach().cpu().numpy()

        q_values[~action_mask] = -1e5
        feasible_actions.sort(key=lambda a: -q_values[a])
        self._unexpanded_actions = feasible_actions

    def expand_one_child(self, agent: DQN) -> Optional["TreeNode"]:
        """Expand one new child node and compute its lower bound."""
        if self.is_terminal:
            return None

        self._init_unexpanded_actions(agent)
        if not self._unexpanded_actions:
            return None

        action_id = self._unexpanded_actions.pop(0)

        new_tool: ClusterTool = copy.deepcopy(self.cluster_tool)
        action_op = new_tool.map_action(action_id)
        _, reward, _, _, done = new_tool.step(action_op)

        child = TreeNode(cluster_tool=new_tool,
                         parent=self,
                         action_from_parent=action_id)

        child.is_terminal = done
        child.immediate_obj_from_parent = -float(reward)

        child.compute_lower_bound(pre_state_lb=self.lower_bound)

        self.children[action_id] = child
        return child

    def select_child_ucb(self, c_ucb: float) -> "TreeNode":
        assert self.children, "select_child_ucb called on node with no children."

        lb_weight = 0.2
        log_N = 2 * math.log(self.num_visited + 1.0)

        infos = []
        raw_scores = []

        for _, child in self.children.items():
            if child.num_visited == 0:
                return child

            child_lb = child.compute_lower_bound(pre_state_lb=self.lower_bound)
            est = child.estimated_obj if child.estimated_obj < float("inf") else child_lb
            raw = (1.0 - lb_weight) * est + lb_weight * child_lb

            infos.append((child, child_lb, est, raw))
            raw_scores.append(raw)

        raw_min = min(raw_scores)
        raw_max = max(raw_scores)
        scale = max(raw_max - raw_min, 10.0)

        best_score = float("inf")
        best_child = None

        for child, child_lb, est, raw in infos:
            exploit = (raw - raw_min) / scale
            explore = c_ucb * math.sqrt(log_N / child.num_visited)
            score = exploit - explore

            child.ucb_value = score
            child.exploit = exploit
            child.explore = explore
            child.lb_term = child_lb

            if score < best_score:
                best_score = score
                best_child = child

        return best_child



    def rollout(self,
                agent: DQN,
                max_rollout_depth: int = 1e6,
                use_lower_bound: bool = False) -> Tuple[float, float]:
        """
        Perform deterministic rollout from the current node.
        """

        sim_tool: ClusterTool = copy.deepcopy(self.cluster_tool)
        depth = 0; rollout_delta = 0.0
        while (not sim_tool.is_done) and depth < max_rollout_depth:
            chamber_state, robot_state = sim_tool.state
            chamber_state = np.copy(chamber_state)
            robot_state = np.copy(robot_state)
            action_mask = sim_tool.get_action_mask()

            action_id = agent.take_action(False, chamber_state, robot_state, action_mask)
            action_op = sim_tool.map_action(action_id)
            _, reward, _, _, done = sim_tool.step(action_op)

            rollout_delta += (-1 * reward)
            depth += 1

        if sim_tool.is_done or (not use_lower_bound):
            return rollout_delta, 0.0

        assert self.lower_bound != float('inf')
        remaining_lb = 0. if sim_tool.is_done else (sim_tool.compute_lower_bound(pre_state_lb=self.lower_bound) - sim_tool.get_now)
        return rollout_delta, remaining_lb


    def backup(self, rollout_obj: float):
        """
        Back-propagate the evaluated objective value.
        """
        node: Optional["TreeNode"] = self
        while node is not None:
            node.num_visited += 1
            if node.estimated_obj == float('inf'):
                node.estimated_obj = rollout_obj
            else:
                node.estimated_obj += (rollout_obj - node.estimated_obj) / node.num_visited
            node = node.parent



    def best_child_by_value(self) -> Optional["TreeNode"]:
        if not self.children:
            return None

        best_child = None
        best_val = float("inf")
        for _, child in self.children.items():
            if child.estimated_obj < best_val:
                best_val = child.estimated_obj
                best_child = child
        return best_child






    """ EOF """