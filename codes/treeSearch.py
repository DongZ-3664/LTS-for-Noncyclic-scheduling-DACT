# -*- coding: utf-8 -*-
# @Time    : 2025/8/26 21:05
# @Author  : Dong Z.
# @File    : treeSearch.py


from wafer import Oper
from Instance import *
from clusterTool import ClusterTool

from DQN import DQN
from treeNode import TreeNode




def _select_and_expand_once(root: TreeNode,
                            agent: DQN,
                            max_tree_depth: int,
                            c_ucb: float) -> Tuple[TreeNode, float, int, bool]:
    """
    Perform one serial selection and expansion pass.

    Returns:
        leaf_node, prefix_obj, depth, whether_expanded.
    """
    node = root
    depth = 0
    prefix_obj = float(root.cluster_tool.get_now)

    while (not node.is_terminal) and depth < max_tree_depth and node.fully_expanded():
        next_node = node.select_child_ucb(c_ucb)
        prefix_obj += float(next_node.immediate_obj_from_parent)
        node = next_node
        depth += 1

    expanded = False
    if (not node.is_terminal) and depth < max_tree_depth:
        child = node.expand_one_child(agent)
        if child is not None:
            prefix_obj += float(child.immediate_obj_from_parent)
            node = child
            depth += 1
            expanded = True

    return node, prefix_obj, depth, expanded



def batched_rollout(agent: DQN,
                    leaf_nodes: List[TreeNode],
                    max_rollout_depth: int = int(1e6),
                    use_lower_bound: bool = False,
                    debug: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Perform batched rollout for a set of leaf nodes.

    Selection, expansion, and backup are performed serially.
    Only the rollout phase is batched.
    """
    batch_size = len(leaf_nodes)
    rollout_deltas = np.zeros(batch_size, dtype=np.float64)
    remaining_lbs = np.zeros(batch_size, dtype=np.float64)

    sim_tools: List[ClusterTool] = [copy.deepcopy(node.cluster_tool) for node in leaf_nodes]
    alive = np.array([not tool.is_done for tool in sim_tools], dtype=bool)

    device = next(agent.q_net.parameters()).device

    q_forward_calls = 0
    total_env_steps = 0
    active_counts: List[int] = []
    used_depth = 0

    for depth in range(int(max_rollout_depth)):
        active_idx = np.flatnonzero(alive)
        if active_idx.size == 0:
            break

        used_depth = depth + 1
        active_counts.append(int(active_idx.size))

        cs_batch = []
        rs_batch = []
        mask_batch = []

        for idx in active_idx:
            chamber_state, robot_state = sim_tools[idx].state
            action_mask = sim_tools[idx].get_action_mask()

            if not np.any(action_mask):
                raise RuntimeError(f"No feasible action in batched_rollout for sample {idx} at depth {depth}.")

            cs_batch.append(np.copy(chamber_state))
            rs_batch.append(np.copy(robot_state))
            mask_batch.append(np.copy(action_mask))

        cs_batch = np.asarray(cs_batch, dtype=np.float32)
        rs_batch = np.asarray(rs_batch, dtype=np.int64)
        mask_batch = np.asarray(mask_batch, dtype=bool)

        with torch.no_grad():
            cs_t = torch.as_tensor(cs_batch, dtype=torch.float32, device=device)
            rs_t = torch.as_tensor(rs_batch, dtype=torch.long, device=device)
            mask_t = torch.as_tensor(mask_batch, dtype=torch.bool, device=device)

            q_values, _, _ = agent.q_net(cs_t, rs_t, mask_t)
            action_ids = torch.argmax(q_values, dim=1).detach().cpu().numpy()

        q_forward_calls += 1

        for row, idx in enumerate(active_idx):
            action_id = int(action_ids[row])
            action_op = sim_tools[idx].map_action(action_id)
            _, reward, _, _, done = sim_tools[idx].step(action_op)
            rollout_deltas[idx] += (-1.0 * float(reward))
            total_env_steps += 1
            if done:
                alive[idx] = False

    if use_lower_bound:
        for i, node in enumerate(leaf_nodes):
            if not sim_tools[i].is_done:
                lb_abs = sim_tools[i].compute_lower_bound(pre_state_lb=node.lower_bound)
                remaining_lbs[i] = max(0.0, float(lb_abs - sim_tools[i].get_now))

    info: Dict[str, Any] = {
        "batch_size": batch_size,
        "q_forward_calls": q_forward_calls,
        "total_env_steps": total_env_steps,
        "used_depth": used_depth,
        "unfinished_after_truncation": int(np.sum(alive)),
        "active_counts": active_counts,
        "avg_active": float(np.mean(active_counts)) if active_counts else 0.0,
        "avg_rollout_delta": float(np.mean(rollout_deltas)) if batch_size > 0 else 0.0,
        "avg_remaining_lb": float(np.mean(remaining_lbs)) if batch_size > 0 else 0.0,
    }

    if debug:
        info["delta_min"] = float(np.min(rollout_deltas)) if batch_size > 0 else 0.0
        info["delta_max"] = float(np.max(rollout_deltas)) if batch_size > 0 else 0.0
        info["lb_min"] = float(np.min(remaining_lbs)) if batch_size > 0 else 0.0
        info["lb_max"] = float(np.max(remaining_lbs)) if batch_size > 0 else 0.0

    return rollout_deltas, remaining_lbs, info



def local_tree_search(args: Dict[str, Any],
                      ins: Instance,
                      cluster_tool: ClusterTool,
                      agent: DQN) -> Tuple[float, List[List[Oper]], List]:
    """
    Perform rolling decision-making using D3QN-guided tree search with batched rollout.
    """
    max_tree_depth: int = args.get("ts_max_depth", 6)
    num_simulations_per_decision: int = args.get("ts_n_sim", 192)
    max_rollout_depth: int = args.get("ts_rollout_depth", 30)
    c_ucb: float = args.get("ts_ucb_c", 0.3)
    use_lower_bound_in_rollout: bool = args.get("ts_use_lower_bound", True)
    debug: bool = args.get("ts_debug", True)

    rollout_batch_size: int = args.get("ts_rollout_batch", 8)
    rollout_batch_size = max(1, int(rollout_batch_size))

    root = TreeNode(cluster_tool=copy.deepcopy(cluster_tool), parent=None, action_from_parent=-1)
    root.compute_lower_bound(pre_state_lb=0.0)

    global_best_obj = float("inf")
    decision_step = 0

    times = []
    st = time.time()
    while not cluster_tool.is_done:
        decision_step += 1

        step_rollouts: List[float] = []
        valid_simulations = 0

        t_sel_exp = 0.0
        t_rollout = 0.0
        t_backup = 0.0
        batch_infos: List[Dict[str, Any]] = []
        leaf_depths: List[int] = []
        leaf_expanded_flags: List[int] = []


        n_done = 0
        while n_done < num_simulations_per_decision:
            cur_batch = min(rollout_batch_size, num_simulations_per_decision - n_done)

            leaves: List[TreeNode] = []
            prefix_objs: List[float] = []

            t0 = time.perf_counter()
            for _ in range(cur_batch):
                node, prefix_obj, depth, expanded = _select_and_expand_once(
                    root=root,
                    agent=agent,
                    max_tree_depth=max_tree_depth,
                    c_ucb=c_ucb,
                )
                leaves.append(node)
                prefix_objs.append(prefix_obj)
                leaf_depths.append(depth)
                leaf_expanded_flags.append(1 if expanded else 0)
            t_sel_exp += (time.perf_counter() - t0)

            t1 = time.perf_counter()
            rollout_deltas, remaining_lbs, info = batched_rollout(
                agent=agent,
                leaf_nodes=leaves,
                max_rollout_depth=max_rollout_depth,
                use_lower_bound=use_lower_bound_in_rollout,
                debug=debug,
            )
            t_rollout += (time.perf_counter() - t1)
            batch_infos.append(info)

            t2 = time.perf_counter()
            for i, node in enumerate(leaves):
                node_obj_val = float(prefix_objs[i] + rollout_deltas[i] + remaining_lbs[i])
                step_rollouts.append(node_obj_val)
                valid_simulations += 1
                global_best_obj = min(global_best_obj, node_obj_val)
                node.backup(node_obj_val)
            t_backup += (time.perf_counter() - t2)

            n_done += cur_batch

        # times.append(time.time() - st)

        best_child = root.best_child_by_value()
        best_action = best_child.action_from_parent
        action_op = cluster_tool.map_action(best_action)
        _, reward, _, _, done = cluster_tool.step(action_op)

        if debug:
            if step_rollouts:
                arr = np.asarray(step_rollouts, dtype=float)
                obj_min = arr.min()
                obj_mean = arr.mean()
            else:
                obj_min = float("nan")
                obj_mean = float("nan")

            search_time = t_sel_exp + t_rollout + t_backup
            print(
                f"[TS] step={decision_step}, now={cluster_tool.get_now}, "
                f"action={best_action}, visits={best_child.num_visited}, "
                f"avg_obj={best_child.estimated_obj:.2f}, "
                f"rollout_min={obj_min:.2f}, rollout_mean={obj_mean:.2f}, "
                f"time={search_time:.3f}s"
            )

        best_child.set_as_root()
        root = best_child

    makespan = cluster_tool.get_now
    print(f"W_{ins.num_wafers}_R_{ins.num_recipes}_TQ_{ins.tq_id} : {makespan} "
          f"(batched tree search, best rollout obj={global_best_obj:.1f})")

    return makespan, cluster_tool.get_schedule(), times