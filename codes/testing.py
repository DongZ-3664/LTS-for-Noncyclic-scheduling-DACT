# -*- coding: utf-8 -*-
# @Time    : 2025/8/24 14:16
# @Author  : Dong Z.
# @File    : testing.py


import math

from Instance import *
from DQN import *
import Solution as sol
from clusterTool import ClusterTool
from treeSearch import local_tree_search

def dqn_testing(args: Dict[str, Any], ins: Instance,
                used_qnet: str = None, with_mcts: bool = False) -> Tuple[List, Tuple[Dict, List[Dict]]]:

    start_time = time.time()

    agent = DQN(method="D3QN",
                hidden_dim=cfg.hidden_dims_dict[args["net_scale"]],
                action_dim=(1 + 3 * 6 + cfg.TOTAL_RECIPES),
                learn_rate=args["qnet_lr"],
                gamma=args["gamma"],
                explore=args["explore_rate"],
                tnet_update=args["tnet_update"],
                num_episodes=args["episodes"],
                device=args["device"])

    if agent.load_qnet_model(used_qnet):
        print(f"Loaded model from {used_qnet}")
    else:
        raise Exception("Warning: No model loaded, the pth does not exist.")

    if agent.q_net.training is True:
        agent.q_net.eval()


    ins = Instance(
        args=args, K=ins.clean_freq,
        W=ins.num_wafers, R=ins.num_recipes, TQ=ins.tq_id, ID=ins.ins_id
    )

    num_episodes = 1
    schedule = None
    log_info: Dict[str, Any] = {
        "episode": np.arange(num_episodes + 1),
        "returns": np.zeros(num_episodes + 1),
        "times": np.zeros(num_episodes + 1)
    }
    time_log ={"times": []}

    start_time_i = time.time()

    cluster_tool = ClusterTool(ins=ins)
    upper_bound = 1e6
    for episode_i in range(num_episodes):


        cluster_tool.reset(ins=ins)
        returns_i = 0

        if with_mcts:
            """Run tree search."""

            returns_i, schedule_i, _ = local_tree_search(
                args=args,
                ins=ins,
                cluster_tool=cluster_tool,
                agent=agent
            )

        else:
            done = False
            while not done:
                chamber_state = np.copy(cluster_tool.state[0])
                robot_state = np.copy(cluster_tool.state[1])

                action_mask = cluster_tool.get_action_mask()

                action_id = agent.take_action((num_episodes > 1), chamber_state, robot_state, action_mask)
                action_op = cluster_tool.map_action(action_id=action_id)
                action_op, reward, next_chamber_state, next_robot_state, done = cluster_tool.step(action_op)

                returns_i += (-1 * reward)

            schedule_i = cluster_tool.get_schedule()

        time_i = time.time() - start_time_i

        if returns_i < upper_bound:
            upper_bound = returns_i
            schedule = schedule_i

        log_info["returns"][episode_i] = returns_i
        log_info["times"][episode_i] = round(time_i, 2)

    if sol.check_feasibility(ins=ins, schedule=schedule):
        print(f"W={ins.num_wafers}, R={ins.num_recipes}, TQ={ins.tq_id}, ins={ins.ins_id}, K={ins.clean_freq},"
              f" best return: {upper_bound}")
    else:
        raise Exception("this solution is infeasible xxxxx !")




    end_time = time.time(); total_time = end_time - start_time
    log_info["times"][-1]=round(total_time, 2)
    print(f"testing 所需总时间: {total_time}秒")
    return schedule, (log_info, [{}])