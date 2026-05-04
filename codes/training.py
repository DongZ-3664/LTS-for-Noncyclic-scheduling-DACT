# -*- coding: utf-8 -*-
# @Time    : 2025/6/9 15:34
# @Author  : Dong Z.
# @File    : training.py.py


from DQN import *
from Instance import *
from clusterTool import ClusterTool


""" dqn_training """

""" 
    Action dimension:
        The action dimension depends on the number of wafer recipes.
        Actions include:
        1) loading a wafer into chamber 1-6 or the output load lock;
        2) unloading a wafer from chamber 1-6 followed by immediate cleaning;
        3) unloading a wafer from chamber 1-6 without immediate cleaning;
        4) unloading a new wafer from chamber 0, where the wafer type must be specified.
        Therefore, the total number of actions is recipes + 7 + 6 + 6.
"""

def dqn_tuning_optimizing(args: Dict[str, Any], ins: Instance) -> Tuple[List, Tuple[Dict, List[Dict]], Dict]:

    hidden_dim = cfg.hidden_dims_dict[args["net_scale"]]
    action_dim = 1 + 3 * 6 + cfg.TOTAL_RECIPES

    device = args["device"]
    method = args["method"]

    learn_rate  = args["qnet_lr"]
    batch_size  = args["batch_size"]
    buffer_size = args["buffer_size"]
    explore = args["explore_rate"]
    m_steps = args["m_steps"]

    num_episodes = args["episodes"]
    gamma = args["gamma"]
    tnet_update  = args["tnet_update"]

    minimal_size = 4000

    replay_buffer = ReplayBuffer(buffer_size, batch_size, m_steps, gamma)
    agent = DQN(method=method,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                learn_rate=learn_rate,
                gamma=gamma,
                explore=explore,
                tnet_update=tnet_update,
                num_episodes=num_episodes,
                device=device )

    schedule: Optional[List] = None
    log_info: Dict[str, Any] = {
        "episode": np.arange(num_episodes),
        "returns": np.zeros(num_episodes),
        "losses" : np.zeros(num_episodes)
    }
    best_qnet_state_dict = {}

    print(ins.waferNumsDict)
    assert ins.num_wafers == sum(ins.waferNumsDict.values()), "Mismatching in the number of wafers."
    ins.print_instance()

    cluster_tool = ClusterTool(ins=ins)
    upper_bound = 1e6
    for episode_i in range(num_episodes):
        returns_i = 0

        cluster_tool.reset(ins=ins)
        done = False
        action_mask = cluster_tool.get_action_mask(); next_action_mask= None
        while not done:
            chamber_state = np.copy(cluster_tool.state[0])
            robot_state = np.copy(cluster_tool.state[1])

            action_id = agent.take_action(True, chamber_state, robot_state, action_mask)
            action_op = cluster_tool.map_action(action_id=action_id)
            action_op, reward, next_chamber_state, next_robot_state, done = cluster_tool.step(action_op)
            assert action_op.type != '-c', "invalid type"
            next_action_mask = cluster_tool.get_action_mask()
            replay_buffer.add(chamber_state, robot_state, action_id, reward,
                              np.copy(next_chamber_state), np.copy(next_robot_state), done,
                              np.copy(action_mask), np.copy(next_action_mask))

            action_mask = next_action_mask.copy()
            returns_i += (-1 * reward)

        if (returns_i < upper_bound) and (3500 < episode_i):
            """Update the upper bound and save the current Q-network parameters."""
            upper_bound = returns_i
            best_qnet_state_dict = agent.q_net.state_dict()


        loss_value_i = 0
        if minimal_size < replay_buffer.size():
            for _ in range(15):
                transition_dict = _build_transition_dict(replay_buffer, m_steps)
                if transition_dict is None:
                    raise Exception("no enough transitions.")

                loss_value_i = agent.update(transition_dict, episode_i + 1)


        log_info["returns"][episode_i] = returns_i
        log_info["losses"][episode_i] = loss_value_i
        average = (np.average(log_info["returns"][episode_i - 19:episode_i + 1])
                   if 20 <= episode_i else np.average(log_info["returns"][0:episode_i + 1]))
        print(f"episode: {episode_i}, returns: {returns_i:.2f}, loss: {loss_value_i:.2f}, average: {average:.1f}")

    schedule = cluster_tool.get_schedule()
    return schedule, (log_info, [{}]), best_qnet_state_dict






""" 
    dqn_training_with_verifying trains the agent using the training instance set
    and periodically evaluates it on the validation instance set.
"""

def dqn_training_with_verifying(args: Dict[str, Any],
                                training_ins_set: List[Instance],
                                verifying_ins_set: List[Instance] )-> Tuple[Any, Tuple[Dict, List[Dict]], Dict]:

    hidden_dim = cfg.hidden_dims_dict[args["net_scale"]]
    action_dim = 1 + 3 * 6 + cfg.TOTAL_RECIPES

    method = args["method"]
    device = args["device"]

    learn_rate  = args["qnet_lr"]
    batch_size  = args["batch_size"]
    buffer_size = args["buffer_size"]
    explore = args["explore_rate"]
    m_steps = args["m_steps"]

    num_episodes = args["episodes"]
    gamma = args["gamma"]
    tnet_update  = args["tnet_update"]

    minimal_size = 4000
    verify_freq  = 100


    replay_buffer = ReplayBuffer(buffer_size, batch_size, m_steps, gamma)
    agent = DQN(method=method,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                learn_rate=learn_rate,
                gamma=gamma,
                explore=explore,
                tnet_update=tnet_update,
                num_episodes=num_episodes,
                device=device )

    pool = None
    if True:
        if os.name == 'nt':
            pass
        else:
            import resource
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
            except:
                pass
        agent.q_net.share_memory()
        pool = mp.Pool(
            processes=min(8, cfg.VER_THREADS),
            initializer=_init_verify_worker
        )



    schedule: Optional[List] = None
    training_log_info: Dict[str, Any] = {
        "episode": np.arange(num_episodes),
        "losses" : np.zeros(num_episodes)
    }
    verifying_log_info: List[Dict[str, Any]] = [{
        "episode": np.arange(0, num_episodes, verify_freq),
        "returns": np.zeros(num_episodes//verify_freq)
    } for _ in range(len(verifying_ins_set))]

    ver_avg_returns, min_avg_return, ver_th = np.zeros(num_episodes//verify_freq), 1e6, 0
    best_qnet_state_dict = {}

    T_cluster_tool  = ClusterTool(training_ins_set[0])
    V_cluster_tools = [ClusterTool(ins) for ins in verifying_ins_set ]
    for episode_i in range(num_episodes):
        t_ins = random.choice(training_ins_set)

        T_cluster_tool.reset(ins=t_ins)
        done = False
        action_mask = T_cluster_tool.get_action_mask(); next_action_mask = None
        while not done:
            chamber_state = np.copy(T_cluster_tool.state[0])
            robot_state = np.copy(T_cluster_tool.state[1])

            action_id = agent.take_action(True, chamber_state, robot_state, action_mask)
            action_op = T_cluster_tool.map_action(action_id=action_id)
            action_op, reward, next_chamber_state, next_robot_state, done = T_cluster_tool.step(action_op)
            assert action_op.type != '-c', "invalid type"
            next_action_mask = T_cluster_tool.get_action_mask()
            replay_buffer.add(chamber_state, robot_state, action_id, reward,
                              np.copy(next_chamber_state), np.copy(next_robot_state), done,
                              np.copy(action_mask), np.copy(next_action_mask))

            action_mask = next_action_mask.copy()


        if  replay_buffer.size() > minimal_size:
            for _ in range(15):
                transition_dict = _build_transition_dict(replay_buffer, m_steps)
                if transition_dict is None:
                    raise Exception("no enough transitions.")

                training_loss_value_i = agent.update(transition_dict, episode_i + 1)
                training_log_info["losses"][episode_i] = training_loss_value_i

        print(f"episode: {episode_i}, loss: {training_log_info['losses'][episode_i]:.2f}", flush=True)

        if episode_i % verify_freq == 0:
            ver_start_time = time.time()

            with torch.no_grad():
                agent.q_net.eval()
                returns_i = [0] * len(verifying_ins_set)

                if pool is not None:
                    process_args = [(agent, v_cluster_tool) for v_cluster_tool in V_cluster_tools]
                    returns_i = pool.map(_verify_single_instance, process_args)

                agent.q_net.train()

                for v_idx, ret in enumerate(returns_i):
                    verifying_log_info[v_idx]["returns"][ver_th] = ret
                ver_avg_returns[ver_th] = np.average(returns_i)

                if ver_avg_returns[ver_th] < min_avg_return:
                    """Save the current Q-network parameters."""
                    min_avg_return = ver_avg_returns[ver_th]
                    best_qnet_state_dict = agent.q_net.state_dict()

            ver_end_time = time.time()
            time_per_ver = ver_end_time - ver_start_time
            print(f"episode: {episode_i}, avgreturn:{ver_avg_returns[ver_th]:.1f}, "
                  f"time spend on {ver_th + 1}-th verifying is: {time_per_ver:.2f} s", flush=True)
            ver_th += 1

    if pool is not None:
        pool.close()
        pool.join()

    return schedule, (training_log_info, verifying_log_info), best_qnet_state_dict




def _verify_single_instance(process_args: Tuple[DQN, ClusterTool] ):

    agent, v_cluster_tool = process_args
    v_cluster_tool.reset(v_cluster_tool.its_ins)

    ret = 0
    with torch.no_grad():
        done = False
        while not done:
            chamber_state = np.copy(v_cluster_tool.state[0])
            robot_state = np.copy(v_cluster_tool.state[1])

            action_mask = v_cluster_tool.get_action_mask()

            action_id = agent.take_action(False, chamber_state, robot_state, action_mask)

            action_op = v_cluster_tool.map_action(action_id=action_id)
            action_op, reward, next_chamber_state, next_robot_state, done = v_cluster_tool.step(action_op)

            ret += (-1 * reward)

    return ret



def _build_transition_dict(replay_buffer, m_steps):
    if m_steps == 1:
        (batch_c_states, batch_r_states,
         batch_actions, batch_rewards,
         batch_nc_states, batch_nr_states,
         batch_done_flags,
         batch_a_masks, batch_na_masks) = replay_buffer.sample_batch_single_step()

        batch_effective_n = np.ones(len(batch_actions), dtype=np.int32)

    else:
        (batch_c_states, batch_r_states,
         batch_actions, batch_rewards,
         batch_nc_states, batch_nr_states,
         batch_done_flags,
         batch_a_masks, batch_na_masks,
         batch_effective_n) = replay_buffer.sample_batch_multi_steps()

    transition_dict = {
        "c_states": batch_c_states,
        "r_states": batch_r_states,
        "action_ids": batch_actions,
        "rewards": batch_rewards,
        "nc_states": batch_nc_states,
        "nr_states": batch_nr_states,
        "done_flags": batch_done_flags,
        "a_masks": batch_a_masks,
        "na_masks": batch_na_masks,
        "effective_n": batch_effective_n
    }

    return transition_dict



def _init_verify_worker():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)



""" EOF """