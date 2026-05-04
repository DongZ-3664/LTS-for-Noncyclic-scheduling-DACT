
# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 9:38
# @Author  : Dong Z.
# @File    : main.py.py


import argparse
from Instance import *
import configuration as cfg
from testing import dqn_testing


def parse_arguments():
    parser = argparse.ArgumentParser(description='user defined arguments for DQN algorithm.')


    """  DQN model running mode  """
    parser.add_argument("-md", dest="run_mode", help="the running mode of dqn model",
                        type=str, choices=["training", "tuning", "optimizing", "testing"], required=True)

    parser.add_argument("-mc", dest="with_mcts", help="with or without monte-carlo tree search",
                        type=str, default="xxxx")       # "mcts"

    parser.add_argument("-mt", dest="method", help="dqn d2qn duqn d3qn",
                        type=str, choices=["DQN", "D2QN", "DuQN", "D3QN"], required=True)

    parser.add_argument("-dv", dest="device", help="cpu or cuda",
                        type=str, choices=["cpu", "cuda:0"], default="cpu")


    """  DQN hyper-parameters  [6] """
    parser.add_argument("-ns", dest="net_scale", help="scale of different net-architectures",
                        type=str, choices=["S", "M", "L", "XL", "XXL"], required=True)

    parser.add_argument("-sp", dest="m_steps", help="multi step dqn",
                        type=int, required=True)

    parser.add_argument("-lr", dest="qnet_lr", help="learning rate",
                        type=float, required=True)

    parser.add_argument("-bs", dest="batch_size", help="batch size",
                        type=int, required=True)

    parser.add_argument("-bf", dest="buffer_size", help="size of replay buffer",
                        type=int, required=True)

    parser.add_argument("-ex", dest="explore_rate", help="epsilon, exploration rate",
                        type=float, required=True)

    parser.add_argument("-es", dest="episodes", help="number of episodes",
                        type=int, required=True)

    parser.add_argument("-gm", dest="gamma", help="gamma, discount factor",
                        type=float, required=True)

    parser.add_argument("-tn", dest="tnet_update", help="frequency of target network updates",
                        type=int, default=15, required=True)

    """  Instance related parameters  """
    parser.add_argument("-W", dest="num_wafers", help="number of wafers",
                        type=int, choices=[10, 15, 20, 25, 50, 75])

    parser.add_argument("-R", dest="num_recipes", help="numbers of types/recipes",
                        type=int, choices=[2, 3, 4, 6, 8])

    parser.add_argument("-TQ", dest="tq_id", help="type-quantity",
                        type=int, choices=[1, 2, 3, 4])

    parser.add_argument("-I", dest="ins_id", help="the I-th unique instance",
                        type=int, choices=[_ for _ in range(1, 21)])

    parser.add_argument("-K", dest="clean_freq", help="cleaning frequency/threshold",
                        type=int, choices=[1, 2, 3, 4, 5])


    """  mcts parameters  """
    parser.add_argument("-X", dest="para_name", help="tree search depth",
                        type=str, choices=["tmd", "tns", "trd", "tuc"], required=True)

    parser.add_argument("-tmd", dest="ts_max_depth", help="tree search depth",
                        type=int, choices=[2, 4, 6, 8, 10, 12, 18, 24, 48], required=True)

    parser.add_argument("-tns", dest="ts_n_sim", help="number of simulations",
                        type=int, choices=[64, 128, 192, 256, 512], required=True)       # "mcts"

    parser.add_argument("-trd", dest="ts_rollout_depth", help="tree search rollout depth",
                        type=int, choices=[15, 30, 45, 60, 80, 120, 160], required=True)

    parser.add_argument("-tuc", dest="ts_ucb_c", help="multiplier of ucb value",
                        type=float, choices=[0.3, 0.5, 1, 1.5], required=True)

    return vars(parser.parse_args())



def main_for_tuning_mcts():
    # 解析命令行参数, 并检验
    args = parse_arguments()

    # 只对主进程的 CPU 计算做限制
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)


    # 先根据命令参数创建instance集，适用于不同模式
    tuning_set: List[Instance] = [
        Instance(args=args, K=args["clean_freq"], W=args["num_wafers"],
                 R=args["num_recipes"], TQ=args["tq_id"], ID=args["ins_id"])
    ]

    well_trained_dqn_path = (
            cfg.root_path
            + f"/nets/D3QN/well_trained/well_trained_qnet" + ".pth"
    )

    for t_idx, t_ins in enumerate(tuning_set):
        t_ins.print_instance()

        schedule, log_info_pair = dqn_testing(args=args, ins=t_ins,
                                              used_qnet=well_trained_dqn_path,
                                              with_mcts=True)

        ## 不用保存调度方案
        # sol.export_sol_to_json(schedule=schedule, filename=single_ins.path_dict["sol_fn"])

        # 保存testing 的log至csv文件中
        testing_log = log_info_pair[0]

        # 调参结果的输出文件
        filename = (
            cfg.root_path + "/logs/mcts/" +
            f"tuning_for_{args['para_name']}"
            f"/R_K_{t_ins.clean_freq}/W_{t_ins.num_wafers}_R_{t_ins.num_recipes}/TQ_{t_ins.tq_id}" +
            f"/mcts_paras_x_" +
            f"{args['ts_max_depth']}_{args['ts_n_sim']}_{args['ts_rollout_depth']}_{args['ts_ucb_c']}" +
            f"_ins_{args['ins_id']}.csv"
        )
        print(filename)
        os.makedirs(Path(filename).parent, exist_ok=True)
        pd.DataFrame(testing_log).to_csv(filename)
        # 结束


def set_random_seed(seed=28, device="cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != "cpu":
        torch.cuda.manual_seed(seed)

    # 以下两行保证卷积、矩阵乘法确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    # 设置进程启动方式
    mp.set_start_method(method='spawn', force=True)

    # 设置随机数种子
    set_random_seed()

    # 运行主函数
    main_for_tuning_mcts()




















