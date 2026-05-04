# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 9:38
# @Author  : Dong Z.
# @File    : main.py.py


import argparse
from Instance import *
import Solution as sol
import configuration as cfg
from testing import *
from training import dqn_tuning_optimizing, dqn_training_with_verifying


def parse_arguments():
    parser = argparse.ArgumentParser(description='user defined arguments for DQN algorithm.')

    """DQN model running mode."""
    parser.add_argument("-md", dest="run_mode", help="the running mode of dqn model",
                        type=str, choices=["training", "tuning", "optimizing", "testing"], required=True)

    parser.add_argument("-mc", dest="with_mcts", help="with or without monte-carlo tree search",
                        type=str, default="xxxx")       # "mcts"

    parser.add_argument("-mt", dest="method", help="dqn d2qn duqn d3qn",
                        type=str, choices=["DQN", "D2QN", "DuQN", "D3QN"], required=True)

    parser.add_argument("-dv", dest="device", help="cpu or cuda",
                        type=str, choices=["cpu", "cuda:0"], default="cpu")


    """DQN hyper-parameters."""
    parser.add_argument("-ns", dest="net_scale", help="scale of different net-architectures",
                        type=str, choices=["S", "M", "L", "XL", "XXL"], required=True)

    parser.add_argument("-sp", dest="m_steps", help="multi step dqn",
                        type=int, required=True)

    parser.add_argument("-lr", dest="qnet_lr", help="learning rate",
                        type=float, required=True)

    parser.add_argument("-bs", dest="batch_size", help="batch size",
                        type=int,  required=True)

    parser.add_argument("-bf", dest="buffer_size", help="size of replay buffer",
                        type=int,  required=True)

    parser.add_argument("-ex", dest="explore_rate", help="epsilon, exploration rate",
                        type=float, required=True)

    parser.add_argument("-es", dest="episodes", help="number of episodes",
                        type=int, required=True)

    parser.add_argument("-gm", dest="gamma", help="gamma, discount factor",
                        type=float, required=True)

    parser.add_argument("-tn", dest="tnet_update", help="frequency of target network updates",
                        type=int, default=15, required=True)

    """Instance-related parameters."""
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

    return vars(parser.parse_args())



def main_for_tuning_training_optimizing_testing():
    args = parse_arguments()

    if args["device"] == "cpu":
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)


    training_set: List[Instance] = [ ]
    verifying_set:List[Instance] = [ ]
    if args["run_mode"] == "training":
        training_set  = ins_set_for_training(args=args)
        verifying_set = ins_set_for_verifying(args=args)
    elif args["run_mode"] in ["tuning", "optimizing", "testing"]:
        training_set = [ Instance(args=args, K=args["clean_freq"], W=args["num_wafers"],
                                  R=args["num_recipes"], TQ=args["tq_id"], ID=args["ins_id"] ) ]


    single_ins = training_set[0]
    schedule, log_info_pair, best_qnet = None, None, None
    if args["run_mode"] == "testing":
        well_trained_dqn_path = (
            cfg.root_path
            + f"/nets/{args['method']}/well_trained/well_trained_qnet" + ".pth"
        )
        well_trained_dqn_path = cfg.root_path + "/nets/D3QN/well_trained/well_trained_qnet.pth"
        schedule, log_info_pair = dqn_testing(args=args, ins=single_ins,
                                         used_qnet=well_trained_dqn_path,
                                         with_mcts=(args.get('with_mcts') == "mcts") )


    elif args["run_mode"] == "training":
        schedule, log_info_pair, best_qnet = dqn_training_with_verifying(args=args,
                                                                         training_ins_set=training_set,
                                                                         verifying_ins_set=verifying_set)

    elif args["run_mode"] in ["tuning", "optimizing"]:
        schedule, log_info_pair, best_qnet = dqn_tuning_optimizing(args=args, ins=single_ins)

    else:
        raise Exception("Unexpected mode.")

    if args["run_mode"] == "testing":
        testing_log = log_info_pair[0]
        filename = single_ins.path_dict["log_fn"]
        filename = str(Path(filename).parent) + f"/ins_{args['ins_id']}.csv"
        os.makedirs(Path(filename).parent, exist_ok=True)
        pd.DataFrame(testing_log).to_csv(filename)

    elif args["run_mode"] == "training":
        assert log_info_pair is not None, "No logs exist."
        training_log, verifying_log = log_info_pair[0], log_info_pair[1]

        filename_t = (
            cfg.root_path + "/logs"
            + f"/{args['method']}/{args['run_mode']}"
            + f"/{args['device']}_{args['net_scale']}_{args['m_steps']}"
            + f"_{args['qnet_lr']}_{args['batch_size']}_{args['buffer_size']}"
            + f"_{args['explore_rate']}_{args['episodes']}"
            + f"_{args['gamma']}_{args['tnet_update']}_"
            + "training.csv"
        )
        os.makedirs(Path(filename_t).parent, exist_ok=True)
        pd.DataFrame(training_log).to_csv(filename_t)

        for v_idx, v_ins in enumerate(verifying_set):
            filename_v = v_ins.path_dict["ver_fn"]
            os.makedirs(Path(filename_v).parent, exist_ok=True)
            pd.DataFrame(verifying_log[v_idx]).to_csv(filename_v)

        best_trained_dqn_path = (
            cfg.root_path
            + f"/nets/{args['method']}/well_trained/"
            + f"well_trained_qnet" + ".pth"
        )
        os.makedirs(Path(best_trained_dqn_path).parent, exist_ok=True)
        torch.save(best_qnet, best_trained_dqn_path)


    elif args["run_mode"] in ["tuning", "optimizing"]:
        sol.export_sol_to_json(schedule=schedule, filename=single_ins.path_dict["sol_fn"])

        tuning_or_optimizing_log = log_info_pair[0]
        filename = single_ins.path_dict["log_fn"]
        os.makedirs(Path(filename).parent, exist_ok=True)
        pd.DataFrame(tuning_or_optimizing_log).to_csv(filename)

    else:
        raise Exception("unexpected runing mode.")





def set_random_seed(seed=28, device="cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != "cpu":
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    mp.set_start_method(method='spawn', force=True)

    set_random_seed()

    main_for_tuning_training_optimizing_testing()