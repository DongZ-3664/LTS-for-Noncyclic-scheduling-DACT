# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 12:17
# @Author  : Dong Z.
# @File    : Instance.py

from imports import *
import configuration as cfg
from collections import namedtuple


"""Instance definition."""

Process = namedtuple("Process", ["exeChamber", "winLower", "winUpper"])


class Instance:
    def __init__(self, args:Dict[str, Any], K: int, W:int, R:int, TQ:int, ID:int):

        self.clean_freq = K
        self.num_wafers = W
        self.num_recipes = R
        self.tq_id = TQ
        self.ins_id = ID

        self.path_dict: Dict[str, str] = {}

        self.waferNumsDict: Dict[str, int] = {}
        self.waferTypeDict: Dict[str, List[Process]] = {}

        self.reserved_id_start : Dict[str, int] = {}
        self.tail_time_dict : Dict[str, List]   = {}

        self._label_all_paths(args=args)
        self._read_data_from_json(self.path_dict["ins_fn"])
        self._get_tail_time_dict()

    def _label_all_paths(self, args):
        folder_ = f"/W_{self.num_wafers}_R_{self.num_recipes}/TQ_{self.tq_id}"
        postfix = f"ins_{self.ins_id}"

        if args.get("method", "MILP") == "MILP":
            self.path_dict["ins_fn"] = (
                    cfg.root_path + "/instances"
                    + folder_ + "/" + postfix + ".json"
            )
            self.path_dict["sol_fn"] = (
                    cfg.root_path + "/solutions"
                    + f"/MILP/"
                    + f"/R_K_{self.clean_freq}"
                    + folder_ + "/" + postfix + ".json"
            )
            self.path_dict["log_fn"] = (
                    cfg.root_path + "/logs"
                    + f"/MILP/"
                    + f"/R_K_{self.clean_freq}"
                    + folder_ + "/" + postfix + ".csv"
            )

            return

        dqn_hypers_ = (
            f"/{args['device']}_{args['net_scale']}_{args['m_steps']}"
            + f"_{args['qnet_lr']}_{args['batch_size']}_{args['buffer_size']}"
            + f"_{args['explore_rate']}_{args['episodes']}"
            + f"_{args['gamma']}_{args['tnet_update']}_"
        )

        mcts_hypers_ = (
            "mcts_paras_x_"
            if args["run_mode"] == "testing" and args.get('with_mcts') == 'mcts' else
            "without_mcts_"
        )

        self.path_dict["ins_fn"] = (
            cfg.root_path + "/instances"
            + folder_ + "/" + postfix + ".json"
        )
        self.path_dict["sol_fn"] = (
            cfg.root_path + "/solutions"
            + f"/{args['method']}/{args['run_mode']}"
            + f"/R_K_{self.clean_freq}"
            + folder_ + "/" + dqn_hypers_ + mcts_hypers_ + postfix + ".json"
        )
        self.path_dict["log_fn"] = (
            cfg.root_path + "/logs"
            + f"/{('mcts' if args['with_mcts'] == 'mcts' else 'D3QN')}/{args['run_mode']}"
            + f"/R_K_{self.clean_freq}"
            + folder_ + "/" + dqn_hypers_ + mcts_hypers_ + postfix + ".csv"
        )
        self.path_dict["ver_fn"] = (
            cfg.root_path + "/logs"
            + f"/{args['method']}/verifying"
            + f"/R_K_{self.clean_freq}"
            + folder_ + "/" + dqn_hypers_ + mcts_hypers_ + postfix + ".csv"
        )
        self.path_dict["qnet_fn"] = (
            cfg.root_path + "/nets"
            + f"/{args['method']}/{args['run_mode']}"
            + f"/R_K_{self.clean_freq}"
            + folder_ + "/" + dqn_hypers_ + mcts_hypers_ + postfix + ".pth"
        )


    def _read_data_from_json(self, filename):
        if not os.path.exists(filename):
            print(f"Error: the file '{filename}' does not exist.", file=sys.stderr)
            sys.exit(1)

        with open(filename, 'r') as ins_data:
            root = json.load(ins_data)

            s = 1
            for wafer_info in root:
                wafer_flow = []
                for p in range(len(wafer_info["route"])):
                    exeCbr = wafer_info["route"][p]
                    Win_L = wafer_info["windows"][p][0]
                    Win_U = wafer_info["windows"][p][1]
                    wafer_flow.append(Process(exeCbr, Win_L, Win_U))

                self.waferTypeDict[wafer_info["id"]] = wafer_flow
                self.waferNumsDict[wafer_info["id"]] = wafer_info["nums"]
                self.reserved_id_start[wafer_info["id"]] = s
                s += wafer_info["nums"]


    def _get_tail_time_dict(self):
        tail_time_dict:Dict[str, List] = {}
        for recipe, route in self.waferTypeDict.items():
            tail_time_dict[recipe] = [int(1e6)] * (cfg.numChambers + 1)
            remaining_time = 2 * cfg.operTime + cfg.rotaTime
            for process in reversed(route):
                tail_time_dict[recipe][process.exeChamber] = remaining_time
                remaining_time += (2 * cfg.operTime + cfg.rotaTime + process.winLower)

            tail_time_dict[recipe][0] = remaining_time

        self.tail_time_dict = tail_time_dict


    def print_instance(self):
        print(f"W={self.num_wafers}, R={self.num_recipes}, TQ={self.tq_id}, K={self.clean_freq}.")
        print(f"Total number of wafers: {sum(self.waferNumsDict.values())}")
        print(f"Cleaning frequency/threshold: {self.clean_freq}")
        for recipe, nums in self.waferNumsDict.items():
            print(f"wafer type: {recipe}, wafer nums: {nums}")
        for recipe, route in self.waferTypeDict.items():
            print(f"wafer type: {recipe}, route:{[_.exeChamber for _ in route]}")
        for recipe, route in self.waferTypeDict.items():
            print(f"wafer type: {recipe}, p_times:{[_.winLower for _ in route]}")


def ins_set_for_training(args: Dict[str, Any]):
    training_set: List[Instance] = [ ]
    for K_ in [1, 2, 3]:
        for W_ in [10, 15, 20, 25, 50, 75]:
            for R_ in ([2, 3, 4] if W_ < 30 else [4, 6, 8]):
                for TQ_ in [1, 2, 3, 4]:
                    training_set.append(
                        Instance(args=args, W=W_, R=R_, TQ=TQ_, ID=11, K=K_)
                    )

    return training_set



def ins_set_for_tuning_mcts(args: Dict[str, Any]):
    training_set: List[Instance] = [ ]
    for K_ in [1, 2, 3]:
        for W_ in [10, 15, 20, 25]:
            for R_ in [2, 3, 4]:
                for TQ_ in [1, 2, 3, 4]:
                    training_set.append(
                        Instance(args=args, W=W_, R=R_, TQ=TQ_, ID=12, K=K_)
                    )

    return training_set



def ins_set_for_verifying(args: Dict[str, Any]):
    verifying_set: List[Instance] = [
        Instance(args=args, W=75, R=4, TQ=1, ID=11, K=3),
        Instance(args=args, W=75, R=4, TQ=2, ID=11, K=2),
        Instance(args=args, W=75, R=6, TQ=3, ID=11, K=3),
        Instance(args=args, W=75, R=6, TQ=4, ID=11, K=2),
        Instance(args=args, W=75, R=8, TQ=1, ID=11, K=3),
        Instance(args=args, W=75, R=8, TQ=2, ID=11, K=3),
        Instance(args=args, W=75, R=8, TQ=3, ID=11, K=2),
        Instance(args=args, W=75, R=8, TQ=4, ID=11, K=1),

        Instance(args=args, W=50, R=4, TQ=1, ID=11, K=3),
        Instance(args=args, W=50, R=4, TQ=2, ID=11, K=2),
        Instance(args=args, W=50, R=6, TQ=3, ID=11, K=3),
        Instance(args=args, W=50, R=6, TQ=4, ID=11, K=2),
        Instance(args=args, W=50, R=8, TQ=1, ID=11, K=3),
        Instance(args=args, W=50, R=8, TQ=2, ID=11, K=3),
        Instance(args=args, W=50, R=8, TQ=3, ID=11, K=2),
        Instance(args=args, W=50, R=8, TQ=4, ID=11, K=1),

        Instance(args=args, W=25, R=2, TQ=1, ID=11, K=1),
        Instance(args=args, W=25, R=2, TQ=2, ID=11, K=1),
        Instance(args=args, W=25, R=3, TQ=3, ID=11, K=2),
        Instance(args=args, W=25, R=3, TQ=4, ID=11, K=2),
        Instance(args=args, W=25, R=4, TQ=1, ID=11, K=3),
        Instance(args=args, W=25, R=4, TQ=2, ID=11, K=3),

        Instance(args=args, W=20, R=2, TQ=1, ID=11, K=1),
        Instance(args=args, W=20, R=2, TQ=2, ID=11, K=1),
        Instance(args=args, W=20, R=3, TQ=3, ID=11, K=2),
        Instance(args=args, W=20, R=3, TQ=4, ID=11, K=2),
        Instance(args=args, W=20, R=4, TQ=2, ID=11, K=3),
        Instance(args=args, W=20, R=4, TQ=4, ID=11, K=3),

        Instance(args=args, W=15, R=2, TQ=4, ID=11, K=1),
        Instance(args=args, W=15, R=2, TQ=3, ID=11, K=1),
        Instance(args=args, W=15, R=3, TQ=2, ID=11, K=2),
        Instance(args=args, W=15, R=3, TQ=1, ID=11, K=2),
        Instance(args=args, W=15, R=4, TQ=3, ID=11, K=3),
        Instance(args=args, W=15, R=4, TQ=1, ID=11, K=3),

        Instance(args=args, W=10, R=2, TQ=1, ID=11, K=1),
        Instance(args=args, W=10, R=2, TQ=2, ID=11, K=1),
        Instance(args=args, W=10, R=3, TQ=3, ID=11, K=2),
        Instance(args=args, W=10, R=3, TQ=4, ID=11, K=2),
        Instance(args=args, W=10, R=4, TQ=1, ID=11, K=3),
        Instance(args=args, W=10, R=4, TQ=4, ID=11, K=3),

        Instance(args=args, W=75, R=6, TQ=1, ID=11, K=3),
        Instance(args=args, W=50, R=6, TQ=2, ID=11, K=3),
        Instance(args=args, W=25, R=4, TQ=3, ID=11, K=2),
        Instance(args=args, W=20, R=4, TQ=4, ID=11, K=2),
        Instance(args=args, W=15, R=4, TQ=4, ID=11, K=2),
        Instance(args=args, W=10, R=4, TQ=4, ID=11, K=2),

        Instance(args=args, W=75, R=8, TQ=1, ID=11, K=3),
        Instance(args=args, W=50, R=8, TQ=2, ID=11, K=3),
        Instance(args=args, W=25, R=3, TQ=3, ID=11, K=2),
        Instance(args=args, W=20, R=3, TQ=4, ID=11, K=2),
        Instance(args=args, W=15, R=2, TQ=4, ID=11, K=2),
        Instance(args=args, W=10, R=2, TQ=4, ID=11, K=2),
    ]

    return verifying_set




""" EOF """
