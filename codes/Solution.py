# -*- coding: utf-8 -*-
# @Time    : 2025/6/9 15:34
# @Author  : Dong Z.
# @File    : Solution.py

from wafer import Oper
from Instance import *


def export_sol_to_json(schedule: List[List[Oper]], filename):
    os.makedirs(Path(filename).parent, exist_ok=True)

    root = [ { "chamber_id": _,
               "operations": [ ] } for _ in range(cfg.numChambers + 2)]

    for cbr, opers in enumerate(schedule):
        for op in opers:
            root[cbr]["operations"].append({
                "wafer_id": op.id,
                "prcs_th": op.prcs_th,
                "type": str(op.type),
                "exe_time": op.exe_time,
                "exe_chamber": op.exe_chamber,
                "reserved_id" : op.reserved_id
            })

    with open(filename, 'w') as of:
        json.dump(root, of, indent=4)


def get_sche_from_json(filename):
    if not os.path.isfile(filename):
        print(f"Error：the file '{filename}' does not exist！", file=sys.stderr)
        sys.exit(1)

    with open(filename, 'r', encoding='utf-8') as file:
        schedule_json = json.load(file)

    schedule_res: List[List[Oper]] = [[] for _ in range(cfg.numChambers + 2)]
    for cbr_info in schedule_json:
        cbr = cbr_info["chamber_id"]
        for op in cbr_info["operations"]:
            schedule_res[cbr].append(
                Oper( op["wafer_id"], op["prcs_th"], op["type"], op["exe_time"])
            )
            schedule_res[cbr][-1].set_cbr(op["exe_chamber"])
            schedule_res[cbr][-1].set_id(op["reserved_id"])
    return schedule_res


def check_time_windows(sche:List[List[Oper]]):
    for c, m in enumerate(sche[1:-1], start=1):
        if not m : continue
        o = 0
        while o < len(m):
            if m[o].type == '+':
                st = m[o].exe_time
                et = m[o + 1].exe_time
                during = et - st - cfg.operTime
                if m[o].winUpper < during:
                    print(f"Time window violation for (chamber, th): ({c},{o//2})", " ", m[o].winUpper, "  ", during)
            o += 1


def check_feasibility(ins:Instance, schedule:List[List[Oper]]):

    for cbr, cbr_opers in enumerate(schedule[1:-1], start=1):
        i = 0
        while i < len(cbr_opers):
            if cbr_opers[i].type == '+':
                op_ld_time = cbr_opers[i].exe_time
                op_un_time = cbr_opers[i + 1].exe_time
                if op_ld_time + cfg.operTime + cbr_opers[i].winLower > op_un_time:
                    print("processing time is insufficient.")
                    return False
                i += 2
            elif cbr_opers[i].type == 'cb':
                cb_time = cbr_opers[i].exe_time
                ce_time = cbr_opers[i + 1].exe_time
                if cb_time + cfg.cleanTime != ce_time:
                    print("cleaning time is insufficient.")
                    return False
                i += 2
            elif cbr_opers[i].type == 'ce' or cbr_opers[i].type == '-':
                raise Exception("type error.")

    total_opers = 0
    for recipe, num in ins.waferNumsDict.items():
        total_opers += (2 * len(ins.waferTypeDict[recipe]) + 2) * num

    th = [0] * (cfg.numChambers + 2)
    clean_remaining = [ins.clean_freq] * (cfg.numChambers + 2)
    arm: List[Optional[Oper]] = [None, None]

    oper_cnt = 0
    while True:
        delta = 1e6
        next_cbr = 0
        next_op  = None
        find_next = False
        for cbr, cbr_opers in enumerate(schedule):
            if th[cbr] == len(cbr_opers) : continue

            if cbr_opers[th[cbr]].type != '+' and cbr_opers[th[cbr]].type != '-' :
                print("operation type error.")
                return False

            if cbr_opers[th[cbr]].exe_time < delta:
                delta = cbr_opers[th[cbr]].exe_time
                next_op = cbr_opers[th[cbr]]
                next_cbr = cbr
                find_next = True

        if find_next is False: break

        th[next_cbr] += 1
        if 1 <= next_cbr <= cfg.numChambers:
            if schedule[next_cbr][th[next_cbr]].type == "cb":
                if schedule[next_cbr][th[next_cbr] + 1].type != "ce":
                    print("no end of cleaning.")
                    return False
                th[next_cbr] += 2

                if th[next_cbr] < len(schedule[next_cbr]) and schedule[next_cbr][th[next_cbr]].type == '-':
                    raise Exception("type error.")

        # 找到next_op之后，令机械手去执行
        cure_op = next_op
        exe_cbr = cure_op.get_cbr()
        if cure_op.type == '-':
            if arm[0] and arm[1]:
                print("arms are both occupied.")
                return False
            used = 0 if arm[0] is None else 1
            arm[used] = cure_op.next_load

            if clean_remaining[exe_cbr] == 0 :
                clean_remaining[exe_cbr] = ins.clean_freq

        elif cure_op.type == '+':
            if arm[0] is None and arm[1] is None:
                print("arms are both empty")
                return False

            used = 0 if arm[0] and arm[0] == cure_op else 1
            if arm[used] is None or arm[used] != cure_op:
                print("operation error.")
                return False

            arm[used] = None

            if exe_cbr <= cfg.numChambers and clean_remaining[exe_cbr] <= 0:
                print("must be cleaned.")
                return False

            clean_remaining[exe_cbr] -= 1


        oper_cnt += 1


    if oper_cnt != total_opers:
        print("the number of operations is incorrect.")
        return False

    # print("By checking, the solution is feasible. √ √ √ √ √ √.")
    return True



def output_MILP_log_to_csv(sol_info_list, filename: str):

    os.makedirs(Path(filename).parent, exist_ok=True)

    headers = ["obj", "lower_bound", "gap", "global_time", "solve_time", "vars", "cons"]
    with open(filename, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)          # headers
        writer.writerow([
            sol_info_list[0],              # obj
            f"{sol_info_list[1]:.2f}",    # lower_bound
            f"{sol_info_list[2]:.4f}",    # gap
            f"{sol_info_list[3]:.1f}",    # global_time
            f"{sol_info_list[4]:.1f}",    # solve_time
            sol_info_list[5],              # vars
            sol_info_list[6]               # cons
        ])


