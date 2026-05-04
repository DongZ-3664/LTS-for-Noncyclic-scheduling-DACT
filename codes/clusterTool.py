# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 9:38
# @Author  : Dong Z.
# @File    : clusterTool.py


from Instance import *
from wafer import Oper


class ClusterTool:
    def __init__(self, ins:Instance):
        self.__ins = ins
        self.__now: int = 0
        self._remaining_dic : Dict[str, int]       = self.__ins.waferNumsDict.copy()
        self._being_carried : List[Optional[Oper]] = [None, None]
        self._oper_performed: List[List[Oper]]     = [[] for _ in range(cfg.numChambers + 2)]
        self.__capacities   : List[int]            = [self.__ins.clean_freq] * (cfg.numChambers + 2)
        self._available_time: List[int]            = [0] * (cfg.numChambers + 2)
        for cbr in range(cfg.numChambers + 2) :
            self.__update_available_times(cbr)

        self.__unm_completed: List[int]            = [0] * (cfg.numChambers + 2)
        self.__num_processes: List[int]            = [0] * (cfg.numChambers + 2)
        self.__num_processes[0] = self.__num_processes[cfg.numChambers + 1] = self.__ins.num_wafers
        for recipe, num in self._remaining_dic.items():
            for p in self.__ins.waferTypeDict[recipe]:
                self.__num_processes[p.exeChamber] += num

        self.max_K = [0] * (cfg.numChambers + 1)
        for cbr in range(1, cfg.numChambers + 1):
            self.max_K[cbr] = int(ceiling(self.__num_processes[cbr] / self.__ins.clean_freq))

        self.chamber_state = np.zeros((5, 6), dtype=float)
        self.robot_state   = np.zeros((2, 2), dtype=int)

    def __update_available_times(self, cbr: int = 0):
        """
        Update the earliest available time of a chamber or station.

        If the last operation is a loading operation or a cleaning-end operation,
        the available time corresponds to the earliest completion time.
        If the last operation is an unloading operation, the available time is
        the unloading execution time.
        """

        if not self._oper_performed[cbr] : return
        last_op = self._oper_performed[cbr][-1]
        self._available_time[cbr] = last_op.exe_time
        if last_op.type == 'cb' :
            self._available_time[cbr] += cfg.cleanTime
        elif last_op.type == '+':
            pt = last_op.winLower if cbr <= cfg.numChambers else 0
            self._available_time[cbr] += (cfg.operTime + pt)

    def __unload_wafer(self, unload: Oper):
        exe_cbr = unload.exe_chamber
        used_arm = 0 if self._being_carried[0] is None else 1

        if self._being_carried[used_arm] is not None:
            raise ValueError(f"Arm {used_arm} is occupied")
        no_wafer = (not self._oper_performed[exe_cbr]
                    or self._oper_performed[exe_cbr][-1].type != '+')
        if exe_cbr > 0 and no_wafer:
            raise ValueError(f"No wafer to unload from chamber {exe_cbr}")

        self.__now = max(self._available_time[exe_cbr], self.__now + cfg.rotaTime)
        unload.exe_time = self.__now
        self._oper_performed[exe_cbr].append(unload)
        self._being_carried[used_arm] = unload.next_load
        self._available_time[exe_cbr] = self.__now
        self.__now += cfg.operTime

        if exe_cbr == 0:
            self.__unm_completed[0] += 1
            self._remaining_dic[unload.id] -= 1


    def __load_wafer(self, load: Oper):
        exe_cbr = load.exe_chamber
        used_arm = 0 if (self._being_carried[0] and self._being_carried[0] == load) else 1

        if self._being_carried[used_arm] != load:
            raise ValueError("No wafer to load")
        have_wafer = (self._oper_performed[exe_cbr]
                      and self._oper_performed[exe_cbr][-1].type == '+')
        if exe_cbr <= cfg.numChambers and have_wafer:
            raise ValueError(f"Chamber {exe_cbr} not available")

        self.__now += cfg.rotaTime
        if self._oper_performed[exe_cbr] and self._oper_performed[exe_cbr][-1].type == 'cb':
            ce_time = self._oper_performed[exe_cbr][-1].exe_time + cfg.cleanTime
            self._oper_performed[exe_cbr].append(Oper(self.__ins, "x", 0, 'ce', ce_time))
            self.__now = max(self.__now, ce_time)

        load.exe_time = self.__now
        self._oper_performed[exe_cbr].append(load)
        self._being_carried[used_arm] = None
        self.__now += cfg.operTime
        self._available_time[exe_cbr] = (self.__now + load.winLower) if exe_cbr <= cfg.numChambers else self.__now
        self.__unm_completed[exe_cbr] += 1
        if 1 <= exe_cbr <= cfg.numChambers:
            self.__capacities[exe_cbr] -= 1


    def __clean_chamber(self, cbr: int):
        unload = self._oper_performed[cbr][-1]
        assert unload.type == '-', "type error."
        assert self.__now == unload.exe_time + cfg.operTime, "timing error."
        self._oper_performed[cbr].append(Oper(self.__ins, "x", 0, 'cb', self.__now))
        self._available_time[cbr] = self.__now + cfg.cleanTime
        self.__capacities[cbr] = self.__ins.clean_freq

    @property
    def is_done(self):
        return (self.__unm_completed[cfg.numChambers + 1] ==
                self.__num_processes[cfg.numChambers + 1])

    @property
    def state(self):
        """
            chamber state = [ (bool)  whether the chamber is processing a wafer,
                              (int)   remaining processing/cleaning time,
                              (float) completion rate of each chamber,
                              (int)   next chamber to be visited,
                              (int)   remaining capacity before next cleaning ] 5x6

            robot state   = [ (int)  load status of each robot arm,
                              (int)  destination of the occupied arm ] 2x2
        """

        for cbr, seq in enumerate(self._oper_performed[1:-1], start=1):
            completion_rate = self.__unm_completed[cbr] / self.__num_processes[cbr]

            luoji_cbr = cbr - 1
            if not seq:
                self.chamber_state[0][luoji_cbr] = 0
                self.chamber_state[1][luoji_cbr] = 0
                self.chamber_state[2][luoji_cbr] = 0.0
                self.chamber_state[3][luoji_cbr] = 0
                self.chamber_state[4][luoji_cbr] = self.__capacities[cbr]
                continue

            last_op = seq[-1]
            remaining_time = max(0, self._available_time[cbr] - self.__now)

            if last_op.type == '-':
                self.chamber_state[0][luoji_cbr] = 0
                self.chamber_state[1][luoji_cbr] = 0
                self.chamber_state[2][luoji_cbr] = completion_rate
                self.chamber_state[3][luoji_cbr] = 0
                self.chamber_state[4][luoji_cbr] = self.__capacities[cbr]

            elif last_op.type == 'cb':
                self.chamber_state[0][luoji_cbr] = 3 if remaining_time > 0 else 0
                self.chamber_state[1][luoji_cbr] = remaining_time
                self.chamber_state[2][luoji_cbr] = completion_rate
                self.chamber_state[3][luoji_cbr] = 0
                self.chamber_state[4][luoji_cbr] = self.__capacities[cbr]
                assert self.chamber_state[4][luoji_cbr] == self.__ins.clean_freq, "not recovered"

            elif last_op.type == '+':
                self.chamber_state[0][luoji_cbr] = 1 if remaining_time > 0 else 2
                self.chamber_state[1][luoji_cbr] = remaining_time
                self.chamber_state[2][luoji_cbr] = completion_rate
                self.chamber_state[3][luoji_cbr] = (
                    cfg.numChambers + 1 if len(self.__ins.waferTypeDict[last_op.id]) < last_op.prcs_th + 1
                    else self.__ins.waferTypeDict[last_op.id][last_op.prcs_th].exeChamber
                )
                self.chamber_state[4][luoji_cbr] = self.__capacities[cbr]

            else:
                raise ValueError(f"Unexpected operation type: {last_op.type} in chamber {cbr}")

        for arm in (0, 1):
            if self._being_carried[arm] is not None:
                self.robot_state[0, arm] = 1
                self.robot_state[1, arm] = self._being_carried[arm].exe_chamber
            else:
                self.robot_state[0, arm] = 0
                self.robot_state[1, arm] = 0

        return self.chamber_state, self.robot_state


    def step(self, action_op: Oper):
        """State transition."""

        exe_cbr, op_type = action_op.exe_chamber, action_op.type
        start_time = self.__now

        if op_type == '+':
            self.__load_wafer(action_op)
        elif op_type == '-':
            self.__unload_wafer(action_op)
        elif op_type == '-c':
            action_op.type = '-'
            self.__unload_wafer(action_op)
            self.__clean_chamber(action_op.exe_chamber)
        elif op_type == 'w':
            self.__now += 0
        else:
            raise Exception("Unexpected opType!")
        delta = self.__now - start_time

        next_cbr_state, next_rbt_state = self.state
        return action_op, (-1 * delta), next_cbr_state, next_rbt_state , self.is_done


    def get_action_mask(self):
        """
            Generate all feasible actions under the current state.

            Action definition:
                a = [0, 1, ..., cfg.numChambers]:
                    load a carried wafer into chamber a % (cfg.numChambers + 1)
                a = [cfg.numChambers + 1, 2 * cfg.numChambers]:
                    unload a wafer from chamber a - cfg.numChambers and clean the chamber
                a = [2 * cfg.numChambers + 1, 3 * cfg.numChambers]:
                    unload a wafer from chamber a - 2 * cfg.numChambers without cleaning
                a = [3 * cfg.numChambers + 1, ...]:
                    unload a new wafer of the specified recipe from the input load lock
        """

        arm_1, arm_2 = self._being_carried[0], self._being_carried[1]

        if arm_1 is None and arm_2 is None:
            return self._get_actions_for_zero_load()

        if arm_1 and arm_2 :
            return self._get_actions_for_double_loads(arm_1, arm_2)

        return self._get_actions_for_single_load(arm_1 if arm_1 else arm_2)

    def _get_actions_for_zero_load(self):
        """Generate feasible actions when both arms are empty."""
        enable_action_ids = np.zeros((3 * cfg.numChambers + 1 + cfg.TOTAL_RECIPES), dtype=bool)
        for recipe, nums in self._remaining_dic.items():
            if nums > 0:
                enable_action_ids[3 * cfg.numChambers + 1 + int(recipe)] = True

        for cbr, seq in enumerate(self._oper_performed[1:-1], start=1):
            if seq and seq[-1].type == '+':
                if self.__capacities[cbr] > 0:
                    enable_action_ids[2 * cfg.numChambers + cbr] = True
                elif self.__capacities[cbr] == 0:
                    enable_action_ids[1 * cfg.numChambers + cbr] = True

        return enable_action_ids

    def _get_actions_for_double_loads(self, arm1:Oper, arm2:Oper):
        """Generate feasible actions when both arms carry wafers."""
        enable_action_ids = np.zeros((3 * cfg.numChambers + 1 + cfg.TOTAL_RECIPES), dtype=bool)

        if arm1.exe_chamber == arm2.exe_chamber:
            enable_action_ids[arm1.exe_chamber % (cfg.numChambers + 1)] = True
            return enable_action_ids

        for op in (arm1, arm2):
            if self.__not_in_processing(cbr=op.exe_chamber):
                enable_action_ids[op.exe_chamber % (cfg.numChambers + 1)] = True

        return enable_action_ids

    def _get_actions_for_single_load(self, arm:Oper):
        """Generate feasible actions when one arm carries a wafer."""
        des = arm.exe_chamber
        enable_action_ids = np.zeros((3 * cfg.numChambers + 1 + cfg.TOTAL_RECIPES), dtype=bool)

        if self.__not_in_processing(cbr=des):
            enable_action_ids[des % (cfg.numChambers + 1)] = True

            for recipe, num in self._remaining_dic.items():
                if num > 0:
                    enable_action_ids[3 * cfg.numChambers + 1 + int(recipe)] = True

            for cbr, seq in enumerate(self._oper_performed[1:-1], start=1):
                if seq and seq[-1].type == '+':
                    if self.__capacities[cbr] > 0:
                        enable_action_ids[2 * cfg.numChambers + cbr] = True
                    elif self.__capacities[cbr] == 0:
                        enable_action_ids[1 * cfg.numChambers + cbr] = True

            return enable_action_ids

        if self.__capacities[des] > 0:
            enable_action_ids[2 * cfg.numChambers + des] = True
        elif self.__capacities[des] == 0:
            enable_action_ids[1 * cfg.numChambers + des] = True

        for cbr, seq in enumerate(self._oper_performed[1:-1], start=1):
            if self.__not_in_processing(cbr=cbr) or cbr == des: continue

            next_cbr =  (
                cfg.numChambers + 1 if len(self.__ins.waferTypeDict[seq[-1].id]) < seq[-1].prcs_th + 1
                else self.__ins.waferTypeDict[seq[-1].id][seq[-1].prcs_th].exeChamber
            )
            if self.__not_in_processing(cbr=next_cbr):
                if self.__capacities[cbr] > 0:
                    enable_action_ids[2 * cfg.numChambers + cbr] = True
                elif self.__capacities[cbr] == 0:
                    enable_action_ids[1 * cfg.numChambers + cbr] = True


        for recipe, num in self._remaining_dic.items():
            if num > 0 and self.__not_in_processing(cbr=self.__ins.waferTypeDict[recipe][0].exeChamber):
                enable_action_ids[3 * cfg.numChambers + 1 + int(recipe)] = True

        return enable_action_ids


    def map_action(self, action_id: int) -> Oper:

        if action_id <= cfg.numChambers:
            load_cbr = (cfg.numChambers + 1) if action_id == 0 else action_id
            arm1, arm2 = self._being_carried[0], self._being_carried[1]
            if arm1 and arm2 and arm1.exe_chamber == arm2.exe_chamber == load_cbr:
                return arm1 if arm1.exe_time < arm2.exe_time else arm2
            load = arm1 if (arm1 and arm1.exe_chamber == load_cbr) else arm2
            assert load.exe_chamber == load_cbr, "Oper chamber does not match the load chamber"
            return load

        if action_id <= (2 * cfg.numChambers):
            unload_cbr = action_id - cfg.numChambers
            unload = self._oper_performed[unload_cbr][-1].next_unload
            unload.type = '-c'
            return unload

        if action_id <= (3 * cfg.numChambers):
            unload_cbr = action_id - 2 * cfg.numChambers
            unload = self._oper_performed[unload_cbr][-1].next_unload
            return unload

        new_unload_recipe = str(action_id - (3 * cfg.numChambers) - 1)
        action = Oper(self.__ins, new_unload_recipe, 0, '-', 0)
        action.set_cbr(0)
        action.set_id(
            self.__ins.reserved_id_start[new_unload_recipe]
            + self.__ins.waferNumsDict[new_unload_recipe]
            - self._remaining_dic[new_unload_recipe]
        )
        return action

    def __not_in_processing(self, cbr : int) -> bool:
        return (not self._oper_performed[cbr]
                or cbr == cfg.numChambers + 1
                or self._oper_performed[cbr][-1].type != '+' )


    def compute_lower_bound(self, pre_state_lb):
        """
            Compute a heuristic lower bound on the remaining makespan from the current state.

            tail_time_dict stores the minimum remaining completion time after a wafer
            of each recipe finishes processing in each chamber.

            pre_state_lb records the lower bound of the previous state. It is used
            to enforce nondecreasing lower bounds along a search path.
        """

        if self.is_done:
            return self.__now

        now = self.__now
        tail_time_dict = self.__ins.tail_time_dict
        complete_all = lambda c: True if self.__unm_completed[c] == self.__num_processes[c] else False

        base = [0] * (cfg.numChambers + 2)
        for cbr, opers in enumerate(self._oper_performed[1:-1], start=1):
            if not opers:
                base[cbr] = cfg.rotaTime
            elif opers[-1].type == '+':
                base[cbr] = max(opers[-1].exe_time + cfg.operTime + opers[-1].winLower, now + cfg.rotaTime)
                if complete_all(cbr):
                    pass
                else :
                    base[cbr] += cfg.operTime
                    base[cbr] += cfg.cleanTime if self.__capacities[cbr] == 0 else cfg.rotaTime
            elif opers[-1].type == '-':
                assert opers[-1].exe_time + cfg.operTime <= now, "timing error."
                base[cbr] = max(opers[-1].exe_time + cfg.operTime, now)
                if complete_all(cbr):
                    base[cbr] -= cfg.operTime
                else:
                    base[cbr] += cfg.rotaTime
            elif opers[-1].type == 'cb':
                if complete_all(cbr):
                    base[cbr] = max(opers[-1].exe_time, now) - cfg.operTime
                else:
                    base[cbr] = max(opers[-1].exe_time + cfg.cleanTime, now + cfg.rotaTime)
            else:
                raise Exception("unexpected type.")

        workload = [0] * (cfg.numChambers + 2)
        unscheds = [0] * (cfg.numChambers + 2)
        cbr_tail = [int(1e6)] * (cfg.numChambers + 2)


        for cbr, opers in enumerate(self._oper_performed[1:-1], start=1):
            if opers and opers[-1].type == '+':
                current = opers[-1]
                while current.exe_chamber <= cfg.numChambers:
                    if current.exe_chamber != cbr:
                        workload[current.exe_chamber] += current.winLower
                        unscheds[current.exe_chamber] += 1
                    cbr_tail[current.exe_chamber] = min(cbr_tail[current.exe_chamber], tail_time_dict[current.id][current.exe_chamber])
                    current = current.next_load

            if complete_all(cbr) and (opers[-1].type == 'cb' or opers[-1].type == '-'):
                cbr_tail[cbr] = min(cbr_tail[cbr], 2 * cfg.operTime + cfg.rotaTime)


        rbt_span = 0
        for rbt, oper in enumerate(self._being_carried):
            if oper is not None:
                if oper.exe_chamber <= cfg.numChambers:
                    s_t = max(base[oper.exe_chamber], now + cfg.rotaTime + cfg.operTime)
                    rbt_span = max(rbt_span,  s_t + oper.winLower + tail_time_dict[oper.id][oper.exe_chamber])

                current = oper
                while current.exe_chamber <= cfg.numChambers:
                    workload[current.exe_chamber] += current.winLower
                    unscheds[current.exe_chamber] += 1
                    the_tail = tail_time_dict[current.id][current.exe_chamber]
                    cbr_tail[current.exe_chamber] = min(cbr_tail[current.exe_chamber], the_tail)
                    current = current.next_load


        for recipe, nums in self._remaining_dic.items():
            if nums <= 0: continue

            for process in self.__ins.waferTypeDict[recipe]:
                cbr = process.exeChamber
                if 1 <= cbr <= cfg.numChambers:
                    workload[cbr] += nums * process.winLower
                    unscheds[cbr] += nums
                    the_tail = tail_time_dict[recipe][process.exeChamber]
                    cbr_tail[cbr] = min(cbr_tail[cbr], the_tail)

        for cbr in range(1, cfg.numChambers + 1):
            if unscheds[cbr] <= self.__capacities[cbr]:
                remain_K = 0
            else:
                remain_K = int(self.__capacities[cbr] != 0) + math.ceil((unscheds[cbr] - self.__capacities[cbr]) / self.__ins.clean_freq) - 1

            workload[cbr] += remain_K * cfg.cleanTime
            workload[cbr] += max(unscheds[cbr] - 1, 0) * (2 * cfg.operTime + cfg.rotaTime)
            workload[cbr] -= remain_K * cfg.rotaTime

        span = [0] * (cfg.numChambers + 2)
        for cbr in range(1, cfg.numChambers + 1):
            span[cbr] =  base[cbr] + (0 if complete_all(cbr) else cfg.operTime)
            span[cbr] += workload[cbr] + (cbr_tail[cbr] if cbr_tail[cbr] < 1e5 else 0)
        lb = max(span[1:-1])
        # lb = max(rbt_span, lb)
        # lb = max(pre_state_lb, lb)

        return lb



    def print_state_description(self):
        print(f"Current time: {self.__now}")
        for idx, opers in enumerate(reversed(self._oper_performed[1:-1])):
            cbr = cfg.numChambers - idx
            if not opers[-1]:
                print(f"Chamber {cbr}: not started")
                continue
            op = opers[-1]
            if op.type == '+':
                print(f"Chamber {cbr}: processing wafer {op.reserved_id} (type {op.id}), "
                      f"processing start time {op.exe_time+cfg.operTime}, "
                      f"remaining processing time {max(0, op.exe_time+cfg.operTime+op.winLower-self.__now)}, "
                      f"expected completion time {op.exe_time+cfg.operTime+op.winLower}, "
                      f"remaining operations {self.__num_processes[cbr] - self.__unm_completed[cbr]}" )
            elif op.type == '-':
                print(f"Chamber {cbr}: completed wafer {op.reserved_id} (type {op.id}), "
                      f"completion time {op.exe_time}, "
                      f"idle time {max(0, self.__now-(op.exe_time+cfg.operTime))}, "
                      f"remaining operations {self.__num_processes[cbr] - self.__unm_completed[cbr]}" )
            elif op.type == 'cb':
                print(f"Chamber {cbr}: cleaning, "
                      f"remaining cleaning time {max(0, op.exe_time+cfg.cleanTime-self.__now)}, "
                      f"idle time {max(0, self.__now-(op.exe_time+cfg.operTime+cfg.cleanTime))}, "
                      f"remaining operations {self.__num_processes[cbr] - self.__unm_completed[cbr]}" )

        for rbt, op in enumerate(self._being_carried):
            if op is None:
                print(f"Robot arm {rbt}: empty")
            else:
                print(f"Robot arm {rbt}: carrying wafer {op.reserved_id} (type {op.id}), "
                      f"unloading time {op.exe_time}, destination chamber {op.exe_chamber}")

        print(f"Unreleased wafers: {self._remaining_dic}")


    def reset(self, ins: Instance):
        self.__ins = ins
        self.__now = 0
        self._remaining_dic  = self.__ins.waferNumsDict.copy()
        self._being_carried  = [None, None]
        self._oper_performed = [[] for _ in range(cfg.numChambers + 2)]
        self.__capacities    = [self.__ins.clean_freq] * (cfg.numChambers + 2)
        self._available_time = [0] * (cfg.numChambers + 2)
        for cbr in range(cfg.numChambers + 2) :
            self.__update_available_times(cbr)

        self.__unm_completed = [0] * (cfg.numChambers + 2)
        self.__num_processes = [0] * (cfg.numChambers + 2)
        self.__num_processes[0] = self.__num_processes[cfg.numChambers + 1] = self.__ins.num_wafers
        for w, num in self._remaining_dic.items():
            for p in self.__ins.waferTypeDict[w]:
                self.__num_processes[p.exeChamber] += num

        self.chamber_state = np.zeros((5, 6), dtype=float)
        self.robot_state   = np.zeros((2, 2), dtype=int)

    def get_schedule(self):
        schedule = self._oper_performed.copy()
        for cbr, cbr_opers in enumerate(schedule[1:-1], start=1):
            if cbr_opers[-1].type == '-':
                cb_time = cbr_opers[-1].exe_time + cfg.operTime
                ce_time = cb_time + cfg.cleanTime
                cbr_opers.append(Oper(self.__ins, "x", 0, 'cb', cb_time))
                cbr_opers.append(Oper(self.__ins, "x", 0, 'ce', ce_time))
            elif cbr_opers[-1].type == 'cb':
                ce_time = cbr_opers[-1].exe_time + cfg.cleanTime
                cbr_opers.append(Oper(self.__ins, "x", 0, 'ce', ce_time))
            elif cbr_opers[-1].type == 'ce':
                pass
            else:
                raise Exception("unexpected oper")

        return schedule



    def print_state(self):
        chamber_state = self.state[0]
        robot_state = self.state[1]

        print(f"the current moment is: {self.__now}")
        for chamber in reversed(range(1, cfg.numChambers + 1)):
            chamber_info = (
                f"Chamber {chamber}: in state {int(chamber_state[0][chamber])}, "
                f"remaining time {int(chamber_state[1][chamber])}, "
                f"available time {int(self._available_time[chamber])}"
            )
            print(chamber_info)

        for robot in [0, 1]:
            robot_info = (
                f"Robot {robot}: load is [{int(robot_state[0][robot])}], "
            )
            print(robot_info)

    @property
    def get_now(self):
        return self.__now


    def __deepcopy__(self, memo):
        new_ins = self.__ins

        new_tool = ClusterTool(new_ins)
        new_tool.__now            = self.__now
        new_tool._remaining_dic   = self._remaining_dic.copy()
        new_tool._being_carried   = copy.deepcopy(self._being_carried)
        new_tool._oper_performed  = copy.deepcopy(self._oper_performed)
        new_tool.__capacities     = self.__capacities.copy()
        new_tool._available_time  = self._available_time.copy()
        new_tool.__unm_completed  = self.__unm_completed.copy()
        new_tool.__num_processes  = self.__num_processes.copy()
        new_tool.max_K            = self.max_K.copy()
        new_tool.chamber_state    = np.copy(self.chamber_state)
        new_tool.robot_state      = np.copy(self.robot_state)

        return new_tool

    @property
    def its_ins(self):
        return self.__ins
""" EOF """