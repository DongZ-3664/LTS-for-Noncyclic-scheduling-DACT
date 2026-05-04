# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 11:23
# @Author  : Dong Z.
# @File    : wafer.py.py


from Instance import *


"""Wafer operation."""

class Oper:
    def __init__(self, ins:Instance, id:str="x", prcs_th:int=0, type:str='', exe_time=0):
        self.__ins = ins
        self.id = id            # recipe
        self.prcs_th = prcs_th  # step
        self.type = type
        self.exe_time = exe_time
        self.reserved_id = 0
        self.exe_chamber = 0

    @property
    def winLower(self) -> int:
        if self.exe_chamber in (0, cfg.numChambers + 1):
             return 0
        return self.__ins.waferTypeDict[self.id][self.prcs_th - 1].winLower

    @property
    def winUpper(self) -> int:
        if self.exe_chamber in (0, cfg.numChambers + 1):
             return 0
        return self.__ins.waferTypeDict[self.id][self.prcs_th - 1].winUpper

    def set_id(self, i) -> None:
        self.reserved_id = i

    def get_id(self) -> int:
        return self.reserved_id

    def set_cbr(self, c) -> None:
        self.exe_chamber = c

    def get_cbr(self) -> int:
        return self.exe_chamber

    def oper_str(self, with_exe_time=False) -> str:
        """Return a formatted string representation of the operation."""
        base_str = f"[{self.reserved_id}, {self.prcs_th}, {self.type}, {self.exe_chamber}]"
        if with_exe_time:
            return base_str + f"{{{self.exe_time}, {self.id}}}"
        return base_str

    @property
    def next_load(self):
        next_op = Oper(self.__ins, self.id, self.prcs_th + 1, '+', self.exe_time)

        next_op.set_cbr(
            cfg.numChambers + 1 if len(self.__ins.waferTypeDict[self.id]) < self.prcs_th + 1
            else self.__ins.waferTypeDict[self.id][self.prcs_th].exeChamber
        )
        next_op.set_id(self.reserved_id)
        return next_op

    @property
    def next_unload(self):
        next_prcs_th = self.prcs_th
        next_exe_cbr = self.exe_chamber

        if self.type == '-':
            next_prcs_th = self.prcs_th + 1
            if len(self.__ins.waferTypeDict[self.id]) <= next_prcs_th - 1:
                raise IndexError("No next unloading operation.")
            next_exe_cbr = self.__ins.waferTypeDict[self.id][next_prcs_th - 1].exeChamber

        next_op = Oper(self.__ins, self.id, next_prcs_th, '-', self.exe_time)
        next_op.set_cbr(next_exe_cbr)
        next_op.set_id(self.reserved_id)
        return next_op

    @property
    def prev_load(self):
        prev_prcs_th = self.prcs_th
        prev_exe_cbr = self.exe_chamber

        if self.type == '+':
            prev_prcs_th = self.prcs_th - 1
            if prev_prcs_th <= 0:
                raise Exception("No previous loading operation.")
            prev_exe_cbr = self.__ins.waferTypeDict[self.id][prev_prcs_th - 1].exeChamber

        prev_op = Oper(self.__ins, self.id, prev_prcs_th, '+', self.exe_time)
        prev_op.set_cbr(prev_exe_cbr)
        prev_op.set_id(self.reserved_id)
        return prev_op

    @property
    def prev_unload(self):
        prev_op = Oper(self.__ins, self.id, self.prcs_th - 1, '-', self.exe_time)
        prev_op.set_cbr(
            0 if (self.prcs_th - 1 <= 0)
            else self.__ins.waferTypeDict[self.id][self.prcs_th - 2].exeChamber
        )
        prev_op.set_id(self.reserved_id)
        return prev_op


    @property
    def description(self) -> str:
        """Get a human-readable description of the operation."""
        if self.type == '-' :
            return (f"unload wafer {self.reserved_id} (type-{self.id}) from chamber {self.exe_chamber}, "
                    f"completing step {self.prcs_th}, at time {self.exe_time}.")
        elif self.type == '+' :
            return (f"load wafer {self.reserved_id} (type-{self.id}) into chamber {self.exe_chamber}, "
                    f"starting step {self.prcs_th}, at time {self.exe_time}. ")


    def __eq__(self, other) -> bool:
        return (
            other.get_id() == self.reserved_id
            and other.prcs_th == self.prcs_th
            and other.type == self.type
        )


    def __ne__(self, other) -> bool:
        return not (self == other)


    def __deepcopy__(self, memo=None):
        new_oper = Oper(
            self.__ins,
            id=self.id,
            prcs_th=self.prcs_th,
            type=self.type,
            exe_time=self.exe_time
        )
        new_oper.reserved_id = self.reserved_id
        new_oper.exe_chamber = self.exe_chamber
        return new_oper

""" EOF """