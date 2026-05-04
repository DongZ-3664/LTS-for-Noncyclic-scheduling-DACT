# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 12:17
# @Author  : Dong Z.
# @File    : Instance.py


from imports import *



"""Global configuration parameters for scheduling instances."""

# Fixed global parameters
root_path = "E:/CBQ" if os.name == 'nt' else "~/CBQ"
numChambers = 6
operTime  = 5
rotaTime  = 5
cleanTime = 120

TOTAL_RECIPES = 10
LR_UPDATE = "Cosine"   # or "Constant"

VER_THREADS = 4

hidden_dims_dict = {
    "S" : [32, 32, 32],
    "M" : [32, 64, 64],
    "L" : [64, 128, 128],
    "XL": [64, 256, 128],
    "XXL": [64, 256, 256],
}







""" EOF """