# LTS-for-Noncyclic-scheduling-DACT

This repository provides the Python implementation of the DQN and Learning-based Tree Search (LTS) algorithms for the paper 《noncyclic scheduling problem of dual-armed cluster tools with multi-type wafer fabrication and chamber cleaning requirements》

## 1.The procedure supports:
      DQN training
      DQN validation during training
      Pure DQN inference
      Learning-based Tree Search
      Schedule feasibility checking
      Logging of training and testing results

## 2.Project Structure


## 3.Path Configuration:
###
Before running the code, set the project root directory in configuration.py:
      root_path = "E:/CBQ" if os.name == "nt" else "/home/ache/dongz/CBQ"

Modify it to your local repository path. For example: 
      root_path = "/home/user/LTS-for-Noncyclic-scheduling-DACT"
or on Windows:
      root_path = "D:/LTS-for-Noncyclic-scheduling-DACT"

All input instances, trained networks, logs, and solution files are loaded from or saved to paths under **root_path**.


## 4.Instance Format
Instances are stored as JSON files. Each file contains a list of wafer types.
Each wafer type includes:

>     id      : wafer type ID
>     nums    : number of wafers of this type
>     route   : processing route
>     windows : windows[i][0] is the processing time required by its $i$-th stage.


Example:

###
[
    {
        "id": "0",
        "nums": 5,
        "route": [1, 2, 3],
        "windows": [[80, 1000], [90, 1000], [75, 1000]]
    },
    {
        "id": "1",
        "nums": 5,
        "route": [4, 5, 6],
        "windows": [[140, 1000], [160, 1000], [150, 1000]]
    }
]
###

The instance file should be placed under:
###
instances/W_{W}_R_{R}/TQ_{TQ}/ins_{ID}.json
###
For example:
###
instances/W_10_R_2/TQ_1/ins_11.json
###


## 5. Command-Line Arguments
The code is executed through **main.py**.
| Argument | Description                                                         | Example   |
| -------- | ------------------------------------------------------------------- | --------- |
| `-md`    | Running mode: `training`, `tuning`, `optimizing`, or `testing`      | `testing` |
| `-mc`    | Whether to use tree search during testing. Use `mcts` to enable LTS | `mcts`    |
| `-mt`    | DQN variant. Recommended: `D3QN`                                    | `D3QN`    |
| `-dv`    | Device: `cpu` or `cuda:0`                                           | `cpu`     |
| `-ns`    | Network scale: `S`, `M`, `L`, `XL`, or `XXL`                        | `XL`      |
| `-sp`    | Number of steps in multi-step learning                              | `3`       |
| `-lr`    | Learning rate                                                       | `0.001`   |
| `-bs`    | Batch size                                                          | `128`     |
| `-bf`    | Replay buffer size                                                  | `10000`   |
| `-ex`    | Initial exploration rate                                            | `0.075`   |
| `-es`    | Number of training episodes                                         | `25000`   |
| `-gm`    | Discount factor                                                     | `0.85`    |
| `-tn`    | Target network update frequency                                     | `25`      |
| `-W`     | Number of wafers                                                    | `10`      |
| `-R`     | Number of wafer types                                               | `2`       |
| `-TQ`    | Type-quantity pattern ID                                            | `1`       |
| `-I`     | Instance ID                                                         | `11`      |
| `-K`     | Cleaning frequency / threshold                                      | `1`       |






