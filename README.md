# LTS-for-Noncyclic-scheduling-DACT

This repository provides the Python procudure of the deep Q-network (DQN) and the Learning-based Tree Search (LTS) algorithm for the paper 
**noncyclic scheduling problem of dual-armed cluster tools with multi-type wafer fabrication and chamber cleaning requirements**

## 1.The procedure supports:
      DQN training
      DQN validation during training
      Pure DQN inference
      Learning-based Tree Search
      Schedule feasibility checking
      Logging of training and testing results

## 2.Project Structure

>      LTS-for-Noncyclic-scheduling-DACT/
>      │
>      ├── codes
>      │   └── xxxx.py
>      ├── instances/
>      │   └── W_10_R_2/
>      │       └── TQ_1/
>      │           └── ins_1.json
>      │
>      ├── nets/
>      │   └── D3QN/
>      │       └── well_trained/
>      │           └── well_trained_qnet.pth
>      │
>      ├── logs/
>      │
>      └── solutions/


## 3.Path Configuration:
Before running the code, set the project root directory in configuration.py:
###      
      root_path = "E:/CBQ" if os.name == "nt" else "~/CBQ"
###
Modify it to your local repository path. For example: 
###
      root_path = "/home/user/LTS-for-Noncyclic-scheduling-DACT"
###
or on Windows:
###
      root_path = "D:/LTS-for-Noncyclic-scheduling-DACT"
###
All input instances, trained networks, logs, and solution files are loaded from or saved to paths under the **root_path**.


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
| `-sp`    | Number of steps in multi-step learning                              | `6`       |
| `-lr`    | Learning rate                                                       | `0.005`   |
| `-bs`    | Batch size                                                          | `128`     |
| `-bf`    | Replay buffer size                                                  | `10000`   |
| `-ex`    | Initial exploration rate                                            | `0.009`   |
| `-es`    | Number of training episodes                                         | `25000`   |
| `-gm`    | Discount factor                                                     | `0.90`    |
| `-tn`    | Target network update frequency                                     | `15`      |
| `-W`     | Number of wafers                                                    | `10`      |
| `-R`     | Number of wafer types                                               | `2`       |
| `-TQ`    | Type-quantity pattern ID                                            | `1`       |
| `-I`     | Instance ID                                                         | `11`      |
| `-K`     | Cleaning frequency / threshold                                      | `1`       |



### 6. Train the DQN Agent
To train the DQN/D3QN agent, run:
###
      python main.py \
            -md training \
            -mt D3QN \
            -dv cpu \
            -ns XL \
            -sp 6 \
            -lr 0.005 \
            -bs 128 \
            -bf 14000 \
            -ex 0.009 \
            -es 25000 \
            -gm 0.90 \
            -tn 15
###


The trained model is saved to:
###
      nets/D3QN/well_trained/well_trained_qnet.pth
###

### 8. Pure DQN Testing

Pure DQN testing uses the trained Q-network directly. At each decision step, the admissible action with the largest Q-value is selected.

Example:
###
      python main.py \
              -md testing \
              -mt D3QN \
              -dv cpu \
              -ns XL \
              -sp 3 \
              -lr 0.001 \
              -bs 128 \
              -bf 10000 \
              -ex 0.075 \
              -es 25000 \
              -gm 0.85 \
              -tn 25 \
              -W 10 \
              -R 2 \
              -TQ 1 \
              -I 11 \
              -K 1
###

The program loads the trained model from:

nets/D3QN/well_trained/well_trained_qnet.pth
The testing log is saved to:

logs/D3QN/testing/R_K_{K}/W_{W}_R_{R}/TQ_{TQ}/ins_{ID}.csv


### 9. LTS Testing

To enable Learning-based Tree Search, run testing mode with:
-mc mcts

python main.py \
  -md testing \
  -mc mcts \
  -mt D3QN \
  -dv cpu \
  -ns XL \
  -sp 3 \
  -lr 0.001 \
  -bs 128 \
  -bf 10000 \
  -ex 0.075 \
  -es 25000 \
  -gm 0.85 \
  -tn 25 \
  -W 10 \
  -R 2 \
  -TQ 1 \
  -I 11 \
  -K 1


The default LTS parameters are defined in treeSearch.py:
| Parameter            | Description                                | Default |
| -------------------- | ------------------------------------------ | ------- |
| `ts_max_depth`       | Maximum tree depth                         | `6`     |
| `ts_n_sim`           | Number of simulations per decision step    | `192`   |
| `ts_rollout_depth`   | Maximum rollout depth                      | `30`    |
| `ts_ucb_c`           | Exploration coefficient in UCB             | `0.3`   |
| `ts_use_lower_bound` | Whether to use lower-bound tail estimation | `True`  |
| `ts_debug`           | Whether to print search diagnostics        | `True`  |
| `ts_rollout_batch`   | Batch size for batched rollout             | `8`     |

Currently, these LTS parameters are set inside treeSearch.py. To modify them from the command line, add the corresponding arguments in parse_arguments() in main.py.


### 10. Output Files
Testing logs

Testing logs are saved as CSV files under:
###
      logs/{method}/testing/R_K_{K}/W_{W}_R_{R}/TQ_{TQ}/ins_{ID}.csv
###
For LTS testing, logs are saved under the mcts testing directory.









