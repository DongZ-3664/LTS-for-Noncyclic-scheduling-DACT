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
###
      **id**: wafer type ID
      **nums**: number of wafers of this type
      **route**: processing route
      **windows**: windows[i][0] is the processing time required by its $i$-th stage.
###
