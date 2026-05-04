# LTS-for-Noncyclic-scheduling-DACT

This repository provides the Python implementation of the DQN and Learning-based Tree Search (LTS) algorithms for the paper 《noncyclic scheduling problem of dual-armed cluster tools with multi-type wafer fabrication and chamber cleaning requirements》

## The procedure supports:
      DQN training
      DQN validation during training
      Pure DQN inference
      Learning-based Tree Search
      Schedule feasibility checking
      Logging of training and testing results

## Project Structure

A recommended project structure is as follows:
      LTS-for-Noncyclic-scheduling-DACT/
      │
      ├── main.py
      ├── configuration.py
      ├── Instance.py
      ├── wafer.py
      ├── clusterTool.py
      ├── DQN.py
      ├── training.py
      ├── testing.py
      ├── treeSearch.py
      ├── treeNode.py
      ├── Solution.py
      │
      ├── instances/
      │   └── W_10_R_2/
      │       └── TQ_1/
      │           └── ins_11.json
      │
      ├── nets/
      │   └── D3QN/
      │       └── well_trained/
      │           └── well_trained_qnet.pth
      │
      ├── logs/
      │
      └── solutions/







