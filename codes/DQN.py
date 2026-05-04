# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 11:12
# @Author  : Dong Z.
# @File    : DQN.py

from imports import *
import configuration as cfg
import torch.nn as nn
import torch.nn.functional as tf


class ReplayBuffer:
    def __init__(self, capacity, batch_size, n_steps, gamma):
        self.capacity = capacity
        self.batch_size = batch_size

        # Circular buffer
        self.buffer: List[Tuple] = [None] * capacity
        self.ptr = 0
        self.size_ = 0

        self._n_steps = n_steps
        self._gamma_ = gamma
        self._gamma_pows = np.array([self._gamma_ ** s for s in range(self._n_steps)], dtype=np.float32)


        # Preallocated NumPy arrays
        self.cbr_states = np.empty((batch_size, 5, 6), dtype=np.float32)
        self.rbt_states = np.empty((batch_size, 2, 2), dtype=np.int32)
        self.actions = np.empty(batch_size, dtype=np.int32)
        self.rewards = np.empty(batch_size, dtype=np.float32)
        self.next_cbr_states = np.empty((batch_size, 5, 6), dtype=np.float32)
        self.next_rbt_states = np.empty((batch_size, 2, 2), dtype=np.int32)
        self.dones = np.empty(batch_size, dtype=np.bool_)
        self.action_masks = np.empty((batch_size, action_dim := 3 * cfg.numChambers + 1 + cfg.TOTAL_RECIPES), dtype=np.bool_)
        self.next_action_masks = np.empty((batch_size, action_dim), dtype=np.bool_)
        self.effective_n = np.empty(batch_size, dtype=np.int32)

    def add(self, cbr_state, rbt_state, action, reward,
            next_cbr_state, next_rbt_state, done, action_mask, next_action_mask):
        self.buffer[self.ptr] = (
            cbr_state, rbt_state, action, reward,
            next_cbr_state, next_rbt_state, done,
            action_mask, next_action_mask
        )
        self.ptr = (self.ptr + 1) % self.capacity
        self.size_ = min(self.size_ + 1, self.capacity)

    def sample_batch_single_step(self):
        if self.size_ < self.batch_size:
            return None

        idx_list = np.random.choice(self.size_, size=self.batch_size, replace=False)

        # Position of the logically oldest transition in the circular buffer
        oldest = 0 if self.size_ < self.capacity else self.ptr

        for i, src_logical in enumerate(idx_list):
            src_physical = (oldest + src_logical) % self.capacity
            (self.cbr_states[i], self.rbt_states[i], self.actions[i], self.rewards[i],
             self.next_cbr_states[i], self.next_rbt_states[i], self.dones[i],
             self.action_masks[i], self.next_action_masks[i]) = self.buffer[src_physical]

        return (self.cbr_states, self.rbt_states, self.actions, self.rewards,
                self.next_cbr_states, self.next_rbt_states, self.dones,
                self.action_masks, self.next_action_masks)

    def sample_batch_multi_steps(self):
        """
        Applicable assumptions:
            1) Transitions are written to the replay buffer episode by episode.
            2) When sampling starts, the latest transition written into the buffer is done=True.
        """

        idx_list = np.random.choice(self.size_, size=self.batch_size, replace=False)


        for i, src_idx in enumerate(idx_list):
            start_idx = src_idx
            first_trans = self.buffer[start_idx]

            self.cbr_states[i] = first_trans[0]
            self.rbt_states[i] = first_trans[1]
            self.actions[i] = first_trans[2]
            self.action_masks[i] = first_trans[7]
            self.next_action_masks[i] = first_trans[8]

            sum_rewards = 0.0
            eff_n = 0
            self.dones[i] = False

            for s in range(self._n_steps):
                physical_idx = (start_idx + s) % self.capacity
                trans = self.buffer[physical_idx]

                sum_rewards += self._gamma_pows[s] * trans[3]
                eff_n = s + 1

                self.next_cbr_states[i] = trans[4]
                self.next_rbt_states[i] = trans[5]
                self.next_action_masks[i] = trans[8]

                if trans[6]:
                    self.dones[i] = True
                    break

            self.rewards[i] = sum_rewards
            self.effective_n[i] = eff_n

        return (self.cbr_states, self.rbt_states, self.actions, self.rewards,
                self.next_cbr_states, self.next_rbt_states, self.dones,
                self.action_masks, self.next_action_masks, self.effective_n)


    def size(self):
        return self.size_



class ChamberRobotNet(nn.Module):
    def __init__(self, hidden_dim: List[int], action_dim: int, dropout_rate: float, use_dueling:bool):
        super(ChamberRobotNet, self).__init__()

        self.use_dueling = use_dueling

        self._h0 = hidden_dim[0]
        self._h1 = hidden_dim[1]
        self._h2 = hidden_dim[2]

        self.chamber_branch_nets = nn.ModuleList([
            nn.Sequential(                          # Chamber state vector 0: categorical status
                nn.Embedding(4, 8),
                nn.Flatten(),
                nn.Linear(8 * 6, self._h0),
                nn.ReLU(),
                nn.LayerNorm(self._h0)
            ),

            nn.Sequential(                          # Chamber state vector 1: remaining processing/cleaning time
                nn.Linear(6, self._h0),
                nn.ReLU(),
                nn.LayerNorm(self._h0),
                nn.Linear(self._h0, self._h0),
                nn.ReLU()
            ),

            nn.Sequential(                          # Chamber state vector 2: completion ratio
                nn.Linear(6, self._h0),
                nn.ReLU(),
                nn.LayerNorm(self._h0),
                nn.Linear(self._h0, self._h0),
                nn.ReLU()
            ),

            nn.Sequential(                          # Chamber state vector 3: next destination category
                nn.Embedding(8, 8),
                nn.Flatten(),
                nn.Linear(8 * 6, self._h0),
                nn.ReLU(),
                nn.LayerNorm(self._h0)
            ),

            nn.Sequential(                          # Chamber state vector 4: remaining capacity before next cleaning
                nn.Embedding(5 + 1, 8),
                nn.Flatten(),
                nn.Linear(8 * 6, self._h0),
                nn.ReLU(),
                nn.LayerNorm(self._h0)
            )
        ])

        self.robot_branch_nets = nn.ModuleList([
            nn.Sequential(                          # Robot state vector 0: arm load status
                nn.Embedding(2, 8),
                nn.Flatten(),
                nn.Linear(8 * 2, self._h0),
                nn.ReLU(),
                nn.LayerNorm(self._h0)
            ),

            nn.Sequential(                          # Robot state vector 1: arm destinations
                nn.Embedding(8, 8),
                nn.Flatten(),
                nn.Linear(8 * 2, self._h0),
                nn.ReLU(),
                nn.LayerNorm(self._h0)
            )
        ])

        self.chamber_joint = nn.Sequential(
            nn.Linear(5 * self._h0, self._h1),
            nn.ReLU(),
            nn.LayerNorm(self._h1),
        )

        self.robot_joint = nn.Sequential(
            nn.Linear(2 * self._h0, self._h1),
            nn.ReLU(),
            nn.LayerNorm(self._h1),
        )

        self.joint_decision = nn.Sequential(
            nn.Linear(2 * self._h1, self._h2),
            nn.ReLU(),
            nn.LayerNorm(self._h2),
            nn.Dropout(dropout_rate),
            nn.Linear(self._h2 // 2, action_dim)
        )

        self.shared_feature = nn.Sequential(
            nn.Linear(2 * self._h1, self._h2),
            nn.ReLU(),
            nn.LayerNorm(self._h2),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self._h2, action_dim)
        )

        self.value_stream = nn.Sequential(
            nn.Linear(self._h2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)


    def forward(self,
                chamber_states,
                robot_states,
                action_masks: Optional[np.ndarray]=None)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        assert chamber_states.size(1) == 5 and chamber_states.size(2) == cfg.numChambers, "Chamber input shape must be (batch_size, 4, 6)"
        assert robot_states.size(1) == 2 and robot_states.size(2) == 2, "Robot input shape must be (batch_size, 2, 2)"

        chamber_inputs = []
        for i, branch_i in enumerate(self.chamber_branch_nets):
            if i in (0, 3, 4):
                chamber_inputs.append(branch_i(chamber_states[:, i, :].long()))
            else:
                chamber_inputs.append(branch_i(chamber_states[:, i, :].float()))

        robot_inputs = [branch(robot_states[:, i, :].long())
                          for i, branch in enumerate(self.robot_branch_nets)]

        chamber_features = self.chamber_joint(torch.cat(chamber_inputs, dim=1))
        robot_features = self.robot_joint(torch.cat(robot_inputs, dim=1))

        if not self.use_dueling:
            q_values = self.joint_decision(torch.cat([chamber_features, robot_features], dim=1))
            return q_values, torch.zeros(q_values.size(0), 1, device=q_values.device), torch.zeros_like(q_values)


        combined_features = torch.cat([chamber_features, robot_features], dim=1)
        shared_features = self.shared_feature(combined_features)
        value = self.value_stream(shared_features)
        advantages = self.advantage_stream(shared_features)

        if action_masks is not None:
            if isinstance(action_masks, np.ndarray):
                action_masks = torch.as_tensor(action_masks, dtype=torch.bool, device=chamber_states.device)

            if action_masks.dim() == 1:
                action_masks = action_masks.unsqueeze(0)

            masked_advantages = advantages * action_masks
            num_feasible = action_masks.sum(dim=1, keepdim=True)
            mean_advantages = masked_advantages.sum(dim=1, keepdim=True) / num_feasible
        else:
            raise Exception("the action mask is required by deuling.")

        centered_advantages = advantages - mean_advantages
        q_values = value + centered_advantages

        if action_masks is not None:
            q_values = q_values.masked_fill(~action_masks, -1e8)

        return q_values, value, centered_advantages



class DQN:
    def __init__(self, method:str, hidden_dim: List[int], action_dim: int,
                 learn_rate: float, explore: float, gamma: float,
                 tnet_update: int, num_episodes:int, device: str):

        self.method = method
        self.__lr = learn_rate
        self._explore = explore
        self.__gamma  = gamma
        self._tnet_up = tnet_update

        self.device = torch.device(device)
        self._up_count = 0
        self._cur_episode = 0
        self.num_episodes = num_episodes
        self._ex_desc = 0.9995
        self._dropout = 0.12

        self.q_net = ChamberRobotNet(hidden_dim, action_dim, self._dropout, use_dueling=(self.method=="D3QN")).to(self.device)
        self.t_net = ChamberRobotNet(hidden_dim, action_dim, self._dropout, use_dueling=(self.method=="D3QN")).to(self.device)
        self.t_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=self.__lr, weight_decay=1e-4)


    def take_action(self, ep_greedy:bool, chamber_state, robot_state, feasibility):
        """ take an action using or not using the epislon-greedy """
        if not any(feasibility):
            raise Exception("No feasible action available (DeadLock)")

        if ep_greedy is True and np.random.random() < self._explore:
            return np.random.choice(np.nonzero(feasibility)[0])

        with torch.no_grad():
            chamber_state = torch.as_tensor(np.expand_dims(chamber_state, axis=0), dtype=torch.float32, device=self.device)
            robot_state   = torch.as_tensor(np.expand_dims(robot_state, axis=0),   dtype=torch.long,    device=self.device)

            q_values, _, _ = self.q_net(chamber_state, robot_state, feasibility)
            q_values = q_values[0]
            q_values[~torch.as_tensor(feasibility, dtype=torch.bool, device=self.device)] = float('-inf')
            return q_values.argmax().item()


    def update(self, transition_dict, episode_th):
        c_states  = torch.as_tensor(transition_dict["c_states"],  dtype=torch.float32,  device=self.device)
        r_states  = torch.as_tensor(transition_dict["r_states"],  dtype=torch.long,     device=self.device)
        action_ids= torch.as_tensor(transition_dict["action_ids"],dtype=torch.long,     device=self.device).unsqueeze(1)
        rewards   = torch.as_tensor(transition_dict["rewards"],   dtype=torch.float32,  device=self.device).unsqueeze(1)
        nc_states = torch.as_tensor(transition_dict["nc_states"], dtype=torch.float32,  device=self.device)
        nr_states = torch.as_tensor(transition_dict["nr_states"], dtype=torch.long,     device=self.device)
        done_flags= torch.as_tensor(transition_dict["done_flags"],dtype=torch.float32,  device=self.device).unsqueeze(1)
        a_masks   = torch.as_tensor(transition_dict["a_masks"], dtype=torch.bool, device=self.device)
        na_masks  = torch.as_tensor(transition_dict["na_masks"], dtype=torch.bool, device=self.device)
        effective_n = torch.as_tensor(transition_dict["effective_n"], dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q, _, _ = self.q_net(c_states, r_states, a_masks)
        current_q = current_q.gather(1, action_ids)

        with torch.no_grad():
            if self.method == "DQN":
                next_q, _, _ = self.t_net(nc_states, nr_states, na_masks)
                next_q = next_q.max(1)[0].unsqueeze(1)
            elif self.method == "D3QN":
                online_next_q, _, _ = self.q_net(nc_states, nr_states, na_masks)
                next_actions = online_next_q.max(1)[1].unsqueeze(1)
                target_next_q, _, _ = self.t_net(nc_states, nr_states, na_masks)
                next_q = target_next_q.gather(1, next_actions)
            else:
                raise Exception("unexpected method.")

            discount_n = torch.pow(
                torch.full_like(rewards, self.__gamma),
                effective_n
            )

            target_q = rewards + discount_n * next_q * (1 - done_flags)

        loss = tf.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        self.optimizer.step()

        self._up_count += 1
        if self._up_count == self._tnet_up:
            self._up_count = 0
            self.t_net.load_state_dict(self.q_net.state_dict())

        if self._cur_episode < episode_th :
            self._cur_episode = episode_th
            self._explore = max(self._explore * self._ex_desc, 0.025)

        if cfg.LR_UPDATE == "Cosine":
            lr_now = self._cosine_decay_lr(episode_th, lr_start=self.__lr, lr_min=1e-4, warmup_steps=750)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_now
        elif cfg.LR_UPDATE == "Constant":
            pass

        return loss.item()




    def _cosine_decay_lr(self, step: int, lr_start: float, lr_min: float, warmup_steps: int):
        if step < warmup_steps:
            return lr_start * (step / warmup_steps)

        progress = (step - warmup_steps) / float(self.num_episodes - warmup_steps)
        return lr_min + 0.5 * (lr_start - lr_min) * (1 + math.cos(math.pi * progress))


    def load_qnet_model(self, qnet_used: str) -> bool:
        """Load Q-network parameters from a file."""
        if os.path.exists(qnet_used):
            self.q_net.load_state_dict(torch.load(qnet_used, map_location=self.device))
            return True
        return False