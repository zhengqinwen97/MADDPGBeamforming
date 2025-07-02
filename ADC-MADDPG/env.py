import torch
import numpy as np
from collections import deque
from utils import compute_reward
from utils import ParseData

class SatelliteEnv:
    def __init__(self, system_config_path,
                 embb_cell_sat_pairing_path, urllc_cell_sat_pairing_path, access_status_path,
                 embb_demand_path, urllc_demand_path,
                 channel_matrix_path):
        self.parser = ParseData(system_config_path,
                                embb_cell_sat_pairing_path, urllc_cell_sat_pairing_path, access_status_path,
                                embb_demand_path, urllc_demand_path,
                                channel_matrix_path)
        self.parser.parse_all()
        self._initialize_environment()

    def _initialize_environment(self):
        config = self.parser.system_config

        # === System config ===
        self.K = config["cell_count"]
        self.N = config["satellite_count"]
        self.B = config["beam_count"]
        self.Nt = config["antenna_element_count"]
        # self.T = config["time_slot_count"]
        self.T = 100
        self.bandwidth = config["bandwidth"]
        self.noise_power = config["noise_power"]
        self.total_power = config["total_power"]
        self.tx_element_gain = config["tx_element_gain"]
        self.rx_gain = config["rx_gain"]

        # === Data tensors ===
        self.embb_demand = torch.tensor(self.parser.embb_demand / 20, dtype=torch.float32)
        self.urllc_arrivals = torch.tensor(self.parser.urllc_arrivals / 10, dtype=torch.float32)

        self.access_status = torch.tensor(self.parser.access_status, dtype=torch.bool)
        self.cell_sat_pairing_embb = torch.tensor(self.parser.cell_sat_pairing_embb, dtype=torch.bool)
        self.cell_sat_pairing_urllc = torch.tensor(self.parser.cell_sat_pairing_urllc, dtype=torch.bool)
        self.channel_matrix = torch.tensor(self.parser.channel_matrix, dtype=torch.float32)

        self.reset()

    def reset(self):
        self.total_reward = torch.zeros(24)
        self.t = 0
        self.queues = [[deque() for _ in range(self.N)] for _ in range(self.K)]

    # === Step function for RL interaction ===
    def step(self, urllc_actions):
        # === Process URLLC arrivals and queues ===
        queue_state_prev = self._aggregate_urllc_queue_lengths()
        self._update_urllc_queues()
        queue_state_mid = self._aggregate_urllc_queue_lengths()

        self.allocate_urllc_users()  # Your custom logic

        # --- Step 0: Initial allocation (uniform across paired cells) ---
        def normalize_non_negative(lst):
            # 步骤 1: 将负数替换为 0
            non_negative = [max(x + 1.0, 0) for x in lst]
            
            # 步骤 2: 归一化，使总和为 1
            total = sum(non_negative)
            
            if total == 0:
                # 如果全是 0，返回均匀分布防止除以零错误
                return [1.0 / len(non_negative)] * len(non_negative)
            
            normalized = [x / total for x in non_negative]
            return normalized
        
        urllc_actions = [normalize_non_negative(x)[:] for x in urllc_actions]
        urllc_actions = np.array([[x * self.total_power for x in xs] for xs in urllc_actions], dtype=np.float32)

        embb_power = self.allocate_embb_power(urllc_actions)  # Custom logic
        urllc_rate, embb_rate = self.compute_all_rates(urllc_actions, embb_power)

        self._serve_urllc_users(urllc_rate)
        queue_state_after = self._aggregate_urllc_queue_lengths()

        # === Compute delta (avoid divide by zero) ===
        eps = 1e-8
        queue_delta = queue_state_mid - queue_state_after

        reward, avg_queue_delta, avg_satisfaction, power_penalty = compute_reward(
            embb_rate=embb_rate,
            embb_demand=self.embb_demand,
            urllc_queues_delta=queue_delta,
            urllc_actions=torch.from_numpy(urllc_actions),
            total_power=self.total_power,
            cell_sat_pairing_embb=self.cell_sat_pairing_embb,
            t=self.t
        )

        self.total_reward += reward
        self.t += 1
        done = self.t >= self.T

        if done:
            return self.total_reward, avg_queue_delta, avg_satisfaction, power_penalty, done
        else:
            return torch.zeros(24), avg_queue_delta, avg_satisfaction, power_penalty, done # 取消即时奖励

    # === Generate agent observations ===
    @torch.no_grad()
    def get_obs(self):
        def min_max_normalize(t, eps=1e-8):
            denom = t.max() - t.min()
            if denom.item() < eps:
                # 如果最大值等于最小值，直接返回全 0 张量
                return torch.zeros_like(t)
            return (t - t.min()) / denom
        
        def normalize_tensor(t, dims, norm_type='minmax', eps=1e-8):
            """
            对张量 t 的指定维度 dims 进行归一化。

            参数：
                t (torch.Tensor): 输入张量
                dims (tuple or list): 要归一化的维度，如 (1, 2) 或 (-2, -1)
                norm_type (str): 'minmax' 或 'zscore'
                eps (float): 防止除零的小值

            返回：
                torch.Tensor: 归一化后的张量
            """
            if isinstance(dims, int):
                dims = (dims,)  # 确保是 tuple

            if norm_type == 'minmax':
                min_val = t.amin(dim=dims, keepdim=True)
                max_val = t.amax(dim=dims, keepdim=True)
                denom = (max_val - min_val).clamp(min=eps)  # 防止为 0
                return (t - min_val) / denom

            elif norm_type == 'zscore':
                mean = t.mean(dim=dims, keepdim=True)
                std = t.std(dim=dims, keepdim=True)
                denom = std.clamp(min=eps)  # 防止为 0
                return (t - mean) / denom

            else:
                raise ValueError(f"不支持的归一化方式: {norm_type}")

        # --- URLLC queue length matrix [K, N] ---
        queue_lengths = torch.tensor(
            [[len(self.queues[k][n]) for k in range(self.K)] for n in range(self.N)],
            dtype=torch.float32
        ).T  # shape: [K, N]

        # --- Channel gain snapshot [K, K, N] ---
        channel_slice = self.channel_matrix[:, :, :, self.t]
        channel_slice = torch.nan_to_num(
            channel_slice,
            nan=10 ** (-200.0 / 10)
        )

        # --- eMBB pairing mask [K, N] ---
        embb_pairing = self.cell_sat_pairing_embb[:, :, self.t].float()
        embb_pairing = torch.transpose(embb_pairing, 0, 1)


        # --- URLLC and eMBB demand ---
        urllc_demand = self.urllc_arrivals[:, :, self.t].float()  # [K, N]
        embb_demand = self.embb_demand.float().flatten()          # [K]
        queue_lengths = normalize_tensor(torch.transpose(queue_lengths, 0, 1), dims=(-1,))
        embb_demand = normalize_tensor(embb_demand, dims=(-1,))
        embb_demand = embb_demand.unsqueeze(0).repeat(queue_lengths.shape[0], 1)  # shape: [N, K]
        embb_demand = embb_demand * embb_pairing
        urllc_demand = normalize_tensor(torch.transpose(urllc_demand, 0, 1), dims=(-1,))
        channel_slice = normalize_tensor(torch.log(channel_slice.permute(2, 0, 1)), dims=(-2, -1))

        one_hot = torch.zeros(self.N, self.T)
        one_hot[:, self.t - 1] = 1.0  # 使用广播机制

        # combine tensors
        channel_flat = channel_slice.mean(dim=2)  # -> [24, 280]
        # channel_flat = channel_slice.view(24, -1)
        combined_tensor = torch.cat([
            one_hot,
            queue_lengths,
            # embb_demand,
            urllc_demand,
            channel_flat
        ], dim=1)
        combined_np = combined_tensor.numpy()
        nested_list = [
            [float(np.array(val)) for val in row]
            for row in combined_np
        ]
        return nested_list

        # # --- Normalize inputs ---
        # q_max = queue_lengths.max() + 1e-6
        # urllc_max = urllc_demand.max() + 1e-6
        # embb_max = embb_demand.max() + 1e-6
        # ch_max = channel_slice.max() + 1e-6

        # --- Build observation for each satellite ---
        # observations = []
        # for n in range(self.N):
        #     obs = torch.cat([
        #         queue_lengths[:, n] / q_max,                           # [K]
        #         channel_slice[:, :, n].flatten() / ch_max,            # [K*K]
        #         embb_pairing[:, n] * (embb_demand / embb_max),        # [K]
        #         urllc_demand[:, n] / urllc_max                        # [K]
        #     ])
        #     observations.append(obs)

        # return torch.stack(observations, dim=0)  # [N, obs_dim]

    # === URLLC user selection per satellite ===
    def allocate_urllc_users(self):
        self.urllc_user_map = [[-1] * self.B for _ in range(self.N)]  # shape [N][B]

        for n in range(self.N):
            arrivals = {
                k: self.queues[k][n][0] if self.queues[k][n] else 0.0
                for k in range(self.K)
                if self.is_urllc_paired(k, n) and len(self.queues[k][n]) > 0
            }
            sorted_users = sorted(arrivals.items(), key=lambda x: x[1])  # sort by arrival size

            for b, (k, _) in enumerate(sorted_users[:self.B]):
                self.urllc_user_map[n][b] = k

    # === Allocate eMBB power for each satellite ===
    def allocate_embb_power_single_satellite(self, urllc_actions, embb_power, n):
        paired_indices = torch.nonzero(self.cell_sat_pairing_embb[:, n, self.t], as_tuple=False).flatten()
        power_budget = (self.total_power - urllc_actions[n].sum()).item()

        if len(paired_indices) > 0 and power_budget > 1e-6:
            signal = np.zeros(len(paired_indices), dtype=np.float32)
            interference_noise = np.zeros(len(paired_indices), dtype=np.float32)
            channel_gain = np.zeros(len(paired_indices), dtype=np.float32)

            for i, k in enumerate(paired_indices.cpu().numpy()):
                _, signal[i], interference_noise[i], channel_gain[i] = self._compute_rate_tensorized(
                    k, n, embb_power[n, k].item(), urllc_actions, embb_power
                )

            noise_level = interference_noise / (channel_gain + 1e-6)
            sorted_indices = np.argsort(noise_level)
            sorted_noise = noise_level[sorted_indices]
            K = len(sorted_noise)
            cumsum_noise = np.cumsum(sorted_noise)

            for i in range(K - 1, -1, -1):
                mu = (power_budget + cumsum_noise[i]) / (i + 1)
                p = mu - sorted_noise[:i + 1]
                if np.all(p > 0):
                    break

            p_opt = np.maximum(mu - noise_level, 0)
            embb_power[n, paired_indices] = torch.tensor(p_opt, dtype=torch.float32)

        return embb_power[n, paired_indices]

    # === Allocate eMBB power based on URLLC actions ===
    def allocate_embb_power(self, urllc_actions, max_outer_iter=5, power_tol=1.0):
        """
        Allocate eMBB power for all satellites based on URLLC actions.
        Iteratively optimize until convergence or max iterations.
        """
        embb_power = torch.zeros((self.N, self.K), dtype=torch.float32)
        for n in range(self.N):
            paired_indices = torch.nonzero(self.cell_sat_pairing_embb[:, n, self.t], as_tuple=False).flatten()
            if len(paired_indices) > 0:
                power_budget = torch.clamp(
                    torch.tensor(self.total_power - urllc_actions[n].sum().item(), dtype=torch.float32),
                    min=0.0
                )
                embb_power[n, paired_indices] = power_budget / len(paired_indices)

        # --- Step 1: Iterative refinement ---
        for _ in range(max_outer_iter):
            embb_power_prev = embb_power.clone()

            for n in range(self.N):
                paired_indices = torch.nonzero(self.cell_sat_pairing_embb[:, n, self.t], as_tuple=False).flatten()
                if len(paired_indices) > 0:
                    embb_power[n, paired_indices] = self.allocate_embb_power_single_satellite(
                        urllc_actions, embb_power, n
                    )

            # --- Convergence check ---
            diff = torch.norm(embb_power - embb_power_prev)
            if diff.item() < power_tol:
                break

        return embb_power

    def compute_all_rates(self, urllc_actions, embb_power):
        """
        Compute URLLC and eMBB data rates for all users across satellites.
        Inputs:
            - urllc_actions: [N, B]
            - embb_power: [N, K]
        Outputs:
            - urllc_rate: [N, B]
            - embb_rate: [N, K]
        """
        urllc_rate = torch.zeros((self.N, self.B), dtype=torch.float32)
        embb_rate = torch.zeros((self.N, self.K), dtype=torch.float32)

        # === Compute URLLC rates ===
        for n in range(self.N):
            for b in range(self.B):
                k = self.urllc_user_map[n][b]
                if k != -1:
                    power = urllc_actions[n, b]
                    rate, _, _, _ = self._compute_rate_tensorized(k, n, power, urllc_actions, embb_power)
                    urllc_rate[n, b] = rate

        # === Compute eMBB rates ===
        for n in range(self.N):
            embb_indices = torch.nonzero(self.cell_sat_pairing_embb[:, n, self.t], as_tuple=False).flatten()
            for k in embb_indices:
                power = embb_power[n, k]
                rate, _, _, _ = self._compute_rate_tensorized(k.item(), n, power, urllc_actions, embb_power)
                embb_rate[n, k] = rate

        return urllc_rate, embb_rate

    # === Compute single-user rate using beamforming and channel state ===
    def _compute_rate_tensorized(self, k, n, power, urllc_actions, embb_power):
        channel_gain = self.channel_matrix[k, k, n, self.t] * self.tx_element_gain * self.rx_gain
        signal = power * channel_gain

        # --- Interfering URLLC ---
        urllc_user_map_t = torch.tensor(self.urllc_user_map, dtype=torch.long)
        access_status_t = self.access_status[k, :, self.t]  # [N]
        k_intf = urllc_user_map_t  # [N, B]

        valid_mask = (k_intf != -1) & (torch.arange(self.N).unsqueeze(1) == n) & (k_intf != k) & access_status_t.unsqueeze(1).expand(self.N, self.B)
        valid_indices = torch.nonzero(valid_mask)

        if valid_indices.shape[0] > 0:
            valid_k_intf = k_intf[valid_mask]
            m_idx, b_idx = valid_indices[:, 0], valid_indices[:, 1]
            k_intf_idx = valid_k_intf
            p_intf = urllc_actions[m_idx, b_idx]
            w_gain = self.channel_matrix[k, k_intf_idx, m_idx, self.t] * self.tx_element_gain * self.rx_gain
            p_intf = torch.as_tensor(p_intf, dtype=torch.float32, device=w_gain.device)
            interference_urllc = torch.sum(p_intf * w_gain)
        else:
            interference_urllc = torch.tensor(0.0, dtype=torch.float32)

        # --- Interfering eMBB ---
        cell_sat_pairing_t = self.cell_sat_pairing_embb[:, :, self.t]  # [K, N]
        k_mat = torch.arange(self.K).unsqueeze(1).expand(self.K, self.N)
        n_mat = torch.full((self.K, self.N), n, dtype=torch.long)
        access_status_t = self.access_status[k, :, self.t]  # [N]

        valid_mask = (cell_sat_pairing_t == 1) & (k_mat != k) & (n_mat == n) & access_status_t.unsqueeze(0).expand(self.K, self.N)
        valid_indices = torch.nonzero(valid_mask)

        if valid_indices.shape[0] > 0:
            k_intf_idx = valid_indices[:, 0]
            m_idx = valid_indices[:, 1]
            p_intf = embb_power[m_idx, k_intf_idx]
            w_gain = self.channel_matrix[k, k_intf_idx, m_idx, self.t] * self.tx_element_gain * self.rx_gain
            interference_embb = torch.sum(p_intf * w_gain)
        else:
            interference_embb = torch.tensor(0.0, dtype=torch.float32)

        # === Total interference + noise
        interference_noise = self.noise_power + interference_urllc + interference_embb

        return self._shannon_rate(signal, interference_noise), signal, interference_noise, channel_gain


    # === Serve URLLC queues based on available rate ===
    def _serve_urllc_users(self, urllc_rate):
        for n in range(self.N):
            for b in range(self.B):
                k = self.urllc_user_map[n][b]
                if k == -1:
                    continue

                rate = urllc_rate[n, b].item()
                queue = self.queues[k][n]
                if not queue:
                    continue

                for i in range(len(queue)):
                    tstamp, volume = queue[i]
                    if rate >= volume:
                        queue[i] = (tstamp, 0)
                        rate -= volume
                    else:
                        queue[i] = (tstamp, volume - rate)
                        break

                while queue and queue[0][1] == 0:
                    queue.popleft()


    # === Process URLLC traffic arrivals ===
    def _update_urllc_queues(self):
        arrivals = self.urllc_arrivals[:, :, self.t]  # [K, N]
        for k in range(self.K):
            for n in range(self.N):
                if arrivals[k, n] > 0:
                    self.queues[k][n].append((self.t, arrivals[k, n].item()))

    # === Aggregate total queue lengths (per satellite) ===
    def _aggregate_urllc_queue_lengths(self):
        return torch.tensor(
            [[sum(v for _, v in self.queues[k][n]) for n in range(self.N)] for k in range(self.K)],
            dtype=torch.float32
        ).sum(dim=0)  # shape: [N]


    # === Shannon rate computation ===
    def _shannon_rate(self, signal, interference_noise):
        sinr = signal / (interference_noise + 1e-8)
        return self.bandwidth * torch.log2(1 + sinr)

    # === Query eMBB pairing status ===
    def is_embb_paired(self, k, n):
        return self.cell_sat_pairing_embb[k, n, self.t].item()

    # === Query URLLC pairing status ===
    def is_urllc_paired(self, k, n):
        return self.cell_sat_pairing_urllc[k, n, self.t].item()