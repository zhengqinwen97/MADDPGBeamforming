import torch
from torch_geometric.data import Data
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
        self.embb_demand = torch.tensor(self.parser.embb_demand / 10, dtype=torch.float32)
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

        # --- Step 0: Initial allocation (uniform across paired cells) ---
        urllc_power = self.total_power * np.array(urllc_actions)

        embb_power = self.allocate_embb_power(urllc_power)  # Custom logic
        urllc_rate, embb_rate = self.compute_all_rates(urllc_power, embb_power)

        self._serve_urllc_users(urllc_rate)
        queue_state_after = self._aggregate_urllc_queue_lengths()

        # === Compute delta (avoid divide by zero) ===
        eps = 1e-8
        queue_delta = (queue_state_mid - queue_state_after) / (queue_state_mid + eps)

        reward, avg_queue_delta, avg_satisfaction = compute_reward(
            embb_rate=embb_rate,
            embb_demand=self.embb_demand,
            urllc_queues_delta=queue_delta,
            cell_sat_pairing_embb=self.cell_sat_pairing_embb,
            cell_sat_pairing_urllc=self.cell_sat_pairing_urllc,
            t=self.t
        )

        self.total_reward += reward
        self.t += 1
        done = self.t >= self.T

        if done:
            return self.total_reward, avg_queue_delta, avg_satisfaction, done
        else:
            return torch.zeros(24), avg_queue_delta, avg_satisfaction, done

    # === Generate agent observations ===
    @torch.no_grad()
    def get_obs(self):
        # Normalize URLLC, eMBB demands and channel gain
        max_urllc_demand = self.urllc_arrivals.max().item()
        max_embb_demand =self.embb_demand.max().item()
        urllc_demand = self.urllc_arrivals[:, :, self.t].numpy() / max_urllc_demand
        embb_demand = self.embb_demand.numpy() / max_embb_demand
        channel_gain = 10*np.log10(self.channel_matrix[:, :, :, self.t].numpy()) / 100

        # Get and normalize queue state
        queue_state = self._aggregate_urllc_queue_lengths()/ max_urllc_demand

        node_feats_dict = {}
        for n in range(self.N):
            each_sat_service_cell_idx = np.where((self.cell_sat_pairing_embb[:, n, self.t] + self.urllc_arrivals[:, n, self.t]>0) > 0)[0]
            zero_to_pad = self.B - len(each_sat_service_cell_idx)
            
            each_sat_embb_demand = embb_demand[0, each_sat_service_cell_idx].tolist() + zero_to_pad * [0.0]
            each_sat_urllc_demand = urllc_demand[each_sat_service_cell_idx, n].tolist() + zero_to_pad * [0.0]
            each_sat_queue_lengths = queue_state[each_sat_service_cell_idx, n].tolist() + zero_to_pad * [0.0]
            each_sat_service_link_gain = channel_gain[:, :, n].diagonal()[each_sat_service_cell_idx].tolist() + zero_to_pad * [-2.0]

            each_sat_node_feats_dict = {
                'embb_demand': np.array(each_sat_embb_demand),
                'urllc_demand': np.array(each_sat_urllc_demand) + np.array(each_sat_queue_lengths),
                'service_link_gain': np.array(each_sat_service_link_gain)
            }
            node_feats_dict[n] = each_sat_node_feats_dict

        edge_list = []
        edge_feats = []
        for n in range(self.N):
            each_sat_edge_list = []
            each_sat_edge_feats = []
            each_sat_service_cell_idx = np.where((self.cell_sat_pairing_embb[:, n, self.t] + self.urllc_arrivals[:, n, self.t]>0) > 0)[0]
            for i in range(len(each_sat_service_cell_idx)):
                for j in range(i + 1, len(each_sat_service_cell_idx)):
                    each_sat_edge_list.append([i, j])
                    each_sat_edge_list.append([j, i])
                    src = each_sat_service_cell_idx[i].item()
                    tgt = each_sat_service_cell_idx[j].item()
                    each_sat_edge_feats.append(channel_gain[src, tgt, n].item())
                    each_sat_edge_feats.append(channel_gain[tgt, src, n].item())
            edge_list.append(each_sat_edge_list)
            edge_feats.append(each_sat_edge_feats)
        
        nested_graph = []
        for n in range(self.N):
            node_feats = np.stack([
                node_feats_dict[n]['embb_demand'],
                node_feats_dict[n]['urllc_demand'],
                node_feats_dict[n]['service_link_gain']
            ], axis=1).tolist()
            nested_graph.append((node_feats, edge_list[n], edge_feats[n]))
        
        return nested_graph

    # === Allocate eMBB power for each satellite ===
    def allocate_embb_power_single_satellite(self, embb_power_budget, embb_link_gain):
        # Equal Allocation for code test
        embb_cell_to_serve_count = len(embb_link_gain)
        avg_power = embb_power_budget / embb_cell_to_serve_count
        embb_power_each_link = torch.full((embb_cell_to_serve_count,), avg_power)
        return embb_power_each_link

    # === Allocate eMBB power based on URLLC actions ===
    def allocate_embb_power(self, urllc_actions):
        embb_power = torch.zeros((self.N, self.B), dtype=torch.float32)
        for n in range(self.N):
            embb_power_budget = torch.tensor(self.total_power - urllc_actions[n].sum())
            if embb_power_budget > 0:
                embb_cell_to_serve_idx = torch.nonzero(self.cell_sat_pairing_embb[:, n, self.t], as_tuple=False).flatten()
                if len(torch.nonzero(self.cell_sat_pairing_embb[:, n, self.t], as_tuple=False).flatten()) > 0:
                    embb_link_gain = self.channel_matrix[embb_cell_to_serve_idx, embb_cell_to_serve_idx, n, self.t] * self.tx_element_gain * self.rx_gain
                    embb_power_each_link = self.allocate_embb_power_single_satellite(embb_power_budget, embb_link_gain)
                    embb_power[n, :len(embb_cell_to_serve_idx)] = embb_power_each_link

        return embb_power.numpy()

    def compute_all_rates(self, urllc_power, embb_power):
        urllc_signal_power = torch.zeros((self.N, self.B), dtype=torch.float32)
        urllc_interference_power = torch.zeros((self.N, self.B), dtype=torch.float32)
        embb_signal_power = torch.zeros((self.N, self.B), dtype=torch.float32)
        embb_interference_power = torch.zeros((self.N, self.B), dtype=torch.float32)
        
        for n in range(self.N):
            channel_gain_matrix = self.channel_matrix[:, :, n, self.t] * self.tx_element_gain * self.rx_gain
            embb_cell_to_serve_idx = torch.nonzero(self.cell_sat_pairing_embb[:, n, self.t], as_tuple=False).flatten()
            urllc_cell_to_serve_idx = torch.nonzero(self.cell_sat_pairing_urllc[:, n, self.t], as_tuple=False).flatten()

            # === Compute URLLC rates ===
            for i in range(len(urllc_cell_to_serve_idx)):
                service_urllc_cell_idx = urllc_cell_to_serve_idx[i]
                urllc_signal_power[n, i] = channel_gain_matrix[service_urllc_cell_idx, service_urllc_cell_idx] * urllc_power[n, i]
                # Interference from URLLC links
                for j in range(i + 1, len(urllc_cell_to_serve_idx)):
                    interference_urllc_cell_idx = urllc_cell_to_serve_idx[j]
                    urllc_interference_power[n, i] += channel_gain_matrix[service_urllc_cell_idx, interference_urllc_cell_idx] * urllc_power[n, j]
                # Interference from eMBB links
                for j in range(len(embb_cell_to_serve_idx)):
                    interference_embb_cell_idx = embb_cell_to_serve_idx[j]
                    urllc_interference_power[n, i] += channel_gain_matrix[service_urllc_cell_idx, interference_embb_cell_idx] * embb_power[n, j]
                urllc_interference_power[n, i] += self.noise_power

            # === Compute eMBB rates ===
            for i in range(len(embb_cell_to_serve_idx)):
                service_embb_cell_idx = embb_cell_to_serve_idx[i]
                embb_signal_power[n, i] = channel_gain_matrix[service_embb_cell_idx, service_embb_cell_idx] * embb_power[n, i]
                # Interference from URLLC links
                for j in range(len(urllc_cell_to_serve_idx)):
                    interference_urllc_cell_idx = urllc_cell_to_serve_idx[j]
                    embb_interference_power[n, i] += channel_gain_matrix[service_embb_cell_idx, interference_urllc_cell_idx] * urllc_power[n, j]
                # Interference from eMBB links
                for j in range(i + 1, len(embb_cell_to_serve_idx)):
                    interference_embb_cell_idx = embb_cell_to_serve_idx[j]
                    embb_interference_power[n, i] += channel_gain_matrix[service_embb_cell_idx, interference_embb_cell_idx] * embb_power[n, j]
                embb_interference_power[n, i] += self.noise_power

        urllc_rate = self._shannon_rate(urllc_signal_power, urllc_interference_power)    
        embb_rate = self._shannon_rate(embb_signal_power, embb_interference_power)  

        return urllc_rate, embb_rate

    # === Serve URLLC queues based on available rate ===
    def _serve_urllc_users(self, urllc_rate):
        for n in range(self.N):
            urllc_cell_to_serve_idx = torch.nonzero(self.cell_sat_pairing_urllc[:, n, self.t], as_tuple=False).flatten()
            for j, each_urllc_cell_to_serve_idx in enumerate(urllc_cell_to_serve_idx):
                rate = urllc_rate[n, j].item()
                queue = self.queues[each_urllc_cell_to_serve_idx][n]
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

    # === Aggregate total queue lengths ===
    def _aggregate_urllc_queue_lengths(self):
        return torch.tensor(
            [[sum(v for _, v in self.queues[k][n]) for n in range(self.N)] for k in range(self.K)],
            dtype=torch.float32
        )

    # === Shannon rate computation ===
    def _shannon_rate(self, signal, interference_noise):
        sinr = torch.zeros_like(signal)
        zero_interference_noise_mask = interference_noise > 0
        sinr[zero_interference_noise_mask] = signal[zero_interference_noise_mask] / interference_noise[zero_interference_noise_mask]

        return self.bandwidth * torch.log2(1 + sinr)