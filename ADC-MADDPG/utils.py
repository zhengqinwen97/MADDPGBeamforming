import torch
from scipy.io import loadmat
from scipy.constants import Boltzmann
import numpy as np
import os


# === Reward Function ===
def compute_reward(
    embb_rate, embb_demand, urllc_queues_delta, urllc_actions,
    total_power, cell_sat_pairing_embb, t,
    alpha=1, beta=1000
):
    # --- Compute per-agent eMBB satisfaction rate ---
    valid_mask = cell_sat_pairing_embb[:, :, t] == 1
    valid_mask = valid_mask.transpose(0, 1)  # Transpose [cell, sat] -> [sat, cell]

    # Avoid division by zero
    satisfaction = embb_rate * valid_mask.float() / (embb_demand + 1e-8)
    # agent_satisfaction = torch.sum(satisfaction)
    satisfaction = torch.clamp(satisfaction, min=0.0, max=1.0)
    agent_satisfaction = torch.mean(satisfaction[valid_mask==True])

    # --- Per-agent URLLC queue delta ---
    urllc_queues_delta = torch.nan_to_num(urllc_queues_delta, nan=0.0)

    # --- Per-agent power penalty ---
    power_used = torch.sum(urllc_actions, dim=1)
    overuse = torch.clamp(power_used - total_power, min=0.0)
    underuse = torch.clamp(-power_used, min=0.0)
    agent_power_penalty = (overuse + underuse) / (total_power + 1e-8)

    # --- Final reward per agent ---
    reward = agent_satisfaction + alpha * (-urllc_queues_delta) - beta * agent_power_penalty

    # --- Log stats ---
    avg_queue_delta = torch.mean(-urllc_queues_delta)
    avg_satisfaction = agent_satisfaction  # Already scalar
    power_penalty = torch.mean(agent_power_penalty)

    return reward, avg_queue_delta, avg_satisfaction, power_penalty


# === MATLAB Dataset Parser ===
class ParseData:
    def __init__(self, system_config_path,
                 embb_cell_sat_pairing_path, urllc_cell_sat_pairing_path, access_status_path,
                 embb_demand_path, urllc_demand_path,
                 channel_matrix_path):
        # --- File path setup ---
        self.system_config_path = system_config_path
        self.embb_cell_sat_pairing_path = embb_cell_sat_pairing_path
        self.urllc_cell_sat_pairing_path = urllc_cell_sat_pairing_path
        self.access_status_path = access_status_path
        self.embb_demand_path = embb_demand_path
        self.urllc_demand_path = urllc_demand_path
        self.channel_matrix_path = channel_matrix_path

        # --- Data containers ---
        self.system_config = None
        self.embb_demand = None
        self.urllc_arrivals = None
        self.channel_matrix = None
        self.access_status = None
        self.cell_sat_pairing_embb = None
        self.cell_sat_pairing_urllc = None

    # === Parse system configuration ===
    def parse_system_config(self):
        mat = loadmat(self.system_config_path)["systemConfig"]

        # --- Extract scalar system parameters ---
        system_config = {
            "beam_count": int(mat["BeamCount"][0, 0].item()),
            "total_power": 10 ** (float(5+mat["TotalPower"][0, 0].item()) / 10),
            "bandwidth": float(mat["Bandwidth"][0, 0].item()),
            "tx_element_gain": 10 ** (float(mat["TxElementGain"][0, 0].item()) / 10),
            "antenna_element_count": int(mat["TxAntennaElementCount"][0, 0].item()),
            "rx_gain": 10 ** (float(mat["RxGain"][0, 0].item()) / 10),
            "time_slot_length": float(mat["TimeSlotLength"][0, 0].item()),
            "time_slot_count": int(mat["TimeSlotCount"][0, 0].item()),
            "noise_temperature": float(mat["NoiseTemperature"][0, 0].item()),
            "satellite_count": int(mat["SatelliteCount"][0, 0].item()),
            "cell_count": int(mat["CellCount"][0, 0].item()),
        }

        # --- Compute thermal noise power ---
        system_config["noise_power"] = (
            system_config["bandwidth"] * system_config["noise_temperature"] * Boltzmann
        )

        self.system_config = system_config

    # === Parse cell-satellite pairing and access status ===
    def parse_cell_sat_pairing(self):
        self.cell_sat_pairing_embb = loadmat(self.embb_cell_sat_pairing_path)["eMBBCellSatPairing"]
        self.cell_sat_pairing_urllc = loadmat(self.urllc_cell_sat_pairing_path)["URLLCCellSatPairing"]
        self.access_status = loadmat(self.access_status_path)["accessStatus"]

    # === Parse eMBB and URLLC demand ===
    def parse_demand(self):
        # --- Load eMBB demand ---
        self.embb_demand = loadmat(self.embb_demand_path)["eMBBDataDemand"]

        # --- Load and map URLLC arrivals to paired satellites ---
        demand_urllc = loadmat(self.urllc_demand_path)["URLLCDataDemand"]
        urllc_arrivals = np.zeros((
            self.system_config["cell_count"],
            self.system_config["satellite_count"],
            self.system_config["time_slot_count"]
        ))
        cell_sat_pairing_urllc = loadmat(self.urllc_cell_sat_pairing_path)["URLLCCellSatPairing"]

        for t in range(self.system_config["time_slot_count"]):
            for i in range(self.system_config["cell_count"]):
                sat_idx = np.where(cell_sat_pairing_urllc[i, :, t] == 1)[0]
                if len(sat_idx) == 1:
                    urllc_arrivals[i, sat_idx[0], t] = demand_urllc[i, t]

        self.urllc_arrivals = urllc_arrivals

    # === Parse channel matrix ===
    def parse_channel_matrix(self):
        channel_matrix = np.zeros((
            self.system_config["cell_count"],
            self.system_config["cell_count"],
            self.system_config["satellite_count"],
            self.system_config["time_slot_count"]
        ))
        for i in range(self.system_config["satellite_count"]):
            channel_matrix_slice_path = os.path.join(self.channel_matrix_path, f'channelGain{i+1}.mat')
            channel_matrix_slice = loadmat(channel_matrix_slice_path)["channelGain"]
            channel_matrix[:, :, i, :] = 10**(channel_matrix_slice/10)  # Convert dB to linear scale

        self.channel_matrix = channel_matrix

    # === Run all parsers in correct order ===
    def parse_all(self):
        self.parse_system_config()
        self.parse_cell_sat_pairing()
        self.parse_demand()
        self.parse_channel_matrix()
