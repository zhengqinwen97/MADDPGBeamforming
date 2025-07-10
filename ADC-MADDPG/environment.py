import numpy as np
from env import SatelliteEnv


class BEAMFORMINGENV:
    def __init__(self):
        # === Path Configuration ===
        system_config_path = './data4DL/systemConfig.mat'
        embb_cell_sat_pairing_path = './data4DL/eMBBCellSatPairing.mat'
        urllc_cell_sat_pairing_path = './data4DL/URLLCCellSatPairing.mat'
        access_status_path = './data4DL/accessStatus.mat'
        embb_demand_path = './data4DL/eMBBDataDemand.mat'
        urllc_demand_path = './data4DL/URLLCDataDemand.mat'
        channel_matrix_path = './data4DL/largeScale'

        # === Environment ===
        self.env = SatelliteEnv(system_config_path,
                        embb_cell_sat_pairing_path, urllc_cell_sat_pairing_path, access_status_path,
                        embb_demand_path, urllc_demand_path,
                        channel_matrix_path)
        self.n = self.env.N
        self.T = self.env.T
        self.cell_count = self.env.K
        self.beam_count = self.env.B
        self.observation_space = [self.cell_count * 4] * self.n
        # self.observation_space = self.n
        self.action_space = [self.beam_count] * self.n
        self.node_feat_dim = 3
        self.edge_feat_dim = 1

    def reset(self):
        self.env.reset()
        return self.env.get_obs()

    def step(self, action_n):
        reward, avg_queue_delta, avg_satisfaction, done = self.env.step(action_n)
        cur_obs = self.env.get_obs()
        
        cur_obs = [[np.float32(x) for x in xs] for xs in cur_obs]
        reward = [x for x in reward.numpy()]
        done = [done] * len(reward)

        return cur_obs, reward, avg_queue_delta, avg_satisfaction, done

def make_env_ax():
    return BEAMFORMINGENV()