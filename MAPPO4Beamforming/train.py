import os
import time
import torch
import pandas as pd
from datetime import datetime
from env import SatelliteEnv
from mappo import MAPPO  # 你需要提供 PyTorch 版本的 MAPPO 类
from utils import compute_gae  # 用之前转成 PyTorch 的版本
from torch.nn.utils import clip_grad_norm_

def run_training():
    # === Path Configuration ===
    system_config_path = 'data4DL/systemConfig.mat'
    embb_cell_sat_pairing_path = 'data4DL/eMBBCellSatPairing.mat'
    urllc_cell_sat_pairing_path = 'data4DL/URLLCCellSatPairing.mat'
    access_status_path = 'data4DL/accessStatus.mat'
    embb_demand_path = 'data4DL/eMBBDataDemand.mat'
    urllc_demand_path = 'data4DL/URLLCDataDemand.mat'
    channel_matrix_path = 'data4DL/largeScale'

    log_dir = 'logs'
    model_dir = 'saved_models'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'train_log_{timestamp}.csv')
    log_data = []

    # === Environment ===
    env = SatelliteEnv(system_config_path,
                       embb_cell_sat_pairing_path, urllc_cell_sat_pairing_path, access_status_path,
                       embb_demand_path, urllc_demand_path,
                       channel_matrix_path)

    K, N, B = env.K, env.N, env.B
    obs_dim = 3 * K + K * K

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = {
        'gamma': 0.99,
        'gae_lambda': 0.9,
        'clip_ratio': 0.1,
        'policy_coef': 1.0,
        'value_coef': 0.01,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'lr': 1e-3,
        'total_power': env.total_power,
        'device': device
    }

    mappo = MAPPO(n_agents=N, obs_dim=obs_dim, beam_count=B, args=args)

    for episode in range(1000):
        env.reset()

        obs_buf, act_buf, logp_buf, val_buf, rew_buf = [], [], [], [], []
        centralized_obs_buf = []
        queue_buf, satisfaction_buf, power_buf = [], [], []

        done = False
        step_count = 0
        start_time = time.time()
        all_action_means = []
        all_action_stds = []

        while not done:
            obs = env.get_obs()  # shape: [N, obs_dim]
            obs_tensor = obs.float().to(device) #  [N, obs_dim]

            actions, log_probs = mappo.act(obs_tensor)  # [N, B], [N]
            all_action_means.append(actions.mean().item())
            all_action_stds.append(actions.std().item())

            obs_buf.append(obs_tensor)
            centralized_obs_buf.append(obs_tensor.view(-1))  # Flatten for centralized critic: [N * obs_dim]
            act_buf.append(actions)
            logp_buf.append(log_probs)

            values = mappo.evaluate_value(centralized_obs_buf[-1].unsqueeze(0)).squeeze(0)  # [N]
            val_buf.append(values)

            reward, avg_queue, avg_satisfaction, power_penalty, done = env.step(actions.detach().cpu().numpy())

            rew_buf.append(torch.tensor(reward, dtype=torch.float32, device=device))  # [N]
            queue_buf.append(torch.tensor(avg_queue))
            satisfaction_buf.append(torch.tensor(avg_satisfaction))
            power_buf.append(torch.tensor(power_penalty))

            step_count += 1
            step_duration = time.time() - start_time
            print(f"[Step {env.t:03d}] Avg Queue Delta: {avg_queue * 100:.4f}%, "
                f"Avg Satisfaction: {avg_satisfaction * 100:.4f}%, "
                f"Avg Power Overuse: {power_penalty * 100:.4f}%, "
                f"Step Time: {step_duration / step_count:.4f}s")

        # === Bootstrap for last value ===
        last_obs = obs_buf[-1]  # [N, obs_dim]
        centralized_last_obs = last_obs.view(-1).unsqueeze(0)  # [1, N * obs_dim]
        last_val = mappo.evaluate_value(centralized_last_obs).squeeze(0)  # [N]
        val_buf.append(last_val)

        # === Stack data ===
        obs_tensor = torch.stack(obs_buf)  # [T, N, obs_dim]
        act_tensor = torch.stack(act_buf)  # [T, N, B]
        logp_tensor = torch.stack(logp_buf)  # [T, N]
        rew_tensor = torch.stack(rew_buf)  # [T, N]
        val_tensor = torch.stack(val_buf)  # [T+1, N]
        centralized_tensor = torch.stack(centralized_obs_buf)  # [T, N * obs_dim]

        # === Compute GAE & Returns ===
        adv_buf, ret_buf = compute_gae(rew_tensor, val_tensor, args['gamma'], args['gae_lambda'])  # [T, N]

        # === Flatten to [T * N, ...] for batch update ===
        T, N = adv_buf.shape
        obs_tensor = obs_tensor.view(T * N, -1)
        act_tensor = act_tensor.view(T * N, -1)
        logp_tensor = logp_tensor.view(-1)
        adv_tensor = adv_buf.view(-1)
        ret_tensor = ret_buf.view(-1)
        centralized_tensor = centralized_tensor.view(T * N, -1)

        # === Loss & Update ===
        loss, policy_loss, value_loss, entropy_loss = mappo.compute_loss(
            obs_tensor, centralized_tensor, act_tensor, adv_tensor, ret_tensor, logp_tensor
        )
        mappo.update(loss)

        # === Logging ===
        avg_reward = rew_tensor.mean().item()
        avg_queue = torch.stack(queue_buf).mean().item()
        avg_satisfaction = torch.stack(satisfaction_buf).mean().item()
        avg_power = torch.stack(power_buf).mean().item()

        print(f"========= EPISODE {episode} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =========")
        print(f"Loss: {loss.item():.4f} | Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f}, Entropy: {-entropy_loss.item():.4f}")
        print(f"Reward: {avg_reward:.4f}, Queue: {avg_queue * 100:.4f}%, Satisfaction: {avg_satisfaction * 100:.4f}%, Power: {avg_power * 100:.4f}%")

        log_data.append({
            'episode': episode,
            'avg_reward': avg_reward,
            'avg_queue': avg_queue,
            'min_satisfaction': avg_satisfaction,
            'power_penalty': avg_power,
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item()
        })

        value_preds = val_tensor[:-1]
        print(f"[DEBUG EPISODE {episode}] Critic value mean: {value_preds.mean().item():.4f}, "
              f"std: {value_preds.std().item():.4f}")
        print(f"[DEBUG] Action mean: {sum(all_action_means)/len(all_action_means):.4f}, "
              f"std: {sum(all_action_stds)/len(all_action_stds):.4f}")
        print(f"[DEBUG] Advantage mean: {adv_tensor.mean().item():.4f}, "
              f"std: {adv_tensor.std().item():.4f}\n")

    # === Save Model ===
    model_path = os.path.join(model_dir, f'mappo_ep{episode}_{timestamp}.pt')
    torch.save(mappo.policy.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    df_log = pd.DataFrame(log_data)
    df_log.to_csv(log_path, index=False)
    print(f"Log saved to {log_path}\n")


if __name__ == "__main__":
    run_training()
