import os
import time
import copy
import torch
import torch.optim as optim
import pandas as pd
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from environment import *
from model import openai_actor, openai_critic

global double_q_delay_fre, double_q_delay_cnt
double_q_delay_fre = 2
double_q_delay_cnt = 1

def print_with_timestamp(message):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S") + f".{now.microsecond // 1000:03d}"
    print(f"[{timestamp}] {message}")

def get_trainers_mix_ax(num_actors, node_feat_dim, edge_feat_dim, num_nodes, action_dim, arglist):
    actors_cur = [openai_actor(node_feat_dim, edge_feat_dim, num_nodes, action_dim, arglist).to(arglist.device) for _ in range(num_actors)]
    critics_cur = [openai_critic(node_feat_dim, edge_feat_dim, num_actors, num_nodes, action_dim, arglist).to(arglist.device) for _ in range(num_actors)]

    actors_tar = [openai_actor(node_feat_dim, edge_feat_dim, num_nodes, action_dim, arglist).to(arglist.device) for _ in range(num_actors)]
    critics_tar = [openai_critic(node_feat_dim, edge_feat_dim, num_actors, num_nodes, action_dim, arglist).to(arglist.device) for _ in range(num_actors)]

    optimizers_a = [optim.Adam(actor.parameters(), arglist.lr_a) for actor in actors_cur]
    optimizers_c = [optim.Adam(critic.parameters(), arglist.lr_c) for critic in critics_cur]

    actors_tar = update_trainers(actors_cur, actors_tar, 1.0)  # update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0)  # update the target par using the cur

    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c

def update_trainers(agents_cur, agents_tar, tao):
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key] * tao + \
                                (1 - tao) * state_dict_t[key]
        agent_t.load_state_dict(state_dict_t)
    return agents_tar

def agents_train_mix_ax(arglist, game_step, update_cnt, memory, obs_size, action_size,
                        actors_cur, actors_tar, critics_cur, critics_tar,
                        optimizers_a, optimizers_c, writer, env, rew_n, run_type, random=True):

    if game_step > arglist.learning_start_step and \
       (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:

        if update_cnt == 0:
            print('\r=Start training...' + ' ' * 100)
        update_cnt += 1

        # Sample from memory
        _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory.sample(arglist.batch_size)
        n_agents = len(actors_cur)

        action_batch = torch.tensor(_action_n, dtype=torch.float, device=arglist.device)
        rew_batch = torch.tensor(_rew_n, dtype=torch.float, device=arglist.device)
        done_mask = torch.tensor(1 - _done_n.astype(int), dtype=torch.float, device=arglist.device)

        # Transpose: [batch][agent] -> [agent][batch]
        obs_n_o_per_agent = list(map(list, zip(*_obs_n_o)))
        obs_n_n_per_agent = list(map(list, zip(*_obs_n_n)))

        obs_n_o = [
            [(torch.tensor(x, dtype=torch.float, device=arglist.device),
              torch.tensor(edge_index, dtype=torch.int64, device=arglist.device).t().contiguous(),
              torch.tensor(edge_attr, dtype=torch.float, device=arglist.device).view(-1, 1)
              ) for (x, edge_index, edge_attr) in agent_obs ]
            for agent_obs in obs_n_o_per_agent
        ]

        obs_n_n = [
            [(torch.tensor(x, dtype=torch.float, device=arglist.device),
              torch.tensor(edge_index, dtype=torch.int64, device=arglist.device).t().contiguous(),
              torch.tensor(edge_attr, dtype=torch.float, device=arglist.device).view(-1, 1)
              ) for (x, edge_index, edge_attr) in agent_obs
            ]
            for agent_obs in obs_n_n_per_agent
        ]

        total_critic_loss = 0.0
        total_actor_loss = 0.0

        for agent_idx in range(n_agents):
            actor_c = actors_cur[agent_idx]
            actor_t = actors_tar[agent_idx]
            critic_c = critics_cur[agent_idx]
            critic_t = critics_tar[agent_idx]
            opt_a = optimizers_a[agent_idx]
            opt_c = optimizers_c[agent_idx]

            # ========== Critic Update ==========
            q_targets = []
            with torch.no_grad():
                for b in range(arglist.batch_size):
                    next_graphs = [obs_n_n[i][b] for i in range(n_agents)]
                    next_actions = torch.cat([
                        actors_tar[i](*obs_n_n[i][b]).unsqueeze(0)
                        for i in range(n_agents)
                    ], dim=-1)
                    q = critic_t(next_graphs, next_actions)
                    q_targets.append(q)
                q_targets = torch.cat(q_targets, dim=0).squeeze()

            # target_value = rew_batch[:, agent_idx] + arglist.gamma * done_mask[:, agent_idx] * q_targets
            target_value = rew_batch[:, agent_idx] + arglist.gamma * q_targets

            q_vals = []
            for b in range(arglist.batch_size):
                current_graphs = [obs_n_o[i][b] for i in range(n_agents)]
                act = action_batch[b].unsqueeze(0)
                q = critic_c(current_graphs, act)
                q_vals.append(q)
            q_vals = torch.cat(q_vals, dim=0).squeeze()

            loss_c = torch.nn.MSELoss()(q_vals, target_value.detach())
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

            # ========== Actor Update ==========
            actor_loss = 0.0
            for b in range(arglist.batch_size):
                new_action = action_batch[b].clone()
                current_action = actors_cur[agent_idx](*obs_n_o[agent_idx][b])
                current_action = current_action.view(-1)

                start, end = action_size[agent_idx]
                new_action[start:end] = current_action

                current_graphs = [obs_n_o[i][b] for i in range(n_agents)]
                q = critic_c(current_graphs, new_action.unsqueeze(0))
                actor_loss += -q

            actor_loss = actor_loss.mean()
            opt_a.zero_grad()
            actor_loss.backward()
            opt_a.step()

            # Accumulate loss
            total_critic_loss += loss_c.item()
            total_actor_loss += actor_loss.item()

        # ========== Logging: average ==========
        avg_critic_loss = total_critic_loss / n_agents
        avg_actor_loss = total_actor_loss / n_agents
        writer.add_scalar('critic loss', avg_critic_loss, game_step)
        writer.add_scalar('actor loss', avg_actor_loss, game_step)
        print_with_timestamp(f"[Step {game_step}] Avg Critic Loss: {avg_critic_loss:.4f} | Avg Actor Loss: {avg_actor_loss:.4f}")

        # ========== Save Model ==========
        if update_cnt > arglist.start_save_model and update_cnt % arglist.fre4save_model == 0:
            time_now = time.strftime('%y%m_%d%H%M')
            print(f'=time:{time_now} step:{game_step}        save')
            model_file_dir = os.path.join(arglist.save_dir, f'{run_type}_{time_now}_{game_step}')
            os.makedirs(model_file_dir, exist_ok=True)
            for idx in range(n_agents):
                torch.save(actors_cur[idx].state_dict(), os.path.join(model_file_dir, f'a_c_{idx}.pt'))
                torch.save(actors_tar[idx].state_dict(), os.path.join(model_file_dir, f'a_t_{idx}.pt'))
                torch.save(critics_cur[idx].state_dict(), os.path.join(model_file_dir, f'c_c_{idx}.pt'))
                torch.save(critics_tar[idx].state_dict(), os.path.join(model_file_dir, f'c_t_{idx}.pt'))

        # ========== Soft Target Update ==========
        actors_tar = update_trainers(actors_cur, actors_tar, arglist.tao)
        critics_tar = update_trainers(critics_cur, critics_tar, arglist.tao)

    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar

def train_mix_ax(arglist, type):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """
    env = make_env_ax()

    print('=============================')
    print('=1 Env {} is right ...'.format("resilient path planning"))
    print('=============================')

    """step2: create agents"""
    obs_shape_n = [env.observation_space[i] for i in range(env.n)]
    action_shape_n = [env.action_space[i] for i in range(env.n)]
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers_mix_ax(env.n, env.node_feat_dim, env.edge_feat_dim, env.beam_count, env.beam_count, arglist)
    memory_t = ReplayBuffer(arglist.memory_size)
    memory_j = ReplayBuffer(arglist.memory_size)

    print('=2 The {} agents are inited ...'.format(env.n))
    print('=============================')

    """step3: init the pars """
    obs_size = []
    action_size = []
    game_step = 0
    # episode_cnt = 0
    update_cnt = 0
    t_start = time.time()
    episode_rewards_t = [0.0]  # sum of rewards for all agents
    episode_rewards_j = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a

    print('=3 starting iterations ...')
    print('=============================')
    obs_n = env.reset()
    new_obs_n = copy.deepcopy(obs_n)
    date = {'mean_rw_t': [],
            'var_rw_t': [],
            'mean_rw_j': [],
            'var_rw_j': []}
    df = pd.DataFrame(date)
    df.to_csv('rw_mix_no_jammer.csv', index=False, mode='w')
    
    print('=4 init tensorboards ...')
    print('=================================')
    writer = SummaryWriter('maddpg_beamforming/loss')
        
    for episode_gone in range(arglist.max_episode):
        for episode_cnt in range(arglist.per_episode_max_len):
            action_n = [
                agent(
                    torch.tensor(node_feats, dtype=torch.float32).to(arglist.device),
                    torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(arglist.device),
                    torch.tensor(edge_feats, dtype=torch.float32).view(-1, 1).to(arglist.device)
                ).detach().cpu().numpy().squeeze(0)
                for agent, (node_feats, edge_list, edge_feats) in zip(actors_cur, obs_n)
            ]
            obs_n = copy.deepcopy(new_obs_n)
            new_obs_n, rew_n, rew_n_avg_queue_delta, rew_n_avg_satisfaction, done_n = env.step(action_n=action_n)

            if game_step % arglist.learning_fre == 0:
                print_with_timestamp(f"episode_gone: {episode_gone}, rew_n: {sum(rew_n)}, rew_n_avg_queue_delta: {rew_n_avg_queue_delta}, rew_n_avg_satisfaction: {rew_n_avg_satisfaction}")

            if (all(done_n)):
                print_with_timestamp(f"=============> episode_gone: {episode_gone}, rew_n: {sum(rew_n)}, rew_n_avg_queue_delta: {rew_n_avg_queue_delta}, rew_n_avg_satisfaction: {rew_n_avg_satisfaction}")
                writer.add_scalar('mean_reward', np.mean(rew_n), game_step)
                writer.add_scalar('urllc--rew_n--avg_queue_delta', rew_n_avg_queue_delta, game_step)
                writer.add_scalar('embb--rew_n--avg_satisfaction', rew_n_avg_satisfaction, game_step)

            memory_t.add(obs_n, action_n, rew_n, new_obs_n, done_n)
            episode_rewards_t[-1] += np.sum(rew_n)
            for i, rew in enumerate(rew_n): agent_rewards[i][-1] += rew

            # train our agents
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train_mix_ax(
                arglist, game_step, update_cnt, memory_t,
                obs_size, action_size,
                actors_cur, actors_tar, critics_cur, critics_tar,
                optimizers_a, optimizers_c, writer, env, rew_n, type)

            # update the obs_n
            game_step += 1
            obs_n = new_obs_n
            done = all(done_n)
            terminal = (episode_cnt >= arglist.per_episode_max_len - 1)
            if done or terminal:
                obs_n = env.reset()
                new_obs_n = copy.deepcopy(obs_n)
                episode_rewards_t.append(0)
                episode_rewards_j.append(0)
                continue