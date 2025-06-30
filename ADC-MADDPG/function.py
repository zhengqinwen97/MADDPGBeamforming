# Time: 2019-11-05
# Author: Zachary
# Name: MADDPG_torch
# File func: main func
import os

import time
import torch
import torch.optim as optim
import pandas as pd
from arguments import parse_args
from replay_buffer import ReplayBuffer
from environment import *
from model import openai_actor, openai_critic, openai_critic_double_q, critic_adq, critic_attention_v3, critic_adq_v2
from itertools import chain

global double_q_delay_fre, double_q_delay_cnt
double_q_delay_fre = 2
double_q_delay_cnt = 1


def get_trainers(n, obs_shape_n, action_shape_n, arglist):
    """
    init the trainers or load the old model
    """
    trainers_cur = []
    trainers_tar = []
    actors_cur = [None for _ in range(n)]
    critics_cur = [None for _ in range(n)]
    actors_tar = [None for _ in range(n)]
    critics_tar = [None for _ in range(n)]
    optimizers_c = [None for _ in range(n)]
    optimizers_a = [None for _ in range(n)]
    input_size_global = sum(obs_shape_n) + sum(action_shape_n)

    # if arglist.restore == True:  # restore the model
    #     for idx in arglist.restore_idxs:
    #         trainers_cur[idx] = torch.load(arglist.old_model_name + 'c_{}'.format(agent_idx))
    #         trainers_tar[idx] = torch.load(arglist.old_model_name + 't_{}'.format(agent_idx))

    # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
    for i in range(n):
        actors_cur[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        critics_cur[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
        actors_tar[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        critics_tar[i] = openai_critic(sum
                                       (obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
        optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)
    actors_tar = update_trainers(actors_cur, actors_tar, 1.0)  # update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0)  # update the target par using the cur
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c


def load_mix_trainers(arglist):
    """ load the model """
    actors_cur = [torch.load(arglist.mix_model_name + '/a_c_{}.pt'.format(agent_idx), map_location=arglist.device) \
                  for agent_idx in range(5)]
    critics_cur = [torch.load(arglist.mix_model_name + '/c_c_{}.pt'.format(agent_idx), map_location=arglist.device) \
                   for agent_idx in range(5)]
    actors_tar = [torch.load(arglist.mix_model_name + '/a_t_{}.pt'.format(agent_idx), map_location=arglist.device) \
                  for agent_idx in range(5)]
    critics_tar = [torch.load(arglist.mix_model_name + '/c_t_{}.pt'.format(agent_idx), map_location=arglist.device) \
                   for agent_idx in range(5)]
    return actors_cur, critics_cur, actors_tar, critics_tar


def get_all_trainers(arglist):
    actors_cur = [None for _ in range(5)]
    critics_cur = [None for _ in range(5)]
    actors_tar = [None for _ in range(5)]
    critics_tar = [None for _ in range(5)]
    optimizers_c = [None for _ in range(5)]
    optimizers_a = [None for _ in range(5)]
    """ load the model """
    for agent_idx in range(5):
        actors_cur[agent_idx] = torch.load(arglist.mix_model_name + 'a_c_{}.pt'.format(agent_idx),
                                           map_location=arglist.device)
        critics_cur[agent_idx] = torch.load(arglist.mix_model_name + 'c_c_{}.pt'.format(agent_idx),
                                            map_location=arglist.device)
        actors_tar[agent_idx] = torch.load(arglist.mix_model_name + 'a_t_{}.pt'.format(agent_idx),
                                           map_location=arglist.device)
        critics_tar[agent_idx] = torch.load(arglist.mix_model_name + 'c_t_{}.pt'.format(agent_idx),
                                            map_location=arglist.device)
        optimizers_a[agent_idx] = optim.Adam(actors_cur[agent_idx].parameters(), arglist.lr_a)
        optimizers_c[agent_idx] = optim.Adam(critics_cur[agent_idx].parameters(), arglist.lr_c)

    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c


def get_trainers_mix(n, obs_shape_n, action_shape_n, arglist):
    """
    init the trainers or load the old model
    """
    trainers_cur = []
    trainers_tar = []
    actors_cur = [None for _ in range(n)]
    critics_cur = [None for _ in range(n)]
    actors_tar = [None for _ in range(n)]
    critics_tar = [None for _ in range(n)]
    optimizers_c = [None for _ in range(n)]
    optimizers_a = [None for _ in range(n)]
    input_size_global = sum(obs_shape_n) + sum(action_shape_n)

    # if arglist.restore == True:  # restore the model
    #     for idx in arglist.restore_idxs:
    #         trainers_cur[idx] = torch.load(arglist.old_model_name + 'c_{}'.format(agent_idx))
    #         trainers_tar[idx] = torch.load(arglist.old_model_name + 't_{}'.format(agent_idx))

    # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
    for i in range(n):
        actors_cur[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        actors_tar[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        if i < 3:
            critics_cur[i] = openai_critic(sum(obs_shape_n[0:3]), sum(action_shape_n[0:3]), arglist).to(arglist.device)
            critics_tar[i] = openai_critic(sum(obs_shape_n[0:3]), sum(action_shape_n[0:3]), arglist).to(arglist.device)
        else:
            critics_cur[i] = openai_critic(sum(obs_shape_n[-2:]), sum(action_shape_n[-2:]), arglist).to(arglist.device)
            critics_tar[i] = openai_critic(sum(obs_shape_n[-2:]), sum(action_shape_n[-2:]), arglist).to(arglist.device)
        optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)

    actors_tar = update_trainers(actors_cur, actors_tar, 1.0)  # update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0)  # update the target par using the cur
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c

OBJ_DIMS = 280 * 4
ACT_DIMS = 16
AGENT_NUM = 24

def get_trainers_mix_ax(n, obs_shape_n, action_shape_n, arglist):
    """
    init the trainers or load the old model
    """
    trainers_cur = []
    trainers_tar = []
    actors_cur = [None for _ in range(n)]
    critics_cur = [None for _ in range(n)]
    actors_tar = [None for _ in range(n)]
    critics_tar = [None for _ in range(n)]
    optimizers_c = [None for _ in range(n)]
    optimizers_a = [None for _ in range(n)]
    input_size_global = sum(obs_shape_n) + sum(action_shape_n)

    # if arglist.restore == True:  # restore the model
    #     for idx in arglist.restore_idxs:
    #         trainers_cur[idx] = torch.load(arglist.old_model_name + 'c_{}'.format(agent_idx))
    #         trainers_tar[idx] = torch.load(arglist.old_model_name + 't_{}'.format(agent_idx))

    # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
    for i in range(n):
        actors_cur[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        actors_tar[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        if i < AGENT_NUM:
            critics_cur[i] = openai_critic(sum(obs_shape_n[0:AGENT_NUM]), sum(action_shape_n[0:AGENT_NUM]), arglist).to(arglist.device)
            critics_tar[i] = openai_critic(sum(obs_shape_n[0:AGENT_NUM]), sum(action_shape_n[0:AGENT_NUM]), arglist).to(arglist.device)
        else:
            critics_cur[i] = openai_critic(sum(obs_shape_n[-2:]), sum(action_shape_n[-2:]), arglist).to(arglist.device)
            critics_tar[i] = openai_critic(sum(obs_shape_n[-2:]), sum(action_shape_n[-2:]), arglist).to(arglist.device)
        optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)

    actors_tar = update_trainers(actors_cur, actors_tar, 1.0)  # update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0)  # update the target par using the cur
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c


def get_trainers_mix_double_q(n, obs_shape_n, action_shape_n, arglist):
    """
    init the trainers or load the old model
    """
    trainers_cur = []
    trainers_tar = []
    actors_cur = [None for _ in range(n)]
    critics_cur = [None for _ in range(n)]
    actors_tar = [None for _ in range(n)]
    critics_tar = [None for _ in range(n)]
    optimizers_c = [None for _ in range(n)]
    optimizers_a = [None for _ in range(n)]
    input_size_global = sum(obs_shape_n) + sum(action_shape_n)

    # if arglist.restore == True:  # restore the model
    #     for idx in arglist.restore_idxs:
    #         trainers_cur[idx] = torch.load(arglist.old_model_name + 'c_{}'.format(agent_idx))
    #         trainers_tar[idx] = torch.load(arglist.old_model_name + 't_{}'.format(agent_idx))

    # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
    for i in range(n):
        actors_cur[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        actors_tar[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        if i < 3:
            critics_cur[i] = openai_critic_double_q(sum(obs_shape_n[0:3]), sum(action_shape_n[0:3]), arglist).to(
                arglist.device)
            critics_tar[i] = openai_critic_double_q(sum(obs_shape_n[0:3]), sum(action_shape_n[0:3]), arglist).to(
                arglist.device)
            optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), 2 * arglist.lr_a)
        else:
            critics_cur[i] = openai_critic(sum(obs_shape_n[-2:]), sum(action_shape_n[-2:]), arglist).to(arglist.device)
            critics_tar[i] = openai_critic(sum(obs_shape_n[-2:]), sum(action_shape_n[-2:]), arglist).to(arglist.device)
            optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)

    actors_tar = update_trainers(actors_cur, actors_tar, 1.0)  # update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0)  # update the target par using the cur
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c


def get_trainers_adq(n, obs_shape_n, action_shape_n, arglist):
    """
    init the trainers or load the old model
    """
    trainers_cur = []
    trainers_tar = []
    actors_cur = [None for _ in range(n)]
    critics_cur = [None for _ in range(n)]
    actors_tar = [None for _ in range(n)]
    critics_tar = [None for _ in range(n)]
    optimizers_c = [None for _ in range(n)]
    optimizers_a = [None for _ in range(n)]
    input_size_global = sum(obs_shape_n) + sum(action_shape_n)

    # if arglist.restore == True:  # restore the model
    #     for idx in arglist.restore_idxs:
    #         trainers_cur[idx] = torch.load(arglist.old_model_name + 'c_{}'.format(agent_idx))
    #         trainers_tar[idx] = torch.load(arglist.old_model_name + 't_{}'.format(agent_idx))

    # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
    for i in range(n):
        actors_cur[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        actors_tar[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        if i < 3:
            critics_cur[i] = critic_adq_v2(sum(obs_shape_n[0:3]), sum(action_shape_n[0:3]), arglist).to(arglist.device)
            critics_tar[i] = critic_adq_v2(sum(obs_shape_n[0:3]), sum(action_shape_n[0:3]), arglist).to(arglist.device)
            optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), 2 * arglist.lr_a)
        else:
            critics_cur[i] = openai_critic(sum(obs_shape_n[-2:]), sum(action_shape_n[-2:]), arglist).to(arglist.device)
            critics_tar[i] = openai_critic(sum(obs_shape_n[-2:]), sum(action_shape_n[-2:]), arglist).to(arglist.device)
            optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)

    actors_tar = update_trainers(actors_cur, actors_tar, 1.0)  # update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0)  # update the target par using the cur
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c


def get_trainers_mix_attention(n, obs_shape_n, action_shape_n, arglist):
    """
    init the trainers or load the old model
    """
    trainers_cur = []
    trainers_tar = []
    actors_cur = [None for _ in range(n)]
    critics_cur = [None for _ in range(n)]
    actors_tar = [None for _ in range(n)]
    critics_tar = [None for _ in range(n)]
    optimizers_c = [None for _ in range(n)]
    optimizers_a = [None for _ in range(n)]
    input_size_global = sum(obs_shape_n) + sum(action_shape_n)

    # if arglist.restore == True:  # restore the model
    #     for idx in arglist.restore_idxs:
    #         trainers_cur[idx] = torch.load(arglist.old_model_name + 'c_{}'.format(agent_idx))
    #         trainers_tar[idx] = torch.load(arglist.old_model_name + 't_{}'.format(agent_idx))

    # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
    for i in range(n):
        actors_cur[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        actors_tar[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        if i < 3:
            critics_cur[i] = critic_attention_v3(sum(obs_shape_n[0:3]), sum(action_shape_n[0:3]), arglist).to(
                arglist.device)
            critics_tar[i] = critic_attention_v3(sum(obs_shape_n[0:3]), sum(action_shape_n[0:3]), arglist).to(
                arglist.device)
        else:
            critics_cur[i] = openai_critic(sum(obs_shape_n[-2:]), sum(action_shape_n[-2:]), arglist).to(arglist.device)
            critics_tar[i] = openai_critic(sum(obs_shape_n[-2:]), sum(action_shape_n[-2:]), arglist).to(arglist.device)
        optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)

    actors_tar = update_trainers(actors_cur, actors_tar, 1.0)  # update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0)  # update the target par using the cur
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c


def get_jammer_trainers(arglist):
    """ load the model """
    actors_tar = [torch.load(arglist.jammer_model_name + 'a_c_{}.pt'.format(agent_idx), map_location=arglist.device) \
                  for agent_idx in range(2)]

    return actors_tar


def get_typical_trainers(arglist):
    """ load the model """
    actors_tar = [torch.load(arglist.typical_model_name + 'a_c_{}.pt'.format(agent_idx), map_location=arglist.device) \
                  for agent_idx in range(3)]

    return actors_tar


def update_trainers(agents_cur, agents_tar, tao):
    """
    update the trainers_tar par using the trainers_cur
    This way is not the same as copy_, but the result is the same
    out:
    |agents_tar: the agents with new par updated towards agents_current
    """
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key] * tao + \
                                (1 - tao) * state_dict_t[key]
        agent_t.load_state_dict(state_dict_t)
    return agents_tar


def agents_train(arglist, game_step, update_cnt, memory, obs_size, action_size, \
                 actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, type):
    """
    use this func to make the "main" func clean
    par:
    |input: the data for training
    |output: the data for next update
    """
    # update all trainers, if not in display or benchmark mode
    if game_step > arglist.learning_start_step and \
            (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:
        if update_cnt == 0: print('\r=start training ...' + ' ' * 100)
        # update the target par using the cur
        update_cnt += 1

        # update every agent in different memory batch
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            if opt_c == None: continue  # jump to the next model update

            # sample the experience
            _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory.sample( \
                arglist.batch_size, agent_idx)  # Note_The func is not the same as others

            # --use the date to update the CRITIC
            rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float)  # set the rew to gpu
            done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device)  # set the rew to gpu
            action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
            obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
            obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)
            action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                                    for idx, a_t in enumerate(actors_tar)], dim=1)
            q = critic_c(obs_n_o, action_cur_o).reshape(-1)  # q
            q_ = critic_t(obs_n_n, action_tar).reshape(-1)  # q_
            tar_value = q_ * arglist.gamma * done_n + rew  # q_*gamma*done + reward
            loss_c = torch.nn.MSELoss()(q, tar_value)  # bellman equation
            opt_c.zero_grad()
            loss_c.backward()
            # nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
            opt_c.step()

            # --use the data to update the ACTOR
            # There is no need to cal other agent's action
            model_out, policy_c_new = actor_c( \
                obs_n_o[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]], model_original_out=True)
            # update the aciton of this agent
            action_cur_o[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new
            loss_pse = torch.mean(torch.pow(model_out, 2))
            loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))

            opt_a.zero_grad()
            (1e-3 * loss_pse + loss_a).backward()
            # nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
            opt_a.step()

        # save the model to the path_dir ---cnt by update number
        if update_cnt > arglist.start_save_model and update_cnt % arglist.fre4save_model == 0:
            time_now = time.strftime('%y%m_%d%H%M')
            print('=time:{} step:{}        save'.format(time_now, game_step))
            model_file_dir = os.path.join(arglist.save_dir, '{}_{}_{}'.format( \
                type, time_now, game_step))
            if not os.path.exists(model_file_dir):  # make the path
                os.mkdir(model_file_dir)
            for agent_idx, (a_c, a_t, c_c, c_t) in \
                    enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

        # update the tar par
        actors_tar = update_trainers(actors_cur, actors_tar, arglist.tao)
        critics_tar = update_trainers(critics_cur, critics_tar, arglist.tao)

    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar

def resize(arr,idx,col_num):
    #根据自身编号把对应的信息提到最前面
    if idx == 0:
        return arr
    else:
        start = idx*col_num
        end = idx*col_num+col_num
        col_select = arr[:,start : end]
        result = np.hstack((col_select, arr[:, :start], arr[:, end:]))
        return result
    
def agents_train_mix(arglist, game_step, update_cnt, memory_t, memory_j, obs_size, action_size, \
                     actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, type, random=True):
    """
    use this func to make the "main" func clean
    par:
    |input: the data for training
    |output: the data for next update
    """
    # update all trainers, if not in display or benchmark mode
    if game_step > arglist.learning_start_step and \
            (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:
        if update_cnt == 0: print('\r=start training ...' + ' ' * 100)
        # update the target par using the cur
        update_cnt += 1

        # update every agent in different memory batch
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            if opt_c == None: continue  # jump to the next model update

            # --use the date to update the CRITIC of typical UAVs
            if agent_idx < 3:
                # sample the experience
                _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory_t.sample( \
                    arglist.batch_size, agent_idx)  # Note_The func is not the same as others
                # 重整状态空间、动作空间等，把自己对应的放第一个

                _obs_n_o = resize(_obs_n_o, agent_idx, 15)
                _action_n = resize(_action_n, agent_idx, 2)
                _obs_n_n = resize(_obs_n_n, agent_idx, 15)

                rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float)  # set the rew to gpu
                done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device)  # set the rew to gpu
                action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
                obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
                obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)

                # action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                #                          for idx, a_t in enumerate(actors_tar[0:3])], dim=1)
                seq = list(range(3))
                agent_seq = [agent_idx] + [idx for idx in seq if idx != agent_idx]
                action_tar = actors_tar[agent_idx](obs_n_n[:, :15]).detach()

                for i,idx in enumerate(agent_seq[1:]):
                    action_tar = torch.cat((action_tar, actors_tar[idx](obs_n_n[:,(i+1)*15 : (i+1)*15+15]).detach()), dim = 1)

                # action_tar = actors_tar[agent_idx](obs_n_n[:, :15])
                # for idx in agent_seq:
                #     if idx == agent_idx:
                #         continue
                #     else:
                #         action_tar = torch.cat((action_tar, actors_tar[idx](obs_n_n[:,idx*15 : idx*15+15])), dim = 1)

                q = critic_c(obs_n_o, action_cur_o).reshape(-1)  # q
                # print(obs_n_o, action_cur_o)
                q_ = critic_t(obs_n_n, action_tar).reshape(-1)  # q_
                tar_value = q_ * arglist.gamma * done_n + rew  # q_*gamma*done + reward
                loss_c = torch.nn.MSELoss()(q, tar_value)  # bellman equation
                opt_c.zero_grad()
                loss_c.backward()
                # nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
                opt_c.step()

                # --use the data to update the ACTOR
                # There is no need to cal other agent's action
                model_out, policy_c_new = actor_c( \
                    obs_n_o[:, :15], model_original_out=True)
                # update the aciton of this agent
                action_cur_o[:,:2] = policy_c_new
                loss_pse = torch.mean(torch.pow(model_out, 2))
                loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))

                opt_a.zero_grad()
                (1e-3 * loss_pse + loss_a).backward()
                # nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
                opt_a.step()

            # --use the date to update the CRITIC of jammers
            else:
                _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory_j.sample( \
                    arglist.batch_size, agent_idx - 3)  # Note_The func is not the same as others
                rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float)  # set the rew to gpu
                done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device)  # set the rew to gpu
                action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
                obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
                obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)

                action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx + 3][0] - 45:obs_size[idx + 3][1] - 45]).detach() \
                                        for idx, a_t in enumerate(actors_tar[-2:])], dim=1)
                
                q = critic_c(obs_n_o, action_cur_o).reshape(-1)  # q
                q_ = critic_t(obs_n_n, action_tar).reshape(-1)  # q_
                tar_value = q_ * arglist.gamma * done_n + rew  # q_*gamma*done + reward
                loss_c = torch.nn.MSELoss()(q, tar_value)  # bellman equation
                opt_c.zero_grad()
                loss_c.backward()
                # nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
                opt_c.step()

                # --use the data to update the ACTOR
                # There is no need to cal other agent's action
                model_out, policy_c_new = actor_c( \
                    obs_n_o[:, obs_size[agent_idx][0] - 45:obs_size[agent_idx][1] - 45], model_original_out=True)
                # update the aciton of this agent
                action_cur_o[:, action_size[agent_idx][0] - 6:action_size[agent_idx][1] - 6] = policy_c_new
                loss_pse = torch.mean(torch.pow(model_out, 2))
                loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))

                opt_a.zero_grad()
                (1e-3 * loss_pse + loss_a).backward()
                # nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
                opt_a.step()

        # save the model to the path_dir ---cnt by update number
        if update_cnt > arglist.start_save_model and update_cnt % arglist.fre4save_model == 0:
            time_now = time.strftime('%y%m_%d%H%M')
            print('=time:{} step:{}        save'.format(time_now, game_step))
            model_file_dir = os.path.join(arglist.save_dir, '{}_{}_{}'.format( \
                type, time_now, game_step))
            if not os.path.exists(model_file_dir):  # make the path
                os.mkdir(model_file_dir)
            for agent_idx, (a_c, a_t, c_c, c_t) in \
                    enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

        # update the tar par
        actors_tar = update_trainers(actors_cur, actors_tar, arglist.tao)
        critics_tar = update_trainers(critics_cur, critics_tar, arglist.tao)

    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar

OBJ_DIMS = 280 * 4
ACT_DIMS = 16
AGENT_NUM = 24

def agents_train_mix_ax(arglist, game_step, update_cnt, memory_t, memory_j, obs_size, action_size, \
                     actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, type, random=True):
    """
    use this func to make the "main" func clean
    par:
    |input: the data for training
    |output: the data for next update
    """
    # update all trainers, if not in display or benchmark mode
    if game_step > arglist.learning_start_step and \
            (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:
        if update_cnt == 0: print('\r=start training ...' + ' ' * 100)
        # update the target par using the cur
        update_cnt += 1

        # update every agent in different memory batch
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            if opt_c == None: continue  # jump to the next model update

            # --use the date to update the CRITIC of typical UAVs
            if agent_idx < AGENT_NUM:
                # sample the experience
                _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory_t.sample( \
                    arglist.batch_size, agent_idx)  # Note_The func is not the same as others
                # 重整状态空间、动作空间等，把自己对应的放第一个

                _obs_n_o = resize(_obs_n_o, agent_idx, OBJ_DIMS)
                _action_n = resize(_action_n, agent_idx, ACT_DIMS)
                _obs_n_n = resize(_obs_n_n, agent_idx, OBJ_DIMS)

                rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float)  # set the rew to gpu
                done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device)  # set the rew to gpu
                action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
                obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
                obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)

                # action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                #                          for idx, a_t in enumerate(actors_tar[0:3])], dim=1)
                seq = list(range(AGENT_NUM))
                agent_seq = [agent_idx] + [idx for idx in seq if idx != agent_idx]
                action_tar = actors_tar[agent_idx](obs_n_n[:, :OBJ_DIMS]).detach()

                for i,idx in enumerate(agent_seq[1:]):
                    action_tar = torch.cat((action_tar, actors_tar[idx](obs_n_n[:,(i+1)*OBJ_DIMS : (i+1)*OBJ_DIMS+OBJ_DIMS]).detach()), dim = 1)

                # action_tar = actors_tar[agent_idx](obs_n_n[:, :15])
                # for idx in agent_seq:
                #     if idx == agent_idx:
                #         continue
                #     else:
                #         action_tar = torch.cat((action_tar, actors_tar[idx](obs_n_n[:,idx*15 : idx*15+15])), dim = 1)

                q = critic_c(obs_n_o, action_cur_o).reshape(-1)  # q
                # print(obs_n_o, action_cur_o)
                q_ = critic_t(obs_n_n, action_tar).reshape(-1)  # q_
                tar_value = q_ * arglist.gamma * done_n + rew  # q_*gamma*done + reward
                loss_c = torch.nn.MSELoss()(q, tar_value)  # bellman equation
                opt_c.zero_grad()
                loss_c.backward()
                # nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
                opt_c.step()

                # --use the data to update the ACTOR
                # There is no need to cal other agent's action
                model_out, policy_c_new = actor_c( \
                    obs_n_o[:, :OBJ_DIMS], model_original_out=True)
                # update the aciton of this agent
                action_cur_o[:,:ACT_DIMS] = policy_c_new
                loss_pse = torch.mean(torch.pow(model_out, ACT_DIMS))
                loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))

                opt_a.zero_grad()
                (1e-3 * loss_pse + loss_a).backward()
                # nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
                opt_a.step()

            # --use the date to update the CRITIC of jammers
            else:
                _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory_j.sample( \
                    arglist.batch_size, agent_idx - 3)  # Note_The func is not the same as others
                rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float)  # set the rew to gpu
                done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device)  # set the rew to gpu
                action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
                obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
                obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)

                action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx + 3][0] - 45:obs_size[idx + 3][1] - 45]).detach() \
                                        for idx, a_t in enumerate(actors_tar[-2:])], dim=1)
                
                q = critic_c(obs_n_o, action_cur_o).reshape(-1)  # q
                q_ = critic_t(obs_n_n, action_tar).reshape(-1)  # q_
                tar_value = q_ * arglist.gamma * done_n + rew  # q_*gamma*done + reward
                loss_c = torch.nn.MSELoss()(q, tar_value)  # bellman equation
                opt_c.zero_grad()
                loss_c.backward()
                # nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
                opt_c.step()

                # --use the data to update the ACTOR
                # There is no need to cal other agent's action
                model_out, policy_c_new = actor_c( \
                    obs_n_o[:, obs_size[agent_idx][0] - 45:obs_size[agent_idx][1] - 45], model_original_out=True)
                # update the aciton of this agent
                action_cur_o[:, action_size[agent_idx][0] - 6:action_size[agent_idx][1] - 6] = policy_c_new
                loss_pse = torch.mean(torch.pow(model_out, 2))
                loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))

                opt_a.zero_grad()
                (1e-3 * loss_pse + loss_a).backward()
                # nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
                opt_a.step()

        # save the model to the path_dir ---cnt by update number
        if update_cnt > arglist.start_save_model and update_cnt % arglist.fre4save_model == 0:
            time_now = time.strftime('%y%m_%d%H%M')
            print('=time:{} step:{}        save'.format(time_now, game_step))
            model_file_dir = os.path.join(arglist.save_dir, '{}_{}_{}'.format( \
                type, time_now, game_step))
            if not os.path.exists(model_file_dir):  # make the path
                os.mkdir(model_file_dir)
            for agent_idx, (a_c, a_t, c_c, c_t) in \
                    enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

        # update the tar par
        actors_tar = update_trainers(actors_cur, actors_tar, arglist.tao)
        critics_tar = update_trainers(critics_cur, critics_tar, arglist.tao)

    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar

def agents_train_mix_double_q(arglist, game_step, update_cnt, memory_t, memory_j, obs_size, action_size, \
                              actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, type):
    """
    use this func to make the "main" func clean
    par:
    |input: the data for training
    |output: the data for next update
    """
    # update all trainers, if not in display or benchmark mode
    global double_q_delay_fre, double_q_delay_cnt
    if game_step > arglist.learning_start_step and \
            (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:
        if update_cnt == 0: print('\r=start training ...' + ' ' * 100)
        # update the target par using the cur
        update_cnt += 1

        # update every agent in different memory batch
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            if opt_c == None: continue  # jump to the next model update

            # --use the date to update the CRITIC of typical UAVs
            if agent_idx < 3:
                # sample the experience
                _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory_t.sample( \
                    arglist.batch_size, agent_idx)  # Note_The func is not the same as others

                _obs_n_o = resize(_obs_n_o, agent_idx, 15)
                _action_n = resize(_action_n, agent_idx, 2)
                _obs_n_n = resize(_obs_n_n, agent_idx, 15)

                rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float)  # set the rew to gpu
                done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device)  # set the rew to gpu
                action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
                obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
                obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)\
                
                seq = list(range(3))
                agent_seq = [agent_idx] + [idx for idx in seq if idx != agent_idx]
                action_tar = actors_tar[agent_idx](obs_n_n[:, :15]).detach()
                for i,idx in enumerate(agent_seq[1:]):
                    action_tar = torch.cat((action_tar, actors_tar[idx](obs_n_n[:,(i+1)*15 : (i+1)*15+15]).detach()), dim = 1)

                q1, q2 = critic_c(obs_n_o, action_cur_o)  # q
                q1 = q1.reshape(-1)
                q2 = q2.reshape(-1)
                # q = torch.min(q1,q2)
                # print(obs_n_o, action_cur_o)
                q1_, q2_ = critic_t(obs_n_n, action_tar)  # q_
                q1_ = q1_.reshape(-1)
                q2_ = q2_.reshape(-1)
                q_ = torch.mean(torch.stack([q1_, q2_], dim=0), dim=0)
                # q_ = torch.min(q1_, q2_)
                tar_value = (q_ * arglist.gamma * done_n + rew).reshape(-1).reshape(-1)  # q_*gamma*done + reward
                loss_c = torch.nn.MSELoss()(q1, tar_value) + torch.nn.MSELoss()(q2, tar_value)  # bellman equation
                opt_c.zero_grad()
                loss_c.backward()
                # nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
                opt_c.step()

                # --use the data to update the ACTOR
                # There is no need to cal other agent's action
                # 延迟更新
                if not double_q_delay_cnt % double_q_delay_fre == 0:
                    model_out, policy_c_new = actor_c( \
                        obs_n_o[:,:15], model_original_out=True)
                    # update the aciton of this agent
                    action_cur_o[:, :2] = policy_c_new
                    loss_pse = torch.mean(torch.pow(model_out, 2))
                    loss_a = torch.mul(-1, torch.mean(critic_c.q1(obs_n_o, action_cur_o)))
                    # loss_a = -torch.mean(critic_c.q1(obs_n_o, action_cur_o))
                    opt_a.zero_grad()
                    (1e-3 * loss_pse + loss_a).backward()
                    # nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
                    opt_a.step()
                    double_q_delay_cnt = 1
                else:
                    double_q_delay_cnt += 1

            # --use the date to update the CRITIC of jammers
            else:
                _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory_j.sample( \
                    arglist.batch_size, agent_idx - 3)  # Note_The func is not the same as others
                rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float)  # set the rew to gpu
                done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device)  # set the rew to gpu
                action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
                obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
                obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)

                action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx + 3][0] - 45:obs_size[idx + 3][1] - 45]).detach() \
                                        for idx, a_t in enumerate(actors_tar[-2:])], dim=1)
                q = critic_c(obs_n_o, action_cur_o).reshape(-1)  # q
                q_ = critic_t(obs_n_n, action_tar).reshape(-1)  # q_
                tar_value = q_ * arglist.gamma * done_n + rew  # q_*gamma*done + reward
                loss_c = torch.nn.MSELoss()(q, tar_value)  # bellman equation
                opt_c.zero_grad()
                loss_c.backward()
                # nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
                opt_c.step()

                # --use the data to update the ACTOR
                # There is no need to cal other agent's action
                model_out, policy_c_new = actor_c( \
                    obs_n_o[:, obs_size[agent_idx][0] - 45:obs_size[agent_idx][1] - 45], model_original_out=True)
                # update the aciton of this agent
                action_cur_o[:, action_size[agent_idx][0] - 6:action_size[agent_idx][1] - 6] = policy_c_new
                loss_pse = torch.mean(torch.pow(model_out, 2))
                loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))

                opt_a.zero_grad()
                (1e-3 * loss_pse + loss_a).backward()
                # nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
                opt_a.step()

        # save the model to the path_dir ---cnt by update number
        if update_cnt > arglist.start_save_model and update_cnt % arglist.fre4save_model == 0:
            time_now = time.strftime('%y%m_%d%H%M')
            print('=time:{} step:{}        save'.format(time_now, game_step))
            model_file_dir = os.path.join(arglist.save_dir, '{}_{}_{}'.format( \
                type, time_now, game_step))
            if not os.path.exists(model_file_dir):  # make the path
                os.mkdir(model_file_dir)
            for agent_idx, (a_c, a_t, c_c, c_t) in \
                    enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

        # update the tar par
        actors_tar = update_trainers(actors_cur, actors_tar, arglist.tao)
        critics_tar = update_trainers(critics_cur, critics_tar, arglist.tao)

    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar


def train_mix(arglist, type="mix"):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """
    env = make_env()

    print('=============================')
    print('=1 Env {} is right ...'.format("resilient path planning"))
    print('=============================')

    """step2: create agents"""
    obs_shape_n = [env.observation_space[i] for i in range(env.n)]
    action_shape_n = [env.action_space[i] for i in range(env.n)]  # no need for stop bit
    # num_adversaries = min(env.n, arglist.num_adversaries)
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers_mix(env.n, obs_shape_n, action_shape_n, arglist)
    # memory = Memory(num_adversaries, arglist)
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
    # rew_n_old = [0.0 for _ in range(env.n)]  # set the init reward
    # agent_info = [[[]]]  # placeholder for benchmarking info
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
    obs_n = env.reset(random=True)
    new_obs_n = copy.deepcopy(obs_n)
    date = {'mean_rw_t': [],
            'var_rw_t': [],
            'mean_rw_j': [],
            'var_rw_j': []}
    df = pd.DataFrame(date)
    df.to_csv('rw_mix.csv', index=False, mode='w')
    for episode_gone in range(arglist.max_episode):
        # cal the reward print the debug data
        if game_step > 1 and game_step % 4000 == 0:
            mean_rw_t = round(np.mean(episode_rewards_t[-400:-1]), 3)
            mean_rw_j = round(np.mean(episode_rewards_j[-400:-1]), 3)
            var_rw_t = round(np.var(episode_rewards_t[-400:-1]), 3)
            var_rw_j = round(np.var(episode_rewards_j[-400:-1]), 3)
            df = pd.read_csv('rw_mix.csv')
            df.loc[int(game_step / 4000), 'mean_rw_t'] = mean_rw_t
            df.loc[int(game_step / 4000), 'var_rw_t'] = var_rw_t
            df.loc[int(game_step / 4000), 'mean_rw_j'] = mean_rw_j
            df.loc[int(game_step / 4000), 'var_rw_j'] = var_rw_j
            print(" " * 3 + 'typical mean reward:{} jammer mean reward:{} at {}'.format(mean_rw_t, mean_rw_j,
                                                                                        game_step / 4000), end='\n')
            df.to_csv('rw_mix.csv', index=False)
        for episode_cnt in range(arglist.per_episode_max_len):
            # get action
            action_n = [agent(torch.tensor(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                        for agent, obs in zip(actors_cur, obs_n)]
            # print(action_n)
            # interact with env
            obs_n = copy.deepcopy(new_obs_n)
            new_obs_n, rew_n, done_n = env.step(action_n=action_n, timestep=episode_cnt, jammer_act=arglist.jammer_act)
            # print(new_obs_n, rew_n, done_n)
            # save the experience
            memory_t.add(obs_n[0:3], np.concatenate(action_n[0:3]), rew_n[0:3], new_obs_n[0:3], done_n[0:3])
            memory_j.add(obs_n[-2:], np.concatenate(action_n[-2:]), rew_n[-2:], new_obs_n[-2:], done_n[-2:])
            episode_rewards_t[-1] += np.sum(rew_n[0:3])
            episode_rewards_j[-1] += np.sum(rew_n[-2:])
            for i, rew in enumerate(rew_n): agent_rewards[i][-1] += rew

            # train our agents
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train_mix( \
                arglist, game_step, update_cnt, memory_t, memory_j, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, \
                type=type)

            # update the obs_n
            game_step += 1
            obs_n = new_obs_n
            done = all(done_n)
            terminal = (episode_cnt >= arglist.per_episode_max_len - 1)
            if done or terminal:
                # print(episode_rewards_t[-1],episode_rewards_j[-1], env.done_cnt)
                # print(episode_gone,game_step)
                obs_n = env.reset(random=True)
                new_obs_n = copy.deepcopy(obs_n)
                episode_rewards_t.append(0)
                episode_rewards_j.append(0)
                continue


def train_mix_no_jammer(arglist, type):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """
    env = make_env()

    print('=============================')
    print('=1 Env {} is right ...'.format("resilient path planning"))
    print('=============================')

    """step2: create agents"""
    obs_shape_n = [env.observation_space[i] for i in range(env.n)]
    action_shape_n = [env.action_space[i] for i in range(env.n)]  # no need for stop bit
    # num_adversaries = min(env.n, arglist.num_adversaries)
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers_mix(env.n, obs_shape_n, action_shape_n, arglist)
    # memory = Memory(num_adversaries, arglist)
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
    # rew_n_old = [0.0 for _ in range(env.n)]  # set the init reward
    # agent_info = [[[]]]  # placeholder for benchmarking info
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
    obs_n = env.reset(jammer=False)
    new_obs_n = copy.deepcopy(obs_n)
    date = {'mean_rw_t': [],
            'var_rw_t': [],
            'mean_rw_j': [],
            'var_rw_j': []}
    df = pd.DataFrame(date)
    df.to_csv('rw_mix_no_jammer.csv', index=False, mode='w')
    for episode_gone in range(arglist.max_episode):
        # cal the reward print the debug data
        if game_step > 1 and game_step % 4000 == 0:
            mean_rw_t = round(np.mean(episode_rewards_t[-400:-1]), 3)
            mean_rw_j = round(np.mean(episode_rewards_j[-400:-1]), 3)
            var_rw_t = round(np.var(episode_rewards_t[-400:-1]), 3)
            var_rw_j = round(np.var(episode_rewards_j[-400:-1]), 3)
            df = pd.read_csv('rw_mix_no_jammer.csv')
            df.loc[int(game_step / 4000), 'mean_rw_t'] = mean_rw_t
            df.loc[int(game_step / 4000), 'var_rw_t'] = var_rw_t
            df.loc[int(game_step / 4000), 'mean_rw_j'] = mean_rw_j
            df.loc[int(game_step / 4000), 'var_rw_j'] = var_rw_j
            print(" " * 3 + 'typical mean reward:{} jammer mean reward:{} at {}'.format(mean_rw_t, mean_rw_j,
                                                                                        game_step / 4000), end='\n')
            df.to_csv('rw_mix_no_jammer.csv', index=False)
        for episode_cnt in range(arglist.per_episode_max_len):
            # get action
            action_n = [agent(torch.tensor(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                        for agent, obs in zip(actors_cur, obs_n)]
            # print(action_n)
            # interact with env
            obs_n = copy.deepcopy(new_obs_n)
            new_obs_n, rew_n, done_n = env.step(action_n=action_n, timestep=episode_cnt, jammer_act=arglist.jammer_act)
            # print(new_obs_n, rew_n, done_n)
            # save the experience
            memory_t.add(obs_n[0:3], np.concatenate(action_n[0:3]), rew_n[0:3], new_obs_n[0:3], done_n[0:3])
            memory_j.add(obs_n[-2:], np.concatenate(action_n[-2:]), rew_n[-2:], new_obs_n[-2:], done_n[-2:])
            episode_rewards_t[-1] += np.sum(rew_n[0:3])
            episode_rewards_j[-1] += np.sum(rew_n[-2:])
            for i, rew in enumerate(rew_n): agent_rewards[i][-1] += rew

            # train our agents
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train_mix( \
                arglist, game_step, update_cnt, memory_t, memory_j, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, type=type)

            # update the obs_n
            game_step += 1
            obs_n = new_obs_n
            done = all(done_n)
            terminal = (episode_cnt >= arglist.per_episode_max_len - 1)
            if done or terminal:
                # print(episode_rewards_t[-1],episode_rewards_j[-1], env.done_cnt)
                # print(episode_gone,game_step)
                obs_n = env.reset(jammer=False)
                new_obs_n = copy.deepcopy(obs_n)
                episode_rewards_t.append(0)
                episode_rewards_j.append(0)
                continue


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
    action_shape_n = [env.action_space[i] for i in range(env.n)]  # no need for stop bit
    # num_adversaries = min(env.n, arglist.num_adversaries)
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers_mix_ax(env.n, obs_shape_n, action_shape_n, arglist)
    # memory = Memory(num_adversaries, arglist)
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
    # rew_n_old = [0.0 for _ in range(env.n)]  # set the init reward
    # agent_info = [[[]]]  # placeholder for benchmarking info
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
    for episode_gone in range(arglist.max_episode):
        # cal the reward print the debug data
        # print(game_step)
        if game_step > 1 and game_step % 20 == 0:
            mean_rw_t = round(np.mean(episode_rewards_t[-40:-1]), 3)
            # mean_rw_j = round(np.mean(episode_rewards_j[-400:-1]), 3)
            var_rw_t = round(np.var(episode_rewards_t[-40:-1]), 3)
            # var_rw_j = round(np.var(episode_rewards_j[-400:-1]), 3)
            df = pd.read_csv('rw_mix_no_jammer.csv')
            df.loc[int(game_step / 200), 'mean_rw_t'] = mean_rw_t
            df.loc[int(game_step / 200), 'var_rw_t'] = var_rw_t
            # df.loc[int(game_step / 200), 'mean_rw_j'] = mean_rw_j
            # df.loc[int(game_step / 200), 'var_rw_j'] = var_rw_j
            print(" " * 3 + 'typical mean reward:{} at {}'.format(mean_rw_t, game_step / 20), end='\n')
            df.to_csv('rw_mix_no_jammer.csv', index=False)

        for episode_cnt in range(arglist.per_episode_max_len):
            # get action
            action_n = [agent(torch.tensor(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                        for agent, obs in zip(actors_cur, obs_n)]
            # print(action_n)
            # interact with env
            obs_n = copy.deepcopy(new_obs_n)
            # new_obs_n, rew_n, done_n = env.step(action_n=action_n)
            new_obs_n, rew_n, rew_n_avg_queue_delta, rew_n_avg_satisfaction, rew_n_power_penalty, done_n = env.step(action_n=action_n)

            if episode_cnt == arglist.per_episode_max_len - 1:
                print(f"episode_gone: {episode_gone}, rew_n: {sum(rew_n)}, rew_n_avg_queue_delta: {rew_n_avg_queue_delta}, rew_n_avg_satisfaction: {rew_n_avg_satisfaction}, rew_n_power_penalty: {rew_n_power_penalty}")

            # print(new_obs_n, rew_n, done_n)
            # # save the experience
            memory_t.add(obs_n, np.concatenate(action_n), rew_n, new_obs_n, done_n)
            # memory_t.add(obs_n[0:3], np.concatenate(action_n[0:3]), rew_n[0:3], new_obs_n[0:3], done_n[0:3])
            # memory_j.add(obs_n[-2:], np.concatenate(action_n[-2:]), rew_n[-2:], new_obs_n[-2:], done_n[-2:])
            episode_rewards_t[-1] += np.sum(rew_n)
            # episode_rewards_j[-1] += np.sum(rew_n[-2:])
            for i, rew in enumerate(rew_n): agent_rewards[i][-1] += rew

            # train our agents
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train_mix_ax( \
                arglist, game_step, update_cnt, memory_t, memory_j, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, type=type)

            # update the obs_n
            game_step += 1
            obs_n = new_obs_n
            done = all(done_n)
            terminal = (episode_cnt >= arglist.per_episode_max_len - 1)
            if done or terminal:
                # print(episode_rewards_t[-1],episode_rewards_j[-1], env.done_cnt)
                # print(episode_gone,game_step)
                obs_n = env.reset()
                new_obs_n = copy.deepcopy(obs_n)
                episode_rewards_t.append(0)
                episode_rewards_j.append(0)
                continue


def train_mix_fixed_jammer(arglist, type):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """
    env = make_env()

    print('=============================')
    print('=1 Env {} is right ...'.format("resilient path planning"))
    print('=============================')

    """step2: create agents"""
    obs_shape_n = [env.observation_space[i] for i in range(env.n)]
    action_shape_n = [env.action_space[i] for i in range(env.n)]  # no need for stop bit
    # num_adversaries = min(env.n, arglist.num_adversaries)
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers_mix(env.n, obs_shape_n, action_shape_n, arglist)
    # memory = Memory(num_adversaries, arglist)
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
    # rew_n_old = [0.0 for _ in range(env.n)]  # set the init reward
    # agent_info = [[[]]]  # placeholder for benchmarking info
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
    obs_n = env.reset(random=True)
    new_obs_n = copy.deepcopy(obs_n)
    date = {'mean_rw_t': [],
            'var_rw_t': [],
            'mean_rw_j': [],
            'var_rw_j': []}
    df = pd.DataFrame(date)
    df.to_csv('rw_mix_fixed_jammer.csv', index=False, mode='w')
    for episode_gone in range(arglist.max_episode):
        # cal the reward print the debug data
        if game_step > 1 and game_step % 4000 == 0:
            mean_rw_t = round(np.mean(episode_rewards_t[-400:-1]), 3)
            mean_rw_j = round(np.mean(episode_rewards_j[-400:-1]), 3)
            var_rw_t = round(np.var(episode_rewards_t[-400:-1]), 3)
            var_rw_j = round(np.var(episode_rewards_j[-400:-1]), 3)
            df = pd.read_csv('rw_mix_fixed_jammer.csv')
            df.loc[int(game_step / 4000), 'mean_rw_t'] = mean_rw_t
            df.loc[int(game_step / 4000), 'var_rw_t'] = var_rw_t
            df.loc[int(game_step / 4000), 'mean_rw_j'] = mean_rw_j
            df.loc[int(game_step / 4000), 'var_rw_j'] = var_rw_j
            print(" " * 3 + 'typical mean reward:{} jammer mean reward:{} at {}'.format(mean_rw_t, mean_rw_j,
                                                                                        game_step / 4000), end='\n')
            df.to_csv('rw_mix_fixed_jammer.csv', index=False)
        for episode_cnt in range(arglist.per_episode_max_len):
            # get action
            action_n = [agent(torch.tensor(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                        for agent, obs in zip(actors_cur, obs_n)]
            # print(action_n)
            # interact with env
            obs_n = copy.deepcopy(new_obs_n)
            new_obs_n, rew_n, done_n = env.step(action_n=action_n, timestep=episode_cnt, jammer_act=arglist.jammer_act)
            # print(new_obs_n, rew_n, done_n)
            # save the experience
            memory_t.add(obs_n[0:3], np.concatenate(action_n[0:3]), rew_n[0:3], new_obs_n[0:3], done_n[0:3])
            memory_j.add(obs_n[-2:], np.concatenate(action_n[-2:]), rew_n[-2:], new_obs_n[-2:], done_n[-2:])
            episode_rewards_t[-1] += np.sum(rew_n[0:3])
            episode_rewards_j[-1] += np.sum(rew_n[-2:])
            for i, rew in enumerate(rew_n): agent_rewards[i][-1] += rew

            # train our agents
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train_mix( \
                arglist, game_step, update_cnt, memory_t, memory_j, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, type=type)

            # update the obs_n
            game_step += 1
            obs_n = new_obs_n
            done = all(done_n)
            terminal = (episode_cnt >= arglist.per_episode_max_len - 1)
            if done or terminal:
                # print(episode_rewards_t[-1],episode_rewards_j[-1], env.done_cnt)
                # print(episode_gone,game_step)
                obs_n = env.reset(random=True)
                new_obs_n = copy.deepcopy(obs_n)
                episode_rewards_t.append(0)
                episode_rewards_j.append(0)
                continue


def train_mix_attention(arglist):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """
    env = make_env()

    print('=============================')
    print('=1 Env {} is right ...'.format("resilient path planning"))
    print('=============================')

    """step2: create agents"""
    obs_shape_n = [env.observation_space[i] for i in range(env.n)]
    action_shape_n = [env.action_space[i] for i in range(env.n)]  # no need for stop bit
    # num_adversaries = min(env.n, arglist.num_adversaries)
    #初始化模型
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers_mix_attention(env.n, obs_shape_n, action_shape_n, arglist)
    #加载模型
    #actors_cur, critics_cur, actors_tar, critics_tar = load_mix_trainers(arglist)
    # actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
    #     get_all_trainers(arglist)
    # memory = Memory(num_adversaries, arglist)
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
    # rew_n_old = [0.0 for _ in range(env.n)]  # set the init reward
    # agent_info = [[[]]]  # placeholder for benchmarking info
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
    obs_n = env.reset(random=True)
    new_obs_n = copy.deepcopy(obs_n)
    date = {'mean_rw_t': [],
            'var_rw_t': [],
            'mean_rw_j': [],
            'var_rw_j': []}
    df = pd.DataFrame(date)
    df.to_csv('rw_attention.csv', index=False, mode='w')
    for episode_gone in range(arglist.max_episode):
        # cal the reward print the debug data
        if game_step > 1 and game_step % 4000 == 0:
            mean_rw_t = round(np.mean(episode_rewards_t[-400:-1]), 3)
            mean_rw_j = round(np.mean(episode_rewards_j[-400:-1]), 3)
            var_rw_t = round(np.var(episode_rewards_t[-400:-1]), 3)
            var_rw_j = round(np.var(episode_rewards_j[-400:-1]), 3)
            df = pd.read_csv('rw_attention.csv')
            df.loc[int(game_step / 4000), 'mean_rw_t'] = mean_rw_t
            df.loc[int(game_step / 4000), 'var_rw_t'] = var_rw_t
            df.loc[int(game_step / 4000), 'mean_rw_j'] = mean_rw_j
            df.loc[int(game_step / 4000), 'var_rw_j'] = var_rw_j
            print(" " * 3 + 'typical mean reward:{} jammer mean reward:{} at {}'.format(mean_rw_t, mean_rw_j,
                                                                                        game_step / 4000), end='\n')
            df.to_csv('rw_attention.csv', index=False)
        # print('=Training: steps:{} episode:{}'.format(game_step, episode_gone), end='\r')
        for episode_cnt in range(arglist.per_episode_max_len):
            # get action
            action_n = [agent(torch.tensor(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                        for agent, obs in zip(actors_cur, obs_n)]
            # print(action_n)
            # interact with env
            obs_n = copy.deepcopy(new_obs_n)
            new_obs_n, rew_n, done_n = env.step(action_n=action_n, timestep=episode_cnt, jammer_act=arglist.jammer_act)
            # print(new_obs_n, rew_n, done_n)
            # save the experience
            memory_t.add(obs_n[0:3], np.concatenate(action_n[0:3]), rew_n[0:3], new_obs_n[0:3], done_n[0:3])
            memory_j.add(obs_n[-2:], np.concatenate(action_n[-2:]), rew_n[-2:], new_obs_n[-2:], done_n[-2:])
            episode_rewards_t[-1] += np.sum(rew_n[0:3])
            episode_rewards_j[-1] += np.sum(rew_n[-2:])
            for i, rew in enumerate(rew_n): agent_rewards[i][-1] += rew

            # train our agents
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train_mix( \
                arglist, game_step, update_cnt, memory_t, memory_j, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, type="attention")

            # update the obs_n
            game_step += 1
            obs_n = new_obs_n
            done = all(done_n)
            terminal = (episode_cnt >= arglist.per_episode_max_len - 1)
            if done or terminal:
                # print(episode_rewards_t[-1],episode_rewards_j[-1], env.done_cnt)
                # print(episode_gone,game_step)
                obs_n = env.reset(random=True)
                new_obs_n = copy.deepcopy(obs_n)
                episode_rewards_t.append(0)
                episode_rewards_j.append(0)
                continue


def train_mix_double_q(arglist):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """
    env = make_env()

    print('=============================')
    print('=1 Env {} is right ...'.format("resilient path planning"))
    print('=============================')

    """step2: create agents"""
    obs_shape_n = [env.observation_space[i] for i in range(env.n)]
    action_shape_n = [env.action_space[i] for i in range(env.n)]  # no need for stop bit
    # num_adversaries = min(env.n, arglist.num_adversaries)
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers_mix_double_q(env.n, obs_shape_n, action_shape_n, arglist)
    # memory = Memory(num_adversaries, arglist)
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
    obs_n = env.reset(random = True)
    new_obs_n = copy.deepcopy(obs_n)
    date = {'mean_rw_t': [],
            'var_rw_t': [],
            'mean_rw_j': [],
            'var_rw_j': []}
    df = pd.DataFrame(date)
    df.to_csv('rw_mix_double_q.csv', index=False, mode='w')
    for episode_gone in range(arglist.max_episode):
        # cal the reward print the debug data
        if game_step > 1 and game_step % 4000 == 0:
            mean_rw_t = round(np.mean(episode_rewards_t[-400:-1]), 3)
            mean_rw_j = round(np.mean(episode_rewards_j[-400:-1]), 3)
            var_rw_t = round(np.var(episode_rewards_t[-400:-1]), 3)
            var_rw_j = round(np.var(episode_rewards_j[-400:-1]), 3)
            df = pd.read_csv('rw_mix_double_q.csv')
            df.loc[int(game_step / 4000), 'mean_rw_t'] = mean_rw_t
            df.loc[int(game_step / 4000), 'var_rw_t'] = var_rw_t
            df.loc[int(game_step / 4000), 'mean_rw_j'] = mean_rw_j
            df.loc[int(game_step / 4000), 'var_rw_j'] = var_rw_j
            print(" " * 3 + 'typical mean reward:{} jammer mean reward:{} at {}'.format(mean_rw_t, mean_rw_j,
                                                                                        game_step / 4000), end='\n')
            df.to_csv('rw_mix_double_q.csv', index=False)
        for episode_cnt in range(arglist.per_episode_max_len):
            # get action
            action_n = [agent(torch.tensor(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                        for agent, obs in zip(actors_cur, obs_n)]
            # print(action_n)
            # interact with env
            obs_n = copy.deepcopy(new_obs_n)
            new_obs_n, rew_n, done_n = env.step(action_n=action_n, timestep=episode_cnt, jammer_act=arglist.jammer_act)
            # print(new_obs_n, rew_n, done_n)
            # save the experience
            memory_t.add(obs_n[0:3], np.concatenate(action_n[0:3]), rew_n[0:3], new_obs_n[0:3], done_n[0:3])
            memory_j.add(obs_n[-2:], np.concatenate(action_n[-2:]), rew_n[-2:], new_obs_n[-2:], done_n[-2:])
            episode_rewards_t[-1] += np.sum(rew_n[0:3])
            episode_rewards_j[-1] += np.sum(rew_n[-2:])
            for i, rew in enumerate(rew_n): agent_rewards[i][-1] += rew

            # train our agents
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train_mix_double_q( \
                arglist, game_step, update_cnt, memory_t, memory_j, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, type="double_q")

            # update the obs_n
            game_step += 1
            obs_n = new_obs_n
            done = all(done_n)
            terminal = (episode_cnt >= arglist.per_episode_max_len - 1)
            if done or terminal:
                # print(episode_rewards_t[-1],episode_rewards_j[-1], env.done_cnt)
                # print(episode_gone,game_step)
                obs_n = env.reset(random=True)
                new_obs_n = copy.deepcopy(obs_n)
                episode_rewards_t.append(0)
                episode_rewards_j.append(0)
                continue


def train_mix_adq(arglist):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """
    env = make_env()

    print('=============================')
    print('=1 Env {} is right ...'.format("resilient path planning"))
    print('=============================')

    """step2: create agents"""
    obs_shape_n = [env.observation_space[i] for i in range(env.n)]
    action_shape_n = [env.action_space[i] for i in range(env.n)]  # no need for stop bit
    # num_adversaries = min(env.n, arglist.num_adversaries)
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers_adq(env.n, obs_shape_n, action_shape_n, arglist)
    # memory = Memory(num_adversaries, arglist)
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
    # t_start = time.time()
    # rew_n_old = [0.0 for _ in range(env.n)]  # set the init reward
    # agent_info = [[[]]]  # placeholder for benchmarking info
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
    obs_n = env.reset(random=True)
    new_obs_n = copy.deepcopy(obs_n)
    date = {'mean_rw_t': [],
            'var_rw_t': [],
            'mean_rw_j': [],
            'var_rw_j': []}
    df = pd.DataFrame(date)
    df.to_csv('rw_adq.csv', index=False, mode='w')
    for episode_gone in range(arglist.max_episode):
        # cal the reward print the debug data
        if game_step > 1 and game_step % 4000 == 0:
            mean_rw_t = round(np.mean(episode_rewards_t[-400:-1]), 3)
            mean_rw_j = round(np.mean(episode_rewards_j[-400:-1]), 3)
            var_rw_t = round(np.var(episode_rewards_t[-400:-1]), 3)
            var_rw_j = round(np.var(episode_rewards_j[-400:-1]), 3)
            df = pd.read_csv('rw_adq.csv')
            df.loc[int(game_step / 4000), 'mean_rw_t'] = mean_rw_t
            df.loc[int(game_step / 4000), 'var_rw_t'] = var_rw_t
            df.loc[int(game_step / 4000), 'mean_rw_j'] = mean_rw_j
            df.loc[int(game_step / 4000), 'var_rw_j'] = var_rw_j
            print(" " * 3 + 'typical mean reward:{} jammer mean reward:{} at {}'.format(mean_rw_t, mean_rw_j,
                                                                                        game_step / 4000), end='\n')
            df.to_csv('rw_adq.csv', index=False)
        for episode_cnt in range(arglist.per_episode_max_len):
            # get action
            action_n = [agent(torch.tensor(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                        for agent, obs in zip(actors_cur, obs_n)]
            # print(action_n)
            # interact with env
            obs_n = copy.deepcopy(new_obs_n)
            new_obs_n, rew_n, done_n = env.step(action_n=action_n, timestep=episode_cnt, jammer_act=arglist.jammer_act)
            # print(new_obs_n, rew_n, done_n)
            # save the experience
            memory_t.add(obs_n[0:3], np.concatenate(action_n[0:3]), rew_n[0:3], new_obs_n[0:3], done_n[0:3])
            memory_j.add(obs_n[-2:], np.concatenate(action_n[-2:]), rew_n[-2:], new_obs_n[-2:], done_n[-2:])
            episode_rewards_t[-1] += np.sum(rew_n[0:3])
            episode_rewards_j[-1] += np.sum(rew_n[-2:])
            for i, rew in enumerate(rew_n): agent_rewards[i][-1] += rew

            # train our agents
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train_mix_double_q( \
                arglist, game_step, update_cnt, memory_t, memory_j, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, type="adq")

            # update the obs_n
            game_step += 1
            obs_n = new_obs_n
            done = all(done_n)
            terminal = (episode_cnt >= arglist.per_episode_max_len - 1)
            if done or terminal:
                # print(episode_rewards_t[-1],episode_rewards_j[-1], env.done_cnt)
                # print(episode_gone,game_step)
                obs_n = env.reset(random=True)
                new_obs_n = copy.deepcopy(obs_n)
                episode_rewards_t.append(0)
                episode_rewards_j.append(0)
                continue

def get_typical_trainers( arglist):
    """ load the model """
    actors_typical = [torch.load(arglist.typical_model_name+'a_c_{}.pt'.format(agent_idx), map_location=arglist.device) \
        for agent_idx in range(3)]
    return actors_typical

def get_mix_trainers(arglist,model_name=''):
    """ load the model """
    if  model_name == '':
        actors = [torch.load(arglist.mix_model_name+'a_c_{}.pt'.format(agent_idx), map_location=arglist.device) \
            for agent_idx in range(5)]
    else:
        actors = [torch.load('models/'+model_name+'/a_c_{}.pt'.format(agent_idx), map_location=arglist.device) \
            for agent_idx in range(5)]
    return actors

def load_mix_trainers(arglist):
    """ load the model """
    actors_cur = [torch.load(arglist.mix_model_name+'/a_c_{}.pt'.format(agent_idx), map_location=arglist.device) \
        for agent_idx in range(5)]
    critics_cur = [torch.load(arglist.mix_model_name+'/c_c_{}.pt'.format(agent_idx), map_location=arglist.device) \
        for agent_idx in range(5)]
    actors_tar = [torch.load(arglist.mix_model_name+'/a_t_{}.pt'.format(agent_idx), map_location=arglist.device) \
        for agent_idx in range(5)]
    critics_tar = [torch.load(arglist.mix_model_name+'/c_t_{}.pt'.format(agent_idx), map_location=arglist.device) \
        for agent_idx in range(5)]
    return actors_cur,critics_cur,actors_tar,critics_tar

def get_jammer_trainers( arglist):
    """ load the model """
    actors_jammer = [torch.load(arglist.jammer_model_name+'a_c_{}.pt'.format(agent_idx), map_location=arglist.device) \
        for agent_idx in range(2)]
    return actors_jammer

# def enjoy_typical(arglist):
#     """
#     This func is used for testing the model
#     """
#     episode_step = 0
#     """ init the env """
#     env = make_env()
#     """ init the agents """
#     actors_tar = get_typical_trainers(arglist)
#     """ interact with the env """
#     obs_n = env.reset(random=True)
#     ax = plt.gca()
#     ax.set_aspect(1)
#     rect = plt.Rectangle((env.NFZones[0][0], env.NFZones[0][2]),
#                          env.NFZones[0][1] - env.NFZones[0][0],
#                          env.NFZones[0][3] - env.NFZones[0][2],
#                          fill=False)
#     ax.add_patch(rect)
#     for node in env.IoTNodes:
#         plt.plot(node.x, node.y, '^')
#     total_reward = 0
#     while(1):
#         # update the episode step number
#         episode_step += 1
#         # get action
#         action_n = []
#         for actor, obs in zip(actors_tar, obs_n):
#             model_out,_ = actor(torch.tensor(obs).to(arglist.device, torch.float),model_original_out=True)
#             action_n.append(model_out.detach().cpu().numpy())
#         action_n.append(np.array([0, 0]))
#         action_n.append(np.array([0, 0]))
#         # interact with env
#         obs_n, rew_n, done_n= env.step(action_n=action_n,timestep=episode_step)
#
#         total_reward += np.sum(rew_n)
#         for uav in env.UAVs:
#             plt.plot(uav.pV[0], uav.pV[1], '.', color=colors[uav.num])
#             #if uav.num==1:print(uav.vV)
#         for jammer in env.JAMMERs:
#             plt.plot(jammer.pV[0], jammer.pV[1], '.', color='red')
#         # update the flag
#         done = all(done_n)
#         terminal = (episode_step >= arglist.per_episode_max_len)
#
#         # reset the env
#         if done or terminal:
#             for node in env.IoTNodes:
#                 print(node.done,node.data)
#             print("传输完成节点个数：", env.done_cnt)
#             print(total_reward)
#             plt.show()
#             break
#
# def enjoy_jammer(arglist):
#     """
#     This func is used for testing the model
#     """
#     episode_step = 0
#     """ init the env """
#     env = make_env()
#     """ init the agents """
#     actors_typical = get_typical_trainers(arglist)
#     actors_jammer = get_jammer_trainers(arglist)
#     """ interact with the env """
#     obs_n = env.reset(random=True)
#     ax = plt.gca()
#     ax.set_aspect(1)
#     rect = plt.Rectangle((env.NFZones[0][0], env.NFZones[0][2]),
#                          env.NFZones[0][1] - env.NFZones[0][0],
#                          env.NFZones[0][3] - env.NFZones[0][2],
#                          fill=False)
#     ax.add_patch(rect)
#     for node in env.IoTNodes:
#         plt.plot(node.x, node.y, '^')
#     total_reward = 0
#     while(1):
#         # update the episode step number
#         episode_step += 1
#         # get action
#         action_n = []
#         for actor, obs in zip(chain(actors_typical,actors_jammer), obs_n):
#             model_out,_ = actor(torch.tensor(obs).to(arglist.device, torch.float),model_original_out=True)
#             action_n.append(model_out.detach().cpu().numpy())
#         #print(action_n)
#         # interact with env
#         obs_n, rew_n, done_n= env.step(action_n=action_n,timestep=episode_step)
#         total_reward += np.sum(rew_n)
#         for uav in env.UAVs:
#             plt.plot(uav.pV[0], uav.pV[1], '.', color=colors[uav.num])
#             #if uav.num==1:print(uav.vV)
#         for jammer in env.JAMMERs:
#             plt.plot(jammer.pV[0], jammer.pV[1], '.', color='red')
#         # update the flag
#         done = all(done_n)
#         terminal = (episode_step >= arglist.per_episode_max_len)
#
#         # reset the env
#         if done or terminal:
#             for node in env.IoTNodes:
#                 print(node.done,node.data)
#             print("传输完成节点个数：", env.done_cnt)
#             print(total_reward)
#             plt.show()
#             break