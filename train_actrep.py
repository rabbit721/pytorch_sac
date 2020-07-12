#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

# from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

import gym
import gym_gvgai
import hydra

GRID_COLS = 16
GRID_ROWS = 16
AVATAR_VALUE = 1

import imageio

from scipy import misc
misc.imread = imageio.imread

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def get_grid_state(env):
    grid = env.env.GVGAI.sso.observationGrid

    cols = len(grid)
    rows = len(grid[0])

    output_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)

    prefix = 10
    for col in range(cols):
        for row in range(rows):
            if grid[col][row][0]:
                output_grid[row][col] = prefix + grid[col][row][0].itype

    avatar_position = np.array(
      (np.array(env.env.GVGAI.sso.avatarPosition, np.int32) /
       env.env.GVGAI.sso.blockSize), np.int32)
    output_grid[avatar_position[1]][avatar_position[0]] = AVATAR_VALUE

    return output_grid

def make_env(env_name):
    """Helper function to create dm_control environment"""
    env = gym_gvgai.make(env_name)
    gym_gvgai.envs.gvgai_env.gvgai.LEARNING_SSO_TYPE.IMAGE = (
        gym_gvgai.envs.gvgai_env.gvgai.LEARNING_SSO_TYPE.BOTH
    )
    return env

def softmax(x):
    return np.exp(x)/sum(np.exp(x))


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.observation_space_shape = (16, 16)
        self.device = device
        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.env = make_env(cfg.env)
        self.max_episode_steps = cfg.max_episode_steps

        cfg.agent.params.obs_dim = self.observation_space_shape
        # SET action_dim = env.action_space.n
        cfg.agent.params.action_dim = (self.env.action_space.n)
        cfg.agent.params.action_range = [
            float(0), float(self.env.action_space.n)
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)
        print(self.agent.decoderf)
        self.replay_buffer = ReplayBuffer(self.observation_space_shape,
                                          (cfg.agent.params.latent_dim),
                                          int(cfg.replay_buffer_capacity),
                                          self.device)
        assert(cfg.agent.params.batch_size <= cfg.replay_buffer_capacity)
        '''
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        '''
        self.step = 0

    def evaluate(self):
        print("evaluate")
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            self.env.reset()
            obs = get_grid_state(self.env)
            self.agent.reset()
            # self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            step_count = 0
            while not done and step_count < self.max_episode_steps:
                with utils.eval_mode(self.agent):
                    latent_vec = self.agent.act(obs, sample=False)

                # TRANSFORM latent_vec to action

                action, action_prob = self.agent.cont_to_prob(torch.from_numpy(latent_vec).float().to(self.device))
                step_count += 1
                _, reward, done, _ = self.env.step(action)
                obs = get_grid_state(self.env)
                # self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            # self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        rewards = []
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                rewards.append(episode_reward)
                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                self.env.reset()
                obs = get_grid_state(self.env)
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                # print("episode", episode)
                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                latent_vec = torch.from_numpy(np.random.normal(0, 1, (1, self.env.action_space.n))).float().to(self.device)
            else:
                with utils.eval_mode(self.agent):
                    latent_vec = torch.from_numpy(self.agent.act(obs, sample=True)).float().to(self.device)

            # TODO: transform latent_vec into action
            action, action_prob = self.agent.cont_to_prob(latent_vec)
            # print("before update")
            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            '''
            for param in self.agent.decoderf.parameters():
                assert(param.requires_grad)
                print(param.name, param.data)
            '''

            # print("after update")
            # print(latent_vec.shape, type(latent_vec), latent_vec)
            _, reward, done, _ = self.env.step(action)
            # if done:
            #    print("done")
            next_obs = get_grid_state(self.env)
            # allow infinite bootstrap
            done = float(done) or episode_step + 1 == self.max_episode_steps
            done_no_max = 0 if episode_step + 1 == self.max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, latent_vec.detach().cpu().numpy(), action.item(), reward, next_obs, done,
                                   done_no_max)

            self.agent.approximate(self.replay_buffer)
            obs = next_obs
            episode_step += 1
            self.step += 1

            if len(rewards) >= 10:
                print("----- Mean Ep Reward ----- ", sum(rewards)/len(rewards))
                rewards = []
            # print("self.step", self.step)


@hydra.main(config_path='config/train-actrep.yaml', strict=True)
def main(cfg):
    torch.set_default_tensor_type(torch.FloatTensor)
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
