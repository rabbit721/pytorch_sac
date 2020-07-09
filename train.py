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

        self.replay_buffer = ReplayBuffer(self.observation_space_shape,
                                          (self.env.action_space.n),
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

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
                    action_vec = self.agent.act(obs, sample=False)

                # TRANSFORM action_vec to action
                action = self.cont_to_disc(action_vec)
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

    def cont_to_disc(self, action_vec):
        # action_vec shape 1 x k, where k == env.action_space.n
        # print(action_vec.shape)
        # print(type(action_vec))
        action_vec_softmax = softmax(action_vec)
        disc_action = list(np.random.multinomial(1, action_vec_softmax, size=1)[0]).index(1)
        return disc_action

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
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
                action_vec = torch.from_numpy(np.random.normal(0, 1, self.env.action_space.n))
            else:
                with utils.eval_mode(self.agent):
                    action_vec = self.agent.act(obs, sample=True)

            # TODO: transform action_vec into action
            action = self.cont_to_disc(action_vec)
            # print("before update")
            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)
            # print("after update")
            # print(action_vec.shape, type(action_vec), action_vec)
            _, reward, done, _ = self.env.step(action)
            if done:
               print("done")
            next_obs = get_grid_state(self.env)
            # allow infinite bootstrap
            done = float(done) or episode_step + 1 == self.max_episode_steps
            done_no_max = 0 if episode_step + 1 == self.max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action_vec, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            # print("self.step", self.step)


@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
