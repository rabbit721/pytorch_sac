import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
from collections import defaultdict, Counter

from agent import Agent

import utils
import hydra

import imageio

from scipy import misc
misc.imread = imageio.imread

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def get_indexed_embeddings(obj_id_to_embedding, vocab_size):
    embed_size = len(list(obj_id_to_embedding.items())[0][1])
    indexed_embedding_map = np.zeros(shape=(vocab_size, embed_size))
    for obj_id in obj_id_to_embedding:
        indexed_embedding_map[obj_id] = obj_id_to_embedding[obj_id]
    return indexed_embedding_map

def get_text_state(grid_state, indexed_embedding_map):
    if len(grid_state.shape) > 2:
        text_state = indexed_embedding_map[grid_state.data.cpu().numpy().astype(int)]
        # print("text_state.shape", text_state.shape)
        _, w, h, c = text_state.shape
        return np.moveaxis(text_state, [0, 1, 2, 3], [0, 2, 3, 1])
    else:
        text_state = indexed_embedding_map[grid_state.data.cpu().numpy().astype(int)]
        # print(text_state.shape)
        return np.moveaxis(text_state, [0, 1, 2], [1, 2, 0])

class ActRepAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obj_id_to_embedding_file, vocab_size,
                 state_embed_size, text_embed_size, obs_dim,
                 action_range, action_dim, latent_dim,
                 critic_cfg, actor_cfg,
                 fusion_cfg, approxg_cfg, decoderf_cfg,
                 discount, init_temperature, fusion_lr,
                 fusion_betas, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency,
                 critic_lr, critic_betas, critic_tau,
                 critic_target_update_frequency,
                 approxg_lr, approxg_betas,
                 decoderf_lr, decoderf_betas,
                 batch_size, learnable_temperature):
        super().__init__()

        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.obj_id_to_embedding = pickle.load(open(obj_id_to_embedding_file, 'rb'))
        self.indexed_embedding_map = get_indexed_embeddings(self.obj_id_to_embedding, vocab_size)

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.fusion = hydra.utils.instantiate(fusion_cfg).to(self.device)

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # Model in 3a, env approximator
        self.approxg = hydra.utils.instantiate(approxg_cfg).to(self.device)
        self.decoderf = hydra.utils.instantiate(decoderf_cfg).to(self.device)

        # optimizers
        self.fusion_optimizer = torch.optim.Adam(self.fusion.parameters(),
                                                lr=fusion_lr,
                                                betas=fusion_betas)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.approxg_optimizer = torch.optim.Adam(self.approxg.parameters(),
                                                  lr=approxg_lr,
                                                  betas=approxg_betas)

        self.decoderf_optimizer = torch.optim.Adam(self.decoderf.parameters(),
                                                   lr=decoderf_lr,
                                                   betas=decoderf_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.fusion.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.approxg.train(training)
        self.decoderf.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        grid_state = torch.from_numpy(obs).unsqueeze(0).long().to(self.device)
        text_state = torch.from_numpy(get_text_state(grid_state, self.indexed_embedding_map)).float().to(self.device)
        dist = self.actor(self.fusion((grid_state, text_state)))
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        # assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def cont_to_prob(self, latent_vec):
        prob = self.decoderf(latent_vec)
        action = prob.multinomial(num_samples=1).data
        return action, prob

    def update_critic(self, obs, action_vec, reward, next_obs, not_done, logger,
                      step):
        # print("obs.shape, next_obs.shape", obs.shape, next_obs.shape)
        grid_state = obs.long().to(self.device)
        text_state = torch.from_numpy(get_text_state(grid_state, self.indexed_embedding_map)).float().to(self.device)

        fused = self.fusion((grid_state, text_state))

        grid_state = next_obs.long().to(self.device)
        text_state = torch.from_numpy(get_text_state(grid_state, self.indexed_embedding_map)).float().to(self.device)

        next_fused = self.fusion((grid_state, text_state))
        # print("next_fused.shape", next_fused.shape)

        dist = self.actor(next_fused)
        next_action_vec = dist.rsample()

        log_prob = dist.log_prob(next_action_vec).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_fused, next_action_vec)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(fused, action_vec)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic and fusion model
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        grid_state = obs.long().to(self.device)
        text_state = torch.from_numpy(get_text_state(grid_state, self.indexed_embedding_map)).float().to(self.device)

        fused = self.fusion((grid_state, text_state))
        dist = self.actor(fused)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(fused, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action_vec, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size)
        # print(type(obs), type(next_obs), obs.shape, next_obs.shape)
        logger.log('train/batch_reward', reward.mean(), step)

        self.fusion_optimizer.zero_grad()
        self.update_critic(obs, action_vec, reward, next_obs, not_done_no_max,
                           logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
        self.fusion_optimizer.step()

    def approximate(self, replay_buffer):
        obs, action_vec, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.get_latest_batch(self.batch_size)
        prev_grid, next_grid = obs.long().to(self.device), next_obs.long().to(self.device)
        prev_text = torch.from_numpy(get_text_state(prev_grid, self.indexed_embedding_map)).float().to(self.device)
        next_text = torch.from_numpy(get_text_state(next_grid, self.indexed_embedding_map)).float().to(self.device)

        prev_fusion = self.fusion((prev_grid, prev_text))
        next_fusion = self.fusion((next_grid, next_text))
        Et = self.approxg((prev_fusion, next_fusion))
        At = self.decoderf(Et)
        # print(At.device)
        hist = defaultdict(list)
        etcounter = dict()
        # print("obs.shape", obs.shape, "At.shape", At.shape)
        for i in range(obs.shape[0]):
            state, next_state = obs[i].detach(), next_obs[i].detach()
            string_rep = " ".join(map(str, state.flatten())) + "; " + \
                         " ".join(map(str, next_state.flatten()))
            hist[string_rep].append(int(action[i].detach().item()))
            etcounter[string_rep] = At[i, :]

        total_loss = torch.zeros(1).to(At.device)
        for string_rep, actions in hist.items():
            state = np.fromstring(string_rep.split(";")[0], dtype=int, sep=' ')
            next_state = np.fromstring(string_rep.split(";")[1], dtype=int, sep=' ')
            counts = Counter(actions)
            # print(counts)

            action_prob = etcounter[string_rep]
            # print(action_prob.device)
            total_act = len(actions)

            curr_loss = torch.zeros(1).to(At.device)
            for act in counts:
                curr_loss += counts[act] * torch.log(action_prob[act])

            total_loss += (curr_loss / total_act)
        print("--- approximator loss: ", total_loss.item(), "---")

        self.approxg_optimizer.zero_grad()
        self.decoderf_optimizer.zero_grad()
        total_loss.backward()
        self.approxg_optimizer.step()
        self.decoderf_optimizer.step()
