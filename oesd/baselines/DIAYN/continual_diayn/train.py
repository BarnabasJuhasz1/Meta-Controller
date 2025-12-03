
import os
import random
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

from models import Agent, Discriminator
from utils import make_env

def train():
    with open("config.yaml", "r") as f: cfg = yaml.safe_load(f)
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() and cfg['logging']['device'] == "cuda" else "cpu")
    print(f"🚀 Training on {device}")

    # Vectorized Environments (Speed!)
    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg['env']['id']) for i in range(cfg['env']['num_envs'])]
    )

    # Initialize Networks
    agent = Agent(envs.single_action_space.n, cfg['agent']['skill_dim']).to(device)
    discriminator = Discriminator(cfg['agent']['skill_dim']).to(device)
    
    optimizer = optim.Adam(agent.parameters(), lr=cfg['agent']['lr'], eps=1e-5)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=cfg['training']['discriminator_lr'])

    # Constants
    num_steps = cfg['training']['num_steps']
    num_envs = cfg['env']['num_envs']
    batch_size = int(num_envs * num_steps)
    skill_dim = cfg['agent']['skill_dim']
    
    # Buffers
    obs_img = torch.zeros((num_steps, num_envs, 3, 7, 7)).to(device)
    obs_state = torch.zeros((num_steps, num_envs, 3)).to(device)
    skills = torch.zeros((num_steps, num_envs, skill_dim)).to(device)
    actions = torch.zeros((num_steps, num_envs)).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # Init Skills
    current_skills = torch.zeros(num_envs, skill_dim).to(device)
    for i in range(num_envs):
        current_skills[i][random.randint(0, skill_dim-1)] = 1.0

    # Start Env
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_img = torch.Tensor(next_obs['image']).to(device)
    next_state = torch.Tensor(next_obs['state']).to(device)
    next_done = torch.zeros(num_envs).to(device)

    num_updates = cfg['training']['total_timesteps'] // batch_size
    print(f"Starting {num_updates} updates...")

    for update in range(1, num_updates + 1):
        # 1. Collect Data (Rollout)
        for step in range(num_steps):
            global_step += num_envs
            obs_img[step] = next_img
            obs_state[step] = next_state
            dones[step] = next_done
            skills[step] = current_skills

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_img, next_state, current_skills)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, termin, trunc, _ = envs.step(action.cpu().numpy())
            next_done = np.logical_or(termin, trunc)
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_img = torch.Tensor(next_obs['image']).to(device)
            next_state = torch.Tensor(next_obs['state']).to(device)
            next_done = torch.Tensor(next_done).to(device)

            # Reset skills on done
            for i, d in enumerate(next_done):
                if d:
                    current_skills[i] = 0
                    current_skills[i][random.randint(0, skill_dim-1)] = 1.0

        # 2. Calculate Rewards (The Magic Sauce)
        with torch.no_grad():
            flat_states = obs_state.view(-1, 3)
            flat_skills = skills.view(-1, skill_dim)
            
            # DIAYN
            disc_logits = discriminator(flat_states)
            p_z = F.softmax(disc_logits, dim=1)
            log_p_z = torch.log(p_z + 1e-6)
            diayn_reward = (log_p_z * flat_skills).sum(dim=1) - np.log(1.0/skill_dim)
            
            # Bonuses
            has_key = flat_states[:, 2] # 1.0 if key
            in_room_2 = (flat_states[:, 0] > 0.6).float() # 1.0 if passed wall
            
            key_bonus = has_key * cfg['training']['key_bonus']
            door_bonus = in_room_2 * cfg['training']['door_bonus']
            
            # Combine (Reshape to steps, envs)
            diayn_reward = diayn_reward.view(num_steps, num_envs)
            key_bonus = key_bonus.view(num_steps, num_envs)
            door_bonus = door_bonus.view(num_steps, num_envs)
            
            # Total = DIAYN + Extrinsic(Win) + Bonuses
            total_rewards = (diayn_reward * cfg['training']['diayn_scale']) +                             (rewards * cfg['training']['win_bonus']) +                             key_bonus + door_bonus

        # 3. Bootstrap (GAE)
        with torch.no_grad():
            next_value = agent.get_value(next_img, next_state, current_skills).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = total_rewards[t] + cfg['agent']['gamma'] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg['agent']['gamma'] * cfg['agent']['gae_lambda'] * nextnonterminal * lastgaelam
            returns = advantages + values

        # 4. Flatten Batch
        b_img = obs_img.reshape((-1, 3, 7, 7))
        b_state = obs_state.reshape((-1, 3))
        b_skills = skills.reshape((-1, skill_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 5. PPO Update
        b_inds = np.arange(batch_size)
        for epoch in range(cfg['training']['update_epochs']):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, 128): # Mini-batch
                end = start + 128
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_img[mb_inds], b_state[mb_inds], b_skills[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1-cfg['agent']['clip_coef'], 1+cfg['agent']['clip_coef'])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                
                loss = pg_loss - cfg['agent']['ent_coef'] * entropy.mean() + v_loss * cfg['agent']['vf_coef']

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg['agent']['max_grad_norm'])
                optimizer.step()

        # 6. Discriminator Update
        disc_pred = discriminator(b_state)
        disc_loss = F.cross_entropy(disc_pred, torch.argmax(b_skills, dim=1))
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

        # 7. Logs
        if update % 10 == 0:
            print(f"Upd {update}/{num_updates} | FPS:{int(global_step/(time.time()-start_time))} | R_Tot:{total_rewards.mean():.2f} | Bonus_Key:{key_bonus.mean():.2f} | Bonus_Door:{door_bonus.mean():.2f} | Win:{rewards.sum():.2f}")
        
        if update % cfg['logging']['save_interval'] == 0:
            torch.save(agent.state_dict(), "continual_agent.pth")

    torch.save(agent.state_dict(), "continual_agent.pth")
    print("Done!")

if __name__ == "__main__":
    train()
