import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from p_utils import *
import joblib
import os

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Dynamic Grasping",
                    help='Pybullet environment')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.95, metavar='G',
                    help='discount factor for reward (default: 0.95)')
parser.add_argument('--tau', type=float, default=0.0005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.0005)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='G',
                    help='learning rate (default: 0.00001)') # first 0.0001 then 0.00005
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# manual save
if not os.path.exists('Trainresult/'): os.mkdir('Trainresult/')
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Environment
env = ENV(GUI=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
num_inputs = 7 + 4 + 6 + 3 # 1object  orientation & position, 1obstacle position & closeset distance, 6 joints states, end-effector position
action_dimensions = 6 # 6 rotational axis angle for arm control
load_Flag = 0 # train model
agent = SAC(num_inputs, action_dimensions, args)
agent.load_checkpoint('D:\Learning\\Newlearning\COMS6998\Bullet_test\PlanA\checkpoints\sac_checkpoint_2021-12-08_10-11-41_')
load_Flag = 1 # evaluate model

#Tesnorboard
writer = SummaryWriter('COMS6998/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
database = {'critic_1':[], 'critic_2':[], 'policy':[], 'entropy':[], 'entropy_alpha':[], 'reward_train':[], 'reward_test':[]}

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False

    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps and load_Flag == 0:
            action = env.random_action()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size and load_Flag == 0:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)

                database['critic_1'].append([updates, critic_1_loss])
                database['critic_2'].append([updates, critic_2_loss])
                database['policy'].append([updates, policy_loss])
                database['entropy'].append([updates, ent_loss])
                database['entropy_alpha'].append([updates, alpha])

                updates += 1

        next_state, reward, done = env.step(action, model='train') # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        # mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        mask = float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    database['reward_train'].append([i_episode, episode_reward])
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done= env.step(action, model='train')
                episode_reward += reward

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)
        database['reward_test'].append([i_episode, avg_reward])

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")
        joblib.dump(database, 'Trainresult/' + current_time)

print('Process Over')
agent.save_checkpoint(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


env.close()

