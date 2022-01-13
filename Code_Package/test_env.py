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

import pybullet as p
import time
import pybullet_data
import numpy as np
from rrt.search_space import SearchSpace
from rrt.rrt_star import RRTStar
from rrt.rrt import RRT

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
parser.add_argument('--lr', type=float, default=0.00005, metavar='G',
                    help='learning rate (default: 0.00001)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
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

#-------------------------------------------------------------------
# Environment
env = ENV(GUI=True)
env.reset()
steps(200)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
num_inputs = 7 + 4 + 6 + 3 # 1object  orientation & position, 1obstacle position & closeset distance, 6 joints states
action_dimensions = 6 # 6 rotational axis angle for arm control
agent = SAC(num_inputs, action_dimensions, args)
agent.load_checkpoint('D:\Learning\\Newlearning\COMS6998\Bullet_test\PlanA\checkpoints\sac_checkpoint_2021-12-08_10-11-41_')

# success
num_trails = 20
success_all = 0
time_box = []

test_mode = 'RRT'

if test_mode == 'RL':
    time.sleep(5)
    for i_episode in range(num_trails):
        start_time = time.time()
        done = False
        state = env.reset()

        while not done:
            action = agent.select_action(state)  # Sample action from policy
            next_state, success, done = env.step(action, 'eval') # Step
            state = next_state
            if success: success_all += 1; time.sleep(0.8)
        if success:
            time_consume = time.time() - start_time
            print(time_consume)
            time_box.append(time_consume)

    print('Process Over')

    success_rate = success_all / num_trails
    mean_time = np.mean(time_box)
    print(success_rate, mean_time)

elif test_mode == 'RRT':
    time.sleep(5)
# rrt--------------------------------
    for i in range(num_trails):
        env.reset()
        start_time = time.time()
        while not env.conveyor.check_done():
            success = env.planner_step()
            if success != None:
                time.sleep(0.8)
                success_all += success
                break;
            if env.ur5.collision_check(env.all_id):
                print('Collision!'); break

        if success:
            time_consume = time.time() - start_time
            print(time_consume)
            time_box.append(time_consume)

    print('end')
    success_rate = success_all / num_trails
    mean_time = np.mean(time_box)
    print(success_rate, mean_time)

# env.close()