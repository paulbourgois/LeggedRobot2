# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import os, sys
import gymnasium as gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

# utils
from env.quadruped_gym_env import QuadrupedGymEnv
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results
from enum import Enum

LEARNING_ALG = "PPO"
interm_dir = "./logs/intermediate_models/"

# ---------- Policy Selector ----------
class PolicyName(str, Enum):
    BASELINE_FLAT = "baseline_flat"
    FLAT_WITH_ALL_NOISE = "flat_with_all_noise"
    FLAT_SELECTIVE_NOISE = "flat_selective_noise"
    FLAT_ADD_MASS = "flat_add_mass"
    SLOPE_NO_NOISE = "slope_no_noise"
    SLOPE_ADD_NOISE_AND_MASS = "slope_add_noise_and_mass"
    DEV_TEST = "dev_test"

POLICY_CONFIG = {
    PolicyName.BASELINE_FLAT: (
        "121625215450_baseline_full",
        {
            "terrain": None,
            "add_noise": False,
            "add_base_mass": False,
        }
    ),
    PolicyName.FLAT_WITH_ALL_NOISE: (
        "121825111022_flat_add_all_noise_best_0.8mpers",
        {
            "terrain": None,
            "add_noise": True,
            "add_base_mass": False,
        }
    ),
    PolicyName.FLAT_SELECTIVE_NOISE: (
        "121825165652_add_selective_noise",
        {
            "terrain": None,
            "add_noise": True,
            "add_base_mass": False,
        }
    ),
    PolicyName.FLAT_ADD_MASS: (
        "121825190758_flat_add_mass_500000",
        {
            "terrain": None,
            "add_noise": True,
            "add_base_mass": True,
        }
    ),
    PolicyName.SLOPE_NO_NOISE: (
        "121725230254_slope4_0.6to0.9_adjusted_learning_curve",
        {
            "terrain": "SLOPES",
            "terrain_difficulty": 4,
            "add_noise": False,
            "add_base_mass": False,
        }
    ),
    PolicyName.SLOPE_ADD_NOISE_AND_MASS: (
        "121825194935_slop_add_mass_1500000",
        {
            "terrain": "SLOPES",
            "terrain_difficulty": 4,
            "add_noise": True,
            "add_base_mass": True,
        }
    ),
    PolicyName.DEV_TEST: (
        "121825231820_slope_noise_mass",
        {
            "terrain": "SLOPES",
            "terrain_difficulty": 4,
            "add_noise": True,
            "add_base_mass": True,
        }
    ),
}

# ---------- Choose policy to test ----------
chosen_policy = PolicyName.SLOPE_ADD_NOISE_AND_MASS  # 👈 change this line to test other policies. 
#### If  you select SLOPES, you can change the terrain difficulty: upto 5 with no noise (4 = 0.2 pitch)

log_subdir, overrides = POLICY_CONFIG[chosen_policy]
log_dir = os.path.join(interm_dir, log_subdir)

# ---------- Build env_config ----------
env_config = {
    "motor_control_mode": "CPG",
    "task_env": "LR_COURSE_TASK",
    "terrain": None,
    "terrain_difficulty": 0,
    "observation_space_mode": "FULL",
    "render": True,
    "record_video": False,
    "add_noise": False,
    "add_base_mass": False,
}
env_config.update(overrides)




# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
# plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + "_" + chosen_policy.value , save_dir="report_images") [TODO]: hey team, use this to save our plots
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + "_" + chosen_policy.value)
plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")



#########################
## Hi TAs, you can use this to change the velocity command that you want
#########################
env.venv.env_method("set_command", 0.80, 0.0, 0.0, randomize=False, override = True)
obs = env.reset()
episode_reward = 0

for i in range(7000):
    #########################
    ## Hi TAs, you can change here also if you want 😉
    #########################
    if chosen_policy in [PolicyName.BASELINE_FLAT, PolicyName.FLAT_ADD_MASS, PolicyName.FLAT_SELECTIVE_NOISE]:
        # change command every 400 steps
        if (i % 800 > 400):
            env.venv.env_method("set_command", 0.5, 0.0, 0.0, randomize=False, override = True)
        else:
            env.venv.env_method("set_command", 0.8, 0.0, 0.0, randomize=False, override = True)

    
    action, _states = model.predict(obs,deterministic=True) # sample at test time? ([TODO]: test if the outputs make sense)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0
