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

"""This file implements the gym environment for a quadruped. """

import os, inspect
# so we can import files
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
import sys

# misc
import time, datetime
import numpy as np
from typing import Optional

# gym
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

# pybullet
import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
import random
random.seed(10)

# quadruped and configs
import quadruped
import configs_a1 as robot_config
from hopf_network import HopfNetwork

# helper functions
def unit_vector(vector):
	""" Returns the unit vector of the vector.  """
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
	""" Returns the angle in radians between vectors 'v1' and 'v2' """
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotation_matrix(theta):
	return np.array([ [np.cos(theta), -np.sin(theta) ], [np.sin(theta), np.cos(theta)] ])

NUM_LEGS = 4
ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
VIDEO_LOG_DIRECTORY = 'videos/' + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f")

##################################################################################
#  Important details!
##################################################################################
"""
Implemented observation spaces for deep reinforcement learning:
  "DEFAULT": motor angles and velocities, body orientation
  "LR_COURSE_OBS": [TODO: what should you include? what is reasonable to measure on the real system? CPG states?]

Tasks to be learned with reinforcement learning:
    - "FWD_LOCOMOTION":
        reward forward progress only
    - "FLAGRUN":
        move to goal, once reached, a new goal is randomly selected. This is just for you test if you want to try moving the robot to custom waypoints.
    - "LR_COURSE_TASK":
        [TODO: what should you train for?]
        Ideally we want to command A1 to run in any direction while expending minimal energy
        It is suggested to first train to run at 3 sample velocities (0.5 m/s, 1 m/s, 1.5 m/s)
        How will you construct your reward function?

Motor control modes:
  - "TORQUE":
        supply raw torques to each motor (12 values)
  - "PD":
        supply desired joint positions to each motor (12 values)
        torques are computed based on the joint position/velocity error
  - "CARTESIAN_PD":
        supply desired foot positions for each leg (12 values)
        torques are computed based on the foot position/velocity error
  - "CPG":
        supply desired CPG state modulations (8 values), mapped to foot positions
        torques are computed based on inverse kinematics + joint PD (or you can add Cartesian PD)
"""

EPISODE_LENGTH = 10   # how long before we reset the environment (max episode length for RL)
MAX_FWD_VELOCITY = 1  # to avoid exploiting simulator dynamics, cap max reward for body velocity

# CPG quantities, matching with ones in hopf_network.py
MU_LOW = 1
MU_UPP = 2
ALPHA =  150
F_MAX = 100 # actually I just put an arbitrary large number here, i think it does not mater anyway actually -- Nathan

class QuadrupedGymEnv(gym.Env):
  """The gym environment for a quadruped {Unitree A1}.
  It simulates the locomotion of a quadrupedal robot.
  The state space, action space, and reward functions can be chosen with:
  observation_space_mode, motor_control_mode, task_env.
  """
  def __init__(
      self,
      robot_config=robot_config,
      isRLGymInterface=True,
      time_step=0.001,
      action_repeat=10,
      motor_control_mode="CPG",
      task_env="FWD_LOCOMOTION",
      observation_space_mode="CPG",
      on_rack=False,
      render=False,
      record_video=True,
      add_noise=True,
      terrain=None,
      terrain_difficulty = 1,
      test_flagrun=False,
      **kwargs): # any extra arguments from legacy
    """Initialize the quadruped gym environment.
    Args:
      robot_config: The robot config file, contains A1 parameters.

      isRLGymInterface: If the gym environment is being run as RL or not. Affects if the actions should be scaled.

      time_step: Simulation time step.

      action_repeat: The number of simulation steps where the same actions are applied.

      motor_control_mode: Whether to use Torque control, PD, Cartesian control or CPG.

      task_env: Task trying to learn (fwd locomotion, task specific, etc.)

      observation_space_mode: what should be in here? Check available functions in quadruped.py. also consider CPG states (amplitudes/phases).

      on_rack: Whether to place the quadruped on rack. This is only used to debug the walking gait. In this mode, the quadruped's base is hanged midair so
      that its walking gait is clearer to visualize.

      render: Whether to render the simulation.

      record_video: Whether to record a video of each trial.

      add_noise: vary coefficient of friction etc.

      terrain: string indicating what kind of terrain ("STAIRS", "SLOPES", "GAPS", "RANDOM"). If you want flat terrain, just put None.

      test_flagrun: follow certain goals in order, fixed coefficient of friction
    """
    self._robot_config = robot_config
    self._isRLGymInterface = isRLGymInterface
    self._time_step = time_step
    self._action_repeat = action_repeat
    self._motor_control_mode = motor_control_mode
    self._TASK_ENV = task_env
    self._observation_space_mode = observation_space_mode
    self._hard_reset = True # must fully reset simulation at init
    self._on_rack = on_rack
    self._is_render = render
    self._is_record_video = record_video
    self._add_noise = add_noise
    self._using_test_env = test_env
    self._test_flagrun = test_flagrun
    self.goal_id = None
    self._terrain = terrain
    self._terrain_difficulty = terrain_difficulty  # to augment our custom terrain
    if self._add_noise:
      self._observation_noise_stdev = 0.01 #
    else:
      self._observation_noise_stdev = 0.0

    self.des_velocity = np.array([1.2, 0, 0])
    self.vx_max = 1.2
    self.vx_min = 0.8
    self.vy_max = 1.2
    self.vy_min = 0.2
    self.wz_max = 1.0


    # reward weights
    #self.w_vx = 0.75
    #self.w_vy = 0.75
    self.w_vx = 1.5
    self.w_vy = 1.4
    self.w_yaw = 0.75
    self.w_vz_pen = 2.0
    self.w_ang_pen = 0.4
    self.w_pos_y_pen = 0.4
    self.w_ry_pen = 0.4
    self.w_work = 0.001

    ### reward terms tracking
    self._rew_terms_step = {}
    self._rew_terms_ep = {}
    self._ep_len = 0
    self._ep_done_flag = False

    # other bookkeeping
    self._num_bullet_solver_iterations = int(300 / action_repeat)
    self._env_step_counter = 0
    self._sim_step_counter = 0
    self._last_base_position = [0, 0, 0]
    self._last_frame_time = 0.0 # for rendering
    self._MAX_EP_LEN = EPISODE_LENGTH # max sim time in seconds, arbitrary
    self._action_bound = 1.0

    # if using CPG
    self.setupCPG()
    self.setupActionSpace()
    self.setupObservationSpace()
    if self._is_render:
      self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
    else:
      self._pybullet_client = bc.BulletClient()
    self._configure_visualizer()
    self.videoLogID = None
    self.seed()
    self.reset()

  def setupCPG(self):
    self._cpg = HopfNetwork(use_RL=True, alpha=ALPHA)

  ######################################################################################
  # RL Observation and Action spaces
  ######################################################################################
  def setupObservationSpace(self):
    """Set up observation space for RL. """
    # observation_high = (np.zeros(20) + OBSERVATION_EPS)
    # observation_low = (np.zeros(20) -  OBSERVATION_EPS)

    # Bounds derived from the above (with a small safety margin)
    r_max        = 1.2 * np.sqrt(MU_UPP)
    rdot_max     = 1.2 * (ALPHA * MU_UPP**2 * r_max)
    theta_max    = 10 * np.pi
    thetadot_max = 1.2 * (10 * np.pi * F_MAX)

    contact_high = np.ones(NUM_LEGS) * (1.0 + OBSERVATION_EPS)
    contact_low  = np.zeros(NUM_LEGS) - OBSERVATION_EPS

    # r ≥ 0
    r_high  = np.ones(NUM_LEGS) * (r_max + OBSERVATION_EPS)
    r_low   = np.zeros(NUM_LEGS) - OBSERVATION_EPS
    rdot_high = np.ones(NUM_LEGS) * (rdot_max + OBSERVATION_EPS)
    rdot_low  = -rdot_high

    # theta [0, 2 pi)
    theta_high = np.ones(NUM_LEGS) * (theta_max + OBSERVATION_EPS)
    theta_low  = np.zeros(NUM_LEGS) - OBSERVATION_EPS

    # theta_dot = omega >=  0
    thetadot_high = np.ones(NUM_LEGS) * (thetadot_max + OBSERVATION_EPS)
    thetadot_low  = np.zeros(NUM_LEGS) - OBSERVATION_EPS

    # desired_vel_high = np.array([self.vx_max, self.vy_max, self.wz_max ]) + OBSERVATION_EPS
    # desired_vel_low = np.array([self.vx_min, self.vy_min, -self.wz_max ])  - OBSERVATION_EPS
    desired_vel_high =  np.array([self.vx_max]) + OBSERVATION_EPS
    desired_vel_low = np.array([self.vx_min]) - OBSERVATION_EPS
    vel_high = 1.3 * np.array([self.vx_max, self.vy_max, self.wz_max ])
    vel_low = np.array([0.2 * self.vx_min, -self.vy_max, -self.wz_max ])
    rpy_high = np.array([np.pi/4, np.pi/4, np.pi/4])
    rpy_low = np.array([-np.pi/4, -np.pi/6, np.pi/4])

    rpy_vel_high = 2 * np.ones(3)
    rpy_vel_low = -2 * np.ones(3)

    #Joints state and vel
    joint_angle_high = self._robot_config.UPPER_ANGLE_JOINT + OBSERVATION_EPS
    joint_angle_low = self._robot_config.LOWER_ANGLE_JOINT - OBSERVATION_EPS
    joint_vel_high = self._robot_config.VELOCITY_LIMITS + OBSERVATION_EPS
    joint_vel_low = -self._robot_config.VELOCITY_LIMITS - OBSERVATION_EPS

    # cpg actions bound
    action_lower_bound = -np.ones(8)
    action_higher_bound = np.ones(8)


    if self._observation_space_mode == "DEFAULT":
      observation_high = (np.concatenate((self._robot_config.UPPER_ANGLE_JOINT,
                                         self._robot_config.VELOCITY_LIMITS,
                                         np.array([1.0]*4))) +  OBSERVATION_EPS)
      observation_low = (np.concatenate((self._robot_config.LOWER_ANGLE_JOINT,
                                         -self._robot_config.VELOCITY_LIMITS,
                                         np.array([-1.0]*4))) -  OBSERVATION_EPS)

    elif self._observation_space_mode == "LR_COURSE_OBS":
      # [TODO] Set observation upper and lower ranges. What are reasonable limits?
      # Note 50 is arbitrary below, you may have more or less
      # If using CPG-RL, remember to include limits on these
      observation_high = (np.zeros(50) + OBSERVATION_EPS)
      observation_low = (np.zeros(50) -  OBSERVATION_EPS)

    elif self._observation_space_mode == "MINIMAL":
      observation_high = np.concatenate([
          contact_high, r_high, rdot_high, theta_high, thetadot_high
      ])
      observation_low = np.concatenate([
          contact_low,  r_low,  rdot_low,  theta_low,  thetadot_low
      ])
    elif self._observation_space_mode == "MEDIUM":
      observation_high = np.concatenate([
        desired_vel_high, contact_high, r_high, rdot_high, theta_high, thetadot_high, vel_high, rpy_high, rpy_vel_high
      ])
      observation_low = np.concatenate([
        desired_vel_low, contact_low,  r_low,  rdot_low,  theta_low,  thetadot_low, vel_low, rpy_low, rpy_vel_low
      ])
    elif self._observation_space_mode == "FULL":
      observation_high = np.concatenate([
          desired_vel_high,
          # action_higher_bound,
          contact_high,
          r_high,
          rdot_high,
          theta_high,
          thetadot_high,
          vel_high,
          rpy_high,
          rpy_vel_high,
          joint_angle_high,
          joint_vel_high
      ])
      observation_low = np.concatenate([
          desired_vel_low,
          # action_lower_bound,
          contact_low,
          r_low,
          rdot_low,
          theta_low,
          thetadot_low,
          vel_low,
          rpy_low,
          rpy_vel_low,
          joint_angle_low,
          joint_vel_low
      ])
    else:
      raise ValueError("observation space not defined or not intended")

    self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)

  def setupActionSpace(self):
    """ Set up action space for RL. """
    if self._motor_control_mode in ["PD","TORQUE", "CARTESIAN_PD"]:
      action_dim = 12
    elif self._motor_control_mode in ["CPG"]:
      action_dim = 8
    else:
      raise ValueError("motor control mode " + self._motor_control_mode + " not implemented yet.")
    action_high = np.array([1] * action_dim)
    self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self._action_dim = action_dim

  def _get_observation(self):
    """Get observation, depending on obs space selected. """
    if self._observation_space_mode == "DEFAULT":
      self._observation = np.concatenate((self.robot.GetMotorAngles(),
                                          self.robot.GetMotorVelocities(),
                                          self.robot.GetBaseOrientation() ))
    elif self._observation_space_mode == "LR_COURSE_OBS":
      # [TODO] Get observation from robot. What are reasonable measurements we could get on hardware?
      # if using the CPG, you can include states with self._cpg.get_r(), for example
      # 50 is arbitrary
      self._observation = np.zeros(50)
    elif self._observation_space_mode == "MINIMAL":
      '''
      The minimal observation space
      consists only of foot contact booleans and the CPG states
      {r,r, θ, ˙ θ˙}. This observation space shows that coordination
      between limbs can be accomplished with very little sensing
      at all, with the only environmental feedback being from foot
      contact booleans. The idea for this space is inspired from the
      force feedback term in traditional CPGs shown to coordinate
      transitions between gaits [3], [4], also known as Tegotae-based
      control [42]. The importance of contacts and limb loading
      has also been shown by Ekeberg and Pearson in a simulation
      of cat locomotion [43]. For this observation space, the task is
      only to move forward at a particular desired velocity v
      ∗
      b,x.
      '''
      _, _, _, contact_bool = self.robot.GetContactInfo()
      self._observation = np.concatenate((contact_bool,
                                          self._cpg.get_r(),
                                          self._cpg.get_theta(),
                                          self._cpg.get_dr(),
                                          self._cpg.get_dtheta(),
                                          ))


    elif self._observation_space_mode == "MEDIUM":
      '''
      The medium observation removes the
      joint state and last action from the full observation. This
      observation space is chosen to show that joint information
      is actually not necessary for omnidirectional locomotion
      through our method. Other states remain the same (i.e. velocity
      commands, body state (Nathan: "Okay here I use linear body velocity + body orientation, foot contact booleans, and CPG states).
      '''
      _, _, _, contact_bool = self.robot.GetContactInfo()
      linear_vel = self.robot.GetBaseLinearVelocity()
      rpy = self.robot.GetBaseOrientationRollPitchYaw()
      rpy_vel = self.robot.GetBaseAngularVelocity()
      self._observation = np.concatenate((self.des_velocity[0],
                                          contact_bool,
                                          self._cpg.get_r(),
                                          self._cpg.get_theta(),
                                          self._cpg.get_dr(),
                                          self._cpg.get_dtheta(),
                                          linear_vel,
                                          rpy,
                                          rpy_vel
                                          ))


    elif self._observation_space_mode == "FULL":
      '''
      The full observation consists of velocity
      commands and measurements reasonably available with
      proprioceptive sensing, and are becoming standard in DRL
      approaches. These include the body state (orientation, linear
      and angular velocities), joint state (positions, velocities), and
      foot contact booleans. The last action chosen by the policy
      network and CPG states {r,r,θ, ˙ θ,˙ (φ,φ˙)} are concatenated
      to the proprioceptive measurements.
      '''
      _, _, _, contact_bool = self.robot.GetContactInfo()
      linear_vel = self.robot.GetBaseLinearVelocity()
      rpy = self.robot.GetBaseOrientationRollPitchYaw()
      rpy_vel = self.robot.GetBaseAngularVelocity()
      joint_state = self.robot.GetMotorAngles()
      joint_vel = self.robot.GetMotorVelocities()
      self._observation = np.concatenate((np.array([self.des_velocity[0]]),
                                          # self._last_action,
                                          contact_bool,
                                          self._cpg.get_r(),
                                          self._cpg.get_theta(),
                                          self._cpg.get_dr(),
                                          self._cpg.get_dtheta(),
                                          linear_vel,
                                          rpy,
                                          rpy_vel,
                                          joint_state,
                                          joint_vel
                                          ))

      pass
    else:
      raise ValueError("observation space not defined or not intended")

    self._add_obs_noise = (np.random.normal(scale=self._observation_noise_stdev, size=self._observation.shape) *
          self.observation_space.high)
    return self._observation

  def _noisy_observation(self):
    self._get_observation()
    observation = np.array(self._observation)
    if self._observation_noise_stdev > 0:
      observation += self._add_obs_noise
    return observation

  def _get_info(self) -> dict:
    return {'base_pos': self.robot.GetBasePosition()}

  ######################################################################################
  # Termination and reward
  ######################################################################################
  def is_fallen(self,dot_prod_min=0.85):
    """Decide whether the quadruped has fallen.

    If the up directions between the base and the world is larger (the dot
    product is smaller than 0.85) or the base is very low on the ground
    (the height is smaller than 0.13 meter), the quadruped is considered fallen.

    Returns:
      Boolean value that indicates whether the quadruped has fallen.
    """
    base_rpy = self.robot.GetBaseOrientationRollPitchYaw()
    orientation = self.robot.GetBaseOrientation()
    rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
    local_up = rot_mat[6:]
    pos = self.robot.GetBasePosition()
    return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < dot_prod_min or pos[2] < self._robot_config.IS_FALLEN_HEIGHT)

  def _termination(self):
    """Decide whether we should stop the episode and reset the environment. """
    return self.is_fallen()

  def _reward_fwd_locomotion(self, des_vel_x=None):
    """Learn forward locomotion at a desired velocity. """
    if des_vel_x is None:
      des_vel_x = self.robot.desired_velocity
    vel_tracking = np.exp(-2.0 * abs(self.robot.GetBaseLinearVelocity()[0] - des_vel_x))        # [0,1]
    lateral_pen = abs(self.robot.GetBasePosition()[1])                                     # 0..?
    height_pen  = abs(self.robot.GetBasePosition()[2] - self._robot_config.INIT_POSITION[2])
    orient_pen  = np.linalg.norm(self.robot.GetBaseOrientation() - np.array([0,0,0,1]))
    energy_pen  = 0
    for tau,vel in zip(self._dt_motor_torques,self._dt_motor_velocities):
      energy_pen += np.abs(np.dot(tau,vel)) * self._time_step

    # reward = 0.05 * vel_tracking - 0.005 * lateral_pen - 0.002 * height_pen - 0.002 * orient_pen - 0.0005 * energy_pen
    reward = 0.05 * vel_tracking
    # reward = max(0.0, reward)
    return reward

  def _reward_fwd_locomotion_cpg(self, des_vel_x=None):
      """Learn forward locomotion at a desired velocity. """
      if des_vel_x is None:
        des_vel_x = self.robot.desired_velocity

      dt = self._time_step * self._action_repeat

      linear_body_vel_tracking_x = np.exp(-4 * np.linalg.norm(des_vel_x - self.robot.GetBaseLinearVelocity()[0])**2)        # [0,1]
      linear_body_vel_tracking_y = np.exp(-4 * np.linalg.norm(0 - self.robot.GetBaseLinearVelocity()[1])**2)        # [0,1]
      angular_body_vel_tracking_z = np.exp(-1 * np.linalg.norm(0 - self.robot.GetBaseAngularVelocity()[2])**2)        # [0,1]
      linear_body_vel_penalty_z = - self.robot.GetBaseLinearVelocity()[2]**2  # closer to 0 is better
      angular_body_vel_penalty_x = - np.linalg.norm(np.array([self.robot.GetBaseAngularVelocity()[0], self.robot.GetBaseAngularVelocity()[1]]))**2  # closer to 0 is better
      work_between_steps = - abs(np.dot(np.array(self._dt_motor_torques[-1]), np.array(self._dt_motor_velocities[-1]))) * dt  # negative work is better


      reward = 0.5*dt*linear_body_vel_tracking_x + 0.75*dt*linear_body_vel_tracking_y + 0.5*dt*angular_body_vel_tracking_z + 2*dt*linear_body_vel_penalty_z + 0.05*dt*angular_body_vel_penalty_x + 0.001*dt*work_between_steps
      return reward

  def get_distance_and_angle_to_goal(self):
    """ Helper to return distance and angle to current goal location. """
    # current object location
    base_pos = self.robot.GetBasePosition()
    yaw = self.robot.GetBaseOrientationRollPitchYaw()[2]
    goal_vec = self._goal_location
    dist_to_goal = np.linalg.norm(base_pos[0:2]-goal_vec)

    # angle to goal (from current heading)
    body_dir_vec = np.matmul( rotation_matrix(yaw), np.array([[1],[0]]) )
    body_goal_vec = goal_vec - base_pos[0:2]
    body_dir_vec = body_dir_vec.reshape(2,)
    body_goal_vec = body_goal_vec.reshape(2,)

    Vn = unit_vector( np.array([0,0,1]) )
    c = np.cross( np.hstack([body_dir_vec,0]), np.hstack([body_goal_vec,0])  )
    angle = angle_between(body_dir_vec, body_goal_vec)
    angle = angle * np.sign( np.dot( Vn , c ) )

    return dist_to_goal, angle

  def _reward_flag_run(self):
    """ Learn to move towards goal location. """
    curr_dist_to_goal, angle = self.get_distance_and_angle_to_goal()

    # minimize distance to goal (we want to move towards the goal)
    dist_reward = 10 * ( self._prev_pos_to_goal - curr_dist_to_goal)

    # minimize yaw deviation to goal (necessary?)
    yaw_reward = 0 # -0.01 * np.abs(angle)

    # minimize energy
    energy_reward = 0
    for tau,vel in zip(self._dt_motor_torques,self._dt_motor_velocities):
      energy_reward += np.abs(np.dot(tau,vel)) * self._time_step

    reward = dist_reward \
            + yaw_reward \
            - 0.001 * energy_reward

    return max(reward,0) # keep rewards positive

  def _cpg_rl_tracking_term(self, error, sigma=0.17):
    return float(np.exp(-np.sum(error**2) / sigma))


  def _reward_lr_course(self):
    """ Implement your reward function here. How will you improve upon the above? """
    # [TODO] add your reward function. -- tick
    lin_vel_body = self.robot.GetBaseLinearVelocity()
    ang_vel_body = self.robot.GetBaseAngularVelocity()
    position = self.robot.GetBasePosition()
    roll, pitch, yaw = self.robot.GetBaseOrientationRollPitchYaw()

    vx, vy, vz = lin_vel_body
    w_roll, w_pitch, w_yaw = ang_vel_body
    vx_des, vy_des, wz_des = self.des_velocity

    r_vx   = self._cpg_rl_tracking_term(vx_des - vx)
    r_vy   = self._cpg_rl_tracking_term(vy_des - vy)
    r_yaw  = self._cpg_rl_tracking_term(wz_des - w_yaw)

    r_vz_pen   = -vz**2                     # vertical vel
    # r_ang_pen  = -(w_roll**2 + w_pitch**2)  # roll/pitch rates
    r_ang_pen  = -(w_roll**2 + w_pitch**2)  # roll/pitch rates
    r_ry_pen  = -(roll**2 + yaw**2)  # roll/yaw angles

    energy_reward = 0
    for tau,vel in zip(self._dt_motor_torques,self._dt_motor_velocities):
      energy_reward += np.abs(np.dot(tau,vel)) * self._time_step

    reward = (
        self.w_vx * r_vx +
        self.w_pos_y_pen * (np.abs(position[1])) +
        self.w_vz_pen * r_vz_pen +
        self.w_ry_pen * r_ry_pen +
        self.w_work * energy_reward
    ) * 0.01

    # reward = (
    #     self.w_vx * r_vx +
    #     self.w_pos_y_pen * (np.abs(position[1]) +
    #     self.w_vy * r_vy +
    #     self.w_yaw * r_yaw +
    #     self.w_vz_pen * r_vz_pen +
    #     self.w_ang_pen * r_ang_pen +
    #     self.w_ry_pen * r_ry_pen +
    #     self.w_work * energy_reward
    # ) * 0.01


    ## Just to log these to track the learning
    self._rew_terms_step = {
        "r_vx_raw": float(r_vx),
        # "r_vy_raw": float(r_vy),
        # "r_yaw_raw": float(r_yaw),
        # "r_vz_pen_raw": float(r_vz_pen),
        # "r_ang_pen_raw": float(r_ang_pen),
        "work_raw": float(energy_reward),
        # weighted contributions:
        "r_vx_w": float(self.w_vx * r_vx * 0.01),
        "r_vy_w": float(self.w_vy * r_vy * 0.01),
        # "r_yaw_w": float(self.w_yaw * r_yaw * 0.01),
        "r_vz_pen_w": float(self.w_vz_pen * r_vz_pen * 0.01),
        # "r_ang_pen_w": float(self.w_ang_pen * r_ang_pen * 0.01),
        "work_w": float(self.w_work * energy_reward * 0.01),
    }

    return reward



  def _reward(self):
    """ Get reward depending on task"""
    if self._TASK_ENV == "FWD_LOCOMOTION":
      return self._reward_fwd_locomotion()
    elif self._TASK_ENV == "LR_COURSE_TASK":
      return self._reward_lr_course()
    elif self._TASK_ENV == "FLAGRUN":
      return self._reward_flag_run()
    elif self._TASK_ENV == "FWD_LOCOMOTION_CPG":
      return self._reward_fwd_locomotion_cpg()
    else:
      raise ValueError("This task mode not implemented yet.")

  ######################################################################################
  # Step simulation, map policy network actions to joint commands, etc.
  ######################################################################################
  def _transform_action_to_motor_command(self, action):
    """ Map actions from RL (i.e. in [-1,1]) to joint commands based on motor_control_mode. """
    # clip actions to action bounds
    action = np.clip(action, -self._action_bound - ACTION_EPS,self._action_bound + ACTION_EPS)

    if self._motor_control_mode == "PD":
      action = self._scale_helper(action, self._robot_config.LOWER_ANGLE_JOINT, self._robot_config.UPPER_ANGLE_JOINT)
      action = np.clip(action, self._robot_config.LOWER_ANGLE_JOINT, self._robot_config.UPPER_ANGLE_JOINT)
    elif self._motor_control_mode == "CARTESIAN_PD":
      action = self.ScaleActionToCartesianPos(action)
    elif self._motor_control_mode == "CPG":
      action = self.ScaleActionToCPGStateModulations(action)
    else:
      raise ValueError("RL motor control mode" + self._motor_control_mode + "not implemented yet.")

    return action

  def _scale_helper(self, action, lower_lim, upper_lim):
    """Helper to linearly scale from [-1,1] to lower/upper limits. """
    new_a = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)

    return np.clip(new_a, lower_lim, upper_lim)

  def ScaleActionToCartesianPos(self,actions):
    """Scale RL action to Cartesian PD ranges.
    Edit ranges, limits etc., but make sure to use Cartesian PD to compute the torques.
    """
    # clip RL actions to be between -1 and 1 (standard RL technique)
    u = np.clip(actions,-1,1)

    # scale to corresponding desired foot positions (i.e. ranges in x,y,z we allow the agent to choose foot positions)
    # [TODO: edit (do you think these should these be increased? How limiting is this?)]
    scale_array = np.array([0.08, 0.04, 0.06]*4)

    # add to nominal foot position in leg frame (what are the final ranges?)
    des_foot_pos = self._robot_config.NOMINAL_FOOT_POS_LEG_FRAME + scale_array*u

    # get Cartesian kp and kd gains (can be modified)
    kpCartesian = self._robot_config.kpCartesian
    kdCartesian = self._robot_config.kdCartesian

    # get current motor velocities
    dq = self.robot.GetMotorVelocities()

    action = np.zeros(12)
    for i in range(4):
      # get Jacobian and foot position in leg frame for leg i (see ComputeJacobianAndPosition() in quadruped.py)
      # [TODO]
      J, foot_pos = self.robot.ComputeJacobianAndPosition(i)

      # desired foot position i (from RL above)
      pd = np.zeros(3) # [TODO]
      pd = des_foot_pos[3*i:3*i+3]

      # desired foot velocity i
      vd = np.zeros(3) # [TODO]

      # foot velocity in leg frame i (Equation 2)
      # [TODO]
      vd_current = J @ dq[3*i:3*i+3]

      # calculate torques with Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      tau = np.zeros(3) # [TODO]
      tau = kpCartesian @ (pd - foot_pos) - kdCartesian @ vd_current
      tau = J.T @ tau
      # print("tau:", tau)
      action[3*i:3*i+3] = tau

    return action

  def ScaleActionToCPGStateModulations(self,actions):
    """Scale RL action to CPG modulation parameters."""
    # clip RL actions to be between -1 and 1 (standard RL technique)
    u = np.clip(actions,-1,1)

    # scale omega to ranges, and set in CPG (range is an example)
    omega = self._scale_helper( u[0:4], 5, 4.5*2*np.pi)
    self._cpg.set_omega_rl(omega)

    # scale mu to ranges, and set in CPG (squared since we converge to the sqrt in the CPG amplitude)
    mus = self._scale_helper( u[4:8], MU_LOW**2, MU_UPP**2)
    self._cpg.set_mu_rl(mus)

    # integrate CPG, get mapping to foot positions
    xs,zs = self._cpg.update()

    # IK parameters
    foot_y = self._robot_config.HIP_LINK_LENGTH
    sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

    # get motor kp and kd gains (can be modified)
    kp = self._robot_config.MOTOR_KP # careful of size!
    kd = self._robot_config.MOTOR_KD

    # get current motor velocities
    q = self.robot.GetMotorAngles()
    dq = self.robot.GetMotorVelocities()

    action = np.zeros(12)
    # loop through each leg
    for i in range(4):
      # get desired foot i pos (xi, yi, zi)
      x = xs[i]
      y = sideSign[i] * foot_y # careful of sign
      z = zs[i]

      # call inverse kinematics to get corresponding joint angles
      # q_des = np.zeros(3) # [TODO] -- tick
      q_des = self.robot.ComputeInverseKinematics(i, np.array([x,y,z]))

      # Add joint PD contribution to tau
      # tau = np.zeros(3) # [TODO] --tick
      tau = kp[3*i:3*i+3] * (q_des - q[3*i:3*i+3]) - kd[3*i:3*i+3] * dq[3*i:3*i+3]  # wait but actually we can improve on this, joint_vel_des needs not to be 0

      # add Cartesian PD contribution (as you wish)
      # J, pos = self.robot.ComputeJacobianAndPosition(i, q[3*i:3*i+3])
      # tau += J.T @ (kp[3*i:3*i+3] * (np.array([x,y,z]) - pos))

      action[3*i:3*i+3] = tau

    return action

  def step(self, action):
    """ Step forward the simulation, given the action. """
    curr_act = action.copy()
    # save motor torques and velocities to compute power in reward function
    self._dt_motor_torques = []
    self._dt_motor_velocities = []

    if "FLAGRUN" in self._TASK_ENV:
      self._prev_pos_to_goal, _ = self.get_distance_and_angle_to_goal()

    for _ in range(self._action_repeat):
      if self._isRLGymInterface:
        proc_action = self._transform_action_to_motor_command(curr_act)
      else:
        proc_action = curr_act

      self.robot.ApplyAction(proc_action)
      self._pybullet_client.stepSimulation()
      self._sim_step_counter += 1
      self._dt_motor_torques.append(self.robot.GetMotorTorques())
      self._dt_motor_velocities.append(self.robot.GetMotorVelocities())

      if self._is_render:
        self._render_step_helper()

    self._last_action = curr_act
    self._env_step_counter += 1
    reward = self._reward()
    truncated = False

    # accumulate episode stats
    self._ep_len += 1
    if self._rew_terms_step:
        for k, v in self._rew_terms_step.items():
            self._rew_terms_ep[k] = self._rew_terms_ep.get(k, 0.0) + v

    terminated = self._termination()
    truncated = truncated or (self.get_sim_time() > self._MAX_EP_LEN and not self._test_flagrun)

    info = self._get_info()

    # always attach instantaneous telemetry (so callbacks can stream scalars)
    # suggested extra diagnostics:
    base_lin = self.robot.GetBaseLinearVelocity()
    base_ang = self.robot.GetBaseAngularVelocity()
    _, _, _, contact_bool = self.robot.GetContactInfo()

    info.update({
        "rew_terms": dict(self._rew_terms_step),     # per-step
        "kine/vx": float(base_lin[0]),
        "kine/vy": float(base_lin[1]),
        "kine/vz": float(base_lin[2]),
        "kine/w_roll": float(base_ang[0]),
        "kine/w_pitch": float(base_ang[1]),
        "kine/w_yaw": float(base_ang[2]),
        "contacts/num_in_contact": int(np.sum(contact_bool)),
        "cpg/r_mean": float(np.mean(self._cpg.get_r())),
        "cpg/omega_mean": float(np.mean(self._cpg.get_dtheta())),  # your omega
    })

    # print(f"updated info with {len(info.keys())} keys {list(info.keys())}")

    # when episode ends, attach episode totals so SB3 Monitor/Callback can log them
    if terminated or truncated:
        info["episode_terms"] = dict(self._rew_terms_ep)  # episodic sums of weighted terms
        info["episode_len"] = self._ep_len
        # reset accumulators for the next episode
        self._rew_terms_ep = {}
        self._ep_len = 0

    if "FLAGRUN" in self._TASK_ENV:
      dist_to_goal, _ = self.get_distance_and_angle_to_goal()

      if dist_to_goal < 0.5:
        self._reset_goal()

    return np.array(self._noisy_observation()), reward, terminated, truncated, info

  ######################################################################################
  # Reset
  ######################################################################################
  def set_desired_velocity(self, desired_velocity):
    self.des_velocity = desired_velocity

  def reset(self, seed: Optional[float] = None):
    """ Set up simulation environment. """
    mu_min = 0.5

    # Update seed
    self.seed(seed)

    # resample desired velovity
    vx = self.np_random.uniform(self.vx_min, self.vx_max)
    # vy = self.np_random.uniform(-self.vy_max, self.vy_max)
    # wz = self.np_random.uniform(-self.wz_max, self.wz_max)
    self.des_velocity = np.array([vx, 0, 0], dtype=np.float32)

    # Disable rendering when setting up models (otherwise too slow)
    if self._is_render:
      self._pybullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

    if self._hard_reset:
      # set up pybullet simulation
      self._pybullet_client.resetSimulation()
      self._pybullet_client.setPhysicsEngineParameter(
          numSolverIterations=int(self._num_bullet_solver_iterations))
      self._pybullet_client.setTimeStep(self._time_step)
      self.plane = self._pybullet_client.loadURDF(pybullet_data.getDataPath()+"/plane.urdf",
                                                  basePosition=[80,0,0]) # to extend available running space (shift)
      self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
      self._pybullet_client.configureDebugVisualizer(
          self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
      self._pybullet_client.setGravity(0, 0, -9.8)

      if self._terrain == "GAPS":
        self._robot_config.INIT_POSITION[2] = 1.305
        self._robot_config.IS_FALLEN_HEIGHT = 1.18

      self.robot = (quadruped.Quadruped(pybullet_client=self._pybullet_client,
                                         robot_config=self._robot_config,
                                         motor_control_mode=self._motor_control_mode,
                                         on_rack=self._on_rack,
                                         render=self._is_render))
      self._ground_mu_k = 1

      if self._add_noise:
        ground_mu_k = mu_min+(1-mu_min)*np.random.random()
        self._ground_mu_k = ground_mu_k
        self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=ground_mu_k)
        # self._add_base_mass_offset()
        if self._is_render:
          print('ground friction coefficient is', ground_mu_k)

      if self._terrain is not None:
        if self._terrain == "SLOPES":
          pitch = 0.05 * self._terrain_difficulty
          self.add_slopes(pitch=pitch)
        elif self._terrain == "STAIRS":
          self.add_stairs(num_stairs=12, stair_height=0.05, stair_width=0.25)
        elif self._terrain == "GAPS":
          self.add_gaps(num_gaps=5, gap_width=0.1, between_gaps_width=2)
        elif self._terrain == "RANDOM":
          self.add_random_boxes()
        elif self._terrain == "CUSTOM_1":
          # Custom critical terrain: combination of stairs, gaps, slopes, random obstacles
          # to break simple open-loop CPG controllers.
          # This will compose several existing primitives into one long, challenging
          # track in front of the robot.
          # You can tune difficulty via the difficulty keyword in env constructor
          # (small ints->easier, larger->harder). We fall back to difficulty=2.
          difficulty = getattr(self, '_terrain_difficulty', None)
          if difficulty is None:
            # try reading from kwargs fallback if user passed it earlier
            difficulty = 2
          self.add_critical_terrain(difficulty=difficulty)
        else:
          print('Terrain',self._terrain,'is not implemented')
      elif self._TASK_ENV == "FLAGRUN":
        self.goal_id = None
        if self._test_flagrun:
          self._ground_mu_k = ground_mu_k = 0.8
          self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=ground_mu_k)
          self._add_noise = False
          self._goal_idx = 0
          self.goal_x = np.arange(np.pi/4, 11, np.pi/2)
          self.goal_y = 0.2 * self.goal_x * np.sin(2*self.goal_x)
        self._reset_goal()
    else:
      self.robot.Reset(reload_urdf=False)

    self._env_step_counter = 0
    self._sim_step_counter = 0
    self._last_base_position = [0, 0, 0]

    # Enable rendering again
    if self._is_render:
      self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                       self._cam_pitch, [0, 0, 0])
      self._pybullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

    self._settle_robot()
    self._last_action = np.zeros(self._action_dim)

    if self._is_record_video:
      self.recordVideoHelper()

    return self._noisy_observation(), self._get_info()

  def _reset_goal(self):
    """Reset goal location for flagrun."""
    try:
      if self.goal_id is not None:
        self._pybullet_client.removeBody(self.goal_id)
    except:
      pass

    if self._test_flagrun:
      self._goal_location = np.array([self.goal_x[self._goal_idx],
                                      self.goal_y[self._goal_idx]])
      self._goal_idx = min(self._goal_idx+1, len(self.goal_x))
    else:
      self._goal_location = 6 * (np.random.random((2,)) - 0.5)
      self._goal_location += self.robot.GetBasePosition()[0:2]

    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
        halfExtents=[0.2,0.2,0.2])
    orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
    self.goal_id=self._pybullet_client.createMultiBody(
                          baseMass=0,
                          baseCollisionShapeIndex = sh_colBox,
                          basePosition = [self._goal_location[0],self._goal_location[1],0.6],
                          baseOrientation=orn)
    # print('goal is at ', self._goal_location)

  def _settle_robot(self):
    """ Settle robot and add noise to init configuration. """
    # change to PD control mode to set initial position, then set back..
    tmp_save_motor_control_mode_ENV = self._motor_control_mode
    tmp_save_motor_control_mode_ROB = self.robot._motor_control_mode
    self._motor_control_mode = "PD"
    self.robot._motor_control_mode = "PD"

    try:
      tmp_save_motor_control_mode_MOT = self.robot._motor_model._motor_control_mode
      self.robot._motor_model._motor_control_mode = "PD"
    except:
      pass

    init_motor_angles = self._robot_config.INIT_MOTOR_ANGLES + self._robot_config.JOINT_OFFSETS

    for _ in range(1000):
      self.robot.ApplyAction(init_motor_angles)

      if self._is_render:
        self._render_step_helper()
      self._pybullet_client.stepSimulation()

    # set control mode back
    self._motor_control_mode = tmp_save_motor_control_mode_ENV
    self.robot._motor_control_mode = tmp_save_motor_control_mode_ROB
    try:
      self.robot._motor_model._motor_control_mode = tmp_save_motor_control_mode_MOT
    except:
      pass

  ######################################################################################
  # Render, record videos, bookkeping, and misc pybullet helpers.
  ######################################################################################
  def startRecordingVideo(self,name):
    self.videoLogID = self._pybullet_client.startStateLogging(
                            self._pybullet_client.STATE_LOGGING_VIDEO_MP4,
                            name)

  def stopRecordingVideo(self):
    self._pybullet_client.stopStateLogging(self.videoLogID)

  def close(self):
    if self._is_record_video:
      self.stopRecordingVideo()

    self._pybullet_client.disconnect()

  def recordVideoHelper(self, extra_filename=None):
    """ Helper to record video, if not already, or end and start a new one """
    # If no ID, this is the first video, so make a directory and start logging
    if self.videoLogID == None:
      directoryName = VIDEO_LOG_DIRECTORY
      assert isinstance(directoryName, str)
      os.makedirs(directoryName, exist_ok=True)
      self.videoDirectory = directoryName
    else:
      # stop recording and record a new one
      self.stopRecordingVideo()

    if extra_filename is not None:
      output_video_filename = self.videoDirectory + '/' + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f") +extra_filename+ ".MP4"
    else:
      output_video_filename = self.videoDirectory + '/' + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f") + ".MP4"

    logID = self.startRecordingVideo(output_video_filename)
    self.videoLogID = logID

  def configure(self, args):
    self._args = args

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _render_step_helper(self):
    """ Helper to configure the visualizer camera during step(). """
    # Sleep, otherwise the computation takes less time than real time,
    # which will make the visualization like a fast-forward video.
    current_time = time.monotonic()
    elapsed = current_time - self._last_frame_time
    self._last_frame_time = current_time
    time_diff = max(0, self._time_step - elapsed)
    # if time_diff > 0:
    #   time.sleep(time_diff)

    base_pos = self.robot.GetBasePosition()
    camInfo = self._pybullet_client.getDebugVisualizerCamera()
    curTargetPos = camInfo[11]
    distance = camInfo[10]
    yaw = camInfo[8]
    pitch = camInfo[9]
    targetPos = [
        0.95 * curTargetPos[0] + 0.05 * base_pos[0], 0.95 * curTargetPos[1] + 0.05 * base_pos[1],
        curTargetPos[2]
    ]
    self._pybullet_client.resetDebugVisualizerCamera(distance, yaw, pitch, base_pos)

  def _configure_visualizer(self):
    """ Remove all visualizer borders, and zoom in """
    # default rendering options
    self._render_width = 960
    self._render_height = 720
    self._cam_dist = 1.0
    self._cam_yaw = 0
    self._cam_pitch = -30

    # get rid of visualizer things
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI,0)

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])

    base_pos = self.robot.GetBasePosition()
    view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                   aspect=float(self._render_width) /
                                                                   self._render_height,
                                                                   nearVal=0.1,
                                                                   farVal=100.0)
    (_, _, px, _,
     _) = self._pybullet_client.getCameraImage(width=self._render_width,
                                               height=self._render_height,
                                               viewMatrix=view_matrix,
                                               projectionMatrix=proj_matrix,
                                               renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def addLine(self,lineFromXYZ,lineToXYZ,lifeTime=0,color=[1,0,0]):
    """ Add line between point A and B for duration lifeTime"""
    self._pybullet_client.addUserDebugLine(lineFromXYZ,
                                            lineToXYZ,
                                            lineColorRGB=color,
                                            lifeTime=lifeTime)

  def get_sim_time(self):
    """ Get current simulation time. """
    return self._sim_step_counter * self._time_step

  def scale_rand(self,num_rand,low,high):
    """ scale number of rand numbers between low and high """
    return low + np.random.random(num_rand) * (high - low)

  def add_random_boxes(self, num_rand=100, z_height=0.04):
    """Add random boxes in front of the robot in x [0.5, 20] and y [-3,3] """
    # x location
    x_low, x_upp = 0.5, 20

    # y location
    y_low, y_upp = -3, 3

    # z location
    z_low, z_upp = 0.005, z_height

    # block dimensions
    block_x_min, block_x_max = 0.1, 1
    block_y_min, block_y_max = 0.1, 1

    # block orientations
    roll_low, roll_upp = -0.01, 0.01
    pitch_low, pitch_upp = -0.01, 0.01
    yaw_low, yaw_upp = -np.pi, np.pi

    x = x_low + np.random.random(num_rand) * (x_upp - x_low)
    y = y_low + np.random.random(num_rand) * (y_upp - y_low)
    z = z_low + np.random.random(num_rand) * (z_upp - z_low)
    block_x = self.scale_rand(num_rand,block_x_min,block_x_max)
    block_y = self.scale_rand(num_rand,block_y_min,block_y_max)
    roll = self.scale_rand(num_rand,roll_low,roll_upp)
    pitch = self.scale_rand(num_rand,pitch_low,pitch_upp)
    yaw = self.scale_rand(num_rand,yaw_low,yaw_upp)

    # loop through
    for i in range(num_rand):
      sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
          halfExtents=[block_x[i]/2,block_y[i]/2,z[i]/2])
      orn = self._pybullet_client.getQuaternionFromEuler([roll[i],pitch[i],yaw[i]])
      block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [x[i],y[i],z[i]/2],baseOrientation=orn)
      # set friction coeff
      self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)

    # add walls
    orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
        halfExtents=[x_upp/2,0.5,0.5])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [x_upp/2,y_low,0.5],baseOrientation=orn)
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [x_upp/2,-y_low,0.5],baseOrientation=orn)

  def add_gaps(self, num_gaps=5, gap_width=0.1, between_gaps_width=2):
    """Add N gaps
      -each gap is gap_width wide
      -platforms between gaps are between_gaps_width wide"""
    orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])

    # start platform
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
            halfExtents=[2,1,0.5])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                              basePosition = [0,0,0.5],baseOrientation=orn)

    # set friction coeff to 1
    self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)
    first_gap = 2
    block_0 = first_gap + gap_width + between_gaps_width / 2

    # keep track of gaps (possibly for RL observation space!)
    self._gap_centers = np.zeros(num_gaps)

    # loop through
    for i in range(num_gaps):
      self._gap_centers[i] = first_gap + gap_width / 2 + i*between_gaps_width
      block_x = block_0 + i * (gap_width + between_gaps_width)
      sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
          halfExtents=[between_gaps_width / 2, 1, 0.5])
      block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [block_x,0,0.5],baseOrientation=orn)

      # set friction coeff to 1
      self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)
    # print("gaps are centered at", self._gap_centers)

    # end platform
    end_platform_size = 2
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
            halfExtents=[end_platform_size,1,0.5])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                              basePosition = [block_x+between_gaps_width/2+end_platform_size/2,0,0.5],baseOrientation=orn)

    # set friction coeff to 1
    self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)

  def add_stairs(self, num_stairs=12, stair_height=0.05, stair_width=0.25):
    """Add N stairs, with stair_height and stair_width. long so can't get around """
    x_upp = 20
    y_low = -3
    y = 6
    curr_z = 0
    block_x = stair_width * np.ones(num_stairs)
    curr_x = 1

    # loop through
    for i in range(num_stairs):
      if i < num_stairs / 2:
        curr_z += stair_height
      else:
        curr_z -= stair_height

      if curr_z > 0:
        sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
            halfExtents=[block_x[i]/2,y/2,curr_z/2])
        orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
        block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                              basePosition = [curr_x,0,curr_z/2],baseOrientation=orn)

        # set friction coeff to 1
        self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)

      curr_x += block_x[i]

    # add walls
    orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
        halfExtents=[x_upp/2,0.5,0.5])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [x_upp/2,y_low,0.5],baseOrientation=orn)
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [x_upp/2,-y_low,0.5],baseOrientation=orn)

  def add_slopes(self, pitch=0.2):
    """Add slopes with platform in center."""
    y = 6
    slope_len = 2
    box_width = 1
    slope_height = 0.01

    # add first slope UP
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
        halfExtents=[slope_len/2,y/2,slope_height])
    orn = self._pybullet_client.getQuaternionFromEuler([0,-pitch,0])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex=sh_colBox,
        basePosition = [1+slope_len/2,0,slope_len/2*np.sin(pitch) - slope_height*np.cos(pitch) ],baseOrientation=orn)
    self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)

    # add middle box
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
        halfExtents=[box_width/2,y/2,slope_len/2*np.sin(pitch)])
    orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex=sh_colBox,
        basePosition = [1+slope_len*np.cos(pitch)+box_width/2,0,slope_len/2*np.sin(pitch)  ],baseOrientation=orn) # + slope_height/2*np.cos(pitch)
    self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)

    # add descending box
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
        halfExtents=[slope_len/2,y/2,slope_height])
    orn = self._pybullet_client.getQuaternionFromEuler([0,pitch,0])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex=sh_colBox,
        basePosition = [1+slope_len*np.cos(pitch)+box_width + slope_len/2 + 2*slope_height*np.sin(-pitch),0,slope_len/2*np.sin(pitch) - slope_height*np.cos(pitch) ],baseOrientation=orn) # + slope_height/2*np.cos(pitch)
    self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)
    self._add_walls()

  def add_critical_terrain(self, difficulty=2):
    """Compose multiple terrain primitives into a single long, challenging track.

    difficulty: integer >=1. Higher means more stairs/gaps/obstacles and lower friction.
    The function reuses existing primitives with parameter choices tuned to the difficulty.
    """

    # Stairs: fewer but taller when difficulty increases
    num_stairs = max(4, 4 + difficulty * 2)
    stair_height = 0.06 + 0.01 * difficulty
    stair_width = 0.2
    self.add_stairs(num_stairs=num_stairs, stair_height=stair_height, stair_width=stair_width)

    '''
    # Gaps: wider gaps when difficulty increases
    num_gaps = max(2, difficulty + 1)
    gap_width = 0.12 + 0.02 * difficulty
    between_gaps_width = 1.6 - 0.05 * difficulty
    # add gaps after stairs
    self.add_gaps(num_gaps=num_gaps, gap_width=gap_width, between_gaps_width=between_gaps_width)

    # Add a steep slope segment
    pitch = 0.18 + 0.02 * difficulty
    self.add_slopes(pitch=min(pitch, 0.45))

    # Narrow corridor with random boxes to force foot placement accuracy
    num_rand = 20 + 20 * difficulty
    self.add_random_boxes(num_rand=min(num_rand, 200), z_height=0.04 + 0.01 * difficulty)

    # finally, cluster of small obstacles to create uneven ground
    self.add_random_boxes(num_rand=30 + 10 * difficulty, z_height=0.02 + 0.01 * difficulty)

    '''

  def _add_walls(self,x_upp=20,y_low=-3):
    # add walls
    orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
        halfExtents=[x_upp/2,0.25,0.5])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [x_upp/2,y_low,0.5],baseOrientation=orn)
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [x_upp/2,-y_low,0.5],baseOrientation=orn)

  def _add_base_mass_offset(self, spec_mass=None, spec_location=None):
    """Attach mass to robot base."""
    quad_base = np.array(self.robot.GetBasePosition())
    quad_ID = self.robot.quadruped
    offset_low = np.array([-0.15, -0.05, -0.05])
    offset_upp = np.array([ 0.15,  0.05,  0.05])

    if spec_location is None:
      block_pos_delta_base_frame = self.scale_rand(3,offset_low,offset_upp)
    else:
      block_pos_delta_base_frame = np.array(spec_location)

    if spec_mass is None:
      base_mass = 8*np.random.random()
    else:
      base_mass = spec_mass

    if self._is_render:
      print('=========================== Random Mass:')
      print('Mass:', base_mass, 'location:', block_pos_delta_base_frame)
      # if rendering, also want to set the halfExtents accordingly
      # 1 kg water is 0.001 cubic meters
      boxSizeHalf = [(base_mass*0.001)**(1/3) / 2]*3
      translationalOffset = [0,0,0.1]
    else:
      boxSizeHalf = [0.05]*3
      translationalOffset = [0]*3

    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
                      halfExtents=boxSizeHalf, collisionFramePosition=translationalOffset)
    base_block_ID=self._pybullet_client.createMultiBody(baseMass=base_mass,
                                    baseCollisionShapeIndex = sh_colBox,
                                    basePosition = quad_base + block_pos_delta_base_frame,
                                    baseOrientation=[0,0,0,1])
    cid = self._pybullet_client.createConstraint(quad_ID, -1, base_block_ID, -1,
          self._pybullet_client.JOINT_FIXED, [0, 0, 0], [0, 0, 0], -block_pos_delta_base_frame)

    # disable self collision between box and each link
    for i in range(-1,self._pybullet_client.getNumJoints(quad_ID)):
      self._pybullet_client.setCollisionFilterPair(quad_ID,base_block_ID, i,-1, 0)

def test_env():
  motor_control_mode = "CPG"
  task_env = "LR_COURSE_TASK"
  observation_space_mode = "MEDIUM"

  env = QuadrupedGymEnv(render=True,
                        task_env= task_env,
                        observation_space_mode = observation_space_mode,
                        on_rack=True,
                        motor_control_mode = motor_control_mode,
                        action_repeat=10,
                        )

  obs = env.reset()
  print(env.robot.GetBaseOrientationMatrix())
  action_dim = env._action_dim
  action_low = -np.ones(action_dim)
  action = action_low.copy()

  print('obs len', len(obs))
  print('act len', action_dim)

  while True:
    action = 2*np.random.rand(action_dim)-1
    obs, reward, terminated, truncated, info = env.step(action)
    print('obs len', len(obs))


if __name__ == "__main__":
  # test out some functionalities
  test_env()
  sys.exit()
