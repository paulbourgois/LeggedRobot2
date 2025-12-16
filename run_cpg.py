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

""" Run CPG """

import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging!
                    isRLGymInterface=False,     # False : not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)

TEST_STEPS = int(2 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot states
X = np.zeros((2,4,TEST_STEPS)) # CPG states
joint_pos_current = np.zeros((12,TEST_STEPS)) # joint angles
joint_pos_desired = np.zeros((12,TEST_STEPS)) # desired joint angles
foot_pos_desired = np.zeros((4,3,TEST_STEPS)) # desired foot positions
foot_pos_current = np.zeros((4,3,TEST_STEPS)) # current foot positions

############## Sample Gains
# joint PD gains
kp=np.array([100,100,100])
kd=np.array([2,2,2])*0.5

# Cartesian PD gains
kpCartesian = np.diag([500]*3)*1.5
kdCartesian = np.diag([20]*3)

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12)

  # get desired foot positions from CPG
  xs,zs = cpg.update()

  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)

    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])
    foot_pos_desired[i,:,j] = leg_xyz

    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = np.zeros(3) # [TODO]
    leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz)
    joint_pos_desired[3*i:3*i+3,j] = leg_q

    """ No VMC ? No Gravity Compensation ? Just joint PD and cartesian PD ?"""

    # Add joint PD contribution to tau for leg i (Equation 4)
    tau += np.zeros(3) # [TODO]
    tau += kp * (leg_q - q[3*i:3*i+3]) - kd * dq[3*i:3*i+3]

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get desired xyz position in leg frame (use ComputeJacobianAndPosition with the joint angles you just found above)
      # [TODO]
      J, pos = env.robot.ComputeJacobianAndPosition(i, leg_q)
      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      # [TODO]
      J_curr, pos_curr = env.robot.ComputeJacobianAndPosition(i, q[3*i:3*i+3])
      foot_pos_current[i,:,j] = pos_curr
      # Get current foot velocity in leg frame (Equation 2)
      # [TODO]
      foot_vel = J @ dq[3*i:3*i+3]

      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      # tau += np.zeros(3) # [TODO]
      tau += J_curr.T @ ( kpCartesian @ (leg_xyz - pos_curr) - kdCartesian @ foot_vel )

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

  # send torques to robot and simulate TIME_STEP seconds
  env.step(action)

  # [TODO] save any CPG or robot states
  X[:,:,j] = cpg.X
  for i in range(4):
    J, pos = env.robot.ComputeJacobianAndPosition(i, q[3*i:3*i+3])
    foot_pos_current[i,:,j] = pos
    joint_pos_current[:,j] = q
# end of simulation loop


#####################################################
# PLOTS
#####################################################
# [TODO] Create your plots

# A plot of the CPG states (r, θ, ˙r, ˙θ) for a trot gait (plots for other gaits are encouraged, but not required). We suggest making subplots for each leg, and make sure these are at a scale where the states are clearly visible (for example 2 gait cycles).
r = np.sqrt(X[0,:,:])
theta = X[1,:,:]
r_dot = np.zeros_like(r)
theta_dot = np.zeros_like(theta)
for i in range(4):
  r_dot[i,:] = np.gradient(r[i,:], TIME_STEP)
  theta_dot[i,:] = np.gradient(theta[i,:], TIME_STEP)
  # correct for numerical issues when theta wraps around 2pi then derivative is very large
  # if theta_dot is discontinuous, set to previous value
  for j in range(1, TEST_STEPS):
    if abs(theta_dot[i,j] - theta_dot[i,j-1]) > 20:
      theta_dot[i,j] = theta_dot[i,j-1]


# # Plot CPG states
# plt.figure(figsize=(10,8))
# for i in range(4):
#   plt.subplot(2,2,i+1)
#   plt.plot(t, r[i,:], label='r', color='b')
#   plt.plot(t, theta[i,:], label='theta (rad)', color='r')
#   plt.plot(t, r_dot[i,:], label='r dot', color='g')
#   plt.plot(t, theta_dot[i,:], label='theta dot (rad/s)', color='m')
#   plt.title(f'Leg {i} CPG States')
#   plt.xlabel('Time (s)')
#   plt.ylabel('States')
#   plt.legend()
#   plt.grid()
# plt.tight_layout()
# plt.show()

# plot comparing the desired foot position vs actual foot position using joint PD with/without Cartesian PD (for one leg is fine)
leg_to_plot = 0
plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(t, foot_pos_desired[leg_to_plot,0,:], label='Desired X', linestyle='--')
plt.plot(t, foot_pos_current[leg_to_plot,0,:], label='Actual X', linestyle='-')
plt.title(f'Leg {leg_to_plot} Foot Position: Desired vs Actual')
plt.ylabel('X Position (m)')
plt.legend()
plt.grid()
plt.show()


print('==============================================================')
print('CPG Test Summary:')
print('mean error in foot position (m):', np.mean(np.abs(foot_pos_desired - foot_pos_current)), '\n')
print('==============================================================')
#  plot comparing the desired joint angles vs actual joint angles using joint PD with/without Cartesian PD (for one leg is fine).
plt.figure(figsize=(10,6))
for joint in range(3):
  plt.subplot(3,1,joint+1)
  plt.plot(t, joint_pos_desired[3*leg_to_plot + joint,:], label=f'Desired Joint {joint} Angle', linestyle='--')
  plt.plot(t, joint_pos_current[3*leg_to_plot + joint,:], label=f'Actual Joint {joint} Angle', linestyle='-')
  plt.title(f'Leg {leg_to_plot} Joint {joint} Angle over Time')
  plt.ylabel('Angle (rad)')
  plt.legend()
  plt.grid()

plt.show()

# plt.figure()
# for i in range(4):
#   plt.plot(t, X[0,i,:], label=f'Leg {i} r')
# plt.title('CPG Amplitudes')
# plt.xlabel('Time (s)')
# plt.ylabel('r')
# plt.legend()
# plt.grid()

# plt.figure()
# for i in range(4):
#   plt.plot(t, X[1,i,:], label=f'Leg {i} theta')
# plt.title('CPG Phases')
# plt.xlabel('Time (s)')
# plt.ylabel('theta (rad)')
# plt.legend()
# plt.grid()

# plt.show()
