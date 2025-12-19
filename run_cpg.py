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
from env.classical_hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)
TEST_STEPS = int(10 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP
gait = "PACE"
# TROT PACE BOUND WALK
def calculate_metrics(base_pos, base_vel, torques, foot_pos_current, contact_history, dq_history, dt):
  """Calculate locomotion performance metrics including velocity, duty cycle, CoT, ground clearance"""

  # 1. Body velocity (forward = x-direction)
  forward_velocities = base_vel[:, 0]
  avg_velocity = np.mean(forward_velocities[500:])  # skip transient
  max_velocity = np.max(forward_velocities[500:])
  min_velocity = np.min(forward_velocities[500:])

  # 2. Distance traveled
  total_distance = np.abs(base_pos[-1, 0] - base_pos[0, 0])

  # 3. Contact detection and duty cycle
  contact_threshold = -0.18  # meters, adjust to match your environment
  if contact_history is not None and contact_history.shape[0] == foot_pos_current.shape[0]:
    contacts = contact_history.astype(float)
  else:
    contacts = (foot_pos_current[:, :, 2] < contact_threshold).astype(float)  # (TEST_STEPS, 4)

  duty_cycles = np.mean(contacts[500:], axis=0)  # per leg, skip transient
  avg_duty_cycle = np.mean(duty_cycles)

  # 4. Step duration (detect stance/swing transitions)
  step_durations = []
  for leg in range(4):
    contact_signal = contacts[:, leg]
    transitions = np.diff(contact_signal)
    stance_starts = np.where(transitions > 0)[0] + 1
    if contact_signal[0] > 0:
      stance_starts = np.insert(stance_starts, 0, 0)
    if len(stance_starts) > 1:
      step_times = np.diff(stance_starts) * dt
      step_durations.extend(step_times)

  avg_step_duration = np.mean(step_durations) if step_durations else 0.0
  stance_time = avg_step_duration * avg_duty_cycle
  swing_time = avg_step_duration * (1 - avg_duty_cycle)

  # 5. Ground clearance (max foot height during swing phase)
  ground_clearances = []
  for leg in range(4):
    swing_mask = contacts[:, leg] == 0
    if np.any(swing_mask):
      swing_heights = foot_pos_current[swing_mask, leg, 2]
      max_clearance = np.max(swing_heights) - contact_threshold
      ground_clearances.append(max_clearance)
  avg_ground_clearance = np.mean(ground_clearances) if ground_clearances else 0.0

  # 6. Ground penetration (negative z-values below ground plane, should be ~0)
  # Assuming ground is at z=0 in world frame, feet shouldn't go below contact_threshold
  min_foot_z = np.min(foot_pos_current[:, :, 2])
  ground_penetration = max(0.0, contact_threshold - min_foot_z)  # positive = penetration

  # 7. Cost of Transport (CoT)
  # Mechanical power = sum of |torque * angular_velocity| over all joints
  total_energy = 0.0
  for t_idx in range(len(torques)):
    if t_idx < dq_history.shape[1]:
      # Power = |tau · omega| for each joint
      power = np.sum(np.abs(torques[t_idx] * dq_history[:, t_idx]))
      total_energy += power * dt

  robot_mass = 12.0  # A1 robot mass in kg
  g = 9.81
  weight = robot_mass * g

  CoT = total_energy / (weight * total_distance) if total_distance > 0 else np.inf

  return {
      'avg_velocity': avg_velocity,
      'max_velocity': max_velocity,
      'min_velocity': min_velocity,
      'total_distance': total_distance,
      'duty_cycle': avg_duty_cycle,
      'duty_cycles_per_leg': duty_cycles,
      'avg_step_duration': avg_step_duration,
      'stance_time': stance_time,
      'swing_time': swing_time,
      'CoT': CoT,
      'total_energy': total_energy,
      'ground_clearance': avg_ground_clearance,
      'ground_penetration': ground_penetration
  }

def run_simulation(kp,kd,kpCartesian,kdCartesian,ADD_CARTESIAN_PD=False):
  env = QuadrupedGymEnv(render=True,              # visualize
                      on_rack=False,              # useful for debugging!
                      isRLGymInterface=False,     # False : not using RL
                      time_step=TIME_STEP,
                      action_repeat=1,
                      motor_control_mode="TORQUE",
                      add_noise=False,    # start in ideal conditions
                      record_video=False
                      )

  # initialize Hopf Network, supply gait TROT PACE BOUND WALK
  cpg = HopfNetwork(gait=gait,time_step=TIME_STEP)
  t = np.arange(TEST_STEPS)*TIME_STEP

  # [TODO] initialize data structures to save CPG and robot states
  X = np.zeros((2,4,TEST_STEPS)) # CPG states
  joint_pos_current = np.zeros((12,TEST_STEPS)) # joint angles
  joint_pos_desired = np.zeros((12,TEST_STEPS)) # desired joint angles
  foot_pos_desired = np.zeros((4,3,TEST_STEPS)) # desired foot positions
  foot_pos_current = np.zeros((4,3,TEST_STEPS)) # current foot positions

  # Add tracking for metrics
  base_positions = np.zeros((TEST_STEPS, 3))
  base_velocities = np.zeros((TEST_STEPS, 3))
  torques = np.zeros((TEST_STEPS, 12))
  foot_contacts = np.zeros((TEST_STEPS, 4))

  ############## Sample Gains

  for j in range(TEST_STEPS):
    # initialize torque array to send to motors
    action = np.zeros(12)

    # get desired foot positions from CPG
    xs,zs = cpg.update()

    # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
    q = env.robot.GetMotorAngles()
    dq = env.robot.GetMotorVelocities()

    # Track base position and velocity
    base_positions[j] = env.robot.GetBasePosition()
    base_velocities[j] = env.robot.GetBaseLinearVelocity()

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

    # Track applied torques
    torques[j] = action

    # [TODO] save any CPG or robot states
    X[:,:,j] = cpg.X
    for i in range(4):
      J, pos = env.robot.ComputeJacobianAndPosition(i, q[3*i:3*i+3])
      foot_pos_current[i,:,j] = pos
      joint_pos_current[:,j] = q
    _, _, _, contact_bool = env.robot.GetContactInfo()
    foot_contacts[j] = contact_bool
  # end of simulation loop

  # Calculate performance metrics
  metrics = calculate_metrics(base_positions, base_velocities, torques,
                              foot_pos_current.transpose(2, 0, 1), # (TEST_STEPS, 4, 3)
                              foot_contacts,
                              joint_pos_current, TIME_STEP)

  env.close()
  return X, joint_pos_current, joint_pos_desired, foot_pos_current, foot_pos_desired, TEST_STEPS, kp, kd, kpCartesian, kdCartesian, metrics

#####################################################
# PLOTS
#####################################################
# [TODO] Create your plots
# joint PD gains
kp=np.array([100,100,100])
kd=np.array([1,1,1])
# Cartesian PD gains
kpCartesian = np.diag([750]*3)
kdCartesian = np.diag([20]*3)


X, joint_pos_current, joint_pos_desired, foot_pos_current, foot_pos_desired, TEST_STEPS, kp, kd, kpCartesian, kdCartesian, metrics = run_simulation(kp,kd,kpCartesian,kdCartesian,ADD_CARTESIAN_PD=True) # run with Cartesian PD

print('Gait is : ',gait)
print("Avg Velocity (m/s): ", metrics['avg_velocity'])
print("Duty Cycle: ", metrics['duty_cycle'])
print("Cost of Transport: ", metrics['CoT'])
print("Avg step duration (s): ", metrics['avg_step_duration'])


print("="*70 + "\n")

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

# print body volocity speed, duty cycle/ratio, time duration of one step(time in stance and swing), cost of transport


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
