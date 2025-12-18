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

TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)
TEST_STEPS = int(2 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

def calculate_metrics(base_pos, base_vel, torques, foot_pos_current, dq_history, dt):
    """Calculate locomotion performance metrics including velocity, duty cycle, CoT, ground clearance"""

    # 1. Body velocity (forward = x-direction)
    forward_velocities = base_vel[:, 0]
    avg_velocity = np.mean(forward_velocities[500:])  # skip transient
    max_velocity = np.max(forward_velocities[500:])
    min_velocity = np.min(forward_velocities[500:])

    # 2. Distance traveled
    total_distance = np.abs(base_pos[-1, 0] - base_pos[0, 0])

    # 3. Contact detection and duty cycle
    # Foot in contact if z-position < threshold (adjust based on your ground plane)
    contact_threshold = -0.18  # meters, adjust to match your environment
    contacts = (foot_pos_current[:, :, 2] < contact_threshold).astype(float)  # (TEST_STEPS, 4)

    duty_cycles = np.mean(contacts[500:], axis=0)  # per leg, skip transient
    avg_duty_cycle = np.mean(duty_cycles)

    # 4. Step duration (detect stance/swing transitions)
    step_durations = []
    for leg in range(4):
        contact_signal = contacts[:, leg]
        transitions = np.diff(contact_signal)
        stance_starts = np.where(transitions > 0)[0]
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
  env = QuadrupedGymEnv(render=False,              # visualize
                      on_rack=False,              # useful for debugging!
                      isRLGymInterface=False,     # False : not using RL
                      time_step=TIME_STEP,
                      action_repeat=1,
                      motor_control_mode="TORQUE",
                      add_noise=False,    # start in ideal conditions
                      record_video=False
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

  # Add tracking for metrics
  base_positions = np.zeros((TEST_STEPS, 3))
  base_velocities = np.zeros((TEST_STEPS, 3))
  torques = np.zeros((TEST_STEPS, 12))

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
  # end of simulation loop

  # Calculate performance metrics
  metrics = calculate_metrics(base_positions, base_velocities, torques,
                              foot_pos_current.transpose(2, 0, 1), # (TEST_STEPS, 4, 3)
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

# vary joint gains for sensitivity analysis
JOINT_K = np.array([[kp,kd]]*10)*np.linspace(0.7,1.3,10)[:,None,None]
CARTESIAN_K = np.array([[kpCartesian, kdCartesian]] * 10) * np.linspace(0.7, 1.3, 10)[:, None, None, None]
MEAN_ERRORS_JOINT_WITHOUT_CARTESIAN = np.zeros(len(JOINT_K))
MEAN_ERRORS_CARTESIAN_WITHOUT_CARTESIAN = np.zeros(len(CARTESIAN_K))
FOOT_POS_CURRENT_WITHOUT_CARTESIAN = np.zeros((len(CARTESIAN_K),4,3,TEST_STEPS))
FOOT_POS_DESIRED_WITHOUT_CARTESIAN = np.zeros((len(CARTESIAN_K),4,3,TEST_STEPS))
JOINT_POS_CURRENT_WITHOUT_CARTESIAN = np.zeros((len(CARTESIAN_K),12,TEST_STEPS))
JOINT_POS_DESIRED_WITHOUT_CARTESIAN = np.zeros((len(CARTESIAN_K),12,TEST_STEPS))

# Add metrics storage
METRICS_WITHOUT_CARTESIAN = []

MEAN_ERRORS_JOINT_WITH_CARTESIAN = np.zeros(len(JOINT_K))
MEAN_ERRORS_CARTESIAN_WITH_CARTESIAN = np.zeros(len(CARTESIAN_K))
FOOT_POS_CURRENT_WITH_CARTESIAN = np.zeros((len(CARTESIAN_K),4,3,TEST_STEPS))
FOOT_POS_DESIRED_WITH_CARTESIAN = np.zeros((len(CARTESIAN_K),4,3,TEST_STEPS))
JOINT_POS_CURRENT_WITH_CARTESIAN = np.zeros((len(CARTESIAN_K),12,TEST_STEPS))
JOINT_POS_DESIRED_WITH_CARTESIAN = np.zeros((len(CARTESIAN_K),12,TEST_STEPS))

# Add metrics storage
METRICS_WITH_CARTESIAN = []

for p in range(len(JOINT_K)):
  kp = JOINT_K[p,0]
  kd = JOINT_K[p,1]
  print(f'Running simulation with Joint Kp: {kp}, Kd: {kd} | Cartesian Kp: {np.diag(kpCartesian)}, Kd: {np.diag(kdCartesian)}')
  X, joint_pos_current, joint_pos_desired, foot_pos_current, foot_pos_desired, TEST_STEPS, kp, kd, kpCartesian, kdCartesian, metrics = run_simulation(kp,kd,kpCartesian,kdCartesian,ADD_CARTESIAN_PD=False) # run without Cartesian PD
  MEAN_ERRORS_JOINT_WITHOUT_CARTESIAN[p] = np.mean(np.abs(foot_pos_desired - foot_pos_current))
  FOOT_POS_CURRENT_WITHOUT_CARTESIAN[p,:,:,:] = foot_pos_current
  FOOT_POS_DESIRED_WITHOUT_CARTESIAN[p,:,:,:] = foot_pos_desired
  JOINT_POS_CURRENT_WITHOUT_CARTESIAN[p,:,:] = joint_pos_current
  JOINT_POS_DESIRED_WITHOUT_CARTESIAN[p,:,:] = joint_pos_desired
  METRICS_WITHOUT_CARTESIAN.append(metrics)

for p in range(len(CARTESIAN_K)):
  best_joint_idx = np.argmin(MEAN_ERRORS_JOINT_WITHOUT_CARTESIAN)
  kp,kd = JOINT_K[best_joint_idx,0], JOINT_K[best_joint_idx,1]
  kpCartesian, kdCartesian = CARTESIAN_K[p,0], CARTESIAN_K[p,1]
  print(f'Running simulation with Joint Kp: {kp}, Kd: {kd} | Cartesian Kp: {np.diag(kpCartesian)}, Kd: {np.diag(kdCartesian)}')
  X, joint_pos_current, joint_pos_desired, foot_pos_current, foot_pos_desired, TEST_STEPS, kp, kd, kpCartesian, kdCartesian, metrics = run_simulation(kp,kd,kpCartesian,kdCartesian,ADD_CARTESIAN_PD=True) # run with Cartesian PD
  MEAN_ERRORS_CARTESIAN_WITH_CARTESIAN[p] = np.mean(np.abs(foot_pos_desired - foot_pos_current))
  FOOT_POS_CURRENT_WITH_CARTESIAN[p,:,:,:] = foot_pos_current
  FOOT_POS_DESIRED_WITH_CARTESIAN[p,:,:,:] = foot_pos_desired
  JOINT_POS_CURRENT_WITH_CARTESIAN[p,:,:] = joint_pos_current
  JOINT_POS_DESIRED_WITH_CARTESIAN[p,:,:] = joint_pos_desired
  METRICS_WITH_CARTESIAN.append(metrics)

# Print performance metrics summary
print("\n" + "="*70)
print("LOCOMOTION PERFORMANCE ANALYSIS")
print("="*70)

print("\n1. HYPERPARAMETERS TUNED:")
print(f"   - Joint PD gains: Kp range [{JOINT_K[0,0][0]:.1f}, {JOINT_K[-1,0][0]:.1f}], Kd range [{JOINT_K[0,1][0]:.2f}, {JOINT_K[-1,1][0]:.2f}]")
print(f"   - Cartesian PD gains: Kp range [{np.diag(CARTESIAN_K[0,0])[0]:.1f}, {np.diag(CARTESIAN_K[-1,0])[0]:.1f}], Kd range [{np.diag(CARTESIAN_K[0,1])[0]:.1f}, {np.diag(CARTESIAN_K[-1,1])[0]:.1f}]")

print("\n2. BODY VELOCITY (m/s) - Without Cartesian PD:")
velocities_no_cart = [m['avg_velocity'] for m in METRICS_WITHOUT_CARTESIAN]
print(f"   - Highest: {np.max(velocities_no_cart):.4f} m/s")
print(f"   - Lowest: {np.min(velocities_no_cart):.4f} m/s")
print(f"   - Average: {np.mean(velocities_no_cart):.4f} m/s")

print("\n3. BODY VELOCITY (m/s) - With Cartesian PD:")
velocities_with_cart = [m['avg_velocity'] for m in METRICS_WITH_CARTESIAN]
print(f"   - Highest: {np.max(velocities_with_cart):.4f} m/s")
print(f"   - Lowest: {np.min(velocities_with_cart):.4f} m/s")
print(f"   - Average: {np.mean(velocities_with_cart):.4f} m/s")

print("\n4. DUTY CYCLE (stance fraction):")
duty_cycles_no_cart = [m['duty_cycle'] for m in METRICS_WITHOUT_CARTESIAN]
duty_cycles_with_cart = [m['duty_cycle'] for m in METRICS_WITH_CARTESIAN]
print(f"   - Without Cartesian PD: {np.mean(duty_cycles_no_cart):.3f} (range: {np.min(duty_cycles_no_cart):.3f} to {np.max(duty_cycles_no_cart):.3f})")
print(f"   - With Cartesian PD: {np.mean(duty_cycles_with_cart):.3f} (range: {np.min(duty_cycles_with_cart):.3f} to {np.max(duty_cycles_with_cart):.3f})")

print("\n5. STEP TIMING:")
step_durations_no_cart = [m['avg_step_duration'] for m in METRICS_WITHOUT_CARTESIAN]
step_durations_with_cart = [m['avg_step_duration'] for m in METRICS_WITH_CARTESIAN]
stance_times_no_cart = [m['stance_time'] for m in METRICS_WITHOUT_CARTESIAN]
swing_times_no_cart = [m['swing_time'] for m in METRICS_WITHOUT_CARTESIAN]
stance_times_with_cart = [m['stance_time'] for m in METRICS_WITH_CARTESIAN]
swing_times_with_cart = [m['swing_time'] for m in METRICS_WITH_CARTESIAN]
print(f"   - Without Cartesian PD:")
print(f"     * Step duration: {np.mean(step_durations_no_cart):.4f} s")
print(f"     * Stance time: {np.mean(stance_times_no_cart):.4f} s")
print(f"     * Swing time: {np.mean(swing_times_no_cart):.4f} s")
print(f"   - With Cartesian PD:")
print(f"     * Step duration: {np.mean(step_durations_with_cart):.4f} s")
print(f"     * Stance time: {np.mean(stance_times_with_cart):.4f} s")
print(f"     * Swing time: {np.mean(swing_times_with_cart):.4f} s")

print("\n6. COST OF TRANSPORT:")
CoTs_no_cart = [m['CoT'] for m in METRICS_WITHOUT_CARTESIAN]
CoTs_with_cart = [m['CoT'] for m in METRICS_WITH_CARTESIAN]
print(f"   - Without Cartesian PD:")
print(f"     * Best CoT: {np.min(CoTs_no_cart):.4f}")
print(f"     * Worst CoT: {np.max(CoTs_no_cart):.4f}")
print(f"     * Average CoT: {np.mean(CoTs_no_cart):.4f}")
print(f"   - With Cartesian PD:")
print(f"     * Best CoT: {np.min(CoTs_with_cart):.4f}")
print(f"     * Worst CoT: {np.max(CoTs_with_cart):.4f}")
print(f"     * Average CoT: {np.mean(CoTs_with_cart):.4f}")

print("\n7. GROUND CLEARANCE (m):")
clearances_no_cart = [m['ground_clearance'] for m in METRICS_WITHOUT_CARTESIAN]
clearances_with_cart = [m['ground_clearance'] for m in METRICS_WITH_CARTESIAN]
print(f"   - Without Cartesian PD: {np.mean(clearances_no_cart):.4f} m (range: {np.min(clearances_no_cart):.4f} to {np.max(clearances_no_cart):.4f})")
print(f"   - With Cartesian PD: {np.mean(clearances_with_cart):.4f} m (range: {np.min(clearances_with_cart):.4f} to {np.max(clearances_with_cart):.4f})")

print("\n8. GROUND PENETRATION (m) - should be ~0:")
penetrations_no_cart = [m['ground_penetration'] for m in METRICS_WITHOUT_CARTESIAN]
penetrations_with_cart = [m['ground_penetration'] for m in METRICS_WITH_CARTESIAN]
print(f"   - Without Cartesian PD: {np.mean(penetrations_no_cart):.4f} m (max: {np.max(penetrations_no_cart):.4f})")
print(f"   - With Cartesian PD: {np.mean(penetrations_with_cart):.4f} m (max: {np.max(penetrations_with_cart):.4f})")

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

# plot comparing the desired foot position vs actual foot position using joint PD with/without Cartesian PD (for one leg is fine)
leg_to_plot = 0
plt.figure(figsize=(10,6))

best_cartesian_idx = np.argmin(MEAN_ERRORS_CARTESIAN_WITH_CARTESIAN)
best_joint_idx = np.argmin(MEAN_ERRORS_JOINT_WITHOUT_CARTESIAN)
print('Best JOINT GAINS : Kp:', JOINT_K[best_joint_idx,0], 'Kd:', JOINT_K[best_joint_idx,1])
print('Best CARTESIAN GAINS : Kp:', np.diag(CARTESIAN_K[best_cartesian_idx,0]), 'Kd:', np.diag(CARTESIAN_K[best_cartesian_idx,1]))
foot_pos_current_without_cartesian = FOOT_POS_CURRENT_WITHOUT_CARTESIAN[best_joint_idx,:,:,:]
foot_pos_desired_without_cartesian = FOOT_POS_DESIRED_WITHOUT_CARTESIAN[best_joint_idx,:,:,:]
foot_pos_current_with_cartesian = FOOT_POS_CURRENT_WITH_CARTESIAN[best_cartesian_idx,:,:,:]
foot_pos_desired_with_cartesian = FOOT_POS_DESIRED_WITH_CARTESIAN[best_cartesian_idx,:,:,:]
joint_pos_current_without_cartesian  = JOINT_POS_CURRENT_WITHOUT_CARTESIAN[best_joint_idx,:,:]
joint_pos_desired_without_cartesian  = JOINT_POS_DESIRED_WITHOUT_CARTESIAN[best_joint_idx,:,:]
joint_pos_current_with_cartesian  = JOINT_POS_CURRENT_WITH_CARTESIAN[best_cartesian_idx,:,:]
joint_pos_desired_with_cartesian  = JOINT_POS_DESIRED_WITH_CARTESIAN[best_cartesian_idx,:,:]

for j in range(3):
  plt.subplot(3,1,j+1)
  plt.plot(t, foot_pos_desired_without_cartesian[leg_to_plot,j,:], label=f'Desired {["X","Y","Z"][j]}', linestyle='--')
  plt.plot(t, foot_pos_current_without_cartesian[leg_to_plot,j,:], label=f'Actual {["X","Y","Z"][j]}', linestyle='-')
  plt.plot(t, foot_pos_desired_with_cartesian[leg_to_plot,j,:], label=f'Desired {["X","Y","Z"][j]} (with Cartesian PD)', linestyle='--')
  plt.plot(t, foot_pos_current_with_cartesian[leg_to_plot,j,:], label=f'Actual {["X","Y","Z"][j]} (with Cartesian PD)', linestyle='-')
  plt.title(f'Leg {leg_to_plot} Foot Position: Desired vs Actual (without Cartesian PD)')
  plt.ylabel(f'{["X","Y","Z"][j]} Position (m)')
  plt.legend()
  plt.grid()
plt.show()


print('==============================================================')
print('CPG Test Summary:')
print('Joint Position Gains kp:', kp, 'Joint Velocity Gains kd:', kd)
if False:
  print('Cartesian Position Gains kpCartesian:', np.diag(kpCartesian), 'Cartesian Velocity Gains kdCartesian:', np.diag(kdCartesian))
print('mean error in foot position (m):', np.mean(np.abs(foot_pos_desired - foot_pos_current)), '\n')
print('==============================================================')
#  plot comparing the desired joint angles vs actual joint angles using joint PD with/without Cartesian PD (for one leg is fine).
plt.figure(figsize=(10,6))
for joint in range(3):
  plt.subplot(3,1,joint+1)
  plt.plot(t, joint_pos_desired_without_cartesian[3*leg_to_plot + joint,:], label=f'Desired Joint {joint} Angle', linestyle='--')
  plt.plot(t, joint_pos_current_without_cartesian[3*leg_to_plot + joint,:], label=f'Actual Joint {joint} Angle', linestyle='-')
  plt.plot(t, joint_pos_desired_with_cartesian[3*leg_to_plot + joint,:], label=f'Desired Joint {joint} Angle (with Cartesian PD)', linestyle='--')
  plt.plot(t, joint_pos_current_with_cartesian[3*leg_to_plot + joint,:], label=f'Actual Joint {joint} Angle (with Cartesian PD)', linestyle='-')
  plt.title(f'Leg {leg_to_plot} Joint {joint} Angle over Time')
  plt.ylabel('Angle (rad)')
  plt.legend()
  plt.grid()


# worst_joint_idx = np.argmax(MEAN_ERRORS_JOINT_WITHOUT_CARTESIAN)
# worst_cartesian_idx = np.argmax(MEAN_ERRORS_CARTESIAN_WITH_CARTESIAN)
# foot_pos_current_without_cartesian_worst = FOOT_POS_CURRENT_WITHOUT_CARTESIAN[worst_joint_idx,:,:,:]
# foot_pos_desired_without_cartesian_worst = FOOT_POS_DESIRED_WITHOUT_CARTESIAN[worst_joint_idx,:,:,:]
# foot_pos_current_with_cartesian_worst = FOOT_POS_CURRENT_WITH_CARTESIAN[worst_cartesian_idx,:,:,:]
# foot_pos_desired_with_cartesian_worst = FOOT_POS_DESIRED_WITH_CARTESIAN[worst_cartesian_idx,:,:,:]
# joint_pos_current_without_cartesian_worst  = JOINT_POS_CURRENT_WITHOUT_CARTESIAN[worst_joint_idx,:,:]
# joint_pos_desired_without_cartesian_worst  = JOINT_POS_DESIRED_WITHOUT_CARTESIAN[worst_joint_idx,:,:]
# joint_pos_current_with_cartesian_worst  = JOINT_POS_CURRENT_WITH_CARTESIAN[worst_cartesian_idx,:,:]
# joint_pos_desired_with_cartesian_worst  = JOINT_POS_DESIRED_WITH_CARTESIAN[worst_cartesian_idx,:,:]
# plt.figure(figsize=(10,6))
# for joint in range(3):
#   plt.subplot(3,1,joint+1)
#   plt.plot(t, joint_pos_desired_without_cartesian[3*leg_to_plot + joint,:], label=f'Desired Joint {joint} Angle', linestyle='--')
#   plt.plot(t, joint_pos_current_without_cartesian[3*leg_to_plot + joint,:], label=f'Actual Joint {joint} Angle', linestyle='-')
#   plt.plot(t, joint_pos_desired_with_cartesian[3*leg_to_plot + joint,:], label=f'Desired Joint {joint} Angle (with Cartesian PD)', linestyle='--')
#   plt.plot(t, joint_pos_current_with_cartesian[3*leg_to_plot + joint,:], label=f'Actual Joint {joint} Angle (with Cartesian PD)', linestyle='-')
#   plt.title(f'Leg {leg_to_plot} Joint {joint} Angle over Time')
#   plt.ylabel('Angle (rad)')
#   plt.legend()
#   plt.grid()

plt.figure(figsize=(10,6))
for joint in range(3):
  plt.subplot(3,1,joint+1)
  for p in range(len(JOINT_K)):
    plt.plot(t, JOINT_POS_CURRENT_WITHOUT_CARTESIAN[p, 3*leg_to_plot + joint, :], label=f'Kp: {np.round(JOINT_K[p,0],3)}, Kd: {np.round(JOINT_K[p,1],3)}', linestyle='-')
  plt.plot(t, joint_pos_desired_without_cartesian[3*leg_to_plot + joint,:], label=f'Desired Joint {joint} Angle', linestyle='--', color='black')
  plt.title(f'Leg {leg_to_plot} Joint {joint} Angle over Time for varying Joint PD gains')
  plt.ylabel('Angle (rad)')
  plt.legend()
  plt.grid()

plt.figure(figsize=(10,6))
for joint in range(3):
  plt.subplot(3,1,joint+1)
  for p in range(len(CARTESIAN_K)):
    plt.plot(t, JOINT_POS_CURRENT_WITH_CARTESIAN[p, 3*leg_to_plot + joint, :], label=f'Kp: {np.round(np.diag(CARTESIAN_K[p,0]),3)}, Kd: {np.round(np.diag(CARTESIAN_K[p,1]),3)}', linestyle='-')
  plt.plot(t, joint_pos_desired_with_cartesian[3*leg_to_plot + joint,:], label=f'Desired Joint {joint} Angle', linestyle='--', color='black')
  plt.title(f'Leg {leg_to_plot} Joint {joint} Angle over Time for varying Cartesian PD gains')
  plt.ylabel('Angle (rad)')
  plt.legend()
  plt.grid()

#display a table of mean errors for only joint PD and then for fixed joint PD + varying Cartesian PD
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title('Mean Foot Position Errors (m) - Varying Joint PD Gains')
bars = plt.bar([f'Kp:{JOINT_K[i,0]},Kd:{JOINT_K[i,1]}' for i in range(len(JOINT_K))], MEAN_ERRORS_JOINT_WITHOUT_CARTESIAN)
plt.bar_label(bars, fmt='%.5f')
plt.xticks(rotation=45, ha='right')

plt.subplot(1,2,2)
plt.title('Mean Foot Position Errors (m) - Varying Cartesian PD Gains')
bars = plt.bar([f'Kp:{CARTESIAN_K[i,0]},Kd:{CARTESIAN_K[i,1]}' for i in range(len(CARTESIAN_K))], MEAN_ERRORS_CARTESIAN_WITH_CARTESIAN)
plt.bar_label(bars, fmt='%.5f')
plt.xticks(rotation=45, ha='right')
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
