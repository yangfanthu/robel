import robel
import gym
import pdb
import numpy as np

# Create a simulation environment for the D'Claw turn task.
env = gym.make('DClawTurnFixed-v0')

# Reset the environent and perform a random action.
obs = env.reset()
env.render()
for i in range(1000):
    action = env.action_space.sample()  # action is basivally torque command
    action = np.ones(9) * 0.3
    env.step(action)