import argparse
import os
import datetime
from typing import final
import gym
import numpy as np
import itertools
import torch
import robel
from sac import SAC
from PIL import Image, ImageDraw, ImageFont
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import imageio
import matplotlib.pyplot as plt

def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="DClawTurnFixed-v0",
                    help='Mujoco Gym environment (default: DClawTurnFixed-v0)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
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
parser.add_argument('--offline', action="store_true")
args = parser.parse_args()
# def save_gif(filename, frames, bounce=False, color_last=False, duration=0.05):
#     print('Save {} frames into video...'.format(len(frames)))
    # inputs = [np.array(frame.copy())/255 for frame in frames]
    # images = []
    # for tensor in inputs:
    #     if not color_last:
    #         tensor = tensor.transpose(0,1).transpose(1,2)
    #     tensor = np.clip(tensor, 0, 1)
    #     images.append((tensor * 255).astype('uint8'))
    # if bounce:
    #     images = images + list(reversed(images[1:-1]))
    # imageio.mimsave(filename, frames)

env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not os.path.exists('videos'):
    os.system('mkdir videos')
# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
folder_name = "2021-03-21_19-22-32_SAC_DClawTurnFixedD4-v0"
# folder_name = "."

#Tesnorboard
# Memory

# Training Loop
total_numsteps = 0
updates = 0
model_names = os.listdir(os.path.join('models',folder_name))
final_model_names = []
for name in model_names:
    if name.startswith('sac_actor'):
        final_model_names.append(name)
final_model_names.sort()
frames = []
x = []
y = []
total_episodes = int(final_model_names[-1][-4:])
figure = plt.figure()
plt.xlim([0, total_episodes])
plt.ylim([-800, 950])
for i, model_name in enumerate(final_model_names):
    episode = int(model_name[-4:])
    if len(x) > 0:
        assert episode - x[-1] >= 50
    x.append(episode)
    
    model_path = os.path.join(os.getcwd(),'models',folder_name, model_name)
    agent.load_model(model_path, None)
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        
        action = agent.select_action(state, evaluate=True)

        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        if args.offline:
            image_data = env.render(mode='rgb_array', width=1024, height=1024,)
            # frames.append(env.render(mode='rgb_array', width=1024, height=1024,))
            img = Image.fromarray(image_data, "RGB")
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype('./sans-serif.ttf', 36)
            draw.text((50, 100), "Instant Reward: {:.2f}".format(reward), (255, 0, 0), font=font)
            draw.text((50, 140), "Episode Reward: {:.2f}".format(episode_reward), (255, 0, 0), font=font)
            draw.text((50, 60), "Episode: {}".format(episode), (255, 0, 0), font=font)
            # img = np.array(img)
            # frames.append(img)
            # img.save(os.path.join(frame_dir, "frame-%.10d.png" % time_step_counter))
            if len(y) > 0:
                plt.xlim([0, total_episodes])
                plt.ylim([-800, 950])
                plt.xlabel("episodes")
                plt.ylabel("reward")
                plt.plot(x[:-1],y)
                plt.scatter(x[:-1], y)
                array_figure = fig2data(figure)
                pil_figure = Image.fromarray(array_figure)
                final_frame = Image.new('RGB',(1024, 1504), color="white")
                final_frame.paste(img,(0,0))
                final_frame.paste(pil_figure,(200, 1024))
                final_frame = np.array(final_frame)
                frames.append(final_frame)
                plt.cla()
                plt.clf()
            else:
                final_frame = Image.new('RGB',(1024, 1504), color="white")
                final_frame.paste(img,(0,0))
                final_frame = np.array(final_frame)
                frames.append(final_frame)
        else:
            env.render()

        state = next_state
    y.append(episode_reward)


    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(i, round(episode_reward, 2)))
    print("----------------------------------------")
if args.offline:
    imageio.mimsave('videos/{}.mp4'.format(args.env_name), frames)
    # save_gif('videos/{}.mp4'.format(args.env_name), frames, color_last=True)
env.close()

