import os
import gym
import numpy as np
import slimevolleygym
from slimevolleygym import SurvivalRewardEnv
from stable_baselines3.dqn import DQN
from stable_baselines3.ppo import PPO
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.record_video import RecordVideo
from atari_wrappers import RenderWrapper, BufferWrapper, ImageToPyTorch

def recordVideoForModel(model):
    env_name = 'SlimeVolleyNoFrameskip-v0'
    env = gym.make(env_name)
    env = RenderWrapper(env)
    env = SurvivalRewardEnv(env)
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env, True)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4, np.uint8)
    env = RecordVideo(env, './videos', name_prefix=model)

    sub_folder = 'ppo/' if model.startswith('ppo') else 'dqn/'
    model, _ = createOrLoadModel(model, env, f'{model}_', '_steps.zip', sub_folder)

    obs = env.reset()

    done = False
    obss = [obs]
    sum_rewards = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        sum_rewards += reward
        obss.append(obs)

    env.close()

def createOrLoadModel(model_name, env, model_prefix, model_suffix, sub_folder=''):
    model = defaultModel(model_name, env)

    if not os.path.exists(f'./models/{sub_folder}'):
        return model, 0
    
    files = os.listdir(f'./models/{sub_folder}')

    if len(files) == 0:
        return model, 0

    try:
        model_files = filter(lambda x: x.endswith('.zip'), files)
        steps = [int(x.removeprefix(model_prefix).removesuffix(model_suffix)) for x in model_files]
        steps.sort()

        model_name = f'./models/{sub_folder}{model_prefix}{steps[-1]}{model_suffix}'
        print(f'Loading {model_name}')

        model.set_parameters(model_name)

        return model, steps[-1]
    except:
        return model, 0

def defaultModel(model, env):
    if model.startswith('ppo'):
        return PPO("CnnPolicy", env, verbose=0)
    return DQN("CnnPolicy", env, verbose=0, buffer_size=50000)
