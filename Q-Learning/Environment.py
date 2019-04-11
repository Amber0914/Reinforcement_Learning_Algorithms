import gym
from gym.envs.registration import register

class Environment():
    def __init__(self):
        pass
    
    def FrozenLakeNoSlippery(self):
        register(
                 id= 'FrozenLakeNoSlippery-v0',
                 entry_point='gym.envs.toy_text:FrozenLakeEnv',
                 kwargs={'map_name' : '4x4', 'is_slippery': False},
                 max_episode_steps=100,
                 reward_threshold=0.82)
        env = gym.make('FrozenLakeNoSlippery-v0')
        return env
