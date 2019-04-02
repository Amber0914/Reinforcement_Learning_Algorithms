import numpy as np
import gym
from gym.envs.registration import register
import random
from argparse import ArgumentParser

class Environment():
    def __init__(self):
        pass
    def FrozenLakeNoSlippery(self):
        register(
                 id= 'FrozenLakeNoSlippery-v0',
                 entry_point='gym.envs.toy_text:FrozenLakeEnv',
                 kwargs={'map_name' : '4x4', 'is_slippery': False},
                 max_episode_steps=100,
                 reward_threshold=0.82
                 )
        env = gym.make('FrozenLakeNoSlippery-v0')
        return env

class QAgent():
    def __init__(self, args, env):
        # set hyperparameters
        self.max_episodes = int(args.max_episodes)
        self.max_actions = int(args.max_actions)
        self.learning_rate = float(args.learning_rate)
        self.discount = float(args.discount)
        self.exploration_rate = float(args.exploration_rate)
        self.exploration_decay = 1.0/float(args.max_episodes)
        
        # get environmnet
        self.env = env
        
        # initialize Q(s, a)
        row = env.observation_space.n
        col = env.action_space.n
        self.Q = np.zeros((row, col))
    
    def _policy(self, mode, state, e_rate=0):
        if mode=='train':
            if random.random() > e_rate:
                return np.argmax(self.Q[state,:]) # exploitation
            else:
                return self.env.action_space.sample() # exploration
        elif mode=='test':
            return np.argmax(self.Q[state,:])
    
    def train(self):
        # get hyper-parameters
        max_episodes = self.max_episodes
        max_actions = self.max_actions
        learning_rate = self.learning_rate
        discount = self.discount
        exploration_rate = self.exploration_rate
        exploration_decay = 1.0/self.max_episodes
        
        # reset Q for initialize
        row = self.env.observation_space.n
        col = self.env.action_space.n
        self.Q = np.zeros((row, col))

        # start training
        for i in range(max_episodes):
            state = self.env.reset() # reset the environment per eisodes
            for a in range(max_actions):
                action = self._policy('train', state, exploration_rate)
                new_state, reward, done, info = self.env.step(action)
                # The formulation of updating Q(s, a)
                self.Q[state, action] = self.Q [state, action] + learning_rate*(reward+discount*np.max(self.Q [new_state, :]) - self.Q [state, action])
                state = new_state # update the current state
                if done == True:  # if fall in the hole or arrive to the goal, then this episode is terminated.
                    break
            if exploration_rate>0.001:
                exploration_rate -= exploration_decay
    def test(self):
        # Setting hyper-parameters
        max_actions = self.max_actions
        state = self.env.reset() # reset the environment
        for a in range(max_actions):
            self.env.render() # show the environment states
            action = np.argmax(self.Q[state,:]) # take action with the Optimal Policy
            new_state, reward, done, info = self.env.step(action) # arrive to next_state after taking the action
            state = new_state # update current state
            if done:
                print("======")
                self.env.render()
                break
            print("======")
        self.env.close()

    def displayQ(self):
        print("Q\n", self.Q)

def arg_parse():
    parser = ArgumentParser()
    parser.add_argument("--max_episodes", help="max training episode", default=20000)
    parser.add_argument("--max_actions", help="max actions per episode", default=99)
    parser.add_argument("--learning_rate", help="learning rate alpha for Q-learning", default=0.83)
    parser.add_argument("--discount", help="discount factpr for Q-learning", default=0.93)
    parser.add_argument("--exploration_rate", help="exploration_rate", default=1.0)
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse() # get hyper-parameters
    env = Environment().FrozenLakeNoSlippery() # construct the environment
    agent = QAgent(args, env) # get agent
    agent.train()
    print("Testing Model")
    agent.test()
