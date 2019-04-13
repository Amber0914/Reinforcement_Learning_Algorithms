# -*- coding: utf-8 -*-
from ExpReplay import ExpReplay
from DoubleDQN import QNET
import tensorflow as tf
import numpy as np

class Agent():
    def __init__(self, env, args):
        # set hyper parameters
        self.max_episodes = int(args.max_episodes)
        self.max_actions = int(args.max_actions)
        self.exploration_rate = float(args.exploration_rate)
        self.exploration_decay = float(args.exploration_decay)
        
        # set environment
        self.env = env
        self.states = env.observation_space.shape[0]
        self.actions = env.action_space.n
        
        # Experience Replay for batch learning
        self.exp = ExpReplay()
        # the number of experience per batch for batch learning
        self.batch_size = int(args.batch_size)
        
        # Deep Q Network
        self.qnet = QNET(self.states, self.actions, self.exp)
        # For execute Deep Q Network
        session = tf.InteractiveSession()
        session.run(tf.global_variables_initializer())
        self.qnet.set_session(session)
        
    def train(self):
        # set hyper parameters
        max_episodes = self.max_episodes
        max_actions = self.max_actions
        exploration_rate = self.exploration_rate
        exploration_decay = self.exploration_decay
        batch_size = self.batch_size
        
        # start training
        record_rewards = []
        for i in range(max_episodes):
            total_rewards = 0
            state = self.env.reset()
            state = state.reshape(1, self.states)
            for j in range(max_actions):
                #self.env.render() # Uncomment this line to render the environment
                action = self.qnet.get_action(state, exploration_rate)
                next_state, reward, done, info = self.env.step(action)
                next_state = next_state.reshape(1, self.states)
                total_rewards += reward
                
                if done:
                    self.exp.add(state, action, (reward-100), next_state, done)
                    self.qnet.batch_train(batch_size)
                    break
                    
                self.exp.add(state, action, reward, next_state, done)
                self.qnet.batch_train(batch_size)
                
                # update target network
                if (j%25)== 0 and j>0:
                    self.qnet.update()
                # next episode
                state = next_state
                
            record_rewards.append(total_rewards)
            exploration_rate = 0.001 + (exploration_rate-0.001)*np.exp(-exploration_decay*(i+1))
            if i%100==0 and i>0:
                average_rewards = np.mean(np.array(record_rewards))
                record_rewards = []
                print("episodes: %i to %i, average_reward: %.3f, exploration: %.3f" %(i-100, i, average_rewards, exploration_rate))
