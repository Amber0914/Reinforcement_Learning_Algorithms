# -*- coding: utf-8 -*-
import numpy as np

class ExpReplay():
    def __init__(self, e_max=15000, e_min=100):
        self._max = e_max # maximum number of experiences
        self._min = e_min # minimum number of experiences for training
        self.exp = {'state':[], 'action':[], 'reward':[], 'next_state':[], 'done':[]} # total experiences the Agent stored
        
    def get_max(self):
        """return the maximum number of experiences"""
        return self._max
    
    def get_min(self):
        """return the minimum number of experiences"""
        return self._min
    
    def get_num(self):
        """return the curren number of experiences"""
        return len(self.exp['state'])
    
    def get_batch(self, batch_size=64):
        """random choose a batch of experiences for training"""
        idx = np.random.choice(self.get_num(), size=batch_size, replace=False)
        state = np.array([self.exp['state'][i] for i in idx])
        action = [self.exp['action'][i] for i in idx]
        reward = [self.exp['reward'][i] for i in idx]
        next_state = np.array([self.exp['next_state'][i] for i in idx])
        done = [self.exp['done'][i] for i in idx]
        return state, action, reward, next_state, done
        
    def add(self, state, action, reward, next_state, done):
        """add single experience"""
        if self.get_num()>self.get_max():
            del self.exp['state'][0]
            del self.exp['action'][0]
            del self.exp['reward'][0]
            del self.exp['next_state'][0]
            del self.exp['done'][0]
        self.exp['state'].append(state)
        self.exp['action'].append(action)
        self.exp['reward'].append(reward)
        self.exp['next_state'].append(next_state)
        self.exp['done'].append(done)
