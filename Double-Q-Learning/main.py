# -*- coding: utf-8 -*-
from Agent import Agent
from argparse import ArgumentParser
import gym

def arg_parse():
    parser = ArgumentParser()
    parser.add_argument("--max_episodes", help="max training episode", default=20000)
    parser.add_argument("--max_actions", help="max actions per episode", default=10000)
    parser.add_argument("--exploration_rate", help="exploration_rate", default=1.0)
    parser.add_argument("--exploration_decay", help="exploration_decay", default=0.0001)
    parser.add_argument("--batch_size", help="batch_size", default=64)
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse() # get hyper-parameters
    env = gym.make('CartPole-v1')
    agent = Agent(env, args)
    agent.train()
