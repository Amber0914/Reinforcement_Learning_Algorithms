from Environment import Environment
import tensorflow as tf
import numpy as np
import random
from argparse import ArgumentParser

class DeepQAgent():
    
    def __init__(self, args, env):
        # set hyperparameters
        self.max_episodes = int(args.max_episodes)
        self.max_actions = int(args.max_actions)
        self.discount = float(args.discount)
        self.exploration_rate = float(args.exploration_rate)
        self.exploration_decay = 1.0/float(args.max_episodes)
        # get envirionment
        self.env = env
    
        # nn_model parameters
        self.in_units = env.observation_space.n
        self.out_units = env.action_space.n
        self.hidden_units = int(args.hidden_units)
        
        # construct nn model
        self._nn_model()
    
        # save nn model
        self.saver = tf.train.Saver()

    def _nn_model(self):
        self.a0 = tf.placeholder(tf.float32, shape=[1, self.in_units]) # input layer
        self.y = tf.placeholder(tf.float32, shape=[1, self.out_units]) # ouput layer
        
        # from input layer to hidden layer
        w1 = tf.Variable(tf.zeros([self.in_units, self.hidden_units], dtype=tf.float32), name='w1') # weight
        b1 = tf.Variable(tf.random_uniform([self.hidden_units], 0, 0.01, dtype=tf.float32), name='b1') # bias
        a1 = tf.nn.relu(tf.matmul(self.a0, w1) + b1) # the ouput of hidden layer
        
        # from hidden layer to output layer
        w2 = tf.Variable(tf.zeros([self.hidden_units, self.out_units], dtype=tf.float32), name='w2') # weight
        b2 = tf.Variable(tf.random_uniform([self.out_units], 0, 0.01, dtype=tf.float32), name='b2') # bias
        
        # Q-value and Action
        self.a2 = tf.matmul(a1, w2) + b2 # the predicted_y (Q-value) of four actions
        self.action = tf.argmax(self.a2, 1) # the agent would take the action which has maximum Q-value
        
        # loss function
        loss = tf.reduce_sum(tf.square(self.a2-self.y))
        
        # upate model, minimizing loss function
        self.update_model =  tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
    
    def train(self):
        # hyper parameter
        max_episodes = self.max_episodes
        max_actions = self.max_actions
        discount = self.discount
        exploration_rate = self.exploration_rate
        exploration_decay = self.exploration_decay
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(max_episodes):
                state = env.reset()
                for j in range(max_actions):
                    action, pred_Q = sess.run([self.action, self.a2],feed_dict={self.a0:np.eye(self.in_units)[state:state+1]})
                    
                    if np.random.rand()<exploration_rate: # exploration
                        action[0] = env.action_space.sample() # take a random action

                    next_state, rewards, done, info = env.step(action[0])
                    next_Q = sess.run(self.a2,feed_dict={self.a0:np.eye(self.in_units)[next_state:next_state+1]})

                    update_Q = pred_Q
                    update_Q [0,action[0]] = rewards + discount*np.max(next_Q)
                    
                    sess.run([self.update_model],
                             feed_dict={self.a0:np.identity(16)[state:state+1],self.y:update_Q})
                    state = next_state
                    
                    if done:
                        if exploration_rate > 0.001:
                            exploration_rate -= exploration_decay
                        break
            save_path = self.saver.save(sess, "./nn_model.ckpt")

    def test(self):
        max_actions = self.max_actions # hyper-parameter
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver=tf.train.import_meta_graph("./nn_model.ckpt.meta") # restore model
            saver.restore(sess, tf.train.latest_checkpoint('./'))# 載入參數
            state = env.reset()
            for j in range(max_actions):
                env.render()
                action, pred_Q = sess.run([self.action, self.a2],feed_dict={self.a0:np.eye(self.in_units)[state:state+1]})
                next_state, rewards, done, info = env.step(action[0])
                state = next_state
                if done:
                    env.render()
                    break

def arg_parse():
    parser = ArgumentParser()
    parser.add_argument("--max_episodes", help="max training episode", default=20000)
    parser.add_argument("--max_actions", help="max actions per episode", default=99)
    parser.add_argument("--discount", help="discount factpr for Q-learning", default=0.95)
    parser.add_argument("--exploration_rate", help="exploration_rate", default=1.0)
    parser.add_argument("--hidden_units", help="hidden units", default=10)
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse() # get hyper-parameters
    env = Environment().FrozenLakeNoSlippery() # construct the environment
    agent = DeepQAgent(args, env) # get agent
    print("START TRAINING...")
    agent.train()
    print("\n==============\nTEST==============\n")
    agent.test()
