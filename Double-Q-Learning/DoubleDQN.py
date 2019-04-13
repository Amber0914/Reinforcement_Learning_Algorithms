# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class TNET():
    """
    Target network is for calculating the maximum estimated Q-value in given action a.
    """
    def __init__(self, in_units, out_units, hidden_units=250):
        self.in_units = in_units
        self.out_units = out_units
        self.hidden_units = hidden_units
        self._model()
        
    def _model(self):
        with tf.variable_scope('tnet'):
            self.x = tf.placeholder(tf.float32, shape=(None, self.in_units))
            
            W1=tf.get_variable('W1', shape=(self.in_units, self.hidden_units), initializer=tf.random_normal_initializer())
            W2=tf.get_variable('W2', shape=(self.hidden_units, self.hidden_units), initializer=tf.random_normal_initializer())
            W3=tf.get_variable('W3', shape=(self.hidden_units, self.out_units), initializer=tf.random_normal_initializer())
            
            b1=tf.get_variable('b1', shape=(self.hidden_units), initializer=tf.zeros_initializer())
            b2=tf.get_variable('b2', shape=(self.hidden_units), initializer=tf.zeros_initializer())
 
            h1=tf.nn.tanh(tf.matmul(self.x, W1)+b1)
            h2=tf.nn.tanh(tf.matmul(h1, W2)+b2)
            self.q=tf.matmul(h2, W3)

            self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='tnet')
            
            
class QNET():
    def __init__(self, in_units, out_units, exp, hidden_units=250):
        # Target Network
        self.tnet = TNET(in_units, out_units)
        
        # Q network architecture
        self.in_units = in_units
        self.out_units = out_units
        self.hidden_units = hidden_units
        self._model()
        self._batch_learning_model()
        self._tnet_update()
        
        # experience replay
        self.exp = exp 
        
    def _model(self):
        """ Q-network architecture """
        with tf.variable_scope('qnet'):
            self.x = tf.placeholder(tf.float32, shape=(None, self.in_units))
            
            W1 = tf.get_variable('W1', shape=(self.in_units, self.hidden_units), initializer=tf.random_normal_initializer())
            W2 = tf.get_variable('W2', shape=(self.hidden_units, self.hidden_units), initializer=tf.random_normal_initializer())
            W3 = tf.get_variable('W3', shape=(self.hidden_units, self.out_units), initializer=tf.random_normal_initializer())
            
            b1 = tf.get_variable('b1', shape=(self.hidden_units), initializer=tf.zeros_initializer())
            b2 = tf.get_variable('b2', shape=(self.hidden_units), initializer=tf.zeros_initializer())
 
            h1 = tf.nn.tanh(tf.matmul(self.x, W1)+b1)
            h2 = tf.nn.tanh(tf.matmul(h1, W2)+b2)
            self.q = tf.matmul(h2, W3)
            
    def _batch_learning_model(self):
        """For batch learning"""
        with tf.variable_scope('qnet'):
            # TD-target
            self.target = tf.placeholder(tf.float32, shape=(None, ))
            # Action index
            self.selected_idx = tf.placeholder(tf.int32, shape=(None, 2))
            # Q-value
            self.selected_q = tf.gather_nd(self.q, self.selected_idx)
            
            self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='qnet')
            
            # Q-network optimization alogrithms
            loss = tf.losses.mean_squared_error(self.target, self.selected_q)
            gradients = tf.gradients(loss, self.params)
            self.train_opt = tf.train.AdamOptimizer(3e-4).apply_gradients(zip(gradients, self.params))

    def _tnet_update(self):
        """ Update Target network by using the parameters of Q-Network"""
        with tf.variable_scope('qnet'):                        
            self.update_opt = [t.assign(q) for t, q in zip(self.tnet.params, self.params)]
    
    def batch_train(self, batch_size=64):
        """Implement Double DQN Algorithm, batch training"""
        if self.exp.get_num() < self.exp.get_min():
            #The number of experiences is not enough for batch training
            return

        # get a batch of experiences
        state, action, reward, next_state, done = self.exp.get_batch(batch_size)
        state = state.reshape(batch_size, self.in_units)
        next_state = next_state.reshape(batch_size, self.in_units)
        
        # get actions by Q-network
        qnet_q_values = self.session.run(self.q, feed_dict={self.x:next_state})
        qnet_actions = np.argmax(qnet_q_values, axis=1)
        
        # calculate estimated Q-values with qnet_actions by using Target-network
        tnet_q_values = self.session.run(self.tnet.q, feed_dict={self.tnet.x:next_state})
        tnet_q = [np.take(tnet_q_values[i], qnet_actions[i]) for i in range(batch_size)]
        
        # Update Q-values of Q-network
        qnet_update_q = [r+0.95*q if not d else r for r, q, d in zip(reward, tnet_q, done)]
        
        # optimization
        indices=[[i,action[i]] for i in range(batch_size)]
        feed_dict={self.x:state, self.target:qnet_update_q, self.selected_idx:indices}
        self.session.run(self.train_opt, feed_dict)
        
    def update(self):
        """ for updatte target network"""
        self.session.run(self.update_opt)
        
    def set_session(self, sess):
        self.session = sess
        
    def get_action(self, state, e_rate):
        """ for training stage of the Agent, exploitation or exploration"""
        if np.random.random()<e_rate:
            return np.random.choice(self.out_units)
        else:
            return np.argmax(self.session.run(self.q, feed_dict={self.x: state}))