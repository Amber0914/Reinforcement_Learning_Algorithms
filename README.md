
# Reinforcement Learning Algorithms Tutorial

## Enviroment
```
macOS 10.13.6
python 3.5.6
```

## Requirements.txt
```
numpy==1.14.5
tensorflow==1.10.1
gym==0.12.1
```
#### Command for install
```
pip3 install -r requirements.txt
```

## Q-Learning
### Medium Tutorial
The complete tutoriol will be released as soon as possible.

### Problem Definition
We use [FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0/) without slippery and define it in class Environment

### Basic Q-Learning (with table method)
```
python frozenlake_unslippery.py ----max_episodes-2000 --max_actions=99 --discount=0.95 --exploration_rate=1.0 --hidden_units=10
```
```output
''' Output
====== # Start at the state S (0, 0)
'S'FFF
FHFH
FFFH
HFFG
====== # Take the Down action, the agent arrives from (0, 0) to (1, 0) 'F'
  (Down)
SFFF
'F'HFH
FFFH
HFFG
====== # Take the Down action, the agent arrives from (1, 0) to (2, 0) 'F'
  (Down)
SFFF
FHFH
'F'FFH
HFFG
====== # Take the Right action, the agent arrives from (2,0) to (2, 1) 'F'
  (Right)
SFFF
FHFH
F'F'FH
HFFG
====== # Take the Down action, the agent arrives from (2, 1) to (3, 1) 'F'
  (Down)
SFFF
FHFH
FFFH
H'F'FG
====== # Take the Right action, the agent arrives from (3, 1) to (3, 2) 'F'
  (Right)
SFFF
FHFH
FFFH
HF'F'G
====== # Take the Right action, the agent arrives the goal 'G'.
  (Right)
SFFF
FHFH
FFFH
HFF'G'
'''
```


### Deep Q-Learning (Deep Q Network) 
```
python deep_frozenlake_unslippery.py ----max_episodes-2000 --max_actions=99 --discount=0.95 --exploration_rate=1.0 --hidden_units=10
```
