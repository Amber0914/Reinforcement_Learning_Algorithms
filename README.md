
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
python frozenlake_unslippery.py ----max_episodes-2000 --max_actions=99 --discount=0.95 --exploration_rate=1.0
```

### Deep Q-Learning (Deep Q Network) 
```
python deep_frozenlake_unslippery.py ----max_episodes-2000 --max_actions=99 --discount=0.95 --exploration_rate=1.0 --hidden_units=10
```
