# gym-RLcourse
Environments used in the RL-course at Uppsala University. Implementing some examples from "Reinforcement Learning - An introduction" by Sutton and Barto, and some other very simple environments for educational purposes. 

# Installation
Requirements: gymnasium and numpy

```
git clone https://github.com/magni84/gym_RLcourse.git
cd gym_RLcourse
pip install .
```
# Attribution 
The code for `GridWorldEnv` is based on the [FrozenLake](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/frozen_lake.py) environment licensed under an MIT license.  

# Environments

* `MultiArmedBandits-v0` - Implements the multi-armed bandits of Section 2.3 in Sutton and Barto.
* `GridWorld-v0` - Implements Example 4.1 in Sutton and Barto.
* `GridWorld-AB-v0` - Implements Example 3.5 in Sutton and Barto.
* `GridWorld-Windy-v0` - Implements Example 6.5 in Sutton and Barto.
* `GridWorld-WindyKing-v0` - Same as `GridWorld-Windy-v0` but with king moves.
* `GridWorld-Maze-v0` - Implement mazes from Chapter 8 in Sutton and Barto. 
