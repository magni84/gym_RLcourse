from gymnasium.envs.registration import register

register(
    id='MultiarmedBandits-v0',
    entry_point='gym_RLcourse.envs:MultiarmedBanditsEnv',
)

register(
    id='GridWorld-v0',
    entry_point='gym_RLcourse.envs:GridWorldEnv',
    max_episode_steps=100
)

register(
    id='GridWorld-AB-v0', 
    entry_point="gym_RLcourse.envs:GridWorldABEnv",
    kwargs={"map_name": "Example 3.5"},   
    max_episode_steps=100
)

register(
    id='GridWorld-Windy-v0', 
    entry_point="gym_RLcourse.envs:GridWorldWindyEnv",
    kwargs={"map_name": "Example 6.5"},   
    max_episode_steps=10000
)
register(
    id='GridWorld-WindyKing-v0', 
    entry_point="gym_RLcourse.envs:GridWorldWindyEnv",
    kwargs={"map_name": "Example 6.5", "nA": 8},   
    max_episode_steps=10000
)
register(
    id='GridWorld-Maze-v0', 
    entry_point="gym_RLcourse.envs:GridWorldMazeEnv",
    kwargs={"map_name": "Figure 8.2"},   
    max_episode_steps=10000
)

register(id='Corridor-v0',
         entry_point='gym_RLcourse.envs:CorridorEnv',
)

register(id='ShortCorridor-v0',
         entry_point='gym_RLcourse.envs:GridWorldCorridorEnv',
         max_episode_steps=1000
)

register(id='DynaMaze-v0',
         entry_point='gym_RLcourse.envs:DynaMazeEnv',
         kwargs={'maze_type' : 'DM'},
)
