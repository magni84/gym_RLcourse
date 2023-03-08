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
    entry_point="gym_RLcourse.envs:GridWorldEnv",
    kwargs={"map_name": "Example 3.5"},   
    max_episode_steps=100
)
