from gym.envs.registration import register
from .envs.connect_four_env import ConnectFourEnv

register(
    id='Connect4',
    entry_point='connect_4_gym.envs:ConnectFourEnv',
)