from gym.envs.registration import register
from .envs.connect_four_env import ConnectFourEnv, Player, RandomPlayer, SavedPlayer, ResultType

register(
    id='Connect4',
    entry_point='connect_4_gym.envs:Connect4Env',
)