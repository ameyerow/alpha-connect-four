import gym
from gym import spaces
import numpy as np

class ConnectFourEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, board_shape=(6, 7), window_width=512, window_height=512):
        super(ConnectFourEnv, self).__init__()
        self.board_shape = board_shape
        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=board_shape,
                                            dtype=int)
        self.action_space = spaces.Discrete(board_shape[1])

        self.__current_player = 1
        self.__board = np.zeros(self.board_shape, dtype=int)

        self.__player_color = 1
        self.__screen = None
        self.__window_width = window_width
        self.__window_height = window_height
        self.__rendered_board = self._update_board_render()

