import gym
from gym import spaces
import numpy as np
from scipy.ndimage import rotate


class ConnectFourEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, board_shape=(6, 7), window_width=512, window_height=512):
        super(ConnectFourEnv, self).__init__()
        self.win_req = 4
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

    # returns winning player where player is as represented in the __board. Player is None if
    # there is no winner
    def check_victory(self):
        # checks rows and columns
        for board in [self.__board, np.transpose(self.__board)]:
            for i in range(board.shape[0]):
                result = self._check_row(board[i])
                if result is not None:
                    return result

        # checks diagonals
        for board in [self.__board, np.rot90(self.__board)]:
            for k in range(-board.shape[0] + 1, board.shape[1]):
                diagonal = np.diag(board, k=k)
                if len(diagonal >= self.win_req):
                    result = self._check_row(diagonal)
                    if result is not None:
                        return result

    # checks for consecutive pieces from the same player where 0 is empty
    def _check_row(self, row):
        player = 0
        consecutive_count = 0
        for i in range(len(row)):
            if row[i] == player:
                consecutive_count += 1
            else:
                player = row[i]
                consecutive_count = 1
            if player != 0 and consecutive_count == self.win_req:
                return player
