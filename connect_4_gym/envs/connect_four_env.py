import gym
from gym import spaces
import numpy as np
from scipy.ndimage import rotate
from player import Player


class ConnectFourEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    WIN_REWARD = 1
    DRAW_REWARD = 0
    LOSS_REWARD = -1

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

    def run(self, player1: Player, player2, board, render=False):
        # make reset function
        finished = False

        while not finished:
            current_player = player1 if self.__current_player == 1 else player2
            action = player1.


    # takes a step with a given action. Returns the new board and the result, which is win, loss, draw, or None if the
    # game has not ended
    def step(self, action):
        self.insert_into_column(action)
        winning_player = self.check_victory()
        if winning_player is not None:
            if winning_player == self.__current_player:
                reward = self.WIN_REWARD
            else:
                reward = self.LOSS_REWARD
        else:
            if len(self.get_allowed_moves()) == 0:
                reward = self.DRAW_REWARD
            else:
                reward = None
        return self.__board.copy(), reward



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

    # assumes action is valid; TODO: add move rejection for human players
    def insert_into_column(self, column):
        assert(self.__board[0][column] == 0)
        for i in reversed(range(self.board_shape[0])):
            if self.__board[i][column] == 0:
                self.__board[i][column] = self.__current_player
                return

    def get_allowed_moves(self):
        return np.nonzero(self.__board[0] == 0)


