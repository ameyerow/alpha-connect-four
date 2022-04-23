import gym
from gym import spaces
import numpy as np
import pygame
from scipy.ndimage import rotate
from render import BoardView
from time import sleep


class ConnectFourEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    WIN_REWARD = 1
    DRAW_REWARD = 0
    LOSS_REWARD = -1

    def __init__(self, board_shape=(7, 6), window_width=512, window_height=512):

        super(ConnectFourEnv, self).__init__()
        self.win_req = 4
        self.board_shape = board_shape
        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=board_shape,
                                            dtype=int)
        self.action_space = spaces.Discrete(board_shape[1])

        self.current_player = 1
        self.__board = np.zeros(self.board_shape, dtype=int)
        # self.__board = np.random.uniform(-1,1,(7, 6)).round()

        self.__player_color = 1
        self.__screen = None
        self.__window_width = window_width
        self.__window_height = window_height

    def run(self):
        # make reset function
        while True:
            action = np.random.choice(self.get_allowed_moves())
            reward = self.step(action)
            if reward is None:
                self.current_player *= -1
            else:
                break
        return self.current_player, reward


    # takes a step with a given action. Returns the new board and the result, which is win, loss, draw, or None if the
    # game has not ended
    def step(self, action):
        self.insert_into_column(action)
        winning_player = self.check_victory()
        if winning_player is not None:
            if winning_player == self.current_player:
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
                self.__board[i][column] = self.current_player
                return

    def get_allowed_moves(self):
        return np.nonzero(self.__board[0] == 0)

    def render(self, mode="human", close=False):
        if mode == 'console':
            replacements = {
                self.__player_color: 'A',
                0: ' ',
                -1 * self.__player_color: 'B'
            }

            def render_line(line):
                return "|" + "|".join(
                    ["{:>2} ".format(replacements[x]) for x in line]) + "|"

            hline = '|---+---+---+---+---+---+---|'
            print(hline)
            for line in np.apply_along_axis(render_line,
                                            axis=1,
                                            arr=self.__board):
                print(line)
            print(hline)

        elif mode == 'human':
            if self.__screen is None:
                pygame.init()
                self.__screen = pygame.display.set_mode(
                    (round(self.__window_width), round(self.__window_height)))

            if close:
                pygame.quit()

            board_view = BoardView(self.__board, self.__window_width)
            board_view.draw(self.__screen)
            pygame.display.update()

if __name__=="__main__":
    env = ConnectFourEnv()
    env.render()
    while True:
        action = np.random.choice(env.get_allowed_moves()[0])
        reward = env.step(action)
        if reward is None:
            env.current_player *= -1
            env.render()
            sleep(1)
        else:
            env.render()
            sleep(1)

