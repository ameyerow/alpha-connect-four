import gym
from gym import spaces
import numpy as np
from time import sleep
from enum import Enum

class Colors(Enum):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)

class ConnectFourEnv(gym.Env):
    metadata = {'render.modes': ['human'], 'render_fps':5}

    WIN_REWARD = 1
    DRAW_REWARD = 0
    LOSS_REWARD = -1

    def __init__(self, board_shape=(6, 7), win_req=4, screen_size=512):

        super(ConnectFourEnv, self).__init__()
        self.win_req = win_req
        self.board_shape = board_shape
        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=board_shape,
                                            dtype=int)
        self.action_space = spaces.Discrete(board_shape[1])

        self.current_player = 1
        self.board = np.zeros(self.board_shape, dtype=int)

        self.player_color = 1
        self.screen = None
        self.clock = None
        self.screen_size = screen_size

    def run(self):
        # make reset function
        while True:
            action = np.random.choice(self.__get_allowed_moves())
            reward = self.step(action)
            if reward is None:
                self.current_player *= -1
            else:
                break
        return self.current_player, reward

    def run_game(self, player1, player2):
        self.reset()
        players = [None, player1, player2]
        while True:
            # TODO: Consider what arguments a player needs to play the game - may need to move 'move' and 'check_winner' functions from here to the players
            action = np.random.choice(players[self.current_player](self.board, self.current_player))
            reward = self.step(action)
            if reward is None:
                self.current_player *= -1
            else:
                break
        return self.current_player, reward


    # takes a step with a given action. Returns the new board and the result, which is win, loss, draw, or None if the
    # game has not ended
    def step(self, action):
        self.__insert_into_column(action)
        winning_player = self.__check_victory()
        if winning_player is not None:
            if winning_player == self.current_player:
                reward = self.WIN_REWARD
            else:
                reward = self.LOSS_REWARD
        else:
            if len(self.__get_allowed_moves()) == 0:
                reward = self.DRAW_REWARD
            else:
                self.current_player *= -1
                reward = None
        done = reward is not None
        info = None

        return self.board.copy(), reward, done, info

    # resets the board
    def reset(self):
        self.current_player = 1
        self.board = np.zeros(self.board_shape, dtype=int)

        self.screen = None
        self.clock = None

    # returns winning player where player is as represented in the __board. Player is None if
    # there is no winner
    def __check_victory(self):
        # checks rows and columns
        for board in [self.board, np.transpose(self.board)]:
            for i in range(board.shape[0]):
                result = self.__check_row(board[i])
                if result is not None:
                    return result

        # checks diagonals
        for board in [self.board, np.rot90(self.board)]:
            for k in range(-board.shape[0] + 1, board.shape[1]):
                diagonal = np.diag(board, k=k)
                if len(diagonal >= self.win_req):
                    result = self.__check_row(diagonal)
                    if result is not None:
                        return result

    # checks for consecutive pieces from the same player where 0 is empty
    def __check_row(self, row):
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
    def __insert_into_column(self, column):
        assert(self.board[0][column] == 0)
        for i in reversed(range(self.board_shape[0])):
            if self.board[i][column] == 0:
                self.board[i][column] = self.current_player
                return

    # Immutably inserts a piece into a board, i.e. returns a new board
    def move(self, board, player, column):
        board = board.copy()
        for i in reversed(range(board.shape[0])):
            if board[i][column] == 0:
                board[i][column] = player
                return board

    def __get_allowed_moves(self):
        return np.nonzero(self.board[0] == 0)
    
    def get_allowed_moves(self):
        return np.nonzero(self.board[0] == 0)

    def render(self, mode="human", close=False):
        if mode == 'console':
            self.__render_text()
        elif mode == 'human':
            self.__render_gui()

    def __render_text(self):
        replacements = {
            self.player_color: 'A',
            0: ' ',
            -1 * self.player_color: 'B'
        }

        def render_line(line):
            return "|" + "|".join(
                ["{:>2} ".format(replacements[x]) for x in line]) + "|"

        hline = '|---+---+---+---+---+---+---|'
        print(hline)
        for line in np.apply_along_axis(render_line,
                                        axis=1,
                                        arr=self.board):
            print(line)
        print(hline)
    
    def __render_gui(self):
        import pygame 

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surface = pygame.Surface((self.screen_size, self.screen_size))
        surface.fill(Colors.WHITE.value)

        longer_board_axis = max(self.board.shape)
        grid_square_size = self.screen_size // (longer_board_axis)

        if self.board.shape[1] > self.board.shape[0]:
            x_offset = (self.screen_size - (self.board.shape[0] * grid_square_size))//2
            y_offset = 0
        else:
            y_offset = (self.screen_size - (self.board.shape[1] * grid_square_size))//2
            x_offset = 0

        for x, row in enumerate(self.board):
            for y, grid_square in enumerate(row):
                filled_colors = {
                    -1: Colors.RED.value,
                    1 : Colors.BLUE.value,
                    0 : Colors.WHITE.value  
                }
                filled_color = filled_colors[grid_square]
                frame_color = Colors.YELLOW.value

                pygame.draw.rect(surface, frame_color, 
                    (grid_square_size*y+y_offset, grid_square_size*x+x_offset, grid_square_size, grid_square_size))
                
                grid_square_center = (grid_square_size*(y+.5)+y_offset, grid_square_size*(x+.5)+x_offset)
                pygame.draw.circle(surface, filled_color, grid_square_center, 0.4*grid_square_size)
                pygame.draw.circle(surface, Colors.BLACK.value, grid_square_center, 0.4*grid_square_size, width=1)

        pygame.draw.rect(surface, Colors.BLACK.value, 
            [y_offset, x_offset, self.board.shape[1]*grid_square_size, self.board.shape[0]*grid_square_size], 1)
    
        self.screen.blit(surface, (0,0))
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def reset_env(self):
        self.board_shape = np.zeros(self.board_shape, dtype=int)
        self.current_player = 1

if __name__=="__main__":
    env = ConnectFourEnv(board_shape=(1,4), win_req=2)
    env.render()
    while True:
        allowed_moves = env.get_allowed_moves()[0]
        action = np.random.choice(allowed_moves)
        _, _, done, _ = env.step(action)
        env.render()
        sleep(1)
        
        if done:
            env.reset()

