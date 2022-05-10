from typing import Any, Tuple
import numpy as np
from enum import Enum
from time import sleep
from overrides import overrides
from numpy import ndarray
from adversarial_env import AdversarialEnv, State
import torch


class ConnectFourEnv(AdversarialEnv):
    metadata = {'render_mode': 'human', 'render_fps':5}

    WIN_REWARD = 1
    DRAW_REWARD = 0
    LOSS_REWARD = -1

    def __init__(self, board_shape=(6, 7), win_req=4, screen_size=512):
        self.win_req = win_req
        self.board_shape = board_shape

        self.screen = None
        self.clock = None
        self.screen_size = screen_size

        state = State(board=np.zeros(self.board_shape, dtype=int), current_player=1)
        super(ConnectFourEnv, self).__init__(state, (board_shape[1]), board_shape)
    
    @overrides
    def observation(self, state: State=None) -> ndarray:
        if state is None:
            state = self.state
        return state.board * self.state.current_player

    @overrides
    def run(self, model, adversarial_model, state: State=None, render=False) -> Tuple[Any, float]:
        def render_if_enabled(state_to_render):
            if render:
                sleep(1)
                self.render(state=state_to_render)
        
        if state is None:
            state = self.state
        
        # Check that the state is not already terminal, if it is return the current player
        # and the correct reward
        winning_player =  self.__check_victory(state)
        if winning_player is not None:
            modifier = 1 if winning_player == state.current_player else -1
            reward = self.WIN_REWARD * modifier
            return state.current_player, reward
        elif len(self.__get_allowed_moves(state)) == 0:
            reward = self.DRAW_REWARD
            return state.current_player, reward

        render_if_enabled(state)
        players = [None, model, adversarial_model]
        while True:
            if render:
                print(state.board)
            # TODO: Consider what arguments a player needs to play the game - may need to move 'move' and 'check_winner' functions from here to the players
            current_player = players[state.current_player]
            action_probs, _ = current_player.forward(self.observation(state=state))
            action_probs = action_probs.detach().numpy()[0] # needed to index zero to remove extra dimension - may require later change
            # Balance probabilities based on some actions being illegal
            for action in range(len(action_probs)):
                if not self.is_legal_action(action, state=state):
                    action_probs[action] = 0
            prob_sum = np.sum(action_probs)
            if prob_sum == 0:
                print("prob sum should probably not equal 0")
                action_probs = np.ones(self.action_space_shape) / len(action_probs) # TODO: maybe change divisor
            else:
                action_probs /= np.sum(action_probs)
            action_probs /= np.sum(action_probs) # for some reason another normalization is required?
            action = np.random.choice(np.arange(self.board_shape[1]), p=action_probs)
            reward, done = self.step(action, state=state)
            render_if_enabled(state)
            if done:
                break

        return self.state.current_player, reward


    # takes a step with a given action. Returns the new board and the result, which is win, loss, draw, or None if the
    # game has not ended
    @overrides
    def step(self, action, state: State=None) -> Tuple[float, bool]:
        if state is None:
            state = self.state

        self.__insert_into_column(action, state)
        winning_player = self.__check_victory(state)

        if winning_player is not None:
            modifier = 1 if winning_player == state.current_player else -1
            reward = self.WIN_REWARD * modifier
        elif len(self.__get_allowed_moves(state)) == 0:
            reward = self.DRAW_REWARD
        else:
            state.current_player *= -1
            reward = None

        done = reward is not None
        return reward, done

    @overrides
    def perform_action_on_state(self, state: State, action) -> State:
        board = state.board.copy()
        for i in reversed(range(board.shape[0])):
            if board[i][action] == 0:
                board[i][action] = state.current_player
                return State(board=board, current_player=-state.current_player)
    
    @overrides
    def is_legal_action(self, action, state: State=None) -> bool:
        if state == None:
            state = self.state
        actions = np.nonzero(state.board[0] == 0)[0]
        return action in actions

    @overrides
    def reset(self):
        self.state = State(board=np.zeros(self.board_shape, dtype=int), current_player=1)

        self.screen = None
        self.clock = None

    @overrides
    def render(self, state: State=None):
        import pygame 

        class Colors(Enum):
            WHITE = (255, 255, 255)
            BLACK = (0, 0, 0)
            RED = (255, 0, 0)
            BLUE = (0, 0, 255)
            YELLOW = (255, 255, 0)

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surface = pygame.Surface((self.screen_size, self.screen_size))
        surface.fill(Colors.WHITE.value)

        if state is None:
            board = self.state.board
        else:
            board = state.board

        longer_board_axis = max(board.shape)
        grid_square_size = self.screen_size // (longer_board_axis)

        if board.shape[1] > board.shape[0]:
            x_offset = (self.screen_size - (board.shape[0] * grid_square_size))//2
            y_offset = 0
        else:
            y_offset = (self.screen_size - (board.shape[1] * grid_square_size))//2
            x_offset = 0

        for x, row in enumerate(board):
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
            [y_offset, x_offset, board.shape[1]*grid_square_size, board.shape[0]*grid_square_size], 1)
    
        self.screen.blit(surface, (0,0))
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    # returns winning player where player is as represented in the __board. Player is None if
    # there is no winner
    def __check_victory(self, state: State):
        # checks rows and columns
        for board in [state.board, np.transpose(state.board)]:
            for i in range(board.shape[0]):
                result = self.__check_row(board[i])
                if result is not None:
                    return result
        # checks diagonals
        for board in [state.board, np.rot90(state.board)]:
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
    def __insert_into_column(self, column: int, state: State):
        assert(state.board[0][column] == 0)
        for i in reversed(range(self.board_shape[0])):
            if state.board[i][column] == 0:
                state.board[i][column] = state.current_player
                return

    def __get_allowed_moves(self, state: State):
        moves = np.nonzero(state.board[0] == 0)
        return moves[0]

if __name__=="__main__":
    class MockModel:
        def forward(self, board: np.array):
            return np.ones([board.shape[1]])/board.shape[1], 0.0001

    env = ConnectFourEnv(board_shape=(1,4), win_req=2)
    num_games_to_play = 10
    for _ in range(num_games_to_play):
        env.run(MockModel(), MockModel(), render=True)
        env.reset()

