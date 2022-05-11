import abc
import sys
from typing import Dict, List, Tuple
import numpy as np
from numpy import ndarray
from enum import Enum
from overrides import overrides
import pygame
import torch
import random
from time import sleep

from adversarial_env import AdversarialEnv
from big_connect_four_model import BigConnectFourModel
from connect_four_env import ConnectFourEnv
from connect_two_model import ConnectTwoModel


SCREEN_SIZE = 512


class ControlType(Enum):
    Player = 0
    Computer = 1
    Restart = 2

class Controller(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def handle_events(self, env: AdversarialEnv) -> ControlType:
        """
        Handle the stream of events generated by the user, update the board accordingly
        """
        raise NotImplementedError

class ComputerController(Controller):
    def __init__(self, model: torch.nn.Module):
        self.model = model

    @overrides
    def handle_events(self, env: AdversarialEnv) -> ControlType:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                sys.exit()
            
        if env.is_terminal_state():
            return ControlType.Computer

        action_probs, _ = self.model.forward(env.observation())
        action_probs = action_probs.squeeze().detach().cpu().numpy()
        def zero_out_impossible_moves(action, action_prob):
            return action_prob if env.is_legal_action(action) else 0
        action_probs = np.array([zero_out_impossible_moves(action, action_prob) 
            for action, action_prob in enumerate(action_probs)])
        action_probs /= np.sum(action_probs)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        sleep(1)
        env.step(action)

        if env.is_terminal_state():
            return ControlType.Restart
        else:
            return ControlType.Player

class PlayerController(Controller):

    def handle_events(self, env: AdversarialEnv) -> ControlType:
        new_control_type: ControlType = None
        mouse_pos = pygame.mouse.get_pos()
        pygame.mouse.get_pressed()
        print(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                sys.exit()

        if pygame.mouse.get_pressed()[0]:
            print("In Player's handle_events", mouse_pos) 
            action = self.get_action_from_click(mouse_pos, env)
            if action is not None:
                env.step(action)
                if env.is_terminal_state():
                    new_control_type = ControlType.Restart
                else:
                    new_control_type = ControlType.Computer

        if new_control_type is None:
            return ControlType.Player
        else:
            return new_control_type
    
    def get_action_from_click(self, mouse_pos: Tuple[float, float], env: AdversarialEnv) -> int:
        board = env.state.board
        longer_board_axis = max(board.shape)
        grid_square_size = SCREEN_SIZE // (longer_board_axis)
        if board.shape[1] > board.shape[0]:
            x_offset = (SCREEN_SIZE - (board.shape[0] * grid_square_size))//2
            y_offset = 0
        else:
            y_offset = (SCREEN_SIZE - (board.shape[1] * grid_square_size))//2
            x_offset = 0
        move_rectangles: List[pygame.Rect] = []
        for y, _ in enumerate(board[0]):
            rect = pygame.Rect(grid_square_size*y+y_offset, grid_square_size*0+x_offset, grid_square_size, grid_square_size)
            move_rectangles.append(rect)
        for action, rectangle in enumerate(move_rectangles):
            if rectangle.collidepoint(mouse_pos) and env.is_legal_action(action):
                return action

class RestartController(Controller):
    def handle_events(self, env: AdversarialEnv) -> ControlType:
        sleep(1)

        env.reset()

        starting_control_types = [ControlType.Player, ControlType.Computer]
        random.shuffle(starting_control_types)
        new_control_type = starting_control_types[0]

        return new_control_type

def render(env: AdversarialEnv, control_type: ControlType, screen):
    class Colors(Enum):
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        RED = (255, 0, 0)
        BLUE = (0, 0, 255)
        YELLOW = (255, 255, 0)
        HINT = (181, 211, 231)

    board = env.state.board

    surface = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE))
    surface.fill(Colors.WHITE.value)

    longer_board_axis = max(board.shape)
    grid_square_size = SCREEN_SIZE // (longer_board_axis)

    if board.shape[1] > board.shape[0]:
        x_offset = (SCREEN_SIZE - (board.shape[0] * grid_square_size))//2
        y_offset = 0
    else:
        y_offset = (SCREEN_SIZE - (board.shape[1] * grid_square_size))//2
        x_offset = 0

    for x, row in enumerate(board):
        for y, grid_square in enumerate(row):
            filled_colors = {
                -1: Colors.RED.value,
                1 : Colors.BLUE.value,
                0 : Colors.HINT.value if x == 0 and control_type is ControlType.Player else Colors.WHITE.value 
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

    screen.blit(surface, (0,0))
    pygame.display.update()
    pygame.event.get()


def main():
    game_type = "Connect4"

    pygame.init()

    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))

    if game_type == "Connect2":
        env = ConnectFourEnv(board_shape=(1, 4), win_req=2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ConnectTwoModel()
        model.eval()
        checkpoint = torch.load("models/connect2_model", map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    elif game_type == "Connect4":
        env = ConnectFourEnv()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BigConnectFourModel()
        model.eval()
        checkpoint = torch.load("models/best_big_connect_four_model", map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

    starting_control_types = [ControlType.Player, ControlType.Computer]
    random.shuffle(starting_control_types)
    active_control_type = starting_control_types[0]
    render(env, active_control_type, screen)

    controllers: Dict[ControlType, Controller] = {}
    controllers[ControlType.Player] = PlayerController()
    controllers[ControlType.Computer] = ComputerController(model)
    controllers[ControlType.Restart] = RestartController()
    
    while True:
        controller: Controller = controllers[active_control_type]
        active_control_type = controller.handle_events(env)
        render(env, active_control_type, screen)

if __name__=="__main__":
    main()
