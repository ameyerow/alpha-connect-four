from tkinter import X
from turtle import width
import pygame
import numpy

from enum import Enum


class Colors(Enum):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)

class BoardView():
    def __init__(self, model, screen_size=512):
        self.model = model
        self.screen_size = screen_size

    def update_model(self, model):
        self.model = model

    def draw(self, screen):
        board = pygame.Surface((self.screen_size, self.screen_size))
        board.fill(Colors.WHITE.value)

        longer_board_axis = max(self.model.shape)
        grid_square_size = self.screen_size // (longer_board_axis)

        if self.model.shape[0] > self.model.shape[1]:
            x_offset = (self.screen_size - (self.model.shape[1] * grid_square_size))//2
            y_offset = 0
        else:
            y_offset = (self.screen_size - (self.model.shape[0] * grid_square_size))//2
            x_offset = 0

        for y, row in enumerate(self.model):
            for x, grid_square in enumerate(row):
                print(grid_square)
                filled_colors = {
                    -1: Colors.RED.value,
                    1 : Colors.BLUE.value,
                    0 : Colors.WHITE.value  
                }
                filled_color = filled_colors[grid_square]
                frame_color = Colors.YELLOW.value

                pygame.draw.rect(board, frame_color, 
                    (grid_square_size*y+y_offset, grid_square_size*x+x_offset, grid_square_size, grid_square_size))
                
                grid_square_center = (grid_square_size*(y+.5)+y_offset, grid_square_size*(x+.5)+x_offset)
                pygame.draw.circle(board, filled_color, grid_square_center, 0.4*grid_square_size)
                pygame.draw.circle(board, Colors.BLACK.value, grid_square_center, 0.4*grid_square_size, width=1)

        pygame.draw.rect(board, Colors.BLACK.value, 
            [y_offset, x_offset, self.model.shape[0]*grid_square_size, self.model.shape[1]*grid_square_size], 1)
        screen.blit(board, (0,0))