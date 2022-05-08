import gym
import numpy as np
import connect_4_gym
from time import sleep


if __name__ == "__main__":
    env = gym.make('Connect4', board_shape=(1,4), win_req=2)
    env.reset()
    env.render()
    while True:
        allowed_moves = env.get_allowed_moves()[0]
        action = np.random.choice(allowed_moves)
        _, _, done, _ = env.step(action)
        env.render()
        sleep(1)
        
        if done:
            env.reset()