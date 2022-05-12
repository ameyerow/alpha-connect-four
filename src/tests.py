import torch
import unittest
import numpy as np

from .monte_carlo_tree_search import MCTS
from .env.adversarial_env import State
from .env.connect_four_env import ConnectFourEnv


class MCTSTests(unittest.TestCase):

    def test_mcts_from_root_with_equal_priors(self):
        class MockModel:
            def eval(self):
                pass
            def forward(self, board):
                return torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=float), 0.0001

        env = ConnectFourEnv(board_shape=(1, 4), win_req=2)
        model = MockModel()
        mcts = MCTS(env, model, num_simulations=100)
        mcts.run()

        # The best move is to play at index 1 or 2.
        self.assertIn(np.argmax(mcts.pi()), [1, 2])

    def test_mcts_from_second_move(self):
        class MockModel:
            def eval(self):
                pass
            def forward(self, board):
                return torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=float), 0.0001

        env = ConnectFourEnv(board_shape=(1, 4), win_req=2)
        env.state = State(np.array([[1, 0, 0, 0]]), -1)
        model = MockModel()
        mcts = MCTS(env, model, num_simulations=100)
        mcts.run()

        # The best move is to play at index 1.
        self.assertEqual(np.argmax(mcts.pi()), 1)
    
    def test_mcts_from_flipped_second_move(self):
        class MockModel:
            def eval(self):
                pass
            def forward(self, board):
                return torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=float), 0.0001

        env = ConnectFourEnv(board_shape=(1, 4), win_req=2)
        env.state = State(np.array([[0, 0, 0, 1]]), -1)
        model = MockModel()
        mcts = MCTS(env, model, num_simulations=100)
        mcts.run()

        # The best move is to play at index 2.
        self.assertEqual(np.argmax(mcts.pi()), 2)

    def test_mcts_winning_move(self):
        class MockModel:
            def eval(self):
                pass
            def forward(self, board):
                return torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=float), 0.0001

        env = ConnectFourEnv(board_shape=(1, 4), win_req=2)
        env.state = State(np.array([[-1, 0, 0, 1]]), 1)
        model = MockModel()
        mcts = MCTS(env, model, num_simulations=100)
        mcts.run()

        # The best move is to play at index 2.
        self.assertEqual(np.argmax(mcts.pi()), 2)

if __name__ == '__main__':
    unittest.main()