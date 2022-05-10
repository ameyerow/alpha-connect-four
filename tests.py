import unittest
import numpy as np
import torch
from connect_four_env import ConnectFourEnv
from monte_carlo_tree_search import Node, MCTS, UCB1


class MCTSTests(unittest.TestCase):

    def test_mcts_from_root_with_equal_priors(self):
        class MockModel:
            def forward(self, board):
                # starting board is:
                # [0, 0, 1, -1]
                return torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=float), 0.0001

        env = ConnectFourEnv(board_shape=(1, 4), win_req=2)
        model = MockModel()
        mcts = MCTS(env, model, num_simulations=100)
        mcts.run()

        # the best move is to play at index 1 or 2
        self.assertIn(np.argmax(mcts.pi()), [1, 2])

if __name__ == '__main__':
    unittest.main()