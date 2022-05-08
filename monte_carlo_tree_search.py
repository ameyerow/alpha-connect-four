# TODO: implement pi, which should generate the sample action probabilities - it should take in a board and player
# TODO: implement predict_move, which should select a move given board and player (and possibly env to get legal moves)
# predict_move and pi will likely be very similar / identical in logic

class Node:
    def __init__(self, prior, current_player):
        self.visitation_count = 0
        self.current_player = current_player
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

class MCTS:
    def __init__(self):
        pass

    def run(self) -> Node:
        pass