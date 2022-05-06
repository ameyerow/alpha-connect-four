class Node:
    def __init__(self, prior, current_player):
        self.visitation_count = 0
        self.current_player = current_player
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

class MCTS():
    def __init__(self):
        pass

    def run(self) -> Node:
        pass