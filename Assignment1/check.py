import Assignment1.player_submission_tests as tests
from Assignment1.test_players import RandomPlayer
from Assignment1.a import alpha_beta, minimax, alpha_beta0


class OpenMoveEvalFn:
    def score(self, game, player):
        return len(game.get_player_moves(player)) - len(game.get_opponent_moves(player))


class CustomEvalFn:
    def score(self, game, player):
        return len(game.get_player_moves(player)) - 2*len(game.get_opponent_moves(player))


def score1(game, player):
    return len(game.get_player_moves(player)) - len(game.get_opponent_moves(player))


def score2(game, player):
    return len(game.get_player_moves(player)) - 3*len(game.get_opponent_moves(player))


class CustomPlayer:
    def __init__(self, search_depth=4, alpha=float('-inf'), beta=float('inf')):
        self.search_depth = search_depth
        self.alpha = alpha
        self.beta = beta

    def move(self, game, time_left):
        # best_move, utility = minimax(self, game, time_left, self.search_depth, True)
        best_move, utility = alpha_beta(self, game, time_left, self.search_depth, self.alpha, self.beta, my_turn=True)
        return best_move

    def utility(self, game, player=None):
        """You can handle special cases here (e.g. endgame)"""
        return score2(game, self)

class CustomPlayer0:
    def __init__(self, search_depth=4, alpha=float('-inf'), beta=float('inf')):
        self.search_depth = search_depth
        self.alpha = alpha
        self.beta = beta

    def move(self, game, time_left):
        # best_move, utility = minimax(self, game, time_left, self.search_depth, True)
        # best_move, utility = alpha_beta(self, game, time_left, self.search_depth, self.alpha, self.beta, my_turn=True)
        best_move, utility = alpha_beta0(self, game, time_left, self.search_depth, self.alpha, self.beta, my_turn=True)
        return best_move

    def utility(self, game, player=None):
        """You can handle special cases here (e.g. endgame)"""
        return score1(game, self)



if __name__ == '__main__':
    win = 0
    for i in range(1):
        p1 = CustomPlayer0(10)
        p2 = CustomPlayer(10)
        winner = tests.beatRandom(p1, p2)
        if winner == 'CustomPlayer - Q2':
            win += 1

    # tests.minimaxTest(CustomPlayer, alpha_beta)
