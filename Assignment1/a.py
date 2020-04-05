import numpy as np


def minimax(player, game, time_left, depth, my_turn=True):
    moves = game.get_player_moves(game.get_active_player())
    vals = [player.utility(game.forecast_move(move)[0]) for move in moves]

    if depth == 1:
        idx = np.argmax(vals) if my_turn else np.argmin(vals)
        move, val = moves[idx], vals[idx]
        return move, val

    best_val = float('-inf') if my_turn else float('inf')
    best_move = (-1, -1)
    for move in moves:
        new_game, over, winner = game.forecast_move(move)
        if over:
            return move, player.utility(new_game)
        temp, val = minimax(player, new_game, time_left, depth - 1, not my_turn)
        if my_turn and val > best_val:
            best_val = val
            best_move = move
        elif not my_turn and val < best_val:
            best_val = val
            best_move = move
        if time_left() < 10:
            break

    return best_move, best_val


def alpha_beta(player, game, time_left, depth, alpha, beta, my_turn=True):
    # alpha-beta pruning search
    def helper(depth, alpha, beta):
        best_score = float('-inf') if my_turn else float('inf')
        best_move = None
        for move in game.get_active_moves():
            new_game, over, winner = game.forecast_move(move)
            if over and my_turn:
                return move, player.utility(new_game)
            elif my_turn:
                val = min_value(new_game, depth - 1, alpha, beta)
                if val > best_score:
                    best_score = val
                    best_move = move
                if best_score >= beta:
                    break
                alpha = max(alpha, best_score)
            else:
                val = max_value(new_game, depth - 1, alpha, beta)
                if val < best_score:
                    best_score = val
                    best_move = move
                if best_score <= alpha:
                    break
                beta = min(beta, best_score)
                if time_left() < 5:
                    break
        return best_move, best_score

    def terminal_state(depth):
        if depth == 0:
            return True
        return False

    def min_value(game, depth, alpha, beta):
        if terminal_state(depth):
            return player.utility(game)
        score = float("inf")
        for move in game.get_active_moves():
            new_game, over, winner = game.forecast_move(move)
            if over:
                return player.utility(new_game)
            val = max_value(new_game, depth - 1, alpha, beta)
            score = min(score, val)
            if score <= alpha:
                break
            beta = min(beta, score)
        return score

    def max_value(game, depth, alpha, beta):
        if terminal_state(depth):
            return player.utility(game)
        score = float("-inf")
        for move in game.get_active_moves():
            new_game, over, winner = game.forecast_move(move)
            if over:
                return player.utility(new_game)
            val = min_value(new_game, depth - 1, alpha, beta)
            score = max(score, val)
            if score >= beta:
                break
            alpha = max(alpha, score)
        return score

    # iterative deepen search
    for d in range(3, depth + 1):
        res = helper(d, alpha, beta)
        if time_left() < 900:
            break
    return res


def alpha_beta0(player, game, time_left, depth, alpha, beta, my_turn=True):
    def terminal_state(depth):
        if depth == 0:
            return True
        return False

    def min_value(game, depth, alpha, beta):
        if terminal_state(depth):
            return player.utility(game)
        score = float("inf")
        for move in game.get_active_moves():
            new_game, over, winner = game.forecast_move(move)
            if over:
                return player.utility(new_game)
            val = max_value(new_game, depth - 1, alpha, beta)
            score = min(score, val)
            if score <= alpha:
                break
            beta = min(beta, score)
        return score

    def max_value(game, depth, alpha, beta):
        if terminal_state(depth):
            return player.utility(game)
        score = float("-inf")
        for move in game.get_active_moves():
            new_game, over, winner = game.forecast_move(move)
            if over:
                return player.utility(new_game)
            val = min_value(new_game, depth - 1, alpha, beta)
            score = max(score, val)
            if score >= beta:
                break
            alpha = max(alpha, score)
        return score

    def helper(depth, alpha, beta):
        best_score = float('-inf') if my_turn else float('inf')
        best_move = None
        for move in game.get_active_moves():
            new_game, over, winner = game.forecast_move(move)
            if over and my_turn:
                return move, player.utility(new_game)
            elif my_turn:
                val = min_value(new_game, depth - 1, alpha, beta)
                if val > best_score:
                    best_score = val
                    best_move = move
                if best_score >= beta:
                    break
                alpha = max(alpha, best_score)
            else:
                val = max_value(new_game, depth - 1, alpha, beta)
                if val < best_score:
                    best_score = val
                    best_move = move
                if best_score <= alpha:
                    break
                beta = min(beta, best_score)
                if time_left() < 5:
                    break
        return best_move, best_score

    for d in range(2, depth + 1):
        res = helper(d, alpha, beta)
        if time_left() < 900:
            break
    return res
