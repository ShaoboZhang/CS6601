#!/usr/bin/env python
from Assignment1.isolation import Board
from Assignment1.test_players import RandomPlayer
from Assignment1.check import OpenMoveEvalFn, CustomEvalFn


def beatRandom(p1, p2):
    try:
        game = Board(p1, p2, 7, 7)
        winner, move_history, termination = game.play_isolation(time_limit=1000, print_moves=True)
        print("\n", winner, " has won. Reason: ", termination)
        return winner
    except NotImplementedError:
        print('CustomPlayer Test: Not Implemented')


def minimaxTest(yourAgent, minimax_fn):
    """Example test to make sure
    your minimax works, using the
    OpenMoveEvalFunction evaluation function.
    This can be used for debugging your code
    with different model Board states.
    Especially important to check alpha_beta
    pruning"""

    # create dummy 5x5 board
    print("Now running the Minimax test.")
    print()
    try:
        def time_left():  # For these testing purposes, let's ignore timeouts
            return 1000

        player = yourAgent()  # using as a dummy player to create a board
        sample_board = Board(player, RandomPlayer())
        # setting up the board as though we've been playing
        board_state = [
            [" ", "X", "X", " ", "X", "X", " "],
            [" ", " ", "X", " ", " ", "X", " "],
            ["X", " ", " ", " ", " ", "Q1", " "],
            [" ", "X", "X", "Q2", "X", " ", " "],
            ["X", " ", "X", " ", " ", " ", " "],
            [" ", " ", "X", " ", "X", " ", " "],
            ["X", " ", "X", " ", " ", " ", " "]
        ]
        sample_board.set_state(board_state, True)

        test_pass = True

        expected_depth_scores = [(1, -2), (2, 1), (3, 4), (4, 3), (5, 5)]

        for depth, exp_score in expected_depth_scores:
            move, score = minimax_fn(player, sample_board, time_left, depth, player.alpha, player.beta, my_turn=True)
            print(exp_score, score)
            if exp_score != score:
                print("Minimax failed for depth: ", depth)
                test_pass = False
            else:
                print("Minimax passed for depth: ", depth)

        test_pass = True
        if test_pass:
            player = yourAgent()
            sample_board = Board(RandomPlayer(), player)
            # setting up the board as though we've been playing
            board_state = [
                [" ", " ", " ", " ", "X", " ", "X"],
                ["X", "X", "X", " ", "X", "Q2", " "],
                [" ", "X", "X", " ", "X", " ", " "],
                ["X", " ", "X", " ", "X", "X", " "],
                ["X", " ", "Q1", " ", "X", " ", "X"],
                [" ", " ", " ", " ", "X", "X", " "],
                ["X", " ", " ", " ", " ", " ", " "]
            ]
            sample_board.set_state(board_state, p1_turn=True)

            test_pass = True

            expected_depth_scores = [(1, -7), (2, -7), (3, -7), (4, -9), (5, -8)]

            for depth, exp_score in expected_depth_scores:
                move, score = minimax_fn(player, sample_board, time_left, depth, player.alpha, player.beta, my_turn=False)
                print(exp_score, score)
                if exp_score != score:
                    print("Minimax failed for depth: ", depth)
                    test_pass = False
                else:
                    print("Minimax passed for depth: ", depth)

        if test_pass:
            print("Minimax Test: Runs Successfully!")

        else:
            print("Minimax Test: Failed")

    except NotImplementedError:
        print('Minimax Test: Not implemented')
