from agents.agent import Agent
from store import register_agent
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
    
    def step(self, chess_board, player, opponent):
        self.max_depth = 3  #we can increase this to improve performance, i'm thinking of doing alpha beta to reduce computations
        self.corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
        self.cornerVal = 100
        _, best_move = self.minimax(chess_board, player, opponent, depth=self.max_depth, isMax=True)
        return best_move
        #return self.greedyFlips(chess_board, player, opponent)
        #todo: not sure if we should try alpha-beta, minimax is working just fine as is
        #todo: try monte-carlo, upper confidence ?
        

    def greedyFlips(self, chess_board, player, opponent):
        max_flips = 0
        start_time = time.time()
        best_move = None
        legal_player = get_valid_moves(chess_board, player)
        print(legal_player)
        for move in legal_player:
            #as per suggestions in the helper function, i'm now cloning the board before making moves
            new_board = deepcopy(chess_board)
            execute_move(new_board, move, player)
            noFlips = count_capture(chess_board, move, player)
            print(f"Move:  {move}, flips: { noFlips}")
            if noFlips > max_flips:
                max_flips = noFlips
                best_move = move
        print("found the best move!! ", best_move)
        time_taken = time.time() - start_time
        print("My greedy AI's turn took ", time_taken, "seconds.")
        return best_move
    
    def minimax(self, chess_board, player, opponent, depth, isMax):
        best_move=None
        is_endgame, player_score, opponent_score = check_endgame(chess_board, player, opponent)
        if depth == 0 or is_endgame:
            score = player_score - opponent_score 
            return ( self.eval(chess_board, player, opponent, score)), best_move ####player_score, opponent_score
        if isMax:  #maximising player
            start_time = time.time()
            value = float('-inf')
            best_move = None
            legal_player_moves = get_valid_moves(chess_board, player)
            print(legal_player_moves)
            for move in legal_player_moves:
                new_board = deepcopy(chess_board)
                execute_move(new_board, move, player)
                move_value = self.minimax(new_board, player, opponent, depth - 1, False)
                print("okay move value you get in player is this:  ", move_value)
                if move_value[0] > value:
                    value = move_value[0]
                    best_move = move
            print("found the best move!! ", best_move)
            time_taken = time.time() - start_time
            print("My minimax AI's turn took ", time_taken, "seconds.")
            return value, best_move 
        else:  # minimising opponsent 
            start_time = time.time()
            value = float('inf')
            best_move = None
            legal_opponent_moves = get_valid_moves(chess_board, opponent)
            print(legal_opponent_moves)
            for move in legal_opponent_moves:
                new_board = deepcopy(chess_board)
                execute_move(new_board, move, opponent)
                move_value = self.minimax(new_board, player, opponent, depth - 1, True)
                print("okay move value you get in opponent is this:  ", move_value)
                if move_value[0] < value:
                    value = move_value[0]
                    best_move = move
            print("found the best move!! ", best_move)
            time_taken = time.time() - start_time
            print("My minimax AI's turn took ", time_taken, "seconds.")
            return value, best_move 
        
    def eval(self, chess_board, player, opponent, score):
        for corner in self.corners:
            if chess_board[corner[0]][corner[1]]== player:
                print("player found a corner")
                score += self.cornerVal
            elif chess_board[corner[0]][corner[1]] == opponent:
                    print("opponent found a corner")
                    score -= self.cornerVal
        return score


