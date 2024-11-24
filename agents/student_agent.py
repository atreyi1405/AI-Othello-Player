from agents.agent import Agent
from store import register_agent
import numpy as np
from copy import deepcopy
import time
import math
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

class Node:
    def __init__(self, chess_board, player=None, move=None, parent=None):
        self.chess_board=chess_board
        self.player=player
        self.parent=parent
        self.move=move
        self.children=[]
        self.visit=0
        self.wins=0
    
    def uct(self,c=1.0):
        if self.visit==0:
            return float('inf')
        exploitation=self.wins/self.visit
        exploration= c * np.sqrt((np.log(self.parent.visits))/self.visit)
        return exploitation + exploration
  

@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
    
    def step(self, chess_board, player, opponent):
        start=time.time()
        self.max_depth = 3  
        self.corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
        self.cornerVal = 100
        for depth in range(1, self.max_depth+1): 
            if time.time() - start >= 1.9:
                break
            _, move = self.alphabeta(chess_board, player, opponent, depth, isMax=True)
            best_move = move
        #return best_move
        #_, best_move = self.alphabeta(chess_board, player, opponent, depth=self.max_depth, isMax=True)
        return best_move
        #return self.greedyFlips(chess_board, player, opponent)
        #todo: make the heuristic stronger
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
            #print(legal_player_moves)
            for move in legal_player_moves:
                new_board = deepcopy(chess_board)
                execute_move(new_board, move, player)
                move_value = self.minimax(new_board, player, opponent, depth - 1, False)
                #noFlips_player = count_capture(chess_board, move, player)
                #print("okay move value you get in player is this:  ", move_value)
                #if move in self.corners:
                    #move_value[0]= move_value[0]+corner_bonus
                    #noFlips_player+=100
                if move_value[0] > value:
                    value = move_value[0]
                    best_move = move
            #print("found the best move!! ", best_move)
            time_taken = time.time() - start_time
            #print("My minimax AI's turn took ", time_taken, "seconds.")
            return value, best_move 
        else:  # minimising opponsent 
            start_time = time.time()
            value = float('inf')
            best_move = None
            legal_opponent_moves = get_valid_moves(chess_board, opponent)
            #print(legal_opponent_moves)
            for move in legal_opponent_moves:
                new_board = deepcopy(chess_board)
                execute_move(new_board, move, opponent)
                move_value = self.minimax(new_board, player, opponent, depth - 1, True)
                #noFlips_opponent = count_capture(chess_board, move, opponent)
                #print("okay move value you get in opponent is this:  ", move_value)
                #if move in self.corners:
                    #noFlips_opponent-=100
                if move_value[0] < value:
                    value = move_value[0]
                    best_move = move
            #print("found the best move!! ", best_move)
            time_taken = time.time() - start_time
            #print("My minimax AI's turn took ", time_taken, "seconds.")
            return value, best_move 

      
    def eval(self, chess_board, player, opponent, score):
        for corner in self.corners:
            if chess_board[corner[0]][corner[1]]== player:
                #print("player found a corner")
                score += self.cornerVal
            elif chess_board[corner[0]][corner[1]] == opponent:
                    #print("opponent found a corner")
                    score -= self.cornerVal
        return score

    def alphabeta(self, chess_board, player, opponent, depth, isMax, alpha=float('-inf'), beta=float('inf')):
        print("We're at depth: ", depth, "...")
        best_move=None
        is_endgame, player_score, opponent_score = check_endgame(chess_board, player, opponent)
        #call eval here, prioritising corners?
        score = self.eval(chess_board, player, opponent, player_score-opponent_score)
        if depth == 0 or is_endgame:
            #score = player_score - opponent_score
            return score, best_move #then just return score, best_move?
        if isMax: #maximising player
            start_time = time.time()
            value = float('-inf')
            legal_player_moves = get_valid_moves(chess_board, player)
            #print(legal_player_moves)
            for move in legal_player_moves:
                new_board = deepcopy(chess_board)
                execute_move(new_board, move, player)
                move_value = self.alphabeta(new_board, player, opponent, depth - 1, False, alpha,beta)
                #print("okay move value you get in opponent is this:  ", move_value)
                if move_value[0] > value:
                    value = move_value[0]
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta: #this is the difference, immediately break: now we're avoiding the repeating branches, significantly reducing time!
                    break
            #print("found the best move!! ", best_move)
            time_taken = time.time() - start_time
            print("My alphabeta AI's turn took ", time_taken, "seconds.")
            return value,best_move
        else:
            start_time = time.time()
            value = float('inf')
            legal_opponent_moves = get_valid_moves(chess_board, opponent)
            for move in legal_opponent_moves:
                new_board = deepcopy(chess_board)
                execute_move(new_board, move, opponent)
                move_value = self.alphabeta(new_board, player, opponent, depth - 1, True, alpha,beta)
                #print("okay move value you get in opponent is this:  ", move_value)
                if move_value[0] < value:
                    value = move_value[0]
                    best_move = move
                beta = min(alpha, value)
                if alpha >= beta:
                    break
            #print("found the best move!! ", best_move)
            time_taken = time.time() - start_time
            print("My alphabeta AI's turn took ", time_taken, "seconds.")
            return value,best_move

