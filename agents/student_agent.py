from agents.agent import Agent
from store import register_agent
import numpy as np
from copy import deepcopy
import time
import math
import psutil #pip install psutil please
import os
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves, get_directions

@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"

    def step(self, chess_board, player, opponent):
        start_time = time.time()
        depth=3 #the tradeoff to being strict with time is digging less deeper, makes it easier for a human to win 
        #self.max_depth = 3, changed this to dynamically adjust depth (suggestion to improve time )
        best_move = None
        _, move= self.alphabeta(chess_board,player,opponent,depth, isMax=True,start_time=start_time)
        if move is not None:
            best_move=move
        time_taken=time.time()-start_time
        print("My alphabeta AI's turn took ", time_taken, "seconds.")
        return best_move
        #return self.greedyFlips(chess_board, player, opponent)
        #some pointers i got from gpt to improve turn time: iterative deepening is fine but it'd be better to dynamically adjust depth based on available time
        

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
    
    
    def eval(self, chess_board, player, opponent):
        corners = [(0, 0), (0, len(chess_board)-1), (len(chess_board)-1, 0), (len(chess_board)-1, len(chess_board)-1)]
        cornerVal = 100
        center = len(chess_board) // 2
        central_positions = [(center-1, center-1), (center-1, center), (center, center-1), (center, center)]

        total_squares = len(chess_board) * len(chess_board)
        total_pieces = np.sum(chess_board != 0)  # Occupied squares
        stage = total_pieces / total_squares

        corner_score = 0
        stability_score = 0
        mobility_score = 0
        piece_score = 0
        central_score = 0

        # Corners are always better
        for corner in corners:
            if chess_board[corner[0]][corner[1]] == player:
                corner_score += cornerVal
            elif chess_board[corner[0]][corner[1]] == opponent:
                corner_score -= cornerVal

        if self.is_stable(chess_board, corner[0], corner[1], player):
            corner_score += cornerVal * 2  # add even more weight to stable corners

        # Edges
        if row == 0 or row == len(chess_board) - 1 or col == 0 or col == len(chess_board[0]) - 1:
            stability_score += 5  # add extra stability weight for edge pieces.

        for pos in central_positions:
            if chess_board[pos[0]][pos[1]] == player:
                central_score += 10

        # Stability measures how vulnerable the disc is to be flanked
        for row in range(len(chess_board)):
            for col in range(len(chess_board[row])):
                if chess_board[row][col] == player and self.is_stable(chess_board, row, col, player):
                    stability_score += 10
                elif chess_board[row][col] == opponent and self.is_stable(chess_board, row, col, opponent):
                    stability_score -= 10

        # Mobility measures how many moves the player has
        player_moves = len(get_valid_moves(chess_board, player))
        opponent_moves = len(get_valid_moves(chess_board, opponent))

        # simulate future moves to calculate future mobility impact
        for move in get_valid_moves(chess_board, player):
            future_board = self.exec(chess_board, move, player)
            future_player_moves = len(get_valid_moves(future_board, player))
            if future_player_moves < player_moves:
                mobility_score -= 20  # penalize moves that reduce mobility.

        # stage-dependent mobility weighting
        if stage < 0.3:  # Early game
            mobility_score = 20 * (player_moves - opponent_moves) # priortize early movement
        elif stage < 0.7:  # Mid game
            mobility_score = 10 * (player_moves - opponent_moves) # less weighting mid-game
        else:  # Late game
            mobility_score = 5 * (player_moves - opponent_moves)  # minimized in late game.

        
        # stage-dependent piece counts weightings
        player_pieces = np.sum(chess_board == player)
        opponent_pieces = np.sum(chess_board == opponent)
        if stage < 0.3:
            piece_score = 0.1 * (player_pieces - opponent_pieces)
        elif stage < 0.7:
            piece_score = 0.5 * (player_pieces - opponent_pieces)
        else:
            piece_score = 1.0 * (player_pieces - opponent_pieces)
        
        return (0.5 * mobility_score + 1.5 * corner_score +
            0.5 * stability_score + 0.3 * central_score + piece_score)
        
    def is_stable(self, chess_board, row, col, player):
        """
        Determine if a piece is stable (not flippable).
        A piece is stable if it is in a corner or fully surrounded by the same color in all directions.
        """
        if (row == 0 or row == len(chess_board) - 1) or (col == 0 or col == len(chess_board[0]) - 1):
            return True  # Corners and edges are inherently more stable
        directions = get_directions()
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < len(chess_board) and 0 <= c < len(chess_board[0]) and chess_board[r][c] != player:
                return False
        return True
    
    def order_moves(self, chess_board, moves, player): #prioritising moves based on eval score
        return sorted(moves, key=lambda move: self.eval(self.exec(chess_board, move, player), player, 3 - player), reverse=True)
    
    def exec(self, chess_board, move, player):
        new_board = deepcopy(chess_board)
        execute_move(new_board, move, player)
        return new_board

    def alphabeta(self, chess_board, player, opponent, depth, isMax, alpha=float('-inf'), beta=float('inf'), start_time=None): #start_time=None
        #print("We're at depth: ", depth, "...")
        if time.time() - start_time >= 1.9:
            return 0, None  #the max turn time was going up till 5 seconds
        best_move = None
        is_endgame, _, _ = check_endgame(chess_board, player, opponent)
        if depth == 0 or is_endgame:
            return self.eval(chess_board, player, opponent), None
        legal_moves = self.order_moves(chess_board, get_valid_moves(chess_board, player if isMax else opponent), player if isMax else opponent)
        #print(legal_moves)
        if isMax:  #maximising player
            value = float('-inf')
            for move in legal_moves:
                new_board = self.exec(chess_board, move, player)
                move_value, _ = self.alphabeta(new_board, player, opponent, depth - 1, False, alpha, beta, start_time)
                if move_value > value:
                    value = move_value
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            #time_taken=time.time()-start_time
            #print("My alphabeta AI's turn took ", time_taken, "seconds.")
            return value, best_move
        else:  #minimising player
            value = float('inf')
            for move in legal_moves:
                new_board = self.exec(chess_board, move, opponent)
                move_value, _ = self.alphabeta(new_board, player, opponent, depth - 1, True, alpha, beta, start_time)
                if move_value < value:
                    value = move_value
                    best_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    break
            #time_taken=time.time()-start_time
            #print("My alphabeta AI's turn took ", time_taken, "seconds.")
            return value, best_move
        


