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
        start=time.time()
        best_move=None
        self.max_depth = 3
        self.corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
        self.cornerVal = 100
        self.edgeVal=50
        #for depth in range(1,self.max_depth + 1):
            #if time.time()-start>=2:
                #break
        _, best_move = self.alphabeta(chess_board, player, opponent, depth=self.max_depth, isMax=True)
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
    

    #i noticed that we're missing obvious wins a lot, trying to implement a move order here, we need to have phases of the game
    def moveorder(self, chess_board,player,opponent):
        legal_player_moves = get_valid_moves(chess_board, player)
        avail = np.sum(np.array(chess_board) == 0)
        board_size = chess_board.shape[0] * chess_board.shape[1]
        early_phase = board_size * 0.75
        mid_phase = board_size * 0.40
        
        order=[]
        for move in legal_player_moves:
            new_board = deepcopy(chess_board)
            execute_move(new_board, move, player)
            #noflips = count_capture(new_board, move, player)
            priority=0
            iscorner= move in self.corners
            isedge = (move[0] == 0 or move[0] == chess_board.shape[0] - 1 or move[1] == 0 or move[1] == chess_board.shape[1] - 1)
            #we can try edges later
            if avail>early_phase: #early
                #print("available tiles are: ", avail, "so this phase is early-corners most valuable")
                #priority = noflips
                if iscorner: #most valuable
                    priority += 100
                if isedge:
                    priority+= 50
            elif avail<=early_phase and avail>mid_phase: #mid
                #print("available tiles are: ", avail, "so this phase is mid-flips are more valuable")
                #priority = noflips*3
                if iscorner: #most valuable
                    priority += 80
                if isedge:
                    priority +=30
            else:
                #print("available tiles are: ", avail, "so this phase is late")
                #priority=noflips*5
                if iscorner: #most valuable
                    priority += 100
                if isedge:
                    priority+= 50
            order.append((move, priority))
        order.sort(reverse=True, key=lambda x: x[1]) #highest priority first
        return [move for move,_ in order]

    def alphabeta(self, chess_board, player, opponent, depth, isMax, alpha=float('-inf'), beta=float('inf')):
        best_move=None
        is_endgame, player_score, opponent_score = check_endgame(chess_board, player, opponent)
        current_score = self.eval(chess_board, player, opponent, player_score-opponent_score)
        if depth == 0 or is_endgame:
            return current_score, best_move
        if isMax: #maximising player
            start_time = time.time()
            value = float('-inf')
            #legal_player_moves = get_valid_moves(chess_board, player)
            order_player= self.moveorder(chess_board, player,opponent)
            #print(legal_player_moves)
            for move in order_player:
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
            #print("My alphabeta AI's turn took ", time_taken, "seconds.")
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
            #print("My alphabeta AI's turn took ", time_taken, "seconds.")
            return value,best_move
        