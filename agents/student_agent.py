from agents.agent import Agent
from store import register_agent
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves, get_directions

class Node:
    def __init__(self, chess_board, player=None, move=None, parent=None, eval_fn = None):
        self.chess_board = chess_board # state of Board after position is evaluated
        self.player = player # which player is playing
        self.parent = parent # parent node 
        self.move = move # action taken prior to transition from the parent to this child node, move = None for root node at game start.
        self.children = [] # children nodes
        self.visit = 0 # number of to. node was visited
        self.wins = 0 # number of times scored / won
        self.valid_moves = None # list of valid moves to take
        self.eval_fn = eval_fn # self function with heuristical application
    
    def is_fully_expanded(self):
        return len(self.children) == len(get_valid_moves(self.chess_board, self.player))

    def uct(self, N, c=1):
        max_eval_stage = 2618
        min_eval_stage = -2618

        if self.visit == 0:
            return float('inf')

        exploitation = self.wins / self.visit
        exploration = c * np.sqrt(np.log(N+1) / (self.visit+1)) # self.parent.visit is used to reflect how much oppportunity to child node

        # Adding a heuristic scoring scheme following the eval function as a bonus 
        # to explore nodes with better board states
        ## NOTE the self.parent.eval_fn vs self.eval_fn
        if self.parent is not None: # in case it is a root node.
            heuristic_score = self.parent.eval_fn(self.chess_board, self.player, -self.player)
        else:
            heuristic_score = self.eval_fn(self.chess_board, self.player, -self.player)
        normalized_heuristic_score = heuristic_score / max(abs(max_eval_stage), abs(min_eval_stage))

        return exploitation + exploration + 1 * normalized_heuristic_score

@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.iterations = 100 # Num of MCTS iterations
        self.time_limit = 1.90 #  Maximum time per move in seconds, lower than 2s to account for variability

    def clone_board(self, board): # might help in reducing run-time vs deepcopy
        return np.copy(board) # computationally faster than deepcopy from copy module

    #This method selects the best move for the agent based on the Alpha-Beta Pruning algorithm.
    def step(self, chess_board, player, opponent):
        start_time = time.time()
        depth=3 #the tradeoff to being strict with time is digging less deeper, makes it easier for a human to win 
        best_move = None
        _, move= self.alphabeta(chess_board,player,opponent,depth, isMax=True,start_time=start_time)
        if move is not None:
            best_move=move
        time_taken=time.time()-start_time
        print("My alphabeta AI's turn took ", time_taken, "seconds.")
        return best_move
        # Solutions implemented can be called like:
        # return self.greedyFlips(chess_board, player, opponent) or self.minimax(chess_board, player, opponent, depth, True) or self.mcts(self,chess_board,player,opponent)
        

    #This method implements a basic heuristic approach where the agent selects the move that maximizes the number of captured pieces. It is computationally efficient but lacks strategic depth.
    def greedyFlips(self, chess_board, player, opponent):
        max_flips = 0
        start_time = time.time()
        best_move = None
        legal_player = get_valid_moves(chess_board, player)
        print(legal_player)
        for move in legal_player:
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
    
    #This method implements the Minimax algorithm. It explores the game tree recursively, evaluating each move at a given depth and selecting the move that maximizes or minimizes the score.
    def minimax(self, chess_board, player, opponent, depth, isMax):
        best_move=None
        is_endgame, player_score, opponent_score = check_endgame(chess_board, player, opponent)
        if depth == 0 or is_endgame:
            score = player_score - opponent_score 
            return ( self.eval(chess_board, player, opponent, score)), best_move 
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
                if move_value[0] > value:
                    value = move_value[0]
                    best_move = move
            #print("found the best move!! ", best_move)
            time_taken = time.time() - start_time
            #print("My minimax AI's turn took ", time_taken, "seconds.")
            return value, best_move 
        else:  #minimising opponsent 
            start_time = time.time()
            value = float('inf')
            best_move = None
            legal_opponent_moves = get_valid_moves(chess_board, opponent)
            #print(legal_opponent_moves)
            for move in legal_opponent_moves:
                new_board = deepcopy(chess_board)
                execute_move(new_board, move, opponent)
                move_value = self.minimax(new_board, player, opponent, depth - 1, True)
                if move_value[0] < value:
                    value = move_value[0]
                    best_move = move
            #print("found the best move!! ", best_move)
            time_taken = time.time() - start_time
            #print("My minimax AI's turn took ", time_taken, "seconds.")
            return value, best_move
    
    #Eval function (Atreyi)- computes a score for a given board state. It takes into account factors such as corner occupancy, stability, mobility, and piece count, and adapts the weighting based on the game stage.
    def eval(self, chess_board, player, opponent):
        corners = [(0, 0), (0, len(chess_board)-1), (len(chess_board)-1, 0), (len(chess_board)-1, len(chess_board)-1)]
        cornerVal = 100
        #game stage for better strategies throughout --https://medium.com/@gmu1233/how-to-write-an-othello-ai-with-alpha-beta-search-58131ffe67eb, chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://barberalec.github.io/pdf/An_Analysis_of_Othello_AI_Strategies.pdf
        total_squares = len(chess_board)*len(chess_board)
        total_pieces = np.sum(chess_board != 0) #occupied
        stage = total_pieces / total_squares 
        corner_score=0 
        stability_score=0
        mobility_score=0 
        piece_score=0 

        #corners are always better
        for corner in corners:
            if chess_board[corner[0]][corner[1]] == player:
                corner_score += cornerVal
            elif chess_board[corner[0]][corner[1]] == opponent:
                corner_score -= cornerVal

        #stability measures how vulnerable the disc is to be flanked
        for row in range(len(chess_board)):
            for col in range(len(chess_board[row])):
                if chess_board[row][col] == player and self.is_stable(chess_board, row, col, player):
                    stability_score += 10
                elif chess_board[row][col] == opponent and self.is_stable(chess_board, row, col, opponent):
                    stability_score -= 10

        #mobility measures how many moves the player has
        player_moves = len(get_valid_moves(chess_board, player))
        opponent_moves = len(get_valid_moves(chess_board, opponent))
        mobility_score = 10 * (player_moves - opponent_moves)

        #piece count- i believe this is not as important--https://www.ultraboardgames.com/othello/strategy.php
        player_pieces = np.sum(chess_board == player)
        opponent_pieces = np.sum(chess_board == opponent)
        piece_score = player_pieces - opponent_pieces

        #it's better to prioritise different things in different stages
        if stage < 0.3:
            #print("total pieces occupied so far: ", total_pieces, "so this is the early stage, corners are imp")
            return 0.5 * mobility_score + 1.5 * corner_score + 0.5 * stability_score #these numbers come from trial and error
        elif stage < 0.7:
            #print("total pieces occupied so far: ", total_pieces, "so this is the mid stage")
            return 1.0 * mobility_score + 1.5 * corner_score + 0.8 * stability_score + 0.2 * piece_score #gets slightly better at attcking
        else:
            #print("total pieces occupied so far: ", total_pieces, "so this is the late stage, corners are imp")
            return 0.2 * mobility_score + 2.0 * corner_score + 1.0 * piece_score + 0.5 * stability_score #it does better against random in the late stages, but might lose to human agent

    #This helper method checks if a piece is stable (if it is in a corner or surrounded by the same color in all directions)
    def is_stable(self, chess_board, row, col, player):
        if (row == 0 or row == len(chess_board) - 1) or (col == 0 or col == len(chess_board[0]) - 1):
            return True
        directions=get_directions() #defined in helper.py
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < len(chess_board) and 0 <= c < len(chess_board[0]) and chess_board[r][c] != player:
                return False
        return True

    #This method orders the valid moves based on their eval score to prioritize better moves
    def order_moves(self, chess_board, moves, player): 
        return sorted(moves, key=lambda move: self.eval(self.exec(chess_board, move, player), player, 3 - player), reverse=True)
    
    #Executes a move by creating a new board state after applying the move.
    def exec(self, chess_board, move, player):
        new_board = deepcopy(chess_board)
        execute_move(new_board, move, player)
        return new_board

    #Strongest algorithm developed so far - recursively explores the game tree while pruning branches that cannot influence the outcome, uses eval function by Atreyi
    def alphabeta(self, chess_board, player, opponent, depth, isMax, alpha=float('-inf'), beta=float('inf'), start_time=None): 
        #print("We're at depth: ", depth, "...")
        if time.time() - start_time >= 1.9:
            return 0, None  #improved the max turn time
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
        
    def mcts(self, chess_board, player, opponent):
        N = 0 # to track global simulations
        # Track number of wins and losses for this turn in the simulations
        wins = 0
        loss = 0 
        draws = 0
        depth = 5
        nSims = 2

        # Timing the MCTS loop in the `step` method
        start_time = time.time() # track time before following operations.

        total_squares = len(chess_board) * len(chess_board)
        total_pieces = np.sum(chess_board != 0)  # Occupied squares
        stage = total_pieces / total_squares
        epsilon = 0.1 if stage < 0.5 else 0.05  # Less randomness in late game
        
        # if stage > 0.7: # late game, computationally expensive, so lower depth & nSims
        #     depth = 3

        root = Node(chess_board=chess_board, player=player, eval_fn=self.evaluation)
        self.expand(root)
        currentNode = self.select(root, N)
        wins, loss, draws = self.simulate(currentNode, nSims=nSims, depth=depth, epsilon=epsilon)
        N += nSims
        self.backpropagate(currentNode, wins, loss, draws)
        N += nSims
        # Prioritize initial moves based on heuristic evaluation
        # valid_moves = get_valid_moves(chess_board, player)
        # sorted_moves = sorted(valid_moves, key=lambda m: self.eval(self.exec(chess_board, m, player), player, opponent), reverse=True)
        
        iteration = 0 # track iterations

        # MCTS Loop
        while time.time() - start_time<self.time_limit and iteration < self.iterations:
            # Execute MCTS Loop

            # Selection, traverses the tree by selecting child node with highest UCT score
            node = self.select(root, N)

            # Expansion, after check to add a child for an unxplored move from current node
            if not check_endgame(node.chess_board, node.player, opponent)[0]:
                self.expand(node)

            # Simulation, staring from expanded node to play to a random game to completion
            wins, loss, draws = self.simulate(node, nSims=nSims, depth=depth, epsilon=epsilon)
            N += nSims

            # Backprogpagation
            self.backpropagate(node, wins, loss, draws)
            N += nSims

            iteration += 1

        print(f"MCTS completed {iteration} iterations in {time.time() - start_time:.4f} seconds")

        # Once all the iterations or time limit up, decide on a child node.
        # This is the child that will be chosen as the agent's next move.
        # Select the child comparative to the root node.
        best_child = max(root.children, key=lambda child: child.visit)
        return best_child.move

    def select(self, node, N):
        """
        Select node to decide which position to move based on UCT scoring. 
        Adjustment factors to the UCT scoring is also applied (incrementally developed and tuned):
        - Mobility: increase available options for subsequent turns
        - Centrality: rewards movement that place pieces near the center of board for strategic control
        - Corner occupancy: rewards corners and penalizes moves on losing corners
        - greed penalty: would penalize captures of too many pieces early as that limits mobility and strategy.

        These adjustment factors provide a deeper understanding for Reversi-specific strategies and decision-making.
        N = # of simulated global games
        """
        while node.children and node.is_fully_expanded():
            node = max(node.children, key=lambda child: child.uct(N))
        return node
    
    def expand(self, node):
        """
        Expand by adding one valid move as a child node. 
        Helps represent a child node within a new gamestate after executing the move
        """
        valid_moves = get_valid_moves(node.chess_board, node.player)
        if not valid_moves:
            return
        
        sorted_moves = sorted(valid_moves, key=lambda m: self.evaluation(self.execution(node.chess_board, m, node.player), node.player, -node.player), reverse=True)
        tried_moves = {child.move for child in node.children}

        for move in sorted_moves:
            if move not in tried_moves:
                new_board = deepcopy(node.chess_board)
                execute_move(new_board, move, node.player)
                new_player = -node.player # Switch player
                child = Node(chess_board=new_board, player=new_player, move = move, parent = node, eval_fn=self.evaluation)
                node.children.append(child)
                break

    def simulate(self, node, nSims = 5, depth = 10, epsilon = 0.1):
        """"
        Simulate according to exploration parameters until terminal state.
        - nSims (default) = 5: number of games simulated starting from given node's state.
        - Depth (default) = 10: Limit depth to speed up simulation.
        - epsilon (default) = 0.1: Chance to select a random move for improved exploration.

        
        Returns the following cumulative counts:
        - wins: Number of wins for the current player.
        - losses: Number of losses for the current player.
        - draws: Number of draws.
        """
        chess_board = self.clone_board(node.chess_board)
        current_player = node.player
        wins = 0
        draws = 0
        loss = 0

        for _ in range(nSims): # perform a simulation
            current_board = self.clone_board(chess_board)
            current_player = node.player
            for depth_iter in range(depth): # depth-limited rollout
                is_endgame, player_score, opponent_score = check_endgame(current_board, current_player, -current_player)
                if is_endgame:
                    # print(f"ENDGAME, player_score: {player_score}; opponent_score: {opponent_score}")
                    if player_score > opponent_score:
                        wins += 1
                    elif player_score < opponent_score:
                        loss += 1
                    else:
                        draws += 1
                    break
                
                valid_moves = get_valid_moves(chess_board, current_player)
                if not valid_moves:
                    current_player = -current_player
                    continue
          
                sorted_moves = sorted(valid_moves, key=lambda m: self.evaluation(self.execution(node.chess_board, m, node.player), node.player, -node.player), reverse=True)
                
                # appplying epsilon-greedy randomization for move selection 
                move = sorted_moves[0] if depth_iter == 0 else valid_moves[np.random.randint(len(valid_moves))]
                # grab the highest heuristically viable move to execute

                execute_move(chess_board, move, current_player)
                current_player = -current_player

        # Return the results of all simulations
        return wins, loss, draws

    def backpropagate(self, node, wins, loss, draws):
        """
        Propagates simulation result up the tree.
        - wins: Number of wins for the current player.
        - losses: Number of losses for the current player.
        - draws: Number of draws.
        """
        # player's turn
        turn = node.player

        while node:
            node.visit += wins + loss + draws  # total games played

            if node.player == turn:
                node.wins += wins
            else:
                node.wins += loss
            
            # move to parent node

            node = node.parent

    #Modified version of Atreyi's eval function, designed by Toufic
    def evaluation(self, chess_board, player, opponent):
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
                
                # Edges
                if row == 0 or row == len(chess_board) - 1 or col == 0 or col == len(chess_board[0]) - 1:
                    stability_score += 5  # add extra stability weight for edge pieces.

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
        
    
    #Atreyi's modified function, designed by Toufic
    def execution(self, chess_board, move, player):
        new_board = self.clone_board(chess_board)
        execute_move(new_board, move, player)
        return new_board


