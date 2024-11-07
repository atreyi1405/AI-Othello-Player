# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves


#Initial idea: Greedy agent, maximises number of flips
#improvement: Greedy agent, prioritises corners, maximises number of flips?
#minimax with alpha beta pruning for better chances at winning

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"
    
  def step(self, chess_board, player, opponent):
    """
    Variables:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    """
    return self.greedyFlips(chess_board, player, opponent)#self.greedyFlips(chess_board, player, opponent) #temporary: we can keep changing this to different methods to evaluate performance
  
  def greedyFlips(self, chess_board, player, opponent):
    max_flips=0
    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    ##### TODO: 
    #greedyStep(self, chess_board, player, opponent), flips, prioritising corners and reducing opponent captures?
    #minimax(self, chess_board, player, opponent)
    #alphabeta(self, chess_board, player, opponent)
    #try MCTS?
    legal_player= get_valid_moves(chess_board, player)
    print(legal_player)
    for move in legal_player:
      noFlips= count_capture(chess_board, move, player) #i want to clone the board here but keep getting NoneType error
      print(f"Move:  {move}, flips: { noFlips}")
      if noFlips > max_flips:
        max_flips= noFlips
        best_move=move
    print("found the best move!! ", best_move)
    time_taken = time.time() - start_time
    print("My greedy AI's turn took ", time_taken, "seconds.")

    return best_move #if best!=None else random_move(chess_board,player)
 
