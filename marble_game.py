'''
Marble Game
Version 3
Tae Rugh
2023.7.5
'''



# Imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product
from collections import deque
from time import time
from datetime import timedelta



# Constants
TIMESTART = time()

LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4

MOVE_NAMES = {LEFT:'LEFT', RIGHT:'RIGHT', UP:'UP', DOWN:'DOWN'}
MOVE_DIRS = {LEFT:(-1,0), RIGHT:(1,0), UP:(0,1), DOWN:(0,-1)}

BOARD = np.matrix((
    (0, 0, 1, 1, 1, 0, 0),
    (0, 0, 1, 1, 1, 0, 0),
    (1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1),
    (0, 0, 1, 1, 1, 0, 0),
    (0, 0, 1, 1, 1, 0, 0)
    ), dtype=np.bool_)

BOARD_SHAPE = BOARD.shape

NEW_BOARD = np.matrix((
    (0, 0, 1, 1, 1, 0, 0),
    (0, 0, 1, 1, 1, 0, 0),
    (1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 0, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1),
    (0, 0, 1, 1, 1, 0, 0),
    (0, 0, 1, 1, 1, 0, 0)
    ), dtype=np.bool_)

WINNING_BOARD = np.matrix((
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0)
    ), dtype=np.bool_)

WINNING_MOVESET = (
    (5, 3, 1), (4, 5, 4), (6, 4, 1), (6, 2, 3), (4, 3, 3), (4, 6, 4), (4, 2, 2), (4, 0, 3),
    (3, 4, 2), (6, 4, 1), (3, 6, 4), (3, 4, 2), (3, 2, 2), (6, 2, 1), (3, 0, 3), (3, 2, 2),
    (1, 4, 2), (2, 6, 4), (2, 4, 2), (5, 4, 1), (3, 4, 4), (2, 2, 2), (5, 2, 1), (2, 0, 3),
    (2, 3, 4), (0, 2, 2), (3, 2, 1), (0, 4, 4), (0, 2, 2), (2, 1, 3), (1, 3, 2)
    )



# Class
class MarbleGame:
    def __init__(self, initial_board_state=NEW_BOARD):
        self.board_state = np.matrix(initial_board_state)
        self.fig, self.ax = plt.subplots(dpi=100, facecolor='w')
        plt.axis('off')
    

    # Public
    def simulate_moveset(self, moveset):
        return self._simulate_moveset(self.board_state, moveset)


    def show_board(self):
        return self._show_board(self.board_state)
    

    def print_legal_moves(self):
        return self._print_legal_moves(self.board_state)
    

    def df_search(self):
        return self._df_search()
    

    # Private
    def _simulate_moveset(self, board_state, moveset):
        self._show_board(board_state)
        for move in moveset:
            if (input('Next? ') != ''): return
            self._move(board_state, move)
            self._show_board(board_state)
            self._show_move(move)
        input('Done? ')
    

    def _show_board(self, board_state):
        self.ax.clear()
        self.ax.scatter(*np.where(BOARD & ~board_state), s=400, facecolors='none', edgecolors='r')
        self.ax.scatter(*np.where(BOARD & board_state), s=400, facecolors='r', edgecolors='r')
        plt.show(block=False)
    

    def _show_move(self, move):
        r,c,dir = move
        self.ax.arrow(r, c, 2*MOVE_DIRS[dir][0], 2*MOVE_DIRS[dir][1], width=.01, color='b')
    

    def _print_legal_moves(self, board_state):
        moves = self._board_legal_moves(board_state)
        for move in moves:
            print(f'({move[0]}, {move[1]}) {MOVE_NAMES[move[2]]}')
    

    def _df_search(self, initial_board_state=NEW_BOARD, final_board_state=WINNING_BOARD):
        potential_movesets = deque([[]])
        for i in range(10000000):
            moveset = potential_movesets.pop()
            board_state = np.copy(initial_board_state); self._move_moveset(board_state, moveset)

            if np.equal(board_state, WINNING_BOARD).all():
                print('-DONE-')
                print(moveset)
                print()
                return moveset
            
            for move in self._board_legal_moves(board_state):
                    potential_movesets.append(moveset + [move])

            if (i%50000==0):
                print('------')
                print('\t', 'iteration:    ', i)
                print('\t', 'runtime:      ', timedelta(seconds=time()-TIMESTART))
                print('\t', 'queue length: ', len(potential_movesets))
                print()


    def _board_legal_moves(self, board_state):
        moves = []
        for r,c in product(range(BOARD_SHAPE[0]), range(BOARD_SHAPE[1])):
            if board_state[r,c]:
                moves.extend(self._piece_legal_moves(board_state, r, c))
        return moves


    def _piece_legal_moves(self, board_state, r, c):
        moves = []
        # down
        if (c >= 2):
            if board_state[r, c-1] & BOARD[r, c-2] & ~board_state[r, c-2]:
                moves.append((r, c, DOWN))
        # up
        if (c <= BOARD_SHAPE[1]-3):
            if board_state[r, c+1] & BOARD[r, c+2] & ~board_state[r, c+2]:
                moves.append((r, c, UP))
        # right
        if (r <= BOARD_SHAPE[0]-3):
            if board_state[r+1, c] & BOARD[r+2, c] & ~board_state[r+2, c]:
                moves.append((r, c, RIGHT))
        # left
        if (r >= 2):
            if board_state[r-1, c] & BOARD[r-2, c] & ~board_state[r-2, c]:
                moves.append((r, c, LEFT))
        return moves


    def _move_moveset(self, board_state, moveset):
        for move in moveset:
            self._move(board_state, move)


    def _move(self, board_state, move):
        r,c,dir = move
        board_state[r,c] = False
        board_state[r+MOVE_DIRS[dir][0], c+MOVE_DIRS[dir][1]] = False
        board_state[r+2*MOVE_DIRS[dir][0], c+2*MOVE_DIRS[dir][1]] = True



# Main
if __name__ == '__main__':
    game = MarbleGame()

    moveset = game.df_search()
    # moveset = WINNING_MOVESET

    game.simulate_moveset(moveset)