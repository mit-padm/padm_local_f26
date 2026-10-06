from copy import copy
import random


# General game state object to work with multiple games
class game_state:
    def __init__(self,board,parent=None):
        self.board = board
        self.parent = parent
        self.v = None
        self.optimal_play = None
        
        # If the parent is not None, the player is inferred from the player of the parent
        if self.parent != None:
            self.player = (parent.player)%2 + 1
        else:
            self.player = 1        
    
    def __str__(self):
        return self.board.__str__() + '\n'
    
    def successors(self):
        succ_states = []
        
        for succ_board in self.board.successors(self.player):
            succ_state = game_state(succ_board,self)
            succ_states += [succ_state]
        
        return succ_states
    
    def terminal_check(self):
        return self.board.terminal_check()
    
    def score(self):
        return self.board.score()

    
# Tic-tac-toe board as an example of a simple and solvable game
class tic_tac_toe_board:
    def __init__(self,moves):
        if len(moves) != 9:
            raise Error
        # The current board (not a move history): 9 symbols ('x', 'o', or ' ')
        # read row by row, so cells are numbered 0 1 2 / 3 4 5 / 6 7 8.
        self.moves = moves
    
    def __str__(self):
        return '{0} | {1} | {2}\n--+---+--\n{3} | {4} | {5}\n--+---+--\n{6} | {7} | {8}'.format( \
       self.moves[0],self.moves[1],self.moves[2],self.moves[3],self.moves[4],self.moves[5],self.moves[6],self.moves[7],self.moves[8])
    
    def terminal_check(self):
        # Empty Board
        if not ' ' in self.moves:
            return True
        
        for symbol in ['o','x']:
            # Rows
            for row in range(0,3):
                if self.moves[3*row] == symbol and self.moves[3*row] == self.moves[3*row+1] and self.moves[3*row] == self.moves[3*row+2]:
                    return True
            
            # Columns
            for col in range(0,3):
                if self.moves[col] == symbol and self.moves[col] == self.moves[col+3] and self.moves[col] == self.moves[col+6]:
                    return True
            
            # Diagionals
            if self.moves[0] == symbol and self.moves[0] == self.moves[4] and self.moves[0] == self.moves[8]:
                return True
            if self.moves[2] == symbol and self.moves[2] == self.moves[4] and self.moves[2] == self.moves[6]:
                return True
            
        return False
    
    def score(self):
        if not self.terminal_check():
            return 0
        
        # Rows
        for row in range(0,3):
            if self.moves[3*row] == 'x' and self.moves[3*row] == self.moves[3*row+1] and self.moves[3*row] == self.moves[3*row+2]:
                return 1
            if self.moves[3*row] == 'o' and self.moves[3*row] == self.moves[3*row+1] and self.moves[3*row] == self.moves[3*row+2]:
                return -1
        
         # Columns
        for col in range(0,3):
            if self.moves[col] == 'x' and self.moves[col] == self.moves[col+3] and self.moves[col] == self.moves[col+6]:
                return 1
            if self.moves[col] == 'o' and self.moves[col] == self.moves[col+3] and self.moves[col] == self.moves[col+6]:
                return -1
        
        # Diagionals
        if self.moves[0] == 'x' and self.moves[0] == self.moves[4] and self.moves[0] == self.moves[8]:
            return 1
        if self.moves[2] == 'x' and self.moves[2] == self.moves[4] and self.moves[2] == self.moves[6]:
            return 1
        if self.moves[0] == 'o' and self.moves[0] == self.moves[4] and self.moves[0] == self.moves[8]:
            return -1
        if self.moves[2] == 'o' and self.moves[2] == self.moves[4] and self.moves[2] == self.moves[6]:
            return -1
        
        # If we get here, there is a tie
        return 0
    
    def successors(self,player):
        # Crosses correspond to player 1, naughts to 2
        if self.terminal_check():
            raise Error
        
        succ_list = []
        
        for i in range(0,9):
            if self.moves[i] == ' ':
                succ_moves = copy(self.moves)
                
                if player == 1:
                    succ_moves[i] = 'x'
                else:
                    succ_moves[i] = 'o'
                
                succ_list += [tic_tac_toe_board(succ_moves)]
        
        return succ_list


# A player that picks uniformly at random among its non-losing moves: the
# successors with the best MiniMax value for the player to move. `value` maps a
# game_state to its MiniMax value (e.g., the score from your MiniMax search).
def random_non_losing_move(state,value,rng=random):
    succ_states = state.successors()
    values = [value(s) for s in succ_states]
    best = max(values) if state.player == 1 else min(values)
    return rng.choice([s for s,v in zip(succ_states,values) if v == best])


# Writes the record of games between the Problem 2A opponent (x) and a random
# non-losing o to opponent_games.txt. Needs the course autograder package, so
# run it inside the course Docker image: python utils.py
if __name__ == '__main__':
    from principles_of_autonomy.notebook_tests.pset_5 import _make_minimax, _make_opponent, _cell_played

    num_games = 200
    rng = random.Random(0)
    minimax = _make_minimax()
    opponent = _make_opponent()

    legend = [
        'Record of {} tic-tac-toe games against the opponent.'.format(num_games),
        '',
        'X is the opponent and always moves first. O chose uniformly at random among',
        'its non-losing moves (the moves with the best MiniMax value for O).',
        '',
        'Cells are numbered row by row:',
        '',
        '    0 | 1 | 2',
        '    --+---+--',
        '    3 | 4 | 5',
        '    --+---+--',
        '    6 | 7 | 8',
        '',
        'Each line below is one game: the game number, the result, and the moves in',
        'order. A move is the player followed by the cell it took, so "X0 O4 X5"',
        'means X took cell 0, then O took cell 4, then X took cell 5. The result is',
        '"x wins", "o wins", or "tie".',
        '',
        '{:>4}  {:<6}  {}'.format('game','result','moves'),
    ]

    with open('opponent_games.txt','w') as f:
        f.write('\n'.join(legend) + '\n')
        for game in range(1,num_games+1):
            state = game_state(tic_tac_toe_board([' ']*9))
            moves = []
            while not state.terminal_check():
                if state.player == 1:
                    succ_state = opponent(state,rng)
                else:
                    succ_state = random_non_losing_move(state,minimax,rng)
                symbol = 'X' if state.player == 1 else 'O'
                moves += [symbol + str(_cell_played(state,succ_state))]
                state = succ_state
            result = {1: 'x wins', 0: 'tie', -1: 'o wins'}[state.score()]
            f.write('{:>4}  {:<6}  {}\n'.format(game,result,' '.join(moves)))
