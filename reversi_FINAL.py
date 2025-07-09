import numpy as np
import random
import copy
import math
import time

# Board constants
EMPTY, BLACK, WHITE = ".", "B", "W"
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

class Reversi:
    def __init__(self):
        self.board = np.full((8, 8), EMPTY)
        self.board[3, 3], self.board[4, 4] = WHITE, WHITE
        self.board[3, 4], self.board[4, 3] = BLACK, BLACK
        self.current_player = BLACK
    
    def print_board(self):
        print("  " + " ".join(str(i) for i in range(8)))
        for i, row in enumerate(self.board):
            print(i, " ".join(row))

    def is_valid_move(self, row, col, player):
        if self.board[row, col] != EMPTY:
            return False
        opponent = WHITE if player == BLACK else BLACK
        for dr, dc in DIRECTIONS: 
            r, c = row + dr, col + dc
            pieces_to_flip = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == opponent:
                pieces_to_flip.append((r, c))
                r += dr
                c += dc
            if pieces_to_flip and 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == player:
                return True
        return False
    
    def get_valid_moves(self, player):
        return [(r, c) for r in range(8) for c in range(8) if self.is_valid_move(r, c, player)]
    
    def make_move(self, row, col):
        if not self.is_valid_move(row, col, self.current_player):
            print("Invalid move!")
            return False
        
        self._place_piece_and_flip(row, col, self.current_player)
        self.current_player = WHITE if self.current_player == BLACK else BLACK
        return True
    
    def _place_piece_and_flip(self, row, col, player):
        """Helper method to place a piece and flip the captured pieces"""
        opponent = WHITE if player == BLACK else BLACK
        self.board[row, col] = player
        
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            pieces_to_flip = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == opponent:
                pieces_to_flip.append((r, c))
                r += dr
                c += dc
            if pieces_to_flip and 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == player:
                for r, c in pieces_to_flip:
                    self.board[r, c] = player
    
    def simulate_move(self, row, col, player):
        """Create a new game state after making the specified move"""
        new_game = copy.deepcopy(self)
        opponent = WHITE if player == BLACK else BLACK
        new_game._place_piece_and_flip(row, col, player)
        new_game.current_player = opponent
        return new_game
    
    def is_game_over(self):
        return not self.get_valid_moves(BLACK) and not self.get_valid_moves(WHITE)
    
    def get_winner(self):
        black_count = np.sum(self.board == BLACK)
        white_count = np.sum(self.board == WHITE)
        if black_count > white_count:
            return BLACK
        elif white_count > black_count:
            return WHITE
        else:
            return "TIE"

    def get_score(self):
        black_count = np.sum(self.board == BLACK)
        white_count = np.sum(self.board == WHITE)
        return black_count, white_count

class randomMoves:
    def __init__(self, player):
        self.player = player  # AI plays as BLACK or WHITE

    def select_move(self, game):
        """Select a random move."""
        valid_moves = game.get_valid_moves(self.player)
        if not valid_moves:
            return None
        # Choose a random move
        random_move = random.choice(valid_moves)
        return random_move
    
# Monte Carlo Tree Search (MCTS) implementation
class MCTSNode:
    def __init__(self, game, move=None, parent=None):
        self.game = copy.deepcopy(game)
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = self.game.get_valid_moves(self.game.current_player)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.0): # Exploration weight can be adjusted, if needed
        if not self.children:
            return None
        
        def ucb_value(child):
            r_t = child.wins
            n_t = child.visits
            n_s = self.visits
            avg_utility = r_t / (n_t + 1e-6) # 1e-6 for numerical stability
            exploration = exploration_weight* math.sqrt((2 * math.log(n_s)) / (n_t + 1e-6)) # 1e-6 for numerical stability
            return avg_utility + exploration
        
        return max(self.children, key=ucb_value)

    def expand(self):
        move = self.untried_moves.pop()
        new_game = self.game.simulate_move(*move, self.game.current_player)
        child_node = MCTSNode(new_game, move, self)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

    def rollout(self):
        current_game = copy.deepcopy(self.game)
        while not current_game.is_game_over():
            valid_moves = current_game.get_valid_moves(current_game.current_player)
            if not valid_moves:
                current_game.current_player = WHITE if current_game.current_player == BLACK else BLACK
                continue
            move = random.choice(valid_moves)
            current_game.make_move(*move)
        
        winner = current_game.get_winner()
        if winner == BLACK:
            return 1.0
        elif winner == WHITE:
            return 0.0
        else:
            return 0.5

class MCTS_AI:
    def __init__(self, player, simulations=100, verbose=False):
        self.player = player
        self.simulations = simulations
        self.verbose = verbose

    def select_move(self, game):
        if self.verbose:
            print(f"AI is thinking... (running {self.simulations} simulations)")
        start_time = time.time()
        
        root = MCTSNode(game)
        for _ in range(self.simulations):
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
            if not node.is_fully_expanded():
                node = node.expand()
            result = node.rollout()
            node.backpropagate(1 if result == (1.0 if self.player == BLACK else 0.0) else 0)
        
        best_child = root.best_child(exploration_weight=0.0)  # Use exploitation only for final move selection
        end_time = time.time()
        if self.verbose:
            print(f"AI took {end_time - start_time:.2f} seconds to decide")
        
        return best_child.move if best_child else None

class MinimaxAI:
    # Strategic weight matrix
    WEIGHT_MATRIX = np.array([
        [100, -20,  10,   5,   5,  10, -20, 100],
        [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
        [ 10,  -2,   1,   1,   1,   1,  -2,  10],
        [  5,  -2,   1,   1,   1,   1,  -2,   5],
        [  5,  -2,   1,   1,   1,   1,  -2,   5],
        [ 10,  -2,   1,   1,   1,   1,  -2,  10],
        [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
        [100, -20,  10,   5,   5,  10, -20, 100]
    ])

    def __init__(self, player, max_depth):
        self.player = player  # AI plays as BLACK or WHITE
        self.max_depth = max_depth  # Maximum search depth
        self.opponent = WHITE if player == BLACK else BLACK

    def evaluate_board(self, game):
        """
        Evaluate the board position from the perspective of self.player
        Returns higher values for better positions for self.player
        """
        if game.is_game_over():
            # If game is over, return based on winner
            winner = game.get_winner()
            if (self.player == BLACK and winner == 1.0) or (self.player == WHITE and winner == 0.0):
                return 1000  # AI wins
            elif winner == 0.5:
                return 0     # Draw
            else:
                return -1000 # AI loses
        
        # Count weighted pieces
        score = 0
        for r in range(8):
            for c in range(8):
                if game.board[r, c] == self.player:
                    score += self.WEIGHT_MATRIX[r, c]
                elif game.board[r, c] == self.opponent:
                    score -= self.WEIGHT_MATRIX[r, c]
        
        # Add mobility factor (number of valid moves)
        player_moves = len(game.get_valid_moves(self.player))
        opponent_moves = len(game.get_valid_moves(self.opponent))
        if player_moves + opponent_moves != 0:
            mobility = (player_moves - opponent_moves) / (player_moves + opponent_moves)
            score += mobility * 10  # Weight mobility
            
        return score

    def minimax(self, game, depth, alpha, beta, maximizing_player):
        """
        Minimax algorithm with alpha-beta pruning
        """
        if depth == 0 or game.is_game_over():
            return self.evaluate_board(game), None
        
        current_player = self.player if maximizing_player else self.opponent
        valid_moves = game.get_valid_moves(current_player)
        
        # If no valid moves, skip turn and continue with opponent
        if not valid_moves:
            # Skip turn (pass to opponent)
            new_game = copy.deepcopy(game)
            new_game.current_player = self.opponent if maximizing_player else self.player
            value, _ = self.minimax(new_game, depth - 1, alpha, beta, not maximizing_player)
            return value, None
        
        best_move = None
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in valid_moves:
                new_game = game.simulate_move(*move, current_player)
                eval_score, _ = self.minimax(new_game, depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in valid_moves:
                new_game = game.simulate_move(*move, current_player)
                eval_score, _ = self.minimax(new_game, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval, best_move

    def select_move(self, game):
        """Select the best move using minimax with alpha-beta pruning"""
        valid_moves = game.get_valid_moves(self.player)
        if not valid_moves:
            return None
        
        # Calculate how many empty spaces are left to determine if we're in endgame
        empty_count = np.sum(game.board == EMPTY)
        
        # Adjust search depth based on game phase
        # Deeper search for endgame when branching factor is lower
        depth = min(self.max_depth, empty_count)
        """
        if empty_count < 10:  # Very near the end
            depth = min(self.max_depth + 3, empty_count)
        """
        _, best_move = self.minimax(game, depth, float('-inf'), float('inf'), True)
        return best_move

# Main game loop with human player
def play_reversi_human_vs_ai():

    # Ask human player to choose color
    while True:
        human_color = input("Choose your color (B for Black, W for White): ").strip().upper()
        if human_color in [BLACK, WHITE]:
            break
        print("Invalid input. Please enter 'B' for Black or 'W' for White.")
    
    ai_color = WHITE if human_color == BLACK else BLACK
    
    # Initialize game and AI
    game = Reversi()
    ai = MCTS_AI(ai_color, simulations=100, verbose=True)  # Adjust simulation count for difficulty
    
    print("\nWelcome to Reversi!")
    print("Enter moves as 'row col' (e.g., '2 3')")
    print("Black moves first\n")
    
    while not game.is_game_over():
        # Display current board and score
        print("\nCurrent board:")
        game.print_board()
        black_score, white_score = game.get_score()
        print(f"Score - Black: {black_score}, White: {white_score}")
        
        current_player_name = "Black (B)" if game.current_player == BLACK else "White (W)"
        print(f"\nCurrent player: {current_player_name}")
        
        valid_moves = game.get_valid_moves(game.current_player)
        
        if not valid_moves:
            print(f"No valid moves for {current_player_name}. Turn passes.")
            game.current_player = WHITE if game.current_player == BLACK else BLACK
            continue
        
        # Show valid moves
        print("Valid moves:", valid_moves)
        
        # Human or AI turn
        if game.current_player == human_color:
            # Human's turn
            while True:
                try:
                    move_input = input("\nYour move (row col): ")
                    row, col = map(int, move_input.split())
                    if 0 <= row < 8 and 0 <= col < 8 and (row, col) in valid_moves:
                        game.make_move(row, col)
                        break
                    else:
                        print("Invalid move! Please choose from the valid moves list.")
                except (ValueError, IndexError):
                    print("Invalid input! Please enter row and column as numbers (0-7).")
        else:
            # AI's turn
            print("\nAI is making a move...")
            move = ai.select_move(game)
            if move:
                print(f"AI moves to: {move}")
                game.make_move(*move)
    
    # Game over, display result
    print("\nGame Over!")
    game.print_board()
    black_score, white_score = game.get_score()
    print(f"Final Score - Black: {black_score}, White: {white_score}")
    
    winner = game.get_winner()
    if winner == "TIE":
        print("The game is a tie!")
    else:
        winner_name = "Black" if winner == BLACK else "White"
        if winner == human_color:
            print(f"Congratulations! You ({winner_name}) won!")
        else:
            print(f"The AI ({winner_name}) won. Better luck next time!")

def play_reversi_ai_vs_ai(mcts_simulations=24, minimax_depth=3):
    game = Reversi()

    ai = MCTS_AI(BLACK, simulations=mcts_simulations)  # AI plays as Black with MCTS
    minimax = MinimaxAI(WHITE, max_depth=minimax_depth)
    
    while not game.is_game_over():
        valid_moves = game.get_valid_moves(game.current_player)
        
        if not valid_moves:
            # Skip turn when no valid moves
            game.current_player = WHITE if game.current_player == BLACK else BLACK
            continue
            
        if game.current_player == BLACK:
            move = ai.select_move(game)
            if move:
                game.make_move(*move)
        else:
            move = minimax.select_move(game)
            if move:
                game.make_move(*move)
    
    return game.get_winner()

# Run the game
if __name__ == "__main__":
    print("Welcome to Reversi!")
    print("1. Human vs AI")
    print("2. AI vs AI\n")

    while True:
        try:
            choice = int(input("Select game mode (1 or 2): "))
            if choice in [1, 2]:
                break
            print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    if choice == 1:
        play_reversi_human_vs_ai()
    else:
        # Ask user if they want to customize settings
        print("\nAI vs AI mode selected")
        userInput = input("Do you want to customize game parameters? (y/n): ").lower().startswith('y')
        
        # Default values
        num_games = 50
        mcts_simulations = 24
        minimax_depth = 3
        
        # Let user customize parameters if desired
        if userInput:
            
            num_games = int(input(f"Number of games to run (default {num_games}): ") or num_games)
            mcts_simulations = int(input(f"MCTS simulations per move (default {mcts_simulations}): ") or mcts_simulations)
            minimax_depth = int(input(f"Minimax search depth (default {minimax_depth}): ") or minimax_depth)
            print(f"\nRunning {num_games} games with MCTS({mcts_simulations} sims) vs Minimax(depth {minimax_depth})")
    
        else:
            print(f"\nUsing default settings: {num_games} games, MCTS({mcts_simulations} sims) vs Minimax(depth {minimax_depth})")
        
    
        
        black_winrate = 0
        start_time = time.time()

        # Modified loop to use the parameters
        for i in range(num_games):
            # Pass the custom parameters to the function
            winner = play_reversi_ai_vs_ai(mcts_simulations, minimax_depth)

            # Convert string winner to numerical value
            if winner == BLACK:
                black_winrate += 1
            elif winner == "TIE":
                black_winrate += 0.5
            
            if i % 5 == 0:
                print(f"Progress: {i}/{num_games} games, current winrate: {black_winrate/(i+1):.2f}")

        print(f"MCTS winrate ({num_games} games): {black_winrate/num_games:.2f}")
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.4f} seconds")