import os
import chess.pgn
import cupy as cp         # GPU acceleration
import numpy as np
import matplotlib.pyplot as plt
import io
from scipy.stats import linregress
from mpmath import zeta

# ---------------------------
# Directories
# ---------------------------
PGN_DIRECTORY = "C:\\CHESS\\"  # Adjust for Bash: "/c/CHESS/"
pgn_files = [f for f in os.listdir(PGN_DIRECTORY) if f.endswith(".pgn")]

if not pgn_files:
    print("No PGN files found in the directory.")
    exit()

OUTPUT_DIRECTORY = os.path.join(PGN_DIRECTORY, "output")
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
    print(f"Created output directory: {OUTPUT_DIRECTORY}")
else:
    print(f"Output directory exists: {OUTPUT_DIRECTORY}")

# ---------------------------
# Move Entropy Mapping
# ---------------------------
move_complexity = {
    "quiet": 1,
    "capture": 3,
    "check": 5,
    "castling": 4,
    "mate": 10,
    "blunder": 7
}

# ---------------------------
# Analysis Functions
# ---------------------------

def analyze_chess_entropy(pgn_text):
    """Processes a chess PGN and computes entropy evolution based on move complexity."""
    print("Starting chess entropy analysis...")
    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        print("Invalid PGN game.")
        return cp.asarray([])
    
    entropy_values = []
    entropy_state = 0
    board = game.board()
    move_number = 0

    for move in game.mainline_moves():
        move_number += 1
        # Print move in SAN format
        try:
            move_san = board.san(move)
        except Exception:
            move_san = "N/A"
        print(f"Move {move_number}: {move_san}")
        
        # Determine move complexity using current board state
        if board.is_capture(move):
            delta = move_complexity["capture"]
        elif board.is_castling(move):
            delta = move_complexity["castling"]
        elif board.gives_check(move):
            delta = move_complexity["check"]
        else:
            delta = move_complexity["quiet"]
        
        board.push(move)
        # If board is checkmate after move, add mate bonus.
        if board.is_checkmate():
            delta += move_complexity["mate"]
        
        entropy_state += delta
        entropy_values.append(entropy_state)
        print(f"After move {move_number}, entropy: {entropy_state}")
    
    print(f"Chess entropy analysis complete. Total moves: {move_number}, Final entropy: {entropy_state}")
    return cp.asarray(entropy_values)

def plot_entropy(entropy_values, filename):
    """Saves the entropy evolution plot to a file."""
    plt.figure(figsize=(8,5))
    entropy_cpu = cp.asnumpy(entropy_values)
    plt.plot(entropy_cpu, marker='o', linestyle='-', color='blue')
    plt.xlabel("Move Number")
    plt.ylabel("Entropy Level")
    plt.title("Stepwise Entropy Evolution in Chess Game")
    plt.grid(True)
    plot_filepath = os.path.join(OUTPUT_DIRECTORY, f"{filename}_entropy.png")
    plt.savefig(plot_filepath)
    plt.close()
    print(f"Saved entropy plot as {filename}_entropy.png")

def analyze_prime_entropy(entropy_values):
    """Extracts entropy values at prime-numbered moves."""
    entropy_cpu = cp.asnumpy(entropy_values)
    prime_moves = [i for i in range(1, len(entropy_cpu)+1)
                   if all(i % p != 0 for p in range(2, int(np.sqrt(i)) + 1))]
    prime_entropy_values = [entropy_cpu[i-1] for i in prime_moves]
    return prime_moves, prime_entropy_values

def regression_summary(entropy_values):
    """Computes linear regression of entropy vs. move number and returns a summary."""
    entropy_cpu = cp.asnumpy(entropy_values)
    n_points = len(entropy_cpu)
    moves = np.arange(1, n_points+1)
    slope, intercept, r_value, p_value, std_err = linregress(moves, entropy_cpu)
    summary = f"Number of Data Points: {n_points}\n"
    summary += f"Slope: {slope:.5f}\n"
    summary += f"Intercept: {intercept:.5f}\n"
    summary += f"R-value: {r_value:.5f}\n"
    summary += f"P-value: {p_value:.5f}\n"
    summary += f"Standard Error: {std_err:.5f}\n"
    return summary

# ---------------------------
# Main Processing Loop
# ---------------------------
summary_filepath = os.path.join(OUTPUT_DIRECTORY, "chess_summary.txt")
with open(summary_filepath, "w") as summary_file:
    summary_file.write("Chess Entropy Analysis Summary\n")
    summary_file.write("===================================\n\n")
    
    for pgn_file in pgn_files:
        filepath = os.path.join(PGN_DIRECTORY, pgn_file)
        with open(filepath) as f:
            pgn_text = f.read()
        
        print(f"\nProcessing game from file: {pgn_file}")
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game:
            white = game.headers.get("White", "Unknown")
            black = game.headers.get("Black", "Unknown")
            header_info = f"Game: {white} vs {black}\nFile: {pgn_file}\n"
            print(header_info)
            summary_file.write(header_info)
            summary_file.flush()
            
            entropy_values = analyze_chess_entropy(pgn_text)
            if entropy_values.size == 0:
                print("No valid game found in file, skipping.")
                summary_file.write("Invalid game.\n\n")
                summary_file.flush()
                continue
            
            # Save plot to file (non-blocking)
            plot_entropy(entropy_values, pgn_file)
            
            prime_moves, prime_entropy_values = analyze_prime_entropy(entropy_values)
            prime_info = f"Prime Move Numbers: {prime_moves}\n"
            prime_info += f"Prime Entropy Values: {prime_entropy_values}\n"
            print(prime_info)
            summary_file.write(prime_info)
            summary_file.flush()
            
            reg_summary = regression_summary(entropy_values)
            summary_file.write("Regression Summary:\n")
            summary_file.write(reg_summary + "\n")
            summary_file.write("------------------------------------------------\n")
            summary_file.flush()
        else:
            print(f"Skipping invalid PGN file: {pgn_file}")
            summary_file.write(f"Invalid game in file: {pgn_file}\n\n")
            summary_file.flush()

print("All PGN files processed. Summary saved to:", summary_filepath)
