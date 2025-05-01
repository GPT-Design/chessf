import chess.pgn
import numpy as np
import pandas as pd
import cupy as xp            # xp = “array library” (GPU)
except ModuleNotFoundError:
    import numpy as xp           # graceful CPU fallback

from pathlib import Path

# src/chessf/core.py  (no side-effects!)

def calc_complexity(board):
    import numpy as np
    legal = board.legal_moves.count() or 1
    return np.log2(legal).item()

from .analysis import analyse_pgn    # ← wherever the real function lives
__all__ = ["calc_complexity", "analyse_pgn"]

# --- keep optional heavy stuff out of global scope -------------
def prime_regression(prime_moves, prime_values):
    try:
        from scipy.stats import linregress
    except ModuleNotFoundError:  # pragma: no cover
        raise ImportError(
            "SciPy required for regression — install with: pip install chessf[stats]"
        )
    return linregress(prime_moves, prime_values)

# --- Analysis ---
def analyse_pgn(pgn_path: str | Path) -> pd.DataFrame:
    game = chess.pgn.read_game(open(pgn_path))
    board = game.board()

    rows = []
    for ply, move in enumerate(game.mainline_moves(), start=1):
        board.push(move)
        rows.append(
            {
                "ply": ply,
                "fen": board.fen(),
                "complexity": calc_complexity(board),
            }
        )
    return pd.DataFrame(rows)
