# src/chessf/core.py
from pathlib import Path
import chess.pgn

# ---------- backend -------------------------------------------------
try:
    import cupy as xp
except ModuleNotFoundError:
    import numpy as xp

# ---------- single-game analysis -----------------------------------
def calc_complexity(board) -> float:
    legal = board.legal_moves.count() or 1
    return xp.log2(legal).item()

def analyse_pgn(pgn_path: Path):
    """Return a DataFrame with ply-by-ply complexity for *one* PGN."""
    import pandas as pd
    with open(pgn_path, "r", encoding="utf-8") as fh:
        game = chess.pgn.read_game(fh)

    board = game.board()
    rows = []
    for ply, move in enumerate(game.mainline_moves(), start=1):
        board.push(move)
        rows.append(
            {"ply": ply, "fen": board.fen(), "complexity": calc_complexity(board)}
        )
    return pd.DataFrame(rows)
