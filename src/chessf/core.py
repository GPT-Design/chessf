# src/chessf/core.py

# ─── array backend ─────────────────────────────────────────────────────────────
from pathlib import Path

# ─── third-party imports ──────────────────────────────────────────────────────
import chess.pgn
import pandas as pd

# ─── array backend ─────────────────────────────────────────────────────────────
try:
    import cupy as xp          # try GPU acceleration
    # Only keep CuPy if a CUDA-capable driver is present
    try:
        if not xp.is_available():
            raise RuntimeError("CuPy is installed but no GPU available")
    except Exception:
        raise ModuleNotFoundError
except ModuleNotFoundError:
    import numpy as xp         # CPU fallback

# ---------- single-game analysis -----------------------------------
def calc_complexity(board) -> float:
    legal = board.legal_moves.count() or 1
    return xp.log2(legal).item()

def analyse_pgn(pgn_path: str | Path) -> pd.DataFrame:
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
