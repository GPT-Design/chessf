import chess.pgn
import numpy as np
import pandas as pd
from pathlib import Path

def calc_complexity(board) -> float:
    """
    Board-complexity metric = log₂(# legal moves).
    Placeholder until we slot in your richer formula.
    """
    return np.log2(board.legal_moves.count() or 1)

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
