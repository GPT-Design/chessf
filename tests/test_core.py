import chess
from chessf.core import calc_complexity

def test_start_position_complexity():
    assert calc_complexity(chess.Board()) > 0
