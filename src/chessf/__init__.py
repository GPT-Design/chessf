# src/chessf/__init__.py

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

from .core import calc_complexity, analyse_pgn
from .analysis import analyse_folder
