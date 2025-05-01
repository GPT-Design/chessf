# src/chessf/cli.py  (the only place that “runs stuff”)
import pathlib, typer, rich
from chessf.analysis import analyse_pgn     # ← updated path

app = typer.Typer(add_completion=False)

@app.command()
def batch(
    pgn_dir: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent.parent / "data",
):
    """Process all PGNs in *pgn_dir* (defaults to repo-local ./data)."""
    analyse_folder(pgn_dir)   # generates plots, summary, etc.

if __name__ == "__main__":
    app()

