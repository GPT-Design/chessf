# src/chessf/cli.py
import pathlib, typer, rich
from chessf.core import analyse_pgn
from chessf.analysis import analyse_folder

app = typer.Typer(add_completion=False)

@app.command()
def analyze(
    pgn: pathlib.Path = typer.Argument(..., exists=True, file_okay=True),
    out: pathlib.Path = pathlib.Path("metrics.parquet"),
):
    """Analyse a single PGN."""
    rich.print(f"[cyan]Analysing {pgn} → {out}")
    analyse_pgn(pgn).to_parquet(out)

@app.command()
def batch(
    pgn_dir: pathlib.Path = typer.Argument(None, exists=False),
):
    """Analyse all PGNs in *pgn_dir*  (defaults to ./data)."""
    if pgn_dir is None:                       # no arg → use ./data
        pgn_dir = pathlib.Path("data")
    if not pgn_dir.exists():
        raise typer.BadParameter(f"{pgn_dir} does not exist")
    analyse_folder(pgn_dir)

if __name__ == "__main__":
    app()
