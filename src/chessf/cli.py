# src/chessf/cli.py
import pathlib, typer, rich
from chessf.core import analyse_pgn
from chessf.analysis import analyse_folder

app = typer.Typer(add_completion=False)

@app.command()
def analyze(pgn: pathlib.Path, out: pathlib.Path = pathlib.Path("metrics.parquet")):
    """Analyse a single PGN."""
    df = analyse_pgn(pgn)
    df.to_parquet(out)
    rich.print(f"[bold green]✔  Saved metrics → {out}")

@app.command()
def batch(pgn_dir: pathlib.Path = pathlib.Path("data")):
    """Analyse all PGNs in *pgn_dir* (defaults ./data)."""
    analyse_folder(pgn_dir)

if __name__ == "__main__":
    app()


