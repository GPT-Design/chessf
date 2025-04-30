import typer, pathlib, rich
from chessf.core import analyse_pgn

app = typer.Typer(add_completion=False)

@app.command()
def analyze(
    pgn: pathlib.Path,
    out: pathlib.Path = typer.Option("metrics.parquet", help="Output file"),
):
    """Compute complexity metrics for a PGN."""
    df = analyse_pgn(pgn)
    df.to_parquet(out)
    rich.print(f"[bold green]✔  Saved metrics → {out}")

if __name__ == "__main__":
    app()
