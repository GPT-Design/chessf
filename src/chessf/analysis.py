from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Iterable, Tuple

import chess.pgn
import rich

# ──────────────────────────────────────────────
# Array backend ─ GPU first, CPU fallback
# ──────────────────────────────────────────────
try:
    import cupy as xp  # CUDA boxes
except ModuleNotFoundError:          # pragma: no cover
    import numpy as xp  # graceful CPU fallback

# Lazy heavy-math helpers
def _safe_linregress():
    try:
        from scipy.stats import linregress
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ImportError(
            "SciPy required for regression — install with: pip install chessf[stats]"
        ) from e
    return linregress


# ──────────────────────────────────────────────
# Move-complexity weights (tweak to taste)
# ──────────────────────────────────────────────
MOVE_COMPLEXITY = {
    "quiet": 1,
    "capture": 3,
    "check": 5,
    "castling": 4,
    "mate": 10,
    "blunder": 7,
}


# ──────────────────────────────────────────────
# Per-game complexity trace
# ──────────────────────────────────────────────
def analyse_game(pgn_text: str) -> xp.ndarray:
    """Return cumulative complexity (length = moves) for one PGN string."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return xp.asarray([])

    board = game.board()
    complexity_state = 0
    trace: list[int] = []

    for move in game.mainline_moves():
        # complexity delta
        if board.is_capture(move):
            delta = MOVE_COMPLEXITY["capture"]
        elif board.is_castling(move):
            delta = MOVE_COMPLEXITY["castling"]
        elif board.gives_check(move):
            delta = MOVE_COMPLEXITY["check"]
        else:
            delta = MOVE_COMPLEXITY["quiet"]

        board.push(move)
        if board.is_checkmate():
            delta += MOVE_COMPLEXITY["mate"]

        complexity_state += delta
        trace.append(complexity_state)

    return xp.asarray(trace)


# ──────────────────────────────────────────────
# Prime-move extraction & regression helpers
# ──────────────────────────────────────────────
def prime_moves(n: int) -> Iterable[int]:
    """Yield primes ≤ n (naïve)."""
    for k in range(2, n + 1):
        if all(k % p for p in range(2, int(k**0.5) + 1)):
            yield k


def prime_slice(arr: xp.ndarray) -> Tuple[list[int], list[int]]:
    """Return (prime_indices, arr[prime_indices])."""
    if arr.size == 0:
        return [], []
    primes = list(prime_moves(len(arr)))
    values = [int(arr[i - 1]) for i in primes]
    return primes, values


def regression_summary(arr: xp.ndarray) -> str:
    linregress = _safe_linregress()
    y = xp.asnumpy(arr)
    x = xp.arange(1, len(y) + 1)
    slope, intercept, r, p, stderr = linregress(x, y)
    return (
        f"Slope: {slope:.5f}, Intercept: {intercept:.5f}, "
        f"R={r:.5f}, p={p:.5g}, stderr={stderr:.5f}"
    )


# ──────────────────────────────────────────────
# Folder-level driver
# ──────────────────────────────────────────────
def analyse_folder(pgn_dir: Path, *, make_plots: bool = True) -> None:
    """Analyse **all** `*.pgn` files in *pgn_dir* and write outputs to *pgn_dir/output*."""
    pgn_files = sorted(pgn_dir.glob("*.pgn"))
    if not pgn_files:
        rich.print(f"[yellow]No PGN files found in {pgn_dir}[/]")
        return

    out_dir = pgn_dir / "output"
    out_dir.mkdir(exist_ok=True)

    import pandas as pd
    import matplotlib.pyplot as plt  # local import (optional extra)

    summary_lines = []

    for pgn_path in pgn_files:
        text = pgn_path.read_text(encoding="utf-8")
        trace = analyse_game(text)
        if trace.size == 0:
            summary_lines.append(f"{pgn_path.name}: invalid PGN")
            continue

        # parquet
        df = pd.DataFrame({"move": range(1, len(trace) + 1), "complexity": xp.asnumpy(trace)})
        parquet_path = out_dir / f"{pgn_path.stem}.parquet"
        df.to_parquet(parquet_path)

        # plot
        if make_plots:
            plt.figure(figsize=(8, 5))
            plt.plot(df["move"], df["complexity"], marker="o")
            plt.title(pgn_path.stem)
            plt.xlabel("Move")
            plt.ylabel("Complexity")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / f"{pgn_path.stem}.png")
            plt.close()

        # prime slice + regression
        primes, prime_vals = prime_slice(trace)
        reg = regression_summary(trace)
        summary_lines.append(
            f"{pgn_path.name}: moves={len(trace)}, final={int(trace[-1])}\n"
            f"  primes {primes[:5]}… → {prime_vals[:5]}  |  {reg}"
        )

    (out_dir / "chess_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    rich.print(f"[green]✔  Analysis complete. Results in {out_dir}[/]")


# ──────────────────────────────────────────────
# CLI helper (allows `python -m chessf.analysis`)
# ──────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    default_dir = Path(__file__).resolve().parent.parent.parent / "data"
    analyse_folder(default_dir)
