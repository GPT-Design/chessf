# src/chessf/analysis.py

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Tuple

import chess.pgn
import rich

# ─── array backend ─────────────────────────────────────────────────────────────
try:
    import cupy as xp          # GPU first
    def _to_numpy(a):
        return a.get()
except ModuleNotFoundError:
    import numpy as xp         # CPU fallback
    def _to_numpy(a):
        return a               # already NumPy

# ─── lazy heavy-math helper ────────────────────────────────────────────────────
def _safe_linregress():
    try:
        from scipy.stats import linregress
    except ModuleNotFoundError as e:
        raise ImportError(
            "SciPy required for regression — install with: pip install chessf[stats]"
        ) from e
    return linregress

# ─── complexity weights ────────────────────────────────────────────────────────
MOVE_COMPLEXITY = {
    "quiet": 1,
    "capture": 3,
    "check": 5,
    "castling": 4,
    "mate": 10,
}

# ─── per-game trace ─────────────────────────────────────────────────────────────
def analyse_game(pgn_text: str) -> xp.ndarray:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return xp.asarray([])

    board = game.board()
    total = 0
    trace: list[int] = []

    for move in game.mainline_moves():
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

        total += delta
        trace.append(total)

    return xp.asarray(trace)

# ─── prime/regression helpers ─────────────────────────────────────────────────
def _primes_up_to(n: int) -> Iterable[int]:
    for k in range(2, n + 1):
        if all(k % p for p in range(2, int(k**0.5) + 1)):
            yield k

def prime_slice(arr: xp.ndarray) -> Tuple[list[int], list[int]]:
    primes = list(_primes_up_to(len(arr)))
    return primes, [int(arr[i - 1]) for i in primes]

def regression_summary(arr: xp.ndarray) -> str:
    linregress = _safe_linregress()
    y = _to_numpy(arr)
    x = _to_numpy(xp.arange(1, len(y) + 1))
    slope, intercept, r, p, stderr = linregress(x, y)
    return (
        f"Slope: {slope:.5f}, Intercept: {intercept:.5f}, "
        f"R={r:.5f}, p={p:.3g}, stderr={stderr:.5f}"
    )

# ─── folder-level driver ───────────────────────────────────────────────────────
def analyse_folder(pgn_dir: Path, *, make_plots: bool = True) -> None:
    pgn_files = sorted(pgn_dir.glob("*.pgn"))
    if not pgn_files:
        rich.print(f"[yellow]No PGN files found in {pgn_dir}[/]")
        return

    out_dir = pgn_dir / "output"
    out_dir.mkdir(exist_ok=True)

    import pandas as pd
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None
        make_plots = False

    summary: list[str] = []

    for pgn_path in pgn_files:
        # ─── robust file read with UTF-8 then Latin-1 fallback ──────────────
        try:
            text = pgn_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = pgn_path.read_bytes()
            text = raw.decode("latin-1", errors="replace")

        trace = analyse_game(text)
        if trace.size == 0:
            summary.append(f"{pgn_path.name}: invalid PGN")
            continue

        df = pd.DataFrame(
            {"move": range(1, len(trace) + 1), "complexity": _to_numpy(trace)}
        )
        df.to_parquet(out_dir / f"{pgn_path.stem}.parquet")

        if make_plots and plt:
            plt.figure(figsize=(8, 5))
            plt.plot(df.move, df.complexity, marker="o")
            plt.title(pgn_path.stem)
            plt.xlabel("Move")
            plt.ylabel("Complexity")
            plt.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.savefig(out_dir / f"{pgn_path.stem}.png")
            plt.close()

        primes, prime_vals = prime_slice(trace)
        summary.append(
            f"{pgn_path.name}: moves={len(trace)}, final={int(trace[-1])}, "
            f"primes {primes[:5]}… → {prime_vals[:5]} | {regression_summary(trace)}"
        )

    (out_dir / "chess_summary.txt").write_text(
        "\n".join(summary), encoding="utf-8"
    )
    rich.print(f"[green]✔  Analysis complete. Results in {out_dir}[/]")
    
# ─── allow `python -m chessf.analysis` ─────────────────────────────────────────
if __name__ == "__main__":
    analyse_folder(Path(__file__).resolve().parents[3] / "data")
