# Chessf

A lightweight toolkit for analyzing chess games in PGN format.

## Features

- Computes a simple complexity metric for each move in a game.
- Optionally accelerates calculations with GPU via CuPy.
- Command line interface for single games or entire folders.

## Quickstart

Install the package from PyPI:

```bash
pip install chessf
```

For GPU support use:

```bash
pip install chessf[gpu]
```

## Usage

### Analyze one PGN file

```bash
chessf analyze path/to/game.pgn
```

This writes a `metrics.parquet` file with per-move complexity scores.

### Analyze a folder of PGN files

```bash
python -m chessf.analysis path/to/pgn_folder
```

All results are saved inside an `output/` directory under the folder.

---

Drop PGN files into the `data/` directory of this repository to experiment
while developing.

