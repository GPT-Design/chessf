# Chessf

![CI](https://github.com/GPT-Design/chessf/actions/workflows/ci.yml/badge.svg)

Chessf is a command-line tool to analyze chess games in PGN format, computing board complexity metrics and generating visual plots and summaries.

## Features

* Compute complexity metrics per move based on legal move counts and custom weights.
* Analyze a single PGN: `chessf analyze path/to/game.pgn`
* Batch process all PGNs in a directory: `chessf batch data/`
* Outputs metrics as Parquet files and PNG plots in `<pgn_dir>/output`.
* Supports GPU acceleration via CuPy, with graceful fallback to NumPy.
* Extendable complexity formulas and visualizations.

## Installation

Install via pip:

```bash
pip install chessf
```

Or install from source with Poetry:

```bash
git clone https://github.com/GPT-Design/chessf.git
cd chessf
poetry install
```

## Usage

### Analyze a single PGN

```bash
chessf analyze data/Waitzkin.pgn --out metrics.parquet
```

### Batch process PGNs

```bash
chessf batch data/
```

This will produce:

```
data/output/
├── GameName.parquet
├── GameName.png
└── chess_summary.txt
```

## Roadmap

Future enhancements include:

* More advanced complexity metrics and weighting schemes.
* Interactive web dashboard for visualizing game complexity.
* JSON/CSV export options.
* Improved plot aesthetics with bold comparisons between players.

## Contributing

Pull requests are welcome! Please open an issue to discuss major changes before contributing.

## License

MIT License © GPT-Design

