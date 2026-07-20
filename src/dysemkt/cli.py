from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import ProcessedData
from .engine import TrainConfig, train
from .preprocess import preprocess_moocradar
from .text import HashTextEncoder, SentenceTransformerTextEncoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dysemkt")
    commands = parser.add_subparsers(dest="command", required=True)

    prep = commands.add_parser("preprocess", help="build a processed MOOCRadar dataset")
    prep.add_argument("--raw-dir", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--interaction-file", default="student-problem-fine.json")
    prep.add_argument("--encoder", choices=["hash", "sentence-transformer"], default="hash")
    prep.add_argument("--embedding-dim", type=int, default=256)
    prep.add_argument("--model-name", default="BAAI/bge-m3")
    prep.add_argument("--batch-size", type=int, default=32)
    prep.add_argument("--device", default=None)
    prep.add_argument("--min-user-interactions", type=int, default=2)
    prep.add_argument("--seed", type=int, default=42)

    fit = commands.add_parser("train", help="train and evaluate DySemKT")
    fit.add_argument("--data-dir", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--split", choices=["temporal", "cold"], default="temporal")
    fit.add_argument("--feature-mode", choices=["semantic", "id", "hybrid"], default="hybrid")
    fit.add_argument("--hidden-dim", type=int, default=128)
    fit.add_argument("--num-heads", type=int, default=4)
    fit.add_argument("--num-layers", type=int, default=2)
    fit.add_argument("--history-length", type=int, default=50)
    fit.add_argument("--dropout", type=float, default=0.1)
    fit.add_argument("--batch-size", type=int, default=256)
    fit.add_argument("--learning-rate", type=float, default=5e-4)
    fit.add_argument("--weight-decay", type=float, default=1e-4)
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--patience", type=int, default=5)
    fit.add_argument("--device", default="auto")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--num-workers", type=int, default=0)

    inspect = commands.add_parser("inspect", help="print processed dataset metadata")
    inspect.add_argument("--data-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "preprocess":
        encoder = (
            HashTextEncoder(args.embedding_dim)
            if args.encoder == "hash"
            else SentenceTransformerTextEncoder(args.model_name, args.batch_size, args.device)
        )
        result = preprocess_moocradar(
            args.raw_dir, args.output_dir, encoder, interaction_file=args.interaction_file,
            min_user_interactions=args.min_user_interactions, seed=args.seed,
        )
    elif args.command == "train":
        config = TrainConfig(**{name: getattr(args, name) for name in TrainConfig.__dataclass_fields__})
        result = train(args.data_dir, args.output_dir, config)
    else:
        result = ProcessedData(args.data_dir).metadata
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
