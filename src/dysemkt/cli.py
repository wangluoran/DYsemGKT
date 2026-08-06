from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import ProcessedData
from .dygkt_engine import DyGKTTrainConfig, train_dygkt
from .engine import TrainConfig, train
from .preprocess import preprocess_moocradar
from .text import APITextEncoder, HashTextEncoder, SentenceTransformerTextEncoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dysemkt")
    commands = parser.add_subparsers(dest="command", required=True)

    prep = commands.add_parser("preprocess", help="build a processed MOOCRadar dataset")
    prep.add_argument("--raw-dir", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--interaction-file", default="student-problem-fine.json")
    prep.add_argument("--encoder", choices=["hash", "sentence-transformer", "api"], default="hash")
    prep.add_argument("--embedding-dim", type=int, default=256)
    prep.add_argument("--model-name", default="BAAI/bge-m3")
    prep.add_argument("--batch-size", type=int, default=32)
    prep.add_argument("--device", default=None)
    prep.add_argument("--min-user-interactions", type=int, default=2)
    prep.add_argument("--seed", type=int, default=42)

    fit = commands.add_parser("train", help="train and evaluate a knowledge tracing model")
    fit.add_argument("--model", choices=["dysemkt", "dygkt"], default="dysemkt",
                     help="model to train (default: dysemkt)")
    fit.add_argument("--config", type=Path, help="optional JSON train configuration")
    fit.add_argument("--data-dir", type=Path)
    fit.add_argument("--output-dir", type=Path)
    fit.add_argument("--split", choices=["temporal", "cold"], default="cold",
                     help="data split (default: cold)")
    fit.add_argument("--feature-mode", choices=["semantic", "id", "hybrid"])
    fit.add_argument("--d-model", type=int)
    fit.add_argument("--history-length", type=int)
    fit.add_argument("--dropout", type=float)
    fit.add_argument("--batch-size", type=int)
    fit.add_argument("--learning-rate", type=float)
    fit.add_argument("--weight-decay", type=float)
    fit.add_argument("--epochs", type=int)
    fit.add_argument("--patience", type=int)
    fit.add_argument("--device")
    fit.add_argument("--seed", type=int)
    fit.add_argument("--num-workers", type=int)
    # DyGKT-specific
    fit.add_argument("--num-neighbors", type=int)
    fit.add_argument("--time-dim", type=int)
    fit.add_argument("--node-dim", type=int)
    fit.add_argument("--ablation", type=str)

    inspect = commands.add_parser("inspect", help="print processed dataset metadata")
    inspect.add_argument("--data-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "preprocess":
        if args.encoder == "hash":
            encoder = HashTextEncoder(args.embedding_dim)
        elif args.encoder == "sentence-transformer":
            encoder = SentenceTransformerTextEncoder(args.model_name, args.batch_size, args.device)
        else:
            encoder = APITextEncoder.from_env(batch_size=args.batch_size)
        result = preprocess_moocradar(
            args.raw_dir, args.output_dir, encoder, interaction_file=args.interaction_file,
            min_user_interactions=args.min_user_interactions, seed=args.seed,
        )
    elif args.command == "train":
        values = {}
        if args.config is not None:
            with args.config.open("r", encoding="utf-8") as handle:
                values = json.load(handle)
            if not isinstance(values, dict):
                parser.error("--config must point to a JSON object")

        data_dir = args.data_dir or values.pop("data_dir", None)
        output_dir = args.output_dir or values.pop("output_dir", None)
        if data_dir is None or output_dir is None:
            parser.error("train requires --data-dir and --output-dir, or a --config containing data_dir and output_dir")

        if args.model == "dygkt":
            defaults = DyGKTTrainConfig()
            train_values = {name: getattr(defaults, name) for name in DyGKTTrainConfig.__dataclass_fields__}
            for name in tuple(values):
                if name not in train_values:
                    parser.error(f"unknown dygkt train config key: {name}")
                train_values[name] = values.pop(name)
            for name in DyGKTTrainConfig.__dataclass_fields__:
                value = getattr(args, name)
                if value is not None:
                    train_values[name] = value
            config = DyGKTTrainConfig(**train_values)
            result = train_dygkt(Path(data_dir), Path(output_dir), config)
        else:
            defaults = TrainConfig()
            train_values = {name: getattr(defaults, name) for name in TrainConfig.__dataclass_fields__}
            for name in tuple(values):
                if name not in train_values:
                    parser.error(f"unknown train config key: {name}")
                train_values[name] = values.pop(name)
            for name in TrainConfig.__dataclass_fields__:
                value = getattr(args, name)
                if value is not None:
                    train_values[name] = value
            config = TrainConfig(**train_values)
            result = train(Path(data_dir), Path(output_dir), config)
    else:
        result = ProcessedData(args.data_dir).metadata
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
