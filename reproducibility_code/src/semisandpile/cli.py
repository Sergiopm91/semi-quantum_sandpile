from __future__ import annotations

import argparse
import json

from .config import load_config
from .io import save_result
from .simulation import simulate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classical / semi-quantum BTW sandpile simulator")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a simulation from YAML configuration")
    run.add_argument("--config", required=True, help="Path to YAML configuration")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        cfg = load_config(args.config)
        result = simulate(cfg)
        out = save_result(result)
        print(json.dumps(result.summary(), indent=2, ensure_ascii=False))
        print(f"Results written to: {out.resolve()}")


if __name__ == "__main__":
    main()
